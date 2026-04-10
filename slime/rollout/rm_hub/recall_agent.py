from __future__ import annotations

import ast
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

from slime.rollout.rm_hub.math_utils import extract_answer, grade_answer_mathd, grade_answer_sympy
from slime.utils.types import Sample

BOX_PREFIXES = ("\\boxed", "\\fbox", "\\box", "boxed{", "box{")
SUBSTITUTIONS = [
    ("an ", ""),
    ("a ", ""),
    (".$", "$"),
    ("\\$", ""),
    (r"\ ", ""),
    (" ", ""),
    ("mbox", "text"),
    (",\\text{and}", ","),
    ("\\text{and}", ","),
    ("\\text{m}", "\\text{}"),
]
REMOVED_EXPRESSIONS = [
    "square",
    "ways",
    "integers",
    "dollars",
    "mph",
    "inches",
    "hours",
    "km",
    "units",
    "\\ldots",
    "sue",
    "points",
    "feet",
    "minutes",
    "digits",
    "cents",
    "degrees",
    "cm",
    "gm",
    "pounds",
    "meters",
    "meals",
    "edges",
    "students",
    "childrentickets",
    "multiples",
    "\\text{s}",
    "\\text{.}",
    "\\text{\ns}",
    "\\text{}^2",
    "\\text{}^3",
    "\\text{\n}",
    "\\text{}",
    r"\mathrm{th}",
    r"^\circ",
    r"^{\circ}",
    r"\;",
    r",\!",
    "{,}",
    '"',
    "\\dots",
    "<|im_end|>",
    "<|endoftext|>",
]
DEBUG_REWARD = os.getenv("RECALL_AGENT_DEBUG_REWARD", "1") == "1"
DEBUG_REWARD_EVERY = max(1, int(os.getenv("RECALL_AGENT_DEBUG_REWARD_EVERY", "1")))
DEBUG_RESPONSE_MAX_CHARS = max(200, int(os.getenv("RECALL_AGENT_DEBUG_RESPONSE_MAX_CHARS", "4000")))
_DEBUG_COUNTER = 0

_BFCL_CHECKER_IMPORTED = False
_BFCL_MULTI_TURN_CHECKER = None
_BFCL_MULTI_TURN_IRRELEVANCE_CHECKER = None


def _strip_special_tokens(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"<\|im_start\|>.*?(?=\n)", " ", text)
    text = re.sub(r"<\|im_end\|>", " ", text)
    text = re.sub(r"<\|endoftext\|>", " ", text)
    text = re.sub(r"</s>", " ", text)
    return text


def _post_think_text(text: str) -> str:
    if "</think>" in text:
        return text.rsplit("</think>", 1)[-1]
    return text


def _strip_tool_markup(text: str) -> str:
    text = re.sub(r"<tool_call>.*?</tool_call>", " ", text, flags=re.DOTALL)
    text = re.sub(r"<tool_response>.*?</tool_response>", " ", text, flags=re.DOTALL)
    return text


def _extract_balanced_content(text: str, open_brace_idx: int) -> str | None:
    depth = 0
    for idx in range(open_brace_idx, len(text)):
        ch = text[idx]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[open_brace_idx + 1 : idx]
    return None


def _last_boxed_only_string(text: str) -> str | None:
    # Align behavior with math reward utils: take only the last complete boxed span.
    best_idx = -1
    for prefix in BOX_PREFIXES:
        idx = text.rfind(prefix)
        if idx > best_idx:
            best_idx = idx
    if best_idx < 0:
        return None

    right_brace_idx = None
    num_left_braces_open = 0
    i = best_idx
    while i < len(text):
        if text[i] == "{":
            num_left_braces_open += 1
        elif text[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx is None:
        return None
    return text[best_idx : right_brace_idx + 1]


def _remove_boxed(boxed: str | None) -> str | None:
    if boxed is None:
        return None
    open_brace_idx = boxed.find("{")
    if open_brace_idx < 0:
        return None
    content = _extract_balanced_content(boxed, open_brace_idx)
    if content is None:
        return None
    return content.strip()


def _extract_last_boxed(text: str) -> str | None:
    return _remove_boxed(_last_boxed_only_string(text))


def _fallback_answer(text: str) -> str:
    stripped = _strip_tool_markup(_strip_special_tokens(text))
    if not stripped:
        return ""
    for line in reversed(stripped.splitlines()):
        candidate = line.strip()
        if not candidate:
            continue
        if candidate.startswith("<|") and candidate.endswith("|>"):
            continue
        if candidate in {"assistant", "user", "system"}:
            continue
        if candidate:
            return candidate
    return " ".join(stripped.strip().split())


def _normalize_string(value: str) -> str:
    value = " ".join(value.strip().split())
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        value = value[1:-1].strip()
    return value.casefold()


def _canonicalize(value: Any):
    if isinstance(value, str):
        value = value.strip()
        try:
            parsed = ast.literal_eval(value)
        except Exception:
            return _normalize_string(value)
        return _canonicalize(parsed)
    if isinstance(value, list):
        return tuple(_canonicalize(item) for item in value)
    if isinstance(value, tuple):
        return tuple(_canonicalize(item) for item in value)
    if isinstance(value, dict):
        return tuple(sorted((_normalize_string(str(k)), _canonicalize(v)) for k, v in value.items()))
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, (int, float)):
        return value
    return _normalize_string(str(value))


def _extract_single_text_value(value: Any) -> str | None:
    """Extract a single text answer from plain/scalar or singleton containers."""
    if isinstance(value, str):
        raw = value.strip()
        try:
            parsed = ast.literal_eval(raw)
        except Exception:
            return raw
        return _extract_single_text_value(parsed)
    if isinstance(value, (list, tuple)):
        if len(value) != 1:
            return None
        return _extract_single_text_value(value[0])
    if isinstance(value, (dict, set)):
        return None
    return str(value)


def _normalize_relaxed_text(value: str) -> str:
    # Keep semantic order, but treat common separators as equivalent.
    text = _normalize_string(value)
    text = re.sub(r"\s*(?:,|;|\|)\s*", " | ", text)
    text = re.sub(r"\s+-\s+", " | ", text)
    return " ".join(text.split())


def _normalize_final_answer(value: str) -> str:
    final_answer = str(value)
    final_answer = final_answer.split("=")[-1]

    for before, after in SUBSTITUTIONS:
        final_answer = final_answer.replace(before, after)
    for expr in REMOVED_EXPRESSIONS:
        final_answer = final_answer.replace(expr, "")

    final_answer = re.sub(r"(.*?)(\$)(.*?)(\$)(.*)", "$\\3$", final_answer)
    final_answer = re.sub(r"(\\text\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\textbf\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\overline\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\boxed\{)(.*)(\})", "\\2", final_answer)

    final_answer = re.sub(r"(frac)([^{])(.)", "frac{\\2}{\\3}", final_answer)
    final_answer = re.sub(r"(sqrt)([^{])", "sqrt{\\2}", final_answer)
    final_answer = final_answer.replace("$", "")

    if final_answer.replace(",", "").isdigit():
        final_answer = final_answer.replace(",", "")
    return final_answer.strip()


def _safe_normalize_final_answer(value: str) -> str:
    try:
        return _normalize_final_answer(value)
    except Exception:
        return str(value).strip()


def _relaxed_match(prediction: str, target: str) -> bool:
    pred_text = _extract_single_text_value(prediction)
    target_text = _extract_single_text_value(target)
    if pred_text is None or target_text is None:
        return False
    return _normalize_relaxed_text(pred_text) == _normalize_relaxed_text(target_text)


def _math_equivalent_match(prediction: str, target: str) -> bool:
    given_answer = (prediction or "").strip()
    if not given_answer:
        return False

    ground_truth = str(target) if target is not None else ""
    if not ground_truth:
        return False

    # Keep behavior aligned with math utils: compare on boxed payload when present.
    if "\\boxed" in ground_truth:
        extracted = extract_answer(ground_truth)
        if extracted:
            ground_truth = extracted

    try:
        return grade_answer_mathd(given_answer, ground_truth) or grade_answer_sympy(given_answer, ground_truth)
    except Exception:
        return False


def _extract_prediction(response: str) -> str:
    cleaned = _strip_special_tokens(_post_think_text(response or ""))
    boxed = _extract_last_boxed(cleaned)
    if boxed is not None:
        return boxed
    return _fallback_answer(cleaned)


def _target_candidates(target: str) -> list[str]:
    cleaned = _strip_special_tokens(_post_think_text(str(target)))
    candidates: list[str] = [str(target)]
    boxed = _extract_last_boxed(cleaned)
    if boxed is not None and boxed not in candidates:
        candidates.append(boxed)
    return candidates


def _load_targets(label: Any) -> list[str]:
    def _expand_targets(value: Any) -> list[str]:
        if value is None:
            return []
        if hasattr(value, "tolist"):
            value = value.tolist()
        if isinstance(value, (list, tuple, set)):
            expanded: list[str] = []
            for item in value:
                expanded.extend(_expand_targets(item))
            return expanded
        if isinstance(value, str):
            raw = value.strip()
            if not raw:
                return []
            # Some datasets serialize answers as strings like '["John Doe"]'.
            parsed: Any | None = None
            for parser in (json.loads, ast.literal_eval):
                try:
                    parsed = parser(raw)
                    break
                except Exception:
                    continue
            if parsed is not None and not isinstance(parsed, str):
                return _expand_targets(parsed)
            return [raw]
        return [str(value)]

    if isinstance(label, dict):
        ground_truth = label.get("ground_truth")
    else:
        ground_truth = label
    return _expand_targets(ground_truth)


def _augment_multi_part_targets(targets: list[str]) -> list[str]:
    """Labels often store an ordered answer as list[str]; models box it as one comma-separated line.

    Without this, matching compares the full prediction against each part only and never succeeds.
    """
    if len(targets) <= 1:
        return list(targets)
    out = list(targets)
    seen = set(targets)
    for sep in (", ", "; ", " | ", ","," "):
        merged = sep.join(targets)
        if merged not in seen:
            seen.add(merged)
            out.append(merged)
    return out


def _prediction_matches_any_target(prediction: str, targets: list[str]) -> bool:
    normalized_prediction = _safe_normalize_final_answer(prediction)
    pred_value = _canonicalize(normalized_prediction)
    raw_pred_value = _canonicalize(prediction)
    for target in targets:
        for candidate in _target_candidates(target):
            normalized_candidate = _safe_normalize_final_answer(candidate)
            if (
                _relaxed_match(prediction, candidate)
                or raw_pred_value == _canonicalize(candidate)
                or pred_value == _canonicalize(candidate)
                or _math_equivalent_match(prediction, candidate)
                or pred_value == _canonicalize(normalized_candidate)
                or _relaxed_match(normalized_prediction, normalized_candidate)
                or _math_equivalent_match(normalized_prediction, normalized_candidate)
            ):
                return True
    return False


def _prompt_to_text(prompt: Any) -> str:
    if isinstance(prompt, str):
        return prompt
    if isinstance(prompt, list):
        chunks = []
        for message in prompt:
            if not isinstance(message, dict):
                chunks.append(str(message))
                continue
            role = message.get("role", "unknown")
            content = message.get("content", "")
            if isinstance(content, list):
                text_parts = []
                for item in content:
                    if isinstance(item, dict):
                        text_parts.append(str(item.get("text", item)))
                    else:
                        text_parts.append(str(item))
                content = "\n".join(text_parts)
            chunks.append(f"[{role}] {content}")
        return "\n".join(chunks)
    return str(prompt)


def _ensure_bfcl_checker_loaded(metadata: dict[str, Any]) -> None:
    global _BFCL_CHECKER_IMPORTED, _BFCL_MULTI_TURN_CHECKER, _BFCL_MULTI_TURN_IRRELEVANCE_CHECKER
    if _BFCL_CHECKER_IMPORTED:
        return

    bfcl_root = metadata.get("bfcl_root") or "/dev/shm/ye/gorilla/berkeley-function-call-leaderboard"
    bfcl_root = str(bfcl_root)
    if bfcl_root not in sys.path:
        sys.path.insert(0, bfcl_root)

    from bfcl_eval.eval_checker.multi_turn_eval.multi_turn_checker import (  # type: ignore
        multi_turn_checker,
        multi_turn_irrelevance_checker,
    )

    _BFCL_MULTI_TURN_CHECKER = multi_turn_checker
    _BFCL_MULTI_TURN_IRRELEVANCE_CHECKER = multi_turn_irrelevance_checker
    _BFCL_CHECKER_IMPORTED = True


def _tool_call_to_python_call(name: str, arguments: dict[str, Any]) -> str:
    if not arguments:
        return f"{name}()"
    joined = ", ".join(f"{key}={repr(value)}" for key, value in arguments.items())
    return f"{name}({joined})"


def _extract_bfcl_tool_calls_from_text(text: str) -> list[str]:
    tool_calls: list[str] = []
    for block in re.findall(r"<tool_call>\s*(.*?)\s*</tool_call>", text or "", flags=re.DOTALL):
        decoder_candidates: list[Any] = []
        block = block.strip()
        if not block:
            continue
        decoder_candidates.append(block)

        balanced_candidates: list[str] = []
        depth = 0
        start = None
        in_string = False
        escaped = False
        for idx, ch in enumerate(block):
            if ch == "\\" and not escaped:
                escaped = True
                continue
            if ch == '"' and not escaped:
                in_string = not in_string
            if not in_string:
                if ch == "{":
                    if start is None:
                        start = idx
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0 and start is not None:
                        balanced_candidates.append(block[start : idx + 1])
                        start = None
            escaped = False
        decoder_candidates.extend(balanced_candidates)

        parsed_payload = None
        for candidate in decoder_candidates:
            try:
                parsed_payload = json.loads(candidate)
                break
            except Exception:
                continue
        if not isinstance(parsed_payload, dict):
            continue

        name = parsed_payload.get("name")
        arguments = parsed_payload.get("arguments", {})
        function_wrapper = parsed_payload.get("function")
        if (not name) and isinstance(function_wrapper, dict):
            name = function_wrapper.get("name")
            arguments = function_wrapper.get("arguments", {})
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except Exception:
                arguments = {}
        if not isinstance(name, str) or not name.strip():
            continue
        if not isinstance(arguments, dict):
            arguments = {}
        tool_calls.append(_tool_call_to_python_call(name.strip(), arguments))
    return tool_calls


def _build_bfcl_model_result_trace(metadata: dict[str, Any]) -> list[list[list[str]]]:
    trace = metadata.get("bfcl_rollout_trace") or {}
    turns = trace.get("turns") or []
    model_result: list[list[list[str]]] = []
    for turn in turns:
        if not isinstance(turn, list):
            model_result.append([])
            continue
        step_results: list[list[str]] = []
        for step in turn:
            if not isinstance(step, dict):
                continue
            response_text = str(step.get("response_text") or "")
            decoded_calls = _extract_bfcl_tool_calls_from_text(response_text)
            if decoded_calls:
                step_results.append(decoded_calls)
        model_result.append(step_results)
    return model_result


def _bfcl_multi_turn_reward(sample: Sample, metadata: dict[str, Any]) -> float:
    _ensure_bfcl_checker_loaded(metadata)

    ground_truth_turns = metadata.get("bfcl_ground_truth_turns") or []
    test_entry = metadata.get("bfcl_test_entry") or {}
    test_category = metadata.get("bfcl_category") or str(test_entry.get("id") or "").rsplit("_", 1)[0]
    model_result_trace = _build_bfcl_model_result_trace(metadata)

    if len(model_result_trace) < len(ground_truth_turns):
        model_result_trace.extend([[] for _ in range(len(ground_truth_turns) - len(model_result_trace))])
    elif len(model_result_trace) > len(ground_truth_turns):
        model_result_trace = model_result_trace[: len(ground_truth_turns)]

    valid = True
    checker_result = _BFCL_MULTI_TURN_CHECKER(
        model_result_trace,
        ground_truth_turns,
        test_entry,
        test_category,
        "slime_bfcl_eval",
    )
    if not checker_result.get("valid", False):
        valid = False

    irrelevance_result = _BFCL_MULTI_TURN_IRRELEVANCE_CHECKER(
        model_result_trace,
        ground_truth_turns,
    )
    if not irrelevance_result.get("valid", False):
        valid = False

    if DEBUG_REWARD:
        print(
            "[bfcl_multi_turn_reward_debug] "
            + json.dumps(
                {
                    "sample_index": metadata.get("index"),
                    "test_category": test_category,
                    "valid": valid,
                    "model_result_trace": model_result_trace,
                    "ground_truth_turns": ground_truth_turns,
                    "checker_result": checker_result,
                    "irrelevance_result": irrelevance_result,
                },
                ensure_ascii=False,
                default=str,
            ),
            flush=True,
        )

    return 1.0 if valid else 0.0


def _shorten_text(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    head_keep = max_chars // 2
    tail_keep = max_chars - head_keep
    return f"{text[:head_keep]}\n...<truncated>...\n{text[-tail_keep:]}"


async def custom_rm(args, sample: Sample, **kwargs) -> float:
    del args, kwargs
    global _DEBUG_COUNTER
    if not isinstance(sample, Sample):
        raise TypeError("sample must be an instance of slime.utils.types.Sample")

    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    if metadata.get("bfcl_eval_mode") or metadata.get("data_source") == "bfcl_multi_turn":
        return _bfcl_multi_turn_reward(sample, metadata)

    cleaned_response = _strip_special_tokens(_post_think_text(sample.response or ""))
    # sample.response in interaction rollout includes appended observation text
    # (e.g., <tool_response>...), so extract answers only from assistant text.
    answer_scope = _strip_tool_markup(cleaned_response)
    boxed_prediction = _extract_last_boxed(answer_scope)
    prediction = boxed_prediction if boxed_prediction is not None else _fallback_answer(answer_scope)
    targets = _augment_multi_part_targets(_load_targets(sample.label))

    # Prefer boxed payload: if it matches ground truth, full credit immediately.
    if boxed_prediction is not None and _prediction_matches_any_target(boxed_prediction, targets):
        reward = 1.0
    else:
        reward = 0.2 if boxed_prediction is not None else 0.0
        if _prediction_matches_any_target(prediction, targets):
            reward += 0.8

    normalized_prediction = _safe_normalize_final_answer(prediction)

    if DEBUG_REWARD:
        _DEBUG_COUNTER += 1
        if _DEBUG_COUNTER % DEBUG_REWARD_EVERY == 0:
            log_payload = {
                "idx": _DEBUG_COUNTER,
                "sample_index": metadata.get("index"),
                "question": metadata.get("question"),
                "stop_reason": metadata.get("stop_reason"),
                "stop_turn": metadata.get("stop_turn"),
                "stop_finish_type": metadata.get("stop_finish_type"),
                "stop_budget": metadata.get("stop_budget"),
                "stop_has_complete_tool_call": metadata.get("stop_has_complete_tool_call"),
                "stop_status": metadata.get("stop_status"),
                "input": _shorten_text(_prompt_to_text(sample.prompt), DEBUG_RESPONSE_MAX_CHARS),
                "output": _shorten_text(sample.response or "", DEBUG_RESPONSE_MAX_CHARS),
                "prediction": prediction,
                "normalized_prediction": normalized_prediction,
                "ground_truth": targets,
                "reward": reward,
            }
            print(
                f"[recall_agent_reward_debug] {json.dumps(log_payload, ensure_ascii=False, default=str)}",
                flush=True,
            )
    return reward
