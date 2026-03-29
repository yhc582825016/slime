from __future__ import annotations

import ast
import json
import os
import re
from typing import Any

from slime.utils.types import Sample

BOX_PREFIXES = ("\\boxed{", "\\box{", "boxed{", "box{")
DEBUG_REWARD = os.getenv("RECALL_AGENT_DEBUG_REWARD", "1") == "1"
DEBUG_REWARD_EVERY = max(1, int(os.getenv("RECALL_AGENT_DEBUG_REWARD_EVERY", "1")))
DEBUG_RESPONSE_MAX_CHARS = max(200, int(os.getenv("RECALL_AGENT_DEBUG_RESPONSE_MAX_CHARS", "4000")))
_DEBUG_COUNTER = 0


def _post_think_text(text: str) -> str:
    if "</think>" in text:
        return text.rsplit("</think>", 1)[-1]
    return text


def _strip_tool_markup(text: str) -> str:
    text = re.sub(r"<tool_call>.*?</tool_call>", " ", text, flags=re.DOTALL)
    text = re.sub(r"<tool_response>.*?</tool_response>", " ", text, flags=re.DOTALL)
    return re.sub(r"\s+", " ", text).strip()


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


def _extract_last_boxed(text: str) -> str | None:
    last_match: tuple[int, str] | None = None
    for prefix in BOX_PREFIXES:
        start = 0
        while True:
            idx = text.find(prefix, start)
            if idx < 0:
                break
            content = _extract_balanced_content(text, idx + len(prefix) - 1)
            if content is not None:
                last_match = (idx, content)
            start = idx + 1
    return last_match[1].strip() if last_match is not None else None


def _fallback_answer(text: str) -> str:
    stripped = _strip_tool_markup(text)
    if not stripped:
        return ""
    for line in reversed(stripped.splitlines()):
        candidate = line.strip()
        if candidate:
            return candidate
    return stripped


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


def _extract_prediction(response: str) -> str:
    cleaned = _post_think_text(response or "")
    boxed = _extract_last_boxed(cleaned)
    if boxed is not None:
        return boxed
    return _fallback_answer(cleaned)


def _load_targets(label: Any) -> list[str]:
    if isinstance(label, dict):
        ground_truth = label.get("ground_truth")
    else:
        ground_truth = label
    if ground_truth is None:
        return []
    if hasattr(ground_truth, "tolist"):
        ground_truth = ground_truth.tolist()
    if isinstance(ground_truth, (list, tuple)):
        return [str(item) for item in ground_truth if item is not None]
    return [str(ground_truth)]


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
    prediction = _extract_prediction(sample.response or "")
    pred_value = _canonicalize(prediction)
    targets = _load_targets(sample.label)
    reward = 0.0
    for target in targets:
        if pred_value == _canonicalize(target):
            reward = 1.0
            break

    if DEBUG_REWARD:
        _DEBUG_COUNTER += 1
        if _DEBUG_COUNTER % DEBUG_REWARD_EVERY == 0:
            log_payload = {
                "idx": _DEBUG_COUNTER,
                "sample_index": metadata.get("index"),
                "question": metadata.get("question"),
                "input": _shorten_text(_prompt_to_text(sample.prompt), DEBUG_RESPONSE_MAX_CHARS),
                "output": _shorten_text(sample.response or "", DEBUG_RESPONSE_MAX_CHARS),
                "prediction": prediction,
                "ground_truth": targets,
                "reward": reward,
            }
            print(f"[recall_agent_reward_debug] {json.dumps(log_payload, ensure_ascii=False)}", flush=True)
    return reward
