import json
import os
import re
from typing import Any

from slime.utils.types import Sample


TOOL_CALL_JSON_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
TOOL_CALL_BLOCK_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
FUNCTION_BLOCK_RE = re.compile(r"(<function\s*=\s*.*?</function>)", re.DOTALL)
FUNCTION_TAG_RE = re.compile(r"<function\s*=\s*([a-zA-Z0-9_\-\.]+)\s*>", re.DOTALL)
PARAM_TAG_RE = re.compile(r"<parameter\s*=\s*([a-zA-Z0-9_\-\.]+)\s*>(.*?)</parameter>", re.DOTALL)
JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)

DEBUG_REWARD = os.getenv("SWE_AGENTIC_DEBUG_REWARD", "0") == "1"
DEBUG_REWARD_EVERY = max(1, int(os.getenv("SWE_AGENTIC_DEBUG_REWARD_EVERY", "1")))
DEBUG_RESPONSE_MAX_CHARS = max(200, int(os.getenv("SWE_AGENTIC_DEBUG_RESPONSE_MAX_CHARS", "2000")))
_DEBUG_COUNTER = 0


def _safe_json_loads(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return value
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except Exception:
        return value


def _canonicalize(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _canonicalize(v) for k, v in sorted(obj.items(), key=lambda x: str(x[0]))}
    if isinstance(obj, list):
        return [_canonicalize(x) for x in obj]
    if isinstance(obj, str):
        return obj.strip()
    return obj


def _post_think_text(response: str) -> str:
    """Use text after the final </think> when present."""
    if "</think>" in response:
        return response.rsplit("</think>", 1)[-1]
    return response


def _extract_from_function_parameter_markup(text: str) -> dict[str, Any] | None:
    match_fn = FUNCTION_TAG_RE.search(text)
    if not match_fn:
        return None

    fn_name = match_fn.group(1).strip()
    args: dict[str, Any] = {}
    for match in PARAM_TAG_RE.finditer(text):
        key = match.group(1).strip()
        val = match.group(2).strip()
        val = _safe_json_loads(val)
        args[key] = val

    return {"name": fn_name, "arguments": args}


def _extract_prediction_action(response: str) -> dict[str, Any] | None:
    if not response:
        return None
    response = _post_think_text(response)

    # 1) Strict JSON in <tool_call>...</tool_call>
    match = TOOL_CALL_JSON_RE.search(response)
    if match:
        payload = _safe_json_loads(match.group(1))
        if isinstance(payload, dict):
            return payload

    # 2) XML-like function/parameter tool call format:
    # <tool_call><function=foo>...<parameter=bar>v</parameter></function></tool_call>
    blocks = TOOL_CALL_BLOCK_RE.findall(response)
    for block in reversed(blocks):
        payload = _extract_from_function_parameter_markup(block)
        if payload is not None:
            return payload

    # 2.1) Fallback for models that emit raw <function=...>...</function>
    # without wrapping it inside <tool_call>...</tool_call>.
    fn_blocks = FUNCTION_BLOCK_RE.findall(response)
    for block in reversed(fn_blocks):
        payload = _extract_from_function_parameter_markup(block)
        if payload is not None:
            return payload

    # 3) Fallback: try to parse last JSON object in plain text response.
    all_objs = JSON_OBJECT_RE.findall(response)
    for raw in reversed(all_objs):
        payload = _safe_json_loads(raw)
        if isinstance(payload, dict) and "name" in payload:
            return payload
    return None


def _normalize_action(action: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(action, dict):
        return None
    name = action.get("name")
    arguments = _safe_json_loads(action.get("arguments", {}))
    if isinstance(arguments, str):
        # keep raw if it's not valid json; still comparable as string
        arguments = arguments.strip()
    return {"name": name, "arguments": _canonicalize(arguments)}


def _shorten_text(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    head_keep = max_chars // 2
    tail_keep = max_chars - head_keep
    return f"{text[:head_keep]}\n...<truncated>...\n{text[-tail_keep:]}"


def _compare_actions(pred: dict[str, Any] | None, exp: dict[str, Any] | None) -> tuple[bool, bool]:
    pred_norm = _normalize_action(pred)
    exp_norm = _normalize_action(exp)
    if not pred_norm or not exp_norm:
        return False, False

    name_ok = pred_norm.get("name") == exp_norm.get("name")
    args_ok = pred_norm.get("arguments") == exp_norm.get("arguments")
    return name_ok, args_ok


async def reward_func(args, sample: Sample, **kwargs) -> float:
    del args, kwargs
    global _DEBUG_COUNTER
    if not isinstance(sample, Sample):
        raise TypeError("sample must be an instance of slime.utils.types.Sample")

    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    expected_action = metadata.get("expected_action")

    pred_action = _extract_prediction_action(sample.response or "")
    name_ok, args_ok = _compare_actions(pred_action, expected_action)

    # Strict reward:
    # - 1.0 full match (tool name + arguments)
    # - 0.2 only tool name matches (helps stabilize early training)
    # - 0.0 otherwise
    reward = 0.0
    if name_ok and args_ok:
        reward = 1.0
    elif name_ok:
        reward = 0.2

    if DEBUG_REWARD:
        _DEBUG_COUNTER += 1
        if _DEBUG_COUNTER % DEBUG_REWARD_EVERY == 0:
            debug_payload = {
                "idx": _DEBUG_COUNTER,
                "reward": reward,
                "name_ok": name_ok,
                "args_ok": args_ok,
                "expected_action": expected_action,
                "pred_action": pred_action,
                "response": _shorten_text(sample.response or "", DEBUG_RESPONSE_MAX_CHARS),
            }
            print(f"[swe_agentic_reward_debug] {json.dumps(debug_payload, ensure_ascii=False)}", flush=True)

    return reward

