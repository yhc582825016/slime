from __future__ import annotations

import json
import logging
import re
import traceback
from copy import deepcopy
from typing import Any

try:
    import orjson  # type: ignore
except Exception:  # pragma: no cover
    orjson = None

from .base_env import BaseInteractionEnv
from slime.utils.types import Sample

logger = logging.getLogger(__name__)

TOOL_CALL_JSON_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
TOOL_CALL_BLOCK_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
FUNCTION_BLOCK_RE = re.compile(r"(<function\s*=\s*.*?</function>)", re.DOTALL)
FUNCTION_TAG_RE = re.compile(r"<function\s*=\s*([a-zA-Z0-9_\-\.]+)\s*>", re.DOTALL)
PARAM_TAG_RE = re.compile(r"<parameter\s*=\s*([a-zA-Z0-9_\-\.]+)\s*>(.*?)</parameter>", re.DOTALL)
EOS_MARKERS = ("<|im_end|>", "<|endoftext|>", "</s>")


def _json_loads(value: str) -> dict[str, Any]:
    loader = orjson.loads if orjson is not None else json.loads
    return loader(value)


def _json_dumps(value: Any) -> str:
    if orjson is not None:
        return orjson.dumps(value, option=orjson.OPT_NON_STR_KEYS).decode("utf-8")
    return json.dumps(value, ensure_ascii=False, default=str)


def _extract_balanced_json(text: str, start: int) -> str | None:
    depth = 0
    in_string = False
    escaped = False
    for idx in range(start, len(text)):
        ch = text[idx]
        if ch == "\\" and not escaped:
            escaped = True
            continue
        if ch == '"' and not escaped:
            in_string = not in_string
        if not in_string:
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[start : idx + 1]
        escaped = False
    return None


def _safe_json_loads(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return value
    if not isinstance(value, str):
        return value
    try:
        return _json_loads(value)
    except Exception:
        return value


def _extract_from_function_parameter_markup(text: str) -> dict[str, Any] | None:
    match_fn = FUNCTION_TAG_RE.search(text)
    if not match_fn:
        return None

    fn_name = match_fn.group(1).strip()
    arguments: dict[str, Any] = {}
    for match in PARAM_TAG_RE.finditer(text):
        key = match.group(1).strip()
        value = _safe_json_loads(match.group(2).strip())
        arguments[key] = value
    return {"name": fn_name, "arguments": arguments}


def _normalize_tool_payload(payload: Any) -> dict[str, Any] | None:
    if not isinstance(payload, dict):
        return None

    name = payload.get("name")
    arguments = payload.get("arguments")

    # Some models emit OpenAI-style function wrapper:
    # {"function": {"name": "...", "arguments": {...}}}
    fn_payload = payload.get("function")
    if (not name) and isinstance(fn_payload, dict):
        name = fn_payload.get("name")
        arguments = fn_payload.get("arguments")

    if not name:
        return None
    if isinstance(arguments, str):
        arguments = _safe_json_loads(arguments)
    if not isinstance(arguments, dict):
        arguments = {}
    return {"name": str(name).strip(), "arguments": arguments}


def _extract_json_candidates(text: str) -> list[str]:
    candidates: list[str] = []
    for idx, ch in enumerate(text):
        if ch != "{":
            continue
        candidate = _extract_balanced_json(text, idx)
        if candidate:
            candidates.append(candidate.strip())
    return candidates


class RecallAgentEnv(BaseInteractionEnv):
    def __init__(self, *, env_code: str, tool_schemas: list[dict[str, Any]], max_turns: int | None = None):
        self.env_code = env_code
        self.tool_schemas = tool_schemas
        self.max_turns = max_turns
        self.turn = 0
        self.runtime_env: dict[str, Any] = {}
        self.init_error: str | None = None
        self.supported_tools = {
            tool.get("function", {}).get("name")
            for tool in tool_schemas
            if isinstance(tool, dict) and tool.get("function", {}).get("name")
        }

    def reset(self):
        self.turn = 0
        self.runtime_env = {}
        self.init_error = None
        if self.env_code.strip():
            try:
                exec(self.env_code, self.runtime_env, self.runtime_env)
            except Exception as exc:
                self.init_error = f"{type(exc).__name__}: {exc}"
                logger.warning("Failed to initialize recall-agent env: %s", self.init_error)
                logger.debug("Env init traceback:\n%s", traceback.format_exc())
        return {}, {"tool_count": len(self.supported_tools), "init_error": self.init_error}

    def close(self):
        self.runtime_env.clear()

    def _extract_tool_calls(self, text: str) -> list[dict[str, Any]]:
        tool_calls: list[dict[str, Any]] = []
        blocks = TOOL_CALL_BLOCK_RE.findall(text)
        for block in blocks:
            found_in_block = False
            for raw_json in _extract_json_candidates(block):
                try:
                    payload = _json_loads(raw_json)
                except Exception:
                    continue
                normalized = _normalize_tool_payload(payload)
                if normalized is not None:
                    tool_calls.append(normalized)
                    found_in_block = True
                    break
            if found_in_block:
                continue
            payload = _extract_from_function_parameter_markup(block)
            if payload is not None:
                tool_calls.append(payload)

        if tool_calls:
            return tool_calls

        matches = TOOL_CALL_JSON_RE.findall(text)
        for raw_json in matches:
            raw_json = (_extract_balanced_json(raw_json.strip(), 0) or raw_json).strip()
            try:
                payload = _json_loads(raw_json)
            except Exception as exc:
                logger.warning("Failed to decode tool call payload: %s", exc)
                continue
            normalized = _normalize_tool_payload(payload)
            if normalized is not None:
                tool_calls.append(normalized)

        fn_blocks = FUNCTION_BLOCK_RE.findall(text)
        for block in fn_blocks:
            payload = _extract_from_function_parameter_markup(block)
            if payload is not None:
                tool_calls.append(payload)

        return tool_calls

    def _has_terminal_eos(self, text: str) -> bool:
        text = text or ""
        return any(marker in text for marker in EOS_MARKERS)

    def _has_tool_call_markup(self, text: str) -> bool:
        text = text or ""
        return "<tool_call>" in text or "</tool_call>" in text

    def _serialize_result(self, value: Any) -> str:
        if isinstance(value, str):
            return value
        try:
            return _json_dumps(value)
        except Exception:
            return str(value)

    def _turn_hint(self) -> str:
        if self.max_turns is None:
            return "Continue reasoning. When you are ready, provide the final answer in \\boxed{...}."
        remaining_turns = self.max_turns - self.turn
        if remaining_turns <= 1:
            return (
                "You are near the end of the rollout budget. "
                "If you already have enough information, stop calling tools and give the final answer in \\boxed{...}."
            )
        return (
            f"You may continue reasoning and call more tools if needed. "
            f"Remaining assistant turns: {remaining_turns}."
        )

    def _execute_tool(self, tool_name: str, arguments: dict[str, Any]) -> str:
        target_fn = self.runtime_env.get(tool_name)
        if not callable(target_fn):
            return f"Error: function `{tool_name}` is not defined in the provided environment."
        try:
            result = target_fn(**arguments) if isinstance(arguments, dict) else target_fn(arguments)
            return self._serialize_result(result)
        except Exception as exc:
            detail = traceback.format_exc(limit=3)
            logger.warning("Tool execution failed for %s: %s", tool_name, exc)
            return f"Error when executing `{tool_name}`: {exc}\n{detail}"

    def step(self, response_text: str):
        self.turn += 1
        done = self.max_turns is not None and self.turn >= self.max_turns
        if self.init_error:
            info: dict[str, Any] = {"tool_executed": False, "init_error": self.init_error}
            obs = {
                "obs_str": (
                    f"<tool_response>Error: environment initialization failed: {self.init_error}</tool_response>\n"
                    "No tools are available for this sample because env init failed.\n"
                    f"{self._turn_hint()}"
                ),
                "role": "tool",
            }
            return obs, done, info

        tool_calls = self._extract_tool_calls(response_text)
        info: dict[str, Any] = {
            "tool_calls": deepcopy(tool_calls),
            "tool_call": deepcopy(tool_calls[0]) if tool_calls else None,
        }

        if not tool_calls:
            if self._has_tool_call_markup(response_text):
                info["tool_executed"] = False
                obs = {
                    "obs_str": (
                        "<tool_response>Error: malformed tool call payload.</tool_response>\n"
                        "You emitted tool_call markup but no valid callable payload was parsed.\n"
                        "Please output a valid JSON tool call payload.\n"
                        f"{self._turn_hint()}"
                    ),
                    "role": "tool",
                }
                return obs, done, info
            # End only when the model emits no tool call and reaches an EOS marker.
            if self._has_terminal_eos(response_text):
                info.update({"tool_executed": False, "final_answer_detected": True})
                return {"obs_str": "Final answer detected via EOS marker.", "role": "tool"}, True, info
            info["tool_executed"] = False
            obs = {
                "obs_str": (
                    "<tool_response>Error: no valid tool call detected.</tool_response>\n"
                    "Please output exactly one valid <tool_call>...</tool_call> block.\n"
                    f"{self._turn_hint()}"
                ),
                "role": "tool",
            }
            return obs, done, info

        response_chunks: list[str] = []
        executed_calls: list[dict[str, Any]] = []
        for idx, tool_call in enumerate(tool_calls, start=1):
            tool_name = str(tool_call.get("name") or "").strip()
            arguments = tool_call.get("arguments") or {}
            if tool_name not in self.supported_tools:
                tool_result = f"Error: unsupported tool `{tool_name}`."
                executed_calls.append(
                    {
                        "index": idx,
                        "tool_executed": False,
                        "tool_name": tool_name,
                        "arguments": arguments,
                        "tool_result": tool_result,
                    }
                )
                response_chunks.append(f"<tool_response>{tool_result}</tool_response>")
                continue
            tool_result = self._execute_tool(tool_name, arguments)
            executed_calls.append(
                {
                    "index": idx,
                    "tool_executed": True,
                    "tool_name": tool_name,
                    "arguments": arguments,
                    "tool_result": tool_result,
                }
            )
            response_chunks.append(f"<tool_response>{tool_result}</tool_response>")

        any_success = any(item["tool_executed"] for item in executed_calls)
        info.update(
            {
                "tool_executed": any_success,
                "executed_tool_calls": executed_calls,
                "tool_name": executed_calls[0]["tool_name"] if executed_calls else None,
                "tool_result": executed_calls[0]["tool_result"] if executed_calls else None,
            }
        )
        obs = {
            "obs_str": f"{''.join(response_chunks)}\n{self._turn_hint()}",
            "role": "tool",
        }
        return obs, done, info


def build_env(sample: Sample | None = None, args: Any | None = None, **_: Any) -> RecallAgentEnv:
    metadata = sample.metadata if sample is not None and isinstance(sample.metadata, dict) else {}
    env_code = str(metadata.get("env") or "")
    tool_schemas = metadata.get("tool_schemas") or metadata.get("tools") or []
    if not isinstance(tool_schemas, list):
        tool_schemas = []

    max_turns = getattr(args, "max_turns", None)
    if max_turns is None:
        raise ValueError("max_turns must be set via --custom-config-path.")
    return RecallAgentEnv(env_code=env_code, tool_schemas=tool_schemas, max_turns=max_turns)
