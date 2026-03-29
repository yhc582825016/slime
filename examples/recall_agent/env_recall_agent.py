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


class RecallAgentEnv(BaseInteractionEnv):
    def __init__(self, *, env_code: str, tool_schemas: list[dict[str, Any]], max_turns: int | None = None):
        self.env_code = env_code
        self.tool_schemas = tool_schemas
        self.max_turns = max_turns
        self.turn = 0
        self.runtime_env: dict[str, Any] = {}
        self.supported_tools = {
            tool.get("function", {}).get("name")
            for tool in tool_schemas
            if isinstance(tool, dict) and tool.get("function", {}).get("name")
        }

    def reset(self):
        self.turn = 0
        self.runtime_env = {}
        if self.env_code.strip():
            exec(self.env_code, self.runtime_env, self.runtime_env)
        return {}, {"tool_count": len(self.supported_tools)}

    def close(self):
        self.runtime_env.clear()

    def _extract_tool_call(self, text: str) -> dict[str, Any] | None:
        matches = TOOL_CALL_JSON_RE.findall(text)
        for raw_json in reversed(matches):
            raw_json = (_extract_balanced_json(raw_json.strip(), 0) or raw_json).strip()
            try:
                payload = _json_loads(raw_json)
            except Exception as exc:
                logger.warning("Failed to decode tool call payload: %s", exc)
                continue
            if not isinstance(payload, dict):
                continue
            arguments = payload.get("arguments") or {}
            if isinstance(arguments, str):
                arguments = _safe_json_loads(arguments)
            return {"name": payload.get("name"), "arguments": arguments}

        blocks = TOOL_CALL_BLOCK_RE.findall(text)
        for block in reversed(blocks):
            payload = _extract_from_function_parameter_markup(block)
            if payload is not None:
                return payload

        fn_blocks = FUNCTION_BLOCK_RE.findall(text)
        for block in reversed(fn_blocks):
            payload = _extract_from_function_parameter_markup(block)
            if payload is not None:
                return payload

        return None

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
        tool_call = self._extract_tool_call(response_text)
        info: dict[str, Any] = {"tool_call": deepcopy(tool_call)}

        if not tool_call:
            info["tool_executed"] = False
            return {"obs_str": "No tool call detected.", "role": "tool"}, True, info

        tool_name = str(tool_call.get("name") or "").strip()
        arguments = tool_call.get("arguments") or {}
        if tool_name not in self.supported_tools:
            info["tool_executed"] = False
            obs = {
                "obs_str": (
                    f"<tool_response>Error: unsupported tool `{tool_name}`.</tool_response>\n"
                    f"{self._turn_hint()}"
                ),
                "role": "tool",
            }
            return obs, done, info

        tool_result = self._execute_tool(tool_name, arguments)
        info.update({"tool_executed": True, "tool_name": tool_name, "tool_result": tool_result})
        obs = {
            "obs_str": f"<tool_response>{tool_result}</tool_response>\n{self._turn_hint()}",
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
