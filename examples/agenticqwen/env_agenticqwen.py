from __future__ import annotations

import ast
import json
import logging
import os
import re
from copy import deepcopy
from typing import Any

from slime.utils.types import Sample

try:
    from examples.recall_agent.base_env import BaseInteractionEnv
except Exception:  # pragma: no cover
    from ..recall_agent.base_env import BaseInteractionEnv

from .llm_client import call_llm_sync
from .message_parser import content_to_text, extract_tool_history, extract_user_conversation
from .prompts import MOCK_USER_PROMPT, TOOL_SIMULATION_PROMPT

logger = logging.getLogger(__name__)

TOOL_CALL_BLOCK_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
QUESTION_RE = re.compile(r"<question>(.*?)</question>", re.DOTALL)
ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
EOS_MARKERS = ("<|im_end|>", "<|endoftext|>", "</s>")


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, default=str)


def _safe_json_loads(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return value
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except Exception:
        try:
            return ast.literal_eval(value)
        except Exception:
            return value


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


def _normalize_tool_payload(payload: Any) -> dict[str, Any] | None:
    payload = _safe_json_loads(payload)
    if not isinstance(payload, dict):
        return None
    name = payload.get("name")
    arguments = payload.get("arguments")
    if arguments is None:
        arguments = payload.get("parameters")
    function = payload.get("function")
    if isinstance(function, dict):
        name = name or function.get("name")
        arguments = arguments if arguments is not None else function.get("arguments") or function.get("parameters")
    elif isinstance(function, str):
        name = name or function
    arguments = _safe_json_loads(arguments)
    if not isinstance(arguments, dict):
        arguments = {}
    if not name:
        return None
    return {"name": str(name).strip(), "arguments": arguments}


def _extract_tool_calls(text: str) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    for block in TOOL_CALL_BLOCK_RE.findall(text or ""):
        block = block.strip()
        if not block:
            continue
        for idx, ch in enumerate(block):
            if ch != "{":
                continue
            raw = _extract_balanced_json(block, idx)
            if not raw:
                continue
            payload = _normalize_tool_payload(raw)
            if payload is not None:
                calls.append(payload)
                break
    return calls


def _extract_tool_schemas_from_system(system_prompt: str) -> list[dict[str, Any]]:
    matches = re.findall(r"<tools>(.*?)</tools>", system_prompt or "", re.DOTALL)
    for raw in reversed(matches):
        raw = raw.strip()
        if not raw:
            continue
        parsed = _safe_json_loads(raw)
        if isinstance(parsed, list):
            return [item for item in parsed if isinstance(item, dict)]
    return []


def _normalise_for_compare(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _normalise_for_compare(v) for k, v in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, list):
        return [_normalise_for_compare(v) for v in value]
    return value


class AgenticQwenEnv(BaseInteractionEnv):
    def __init__(self, *, sample: Sample, max_turns: int | None = None):
        self.sample = sample
        self.metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
        self.max_turns = max_turns
        self.turn = 0
        self.messages: list[dict[str, Any]] = []
        self.tool_call_history: list[dict[str, Any]] = []

    def reset(self):
        self.turn = 0
        self.tool_call_history = []
        self.messages = deepcopy(self.metadata.get("initial_messages") or [])
        if not self.messages:
            self.messages = deepcopy(self.sample.prompt if isinstance(self.sample.prompt, list) else [])
        self.metadata["agenticqwen_rollout_messages"] = self.messages
        self.metadata["agenticqwen_tool_call_history"] = self.tool_call_history
        return {}, {"tool_count": len(self.metadata.get("tool_schemas") or [])}

    def _turn_hint(self) -> str:
        if self.max_turns is None:
            return "Continue. When done, finish with <answer>...</answer> and ###STOP, or ###TRANSFER_TO_HUMAN if escalation is required."
        remaining = self.max_turns - self.turn
        return (
            f"Remaining assistant turns: {max(remaining, 0)}. "
            "When done, finish with <answer>...</answer> and ###STOP, or ###TRANSFER_TO_HUMAN if escalation is required."
        )

    def _lookup_expected_tool_response(self, tool_call: dict[str, Any]) -> Any | None:
        expected = _safe_json_loads(self.metadata.get("tool_return_expected") or {})
        if not isinstance(expected, dict):
            return None

        name = tool_call.get("name")
        args = _normalise_for_compare(tool_call.get("arguments") or {})
        fallback_by_name: Any | None = None
        for path_name in ("normal_path", "hack_path"):
            path = expected.get(path_name) or []
            if not isinstance(path, list):
                continue
            for item in path:
                if not isinstance(item, dict) or item.get("tool_name") != name:
                    continue
                fallback_by_name = item.get("expected_output")
                if _normalise_for_compare(item.get("input") or {}) == args:
                    return item.get("expected_output")
        return fallback_by_name

    def _simulate_tool_response_with_llm(self, tool_call: dict[str, Any]) -> str:
        tool_schemas = self.metadata.get("tool_schemas") or []
        tool_description = "\n".join(_json_dumps(tool) for tool in tool_schemas)
        user_prompt = TOOL_SIMULATION_PROMPT.format(
            query=_json_dumps(tool_call),
            tools=tool_description,
            history=extract_tool_history(self.messages),
            world_state=self.metadata.get("tool_return_expected") or "",
        )
        response = call_llm_sync(
            user_prompt=user_prompt,
            api_base=os.environ.get("MOCK_TOOL_API_BASE") or os.environ.get("AGENTICQWEN_MOCK_TOOL_API_BASE"),
            api_key=os.environ.get("MOCK_TOOL_API_KEY") or os.environ.get("AGENTICQWEN_MOCK_TOOL_API_KEY"),
            model_name=os.environ.get("MOCK_TOOL_MODEL_NAME") or os.environ.get("AGENTICQWEN_MOCK_TOOL_MODEL_NAME", ""),
            max_tokens=int(os.environ.get("AGENTICQWEN_MOCK_TOOL_MAX_TOKENS", "2048")),
            timeout=float(os.environ.get("AGENTICQWEN_MOCK_TOOL_TIMEOUT", "300")),
        )
        matches = re.findall(r"<simulated_tool_response>(.*?)</simulated_tool_response>", response, re.DOTALL)
        return matches[-1].strip() if matches else response.strip()

    def _execute_tool(self, tool_call: dict[str, Any]) -> str:
        expected = self._lookup_expected_tool_response(tool_call)
        if expected is not None:
            return expected if isinstance(expected, str) else _json_dumps(expected)
        try:
            return self._simulate_tool_response_with_llm(tool_call)
        except Exception as exc:
            logger.warning("LLM tool simulation failed for %s: %s", tool_call, exc)
            return "No Useful Information Found"

    def _mock_user_response(self) -> str:
        if "clarification case" in str(self.metadata.get("test_policy") or ""):
            test_policy = ""
        else:
            test_policy = self.metadata.get("test_policy") or ""
        prompt = MOCK_USER_PROMPT.format(
            task_background=self.metadata.get("task_background") or "",
            test_policy=test_policy,
            user_escape_strategy=self.metadata.get("user_escape_strategy") or "",
            conversation_history=extract_user_conversation(self.messages),
        )
        response = call_llm_sync(
            user_prompt=prompt,
            api_base=os.environ.get("MOCK_USER_API_BASE") or os.environ.get("AGENTICQWEN_MOCK_USER_API_BASE"),
            api_key=os.environ.get("MOCK_USER_API_KEY") or os.environ.get("AGENTICQWEN_MOCK_USER_API_KEY"),
            model_name=os.environ.get("MOCK_USER_MODEL_NAME") or os.environ.get("AGENTICQWEN_MOCK_USER_MODEL_NAME", ""),
            max_tokens=int(os.environ.get("AGENTICQWEN_MOCK_USER_MAX_TOKENS", "2048")),
            timeout=float(os.environ.get("AGENTICQWEN_MOCK_USER_TIMEOUT", "300")),
        )
        matches = re.findall(r"<reply>(.*?)</reply>", response, re.DOTALL)
        return matches[-1].strip() if matches else response.strip()

    def step(self, response_text: str):
        self.turn += 1
        self.messages.append({"role": "assistant", "content": response_text})
        done_by_turn = self.max_turns is not None and self.turn >= self.max_turns

        tool_calls = _extract_tool_calls(response_text)
        if tool_calls:
            chunks: list[str] = []
            for tool_call in tool_calls:
                tool_response = self._execute_tool(tool_call)
                self.tool_call_history.append({"tool_call": deepcopy(tool_call), "tool_response": tool_response})
                chunks.append(f"<tool_response>{tool_response}</tool_response>")
            obs_text = "".join(chunks) + "\n" + self._turn_hint()
            self.messages.append({"role": "user", "content": obs_text})
            return {"obs_str": obs_text, "role": "tool"}, done_by_turn, {"tool_calls": tool_calls, "tool_executed": True}

        if "###TRANSFER_TO_HUMAN" in response_text or "###STOP" in response_text or ANSWER_RE.search(response_text):
            return {"obs_str": "Final answer detected.", "role": "user"}, True, {"final_answer_detected": True}

        if QUESTION_RE.search(response_text):
            try:
                user_reply = self._mock_user_response()
            except Exception as exc:
                logger.warning("Mock user failed: %s", exc)
                user_reply = "I am not sure, but please continue with the information you already have."
            if "###STOP" in user_reply:
                self.messages.append({"role": "user", "content": user_reply})
                return {"obs_str": user_reply, "role": "user"}, True, {"mock_user_stop": True}
            self.messages.append({"role": "user", "content": user_reply})
            return {"obs_str": user_reply, "role": "user"}, done_by_turn, {"mock_user_response": True}

        if any(marker in response_text for marker in EOS_MARKERS):
            return {"obs_str": "Final answer detected.", "role": "user"}, True, {"final_answer_detected": True}

        obs_text = "If you are done, provide <answer>...</answer> followed by ###STOP. If you need user information, ask inside <question>...</question>."
        self.messages.append({"role": "user", "content": obs_text})
        return {"obs_str": obs_text, "role": "user"}, done_by_turn, {"tool_executed": False}

    def format_observation(self, observation: dict) -> dict:
        return {"role": observation.get("role", "user"), "content": observation.get("obs_str", "")}


def build_env(sample: Sample | None = None, args: Any | None = None, **_: Any) -> AgenticQwenEnv:
    if sample is None:
        raise ValueError("sample is required")
    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    if not metadata.get("tool_schemas"):
        system = ""
        messages = metadata.get("initial_messages") or []
        if messages and isinstance(messages[0], dict):
            system = content_to_text(messages[0].get("content", ""))
        metadata["tool_schemas"] = _extract_tool_schemas_from_system(system)
    max_turns = getattr(args, "max_turns", None)
    return AgenticQwenEnv(sample=sample, max_turns=max_turns)

