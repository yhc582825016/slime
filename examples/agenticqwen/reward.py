from __future__ import annotations

import json
import logging
import os
import re
from typing import Any

from slime.utils.types import Sample

from .llm_client import call_llm_async
from .message_parser import extract_solution_summary
from .prompts import JUDGE_SYSTEM_PROMPT, RUBRICS_JUDGE_01_PROMPT

logger = logging.getLogger(__name__)


def _to_builtin(value: Any) -> Any:
    if hasattr(value, "as_py"):
        value = value.as_py()
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, dict):
        return {str(k): _to_builtin(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_builtin(v) for v in value]
    return value


def _json_loads_if_needed(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except Exception:
            return value
    return value


def _metadata(sample: Sample) -> dict[str, Any]:
    metadata = _to_builtin(_json_loads_if_needed(sample.metadata or {}))
    return metadata if isinstance(metadata, dict) else {}


async def _judge_completion(question: str, cleaned_solution: str, rubrics: str) -> tuple[float, str]:
    response = await call_llm_async(
        user_prompt=RUBRICS_JUDGE_01_PROMPT.format(
            question=question,
            cleaned_solution=cleaned_solution,
            rubrics=rubrics,
        ),
        system_prompt=JUDGE_SYSTEM_PROMPT,
        api_base=os.environ.get("REWARD_API_BASE") or os.environ.get("AGENTICQWEN_REWARD_API_BASE"),
        api_key=os.environ.get("REWARD_API_KEY") or os.environ.get("AGENTICQWEN_REWARD_API_KEY"),
        model_name=os.environ.get("REWARD_MODEL_NAME") or os.environ.get("AGENTICQWEN_REWARD_MODEL_NAME", ""),
        max_tokens=int(os.environ.get("AGENTICQWEN_REWARD_MAX_TOKENS", "10240")),
        temperature=float(os.environ.get("AGENTICQWEN_REWARD_TEMPERATURE", "0.1")),
    )
    match = re.search(r"<final_judgment>(.*?)</final_judgment>", response, re.DOTALL)
    verdict = match.group(1).lower() if match else response.lower()
    return (1.0 if "task completed" in verdict else 0.0), response


async def reward_func(args, sample: Sample | list[Sample], **kwargs):
    del args, kwargs
    if isinstance(sample, list):
        return [await reward_func(None, item) for item in sample]

    metadata = _metadata(sample)
    response = sample.response or ""
    messages = metadata.get("agenticqwen_rollout_messages") or metadata.get("initial_messages") or []
    if not isinstance(messages, list):
        messages = []

    terminal = ("###STOP" in response) or ("###TRANSFER_TO_HUMAN" in response) or ("<answer>" in response)
    has_tool_or_transfer = (
        "<tool_response>" in response
        or bool(metadata.get("agenticqwen_tool_call_history"))
        or "###TRANSFER_TO_HUMAN" in response
    )
    if not terminal or not has_tool_or_transfer:
        return {"score": 0.0, "terminal": terminal, "has_tool_or_transfer": has_tool_or_transfer}

    cleaned_solution = extract_solution_summary(messages, response="" if messages else response)
    try:
        score, judge_response = await _judge_completion(
            question=str(metadata.get("question") or ""),
            cleaned_solution=cleaned_solution,
            rubrics=str(metadata.get("rubrics") or ""),
        )
        return {"score": score, "judge_response": judge_response[:2000]}
    except Exception as exc:
        logger.warning("AgenticQwen reward judge failed: %s", exc)
        return {"score": 0.0, "judge_error": str(exc)}


async def batched_reward_func(args, samples: list[Sample], **kwargs):
    return [await reward_func(args, sample, **kwargs) for sample in samples]
