"""Slime adapter for the migrated Reasoning360 multi-domain rewards.

Use with:
  --custom-rm-path examples.mix_rl.reward.reward_func

The expected sample layout is:
  sample.label = {"ground_truth": ..., "style": "rule"}
  sample.metadata = {"data_source": ..., "extra_info": {...}}
"""

from __future__ import annotations

import json
from typing import Any

from slime.utils.types import Sample

from .reward_score import default_compute_score


def _json_loads_if_needed(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except Exception:
            return value
    return value


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


def _extract_reward_inputs(sample: Sample) -> tuple[str, Any, dict[str, Any] | None]:
    metadata = _to_builtin(_json_loads_if_needed(sample.metadata or {}))
    label = _to_builtin(_json_loads_if_needed(sample.label))

    if not isinstance(metadata, dict):
        metadata = {"data_source": metadata}

    data_source = metadata.get("data_source")
    if not data_source and metadata.get("source_dataset"):
        data_source = "reasoning_gym"
    extra_info = metadata.get("extra_info", metadata)

    if isinstance(label, dict):
        ground_truth = label.get("ground_truth", label)
        data_source = data_source or label.get("data_source")
        extra_info = label.get("extra_info", extra_info)
    else:
        ground_truth = label

    if not data_source:
        raise ValueError(
            "Missing data_source. Convert Reasoning360/Guru data with "
            "examples.mix_rl.convert_guru_to_slime or include data_source in sample.metadata."
        )

    if not isinstance(extra_info, dict):
        extra_info = None

    return str(data_source), ground_truth, extra_info


def _prompt_to_question(prompt: Any) -> str | None:
    if isinstance(prompt, str):
        return prompt
    if isinstance(prompt, list):
        for message in reversed(prompt):
            if isinstance(message, dict) and message.get("role") == "user":
                content = message.get("content")
                if isinstance(content, str):
                    return content
    return None


def compute_score_for_sample(sample: Sample) -> float | dict[str, Any]:
    data_source, ground_truth, extra_info = _extract_reward_inputs(sample)
    if data_source == "reasoning_gym":
        extra_info = dict(extra_info or {})
        extra_info.setdefault("question", _prompt_to_question(sample.prompt))
    return default_compute_score(
        data_source=data_source,
        solution_str=sample.response or "",
        ground_truth=ground_truth,
        extra_info=extra_info,
    )


def _score_value(result: float | int | bool | dict[str, Any]) -> float | dict[str, Any]:
    if isinstance(result, dict):
        if "score" in result:
            return result
        if "acc" in result:
            result = dict(result)
            result["score"] = result["acc"]
            return result
        return result
    return float(result)


async def reward_func(args, sample: Sample | list[Sample], **kwargs) -> float | dict[str, Any] | list[float | dict[str, Any]]:
    del args, kwargs
    if isinstance(sample, list):
        return [_score_value(compute_score_for_sample(item)) for item in sample]
    return _score_value(compute_score_for_sample(sample))


async def batched_reward_func(args, samples: list[Sample], **kwargs) -> list[float | dict[str, Any]]:
    result = await reward_func(args, samples, **kwargs)
    assert isinstance(result, list)
    return result
