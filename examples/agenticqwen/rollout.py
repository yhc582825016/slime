"""slime custom rollout entry for AgenticQwen multi-turn tool-use tasks."""

from __future__ import annotations

from typing import Any


async def generate(args: Any, sample: Any, sampling_params: dict):
    from examples.recall_agent.rollout import generate as recall_generate

    return await recall_generate(args, sample, sampling_params)

__all__ = ["generate"]
