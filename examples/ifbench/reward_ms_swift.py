import inspect
import json
import os
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

from slime.utils.types import Sample


def _to_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if hasattr(value, "tolist"):
        converted = value.tolist()
        return converted if isinstance(converted, list) else [converted]
    return [value]


def _normalize_value(value: Any) -> Any:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, list):
        return [_normalize_value(v) for v in value]
    if isinstance(value, dict):
        return {k: _normalize_value(v) for k, v in value.items()}
    return value


def _parse_json_like(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except Exception:
            return value
    return value


def _candidate_ifbench_dirs() -> list[Path]:
    repo_root = Path(__file__).resolve().parents[2]
    candidates = []

    env_path = os.environ.get("MS_SWIFT_IFBENCH_DIR")
    if env_path:
        candidates.append(Path(env_path))

    candidates.append(repo_root.parent / "ms-swift" / "plugin" / "IFbench")
    candidates.append(Path("/dev/shm/ye/ms-swift/plugin/IFbench"))
    return candidates


@lru_cache(maxsize=1)
def _load_instructions_registry():
    for candidate in _candidate_ifbench_dirs():
        if not candidate.exists():
            continue
        candidate_str = str(candidate)
        if candidate_str not in sys.path:
            sys.path.insert(0, candidate_str)
        import instructions_registry

        return instructions_registry

    searched = ", ".join(str(path) for path in _candidate_ifbench_dirs())
    raise ImportError(
        "Unable to locate ms-swift IFBench plugin directory. "
        "Set MS_SWIFT_IFBENCH_DIR to the directory containing instructions_registry.py. "
        f"Searched: {searched}"
    )


def _extract_info(sample: Sample) -> dict[str, Any]:
    candidate_payloads: list[Any] = []

    if isinstance(sample.metadata, dict):
        candidate_payloads.append(sample.metadata)
        candidate_payloads.append(sample.metadata.get("extra_info"))
    if isinstance(sample.label, dict):
        candidate_payloads.append(sample.label)

    for payload in candidate_payloads:
        payload = _parse_json_like(payload)
        if not isinstance(payload, dict):
            continue
        if "instruction_id_list" in payload:
            return payload

    return {}


def _strict_hits(solution_str: str, instruction_id_list: list[str], kwargs_list: list[dict[str, Any]]) -> list[bool]:
    if len(instruction_id_list) != len(kwargs_list):
        raise ValueError("instruction_id_list and instruction_kwargs length mismatch")

    instructions_registry = _load_instructions_registry()
    hits: list[bool] = []
    for idx, inst_id in enumerate(instruction_id_list):
        inst_cls = instructions_registry.INSTRUCTION_DICT[inst_id]
        inst = inst_cls(inst_id)

        raw_kwargs = kwargs_list[idx] or {}
        if not isinstance(raw_kwargs, dict):
            raw_kwargs = {}
        accepted = set(inspect.signature(inst.build_description).parameters.keys())
        filtered_kwargs = {k: _normalize_value(v) for k, v in raw_kwargs.items() if k in accepted}
        inst.build_description(**filtered_kwargs)

        hits.append(bool(solution_str.strip()) and bool(inst.check_following(solution_str)))
    return hits


async def reward_func(args, sample: Sample, **kwargs) -> float:
    del args, kwargs

    if not isinstance(sample, Sample):
        raise TypeError("sample must be an instance of slime.utils.types.Sample")

    info = _extract_info(sample)
    instruction_id_list = [str(x) for x in _to_list(_parse_json_like(info.get("instruction_id_list", [])))]
    raw_kwargs = info.get("instruction_kwargs", info.get("kwargs", []))
    kwargs_list = _to_list(_parse_json_like(raw_kwargs))

    if not instruction_id_list:
        return 0.0

    try:
        hits = _strict_hits(sample.response or "", instruction_id_list, kwargs_list)
    except Exception:
        return 0.0

    return 1.0 if hits and all(hits) else 0.0


async def batched_reward_func(args, samples: list[Sample], **kwargs) -> list[float]:
    return [await reward_func(args, sample, **kwargs) for sample in samples]
