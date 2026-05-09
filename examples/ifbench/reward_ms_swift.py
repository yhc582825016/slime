import inspect
import importlib.util
import json
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
        # try:
        return json.loads(value)
        # except Exception:
            # return value
    return value


@lru_cache(maxsize=1)
def _load_instructions_registry():
    # Force-load the legacy IFBench instruction stack from this directory so
    # it won't be confused with the newer Evaluation/IFBench modules.
    this_dir = Path(__file__).resolve().parent
    module_paths = {
        "instructions_util": this_dir / "instructions_util.py",
        "instructions": this_dir / "instructions.py",
    }
    saved_modules = {name: sys.modules.get(name) for name in (*module_paths.keys(), "instructions_registry")}

    try:
        for module_name, module_path in module_paths.items():
            spec = importlib.util.spec_from_file_location(module_name, module_path)
            if spec is None or spec.loader is None:
                raise ImportError(f"Unable to load {module_name} from {module_path}")
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)

        registry_path = this_dir / "instructions_registry.py"
        registry_name = "_slime_ifbench_legacy_instructions_registry"
        spec = importlib.util.spec_from_file_location(registry_name, registry_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Unable to load instructions_registry from {registry_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        for module_name, old_module in saved_modules.items():
            if old_module is None:
                sys.modules.pop(module_name, None)
            else:
                sys.modules[module_name] = old_module


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


def _extract_rm_type(sample: Sample) -> str:
    if isinstance(sample.metadata, dict):
        raw = sample.metadata.get("rm_type")
        if raw is not None:
            return str(raw).strip()
        extra_info = _parse_json_like(sample.metadata.get("extra_info"))
        if isinstance(extra_info, dict) and extra_info.get("rm_type") is not None:
            return str(extra_info["rm_type"]).strip()
    return ""


def _extract_prompt_text(sample: Sample) -> str:
    prompt = sample.prompt
    if isinstance(prompt, str):
        return prompt
    if isinstance(prompt, list):
        parts: list[str] = []
        for turn in prompt:
            if isinstance(turn, dict):
                content = turn.get("content")
                if content is not None:
                    parts.append(str(content))
        return "\n".join(parts)
    return str(prompt or "")


def _should_use_builtin_ifbench(sample: Sample, info: dict[str, Any]) -> bool:
    rm_type = _extract_rm_type(sample)
    if rm_type == "ifbench":
        return True

    # Legacy train data carries these extra fields, while eval data typically does not.
    if "constraint" in info or "constraint_type" in info:
        return False

    return any(key in info for key in ("record_id", "prompt_text", "kwargs"))


def _build_builtin_ifbench_metadata(sample: Sample, info: dict[str, Any]) -> dict[str, Any]:
    instruction_ids = [str(x) for x in _to_list(_parse_json_like(info.get("instruction_id_list", [])))]
    raw_kwargs = info.get("kwargs", info.get("instruction_kwargs", []))
    kwargs_list = [_normalize_value(v) for v in _to_list(_parse_json_like(raw_kwargs))]

    metadata: dict[str, Any] = {
        "instruction_id_list": instruction_ids,
        "kwargs": kwargs_list,
        "prompt_text": str(info.get("prompt_text") or _extract_prompt_text(sample)),
    }
    if info.get("record_id") is not None:
        metadata["record_id"] = info.get("record_id")
    return metadata


def _extract_answer(response: str) -> str:
    """Extract the answer after </think> for thinking models.

    If no </think> is found but <think> is present (e.g. degenerate generation
    that never closes the thinking block), fall back to the text after the last
    <think> tag so we don't accidentally evaluate raw thinking tokens as the answer.
    If neither tag is present, return the full response unchanged.
    """
    end_think_idx = response.rfind("</think>")
    if end_think_idx != -1:
        return response[end_think_idx + len("</think>"):].strip()
    # Degenerate case: model opened <think> but never closed it
    start_think_idx = response.rfind("<think>")
    if start_think_idx != -1:
        return response[start_think_idx + len("<think>"):].strip()
    return response


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
    if _should_use_builtin_ifbench(sample, info):
        from slime.rollout.rm_hub.ifbench import compute_ifbench_reward

        answer = _extract_answer(sample.response or "")
        metadata = _build_builtin_ifbench_metadata(sample, info)
        return float(compute_ifbench_reward(answer, sample.label, metadata=metadata))

    instruction_id_list = [str(x) for x in _to_list(_parse_json_like(info.get("instruction_id_list", [])))]
    raw_kwargs = info.get("instruction_kwargs", info.get("kwargs", []))
    kwargs_list = _to_list(_parse_json_like(raw_kwargs))

    if not instruction_id_list:
        return 0.0

    # try:
    answer = _extract_answer(sample.response or "")
    hits = _strict_hits(answer, instruction_id_list, kwargs_list)
    # except Exception:
    #     return 0.0

    return 1.0 if hits and all(hits) else 0.0


async def batched_reward_func(args, samples: list[Sample], **kwargs) -> list[float]:
    return [await reward_func(args, sample, **kwargs) for sample in samples]
