"""Reasoning Gym reward wrapper matching NeMo Gym's verifier logic."""

from __future__ import annotations

import json
import re
from typing import Any

import reasoning_gym
from reasoning_gym.utils import extract_answer


def _json_loads_if_needed(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except Exception:
            return value
    return value


def _normalize_answer_for_scoring(value: Any) -> str:
    """Coerce oracle/model answers into strings that reasoning_gym scorers expect."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, bool):
        return "True" if value else "False"
    if isinstance(value, (int, float)):
        if isinstance(value, float) and value.is_integer():
            return str(int(value))
        return str(value)
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value).strip()


def _extract_answer_from_solution(solution_str: str) -> str:
    """Match resources_servers/reasoning_gym/app.py answer extraction."""
    extracted = extract_answer(solution_str, tag_name="answer")
    if extracted is not None:
        return _normalize_answer_for_scoring(extracted)

    boxed_match = re.search(r"\\boxed\{([^}]+)\}", solution_str)
    if boxed_match:
        return boxed_match.group(1).strip()

    return solution_str.strip() if solution_str.strip() else ""


def _metadata_from_extra_info(extra_info: Any) -> dict[str, Any]:
    extra_info = _json_loads_if_needed(extra_info)
    if not isinstance(extra_info, dict):
        return {}

    metadata = _json_loads_if_needed(extra_info.get("metadata", extra_info))
    if isinstance(metadata, dict):
        return dict(metadata)
    return {}


def _entry_from_inputs(ground_truth: Any, extra_info: Any) -> dict[str, Any]:
    ground_truth = _json_loads_if_needed(ground_truth)
    extra_info = _json_loads_if_needed(extra_info)

    if isinstance(extra_info, dict):
        entry = _json_loads_if_needed(extra_info.get("entry"))
        if isinstance(entry, dict):
            entry = dict(entry)
        else:
            entry = {}

        question = extra_info.get("question")
        answer = extra_info.get("answer", ground_truth)
    else:
        entry = {}
        question = None
        answer = ground_truth

    if isinstance(ground_truth, dict):
        entry = {**ground_truth, **entry}
        question = entry.get("question", question)
        answer = entry.get("answer", answer)

    metadata = _metadata_from_extra_info(extra_info)
    if isinstance(entry.get("metadata"), dict):
        metadata = {**entry["metadata"], **metadata}

    entry["question"] = question or entry.get("question", "")
    entry["answer"] = answer
    entry["metadata"] = metadata
    return entry


def _normalize_entry_for_scoring(entry: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(entry)
    if "answer" in normalized:
        normalized["answer"] = _normalize_answer_for_scoring(normalized["answer"])
    return normalized


def _model_answer_variants(model_answer: str) -> list[str]:
    variants: list[str] = []
    for candidate in (
        model_answer,
        _normalize_answer_for_scoring(model_answer),
        re.sub(r"\s+", "", _normalize_answer_for_scoring(model_answer)),
        _normalize_answer_for_scoring(model_answer).replace(",", ""),
    ):
        if candidate and candidate not in variants:
            variants.append(candidate)
    return variants or [""]


def _entry_variants(entry: dict[str, Any]) -> list[dict[str, Any]]:
    variants: list[dict[str, Any]] = []
    for candidate in (entry, _normalize_entry_for_scoring(entry)):
        key = json.dumps(candidate, sort_keys=True, default=str)
        if key not in {json.dumps(v, sort_keys=True, default=str) for v in variants}:
            variants.append(candidate)
    return variants


def _task_name(entry: dict[str, Any], extra_info: Any) -> str:
    extra_info = _json_loads_if_needed(extra_info)
    metadata = entry.get("metadata") if isinstance(entry.get("metadata"), dict) else {}

    candidates = [
        metadata.get("source_dataset"),
        metadata.get("task"),
        metadata.get("task_name"),
    ]
    if isinstance(extra_info, dict):
        candidates.extend(
            [
                extra_info.get("source_dataset"),
                extra_info.get("task"),
                extra_info.get("task_name"),
            ]
        )
    for candidate in candidates:
        if candidate:
            return str(candidate)
    raise ValueError("Reasoning Gym reward requires metadata.source_dataset or task in extra_info.")


def _score_with_format_retries(
    score_fn: Any,
    model_answer: str,
    entry: dict[str, Any],
    task_name: str,
) -> tuple[float, str]:
    """Try raw and normalized Reasoning Gym inputs; keep the best score."""
    best_score = 0.0
    best_answer = _normalize_answer_for_scoring(model_answer)
    last_error: Exception | None = None
    had_successful_call = False

    for entry_variant in _entry_variants(entry):
        for answer_variant in _model_answer_variants(model_answer):
            try:
                score = float(score_fn(answer=answer_variant, entry=entry_variant))
                had_successful_call = True
            except Exception as exc:
                last_error = exc
                continue

            if score > best_score:
                best_score = score
                best_answer = answer_variant
            if best_score >= 1.0:
                return best_score, best_answer

    if last_error is not None and best_score == 0.0 and not had_successful_call:
        print(f"Error scoring Reasoning Gym answer for task {task_name}: {last_error}")

    return best_score, best_answer


def compute_score(solution_str: str, ground_truth: Any, extra_info: Any = None) -> dict[str, float]:
    """Compute Reasoning Gym score for a slime sample.

    Expected data is the NeMo Gym Reasoning Gym JSONL shape:
    question, answer, and metadata.source_dataset. In slime this usually means
    label/ground_truth is the answer and metadata or extra_info is the original
    metadata dict.
    """
    entry = _entry_from_inputs(ground_truth, extra_info)
    task_name = _task_name(entry, extra_info)
    model_answer = _extract_answer_from_solution(solution_str)

    score_fn = reasoning_gym.get_score_answer_fn(task_name)
    score, model_answer = _score_with_format_retries(score_fn, model_answer, entry, task_name)

    return {
        "score": score,
        "acc": score,
        "extracted_answer": model_answer,
        "task_name": task_name,
    }
