import json
from typing import Any


def _normalize_json_str(value: Any) -> Any:
    if isinstance(value, str):
        text = value.strip()
        if "</think>" in text:
            text = text.split("</think>")[-1].strip()
        try:
            return json.loads(text)
        except Exception:
            return text
    return value


def compute_score(solution_str, ground_truth, extra_info=None):
    del extra_info
    pred = _normalize_json_str(solution_str)
    gold = _normalize_json_str(ground_truth)
    return float(pred == gold)
