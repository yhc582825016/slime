from __future__ import annotations

import argparse
import ast
import json
import random
import re
from pathlib import Path
from typing import Any

import pandas as pd


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def _safe_loads(value: Any) -> Any:
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


def _extract_tool_schemas(system_prompt: str) -> list[dict[str, Any]]:
    matches = re.findall(r"<tools>(.*?)</tools>", system_prompt or "", re.DOTALL)
    for raw in reversed(matches):
        raw = raw.strip()
        if not raw:
            continue
        parsed = _safe_loads(raw)
        if isinstance(parsed, list):
            return [item for item in parsed if isinstance(item, dict)]
    return []


def _render_system_prompt(system_prompt: str) -> str:
    suffix = (
        "\n\n### Interaction protocol\n"
        "Use <tool_call>{\"name\": \"tool_name\", \"arguments\": {...}}</tool_call> when calling a tool.\n"
        "Tool results will be returned inside <tool_response></tool_response>.\n"
        "If you need more information from the user, ask exactly one concise question inside <question></question>.\n"
        "When the task is complete, provide your final response inside <answer></answer> and include ###STOP.\n"
        "If policy requires escalation, include ###TRANSFER_TO_HUMAN."
    )
    return system_prompt if "### Interaction protocol" in system_prompt else system_prompt + suffix


def _messages_from_row(row: dict[str, Any]) -> list[dict[str, str]]:
    messages = _safe_loads(row.get("messages_json"))
    if isinstance(messages, list) and len(messages) >= 2:
        normalized = []
        for message in messages:
            if isinstance(message, dict):
                normalized.append({"role": str(message.get("role", "user")), "content": str(message.get("content", ""))})
        if normalized:
            normalized[0]["content"] = _render_system_prompt(normalized[0]["content"])
            return normalized
    return [
        {"role": "system", "content": _render_system_prompt(str(row.get("system") or ""))},
        {"role": "user", "content": str(row.get("user") or "")},
    ]


def _convert_row(row: dict[str, Any], row_idx: int, split: str) -> dict[str, Any]:
    messages = _messages_from_row(row)
    system_prompt = messages[0]["content"] if messages and messages[0]["role"] == "system" else str(row.get("system") or "")
    tool_schemas = _extract_tool_schemas(system_prompt)
    tool_return_expected = _safe_loads(row.get("tool_return_expected_json"))
    if not isinstance(tool_return_expected, dict):
        tool_return_expected = {}

    metadata = {
        "data_source": "agenticqwen_virtual_tool",
        "ability": "tool_use",
        "split": split,
        "index": row.get("id", row_idx),
        "question": str(row.get("user") or (messages[-1]["content"] if messages else "")),
        "rubrics": str(row.get("rubrics") or ""),
        "policy": str(row.get("system") or ""),
        "task_background": str(row.get("task_background") or ""),
        "test_policy": str(row.get("test_policy") or ""),
        "user_escape_strategy": str(row.get("user_escape_strategy") or ""),
        "tool_return_expected": _json_dumps(tool_return_expected),
        "tool_schemas": tool_schemas,
        "tools": tool_schemas,
        "initial_messages": messages,
    }

    return {
        "prompt": messages,
        "tools": tool_schemas,
        "label": {"ground_truth": "None", "style": "rule"},
        "metadata": metadata,
    }


def _split_dataframe(df: pd.DataFrame, train_ratio: float, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    indices = list(range(len(df)))
    random.Random(seed).shuffle(indices)
    cut = int(len(indices) * train_ratio)
    train_idx = indices[:cut]
    test_idx = indices[cut:]
    return df.iloc[train_idx].reset_index(drop=True), df.iloc[test_idx].reset_index(drop=True)


def _write_jsonl(df: pd.DataFrame, path: Path, split: str, limit: int | None = None) -> None:
    if limit is not None:
        df = df.iloc[:limit]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fout:
        for idx, row in enumerate(df.to_dict("records")):
            fout.write(_json_dumps(_convert_row(row, idx, split)) + "\n")
    print(f"wrote {len(df)} {split} rows -> {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert AgenticQwen parquet into slime jsonl prompt-data.")
    parser.add_argument("--input", default="/dev/shm/ye/rl-data/AgenticQwen-Data/agenticqwen_synthetic_data.parquet")
    parser.add_argument("--train-input", default=None, help="Optional pre-split train parquet.")
    parser.add_argument("--test-input", default=None, help="Optional pre-split test parquet.")
    parser.add_argument("--output-dir", default="/dev/shm/ye/slime/examples/agenticqwen/data")
    parser.add_argument("--train-ratio", type=float, default=0.98)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    out = Path(args.output_dir)
    if args.train_input and args.test_input:
        train_df = pd.read_parquet(args.train_input)
        test_df = pd.read_parquet(args.test_input)
    else:
        df = pd.read_parquet(args.input)
        train_df, test_df = _split_dataframe(df, args.train_ratio, args.seed)

    _write_jsonl(train_df, out / "train.jsonl", "train", args.limit)
    _write_jsonl(test_df, out / "test.jsonl", "test", args.limit)


if __name__ == "__main__":
    main()

