import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _load_extra_info(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            return {}
    return {}


def _normalize_ground_truth(value: Any) -> list[str]:
    if value is None:
        return []
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value if item is not None]
    return [str(value)]


def _normalize_tool_schemas(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except Exception:
            return []
    if value is None:
        return []
    if not isinstance(value, list):
        value = [value]
    return [item for item in value if isinstance(item, dict)]


def _render_system_prompt(tool_schemas: list[dict[str, Any]]) -> str:
    tool_block = "\n".join(_json_dumps(tool) for tool in tool_schemas)
    return (
        "You are a helpful tool-using assistant.\n\n"
        "Solve the user's task by reasoning step by step.\n"
        "When you need a tool, output exactly one tool call in the XML format below:\n"
        "<tool_call>\n"
        '{"name": "<function-name>", "arguments": {"arg": "value"}}\n'
        "</tool_call>\n\n"
        "After each tool call, you will receive the execution result inside <tool_response></tool_response>.\n"
        "When you have enough information, stop calling tools and provide the final answer as "
        "\\boxed{your final answer}.\n"
        "Do not put the final answer anywhere except inside the last box.\n\n"
        "Available tools are provided below:\n"
        "<tools>\n"
        f"{tool_block}\n"
        "</tools>"
    )


def _convert_row(row: dict[str, Any], row_idx: int) -> dict[str, Any]:
    extra_info = _load_extra_info(row.get("extra_info"))
    tool_schemas = _normalize_tool_schemas(extra_info.get("tool_schemas") or extra_info.get("func_schemas"))
    reward_model = row.get("reward_model") or {}
    if not isinstance(reward_model, dict):
        reward_model = {"ground_truth": reward_model}

    ground_truth = _normalize_ground_truth(reward_model.get("ground_truth"))
    question = str(row.get("question") or "")

    prompt = [
        {"role": "system", "content": _render_system_prompt(tool_schemas)},
        {"role": "user", "content": question},
    ]

    metadata = {
        "data_source": str(row.get("data_source") or "syntool_recall"),
        "ability": str(row.get("ability") or "re_call"),
        "index": extra_info.get("index", row_idx),
        "env": str(extra_info.get("env") or ""),
        "tool_schemas": tool_schemas,
        "tools": tool_schemas,
        "question": question,
    }

    return {
        "prompt": prompt,
        "label": {
            "ground_truth": ground_truth,
            "style": reward_model.get("style", "rule"),
        },
        "metadata": metadata,
    }


def _validate_env_code(env_code: str) -> tuple[bool, str | None]:
    if not env_code.strip():
        return True, None
    runtime: dict[str, Any] = {}
    try:
        exec(env_code, runtime, runtime)
        return True, None
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


def convert_file(
    input_path: Path, output_path: Path, limit: int | None = None, drop_invalid_env: bool = True
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataframe = pd.read_parquet(input_path)
    if limit is not None:
        dataframe = dataframe.iloc[:limit]

    written = 0
    skipped_invalid = 0
    with output_path.open("w", encoding="utf-8") as fout:
        for row_idx, row in enumerate(dataframe.to_dict("records")):
            converted = _convert_row(row, row_idx)
            env_code = str(converted.get("metadata", {}).get("env") or "")
            is_valid, err = _validate_env_code(env_code)
            if drop_invalid_env and not is_valid:
                skipped_invalid += 1
                if skipped_invalid <= 5:
                    print(
                        f"[warn] skip invalid env sample row={row_idx} "
                        f"question={converted.get('metadata', {}).get('question', '')!r} err={err}"
                    )
                continue
            fout.write(_json_dumps(converted) + "\n")
            written += 1

    print(
        f"converted {written}/{len(dataframe)} rows from {input_path} -> {output_path} "
        f"(skipped_invalid_env={skipped_invalid})"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert syntool recall parquet files to slime jsonl prompt-data.")
    parser.add_argument("--train-input", required=True, help="Path to the raw training parquet file.")
    parser.add_argument("--test-input", required=True, help="Path to the raw evaluation parquet file.")
    parser.add_argument("--output-dir", required=True, help="Directory for the converted train/test jsonl files.")
    parser.add_argument("--limit", type=int, default=None, help="Optional limit for quick debugging.")
    parser.add_argument(
        "--keep-invalid-env",
        action="store_true",
        help="Keep samples whose env code fails to execute during preprocessing.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    drop_invalid_env = not args.keep_invalid_env
    convert_file(Path(args.train_input), output_dir / "train.jsonl", limit=args.limit, drop_invalid_env=drop_invalid_env)
    convert_file(Path(args.test_input), output_dir / "test.jsonl", limit=args.limit, drop_invalid_env=drop_invalid_env)


if __name__ == "__main__":
    main()
