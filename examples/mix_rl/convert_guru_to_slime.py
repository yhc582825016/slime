"""Convert Reasoning360/Guru parquet files into slime-friendly JSONL.

Reasoning360 stores reward inputs across `data_source`, `reward_model`, and
`extra_info`. Slime reads one label key and one metadata key, so this converter
packs the needed reward fields into:
  {"prompt": ..., "reward_model": ..., "metadata": {"data_source": ..., "extra_info": ...}}
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq


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


def convert_file(src: Path, dst: Path, limit: int | None = None) -> int:
    dst.parent.mkdir(parents=True, exist_ok=True)
    count = 0

    with dst.open("w", encoding="utf-8") as out:
        parquet_file = pq.ParquetFile(src)
        for batch in parquet_file.iter_batches():
            for row in batch.to_pylist():
                row = _to_builtin(row)
                record = {
                    "prompt": row["prompt"],
                    "reward_model": row.get("reward_model"),
                    "metadata": {
                        "data_source": row.get("data_source"),
                        "ability": row.get("ability"),
                        "extra_info": row.get("extra_info"),
                    },
                }
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
                count += 1
                if limit is not None and count >= limit:
                    return count

    return count


def iter_sources(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    return sorted(input_path.rglob("*.parquet"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="A parquet file or directory containing parquet files.")
    parser.add_argument("--output", required=True, help="Output JSONL file or directory.")
    parser.add_argument("--limit", type=int, default=None, help="Optional max rows per source file for smoke tests.")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    sources = iter_sources(input_path)
    if not sources:
        raise FileNotFoundError(f"No parquet files found under {input_path}")

    for src in sources:
        if input_path.is_file():
            dst = output_path
        else:
            rel = src.relative_to(input_path).with_suffix(".jsonl")
            dst = output_path / rel
        count = convert_file(src, dst, limit=args.limit)
        print(f"{src} -> {dst} ({count} rows)")


if __name__ == "__main__":
    main()
