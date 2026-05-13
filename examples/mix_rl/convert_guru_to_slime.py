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

READ_COLUMNS = ["prompt", "reward_model", "data_source", "ability", "extra_info"]


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


def _iter_records(src: Path, limit: int | None = None):
    count = 0
    parquet_file = pq.ParquetFile(src)
    columns = [column for column in READ_COLUMNS if column in parquet_file.schema_arrow.names]
    for batch in parquet_file.iter_batches(batch_size=1024, columns=columns):
        for row in batch.to_pylist():
            row = _to_builtin(row)
            yield {
                "prompt": row["prompt"],
                "reward_model": row.get("reward_model"),
                "metadata": {
                    "data_source": row.get("data_source"),
                    "ability": row.get("ability"),
                    "extra_info": row.get("extra_info"),
                },
            }
            count += 1
            if limit is not None and count >= limit:
                return


def convert_file(src: Path, dst: Path, limit: int | None = None) -> int:
    dst.parent.mkdir(parents=True, exist_ok=True)
    count = 0

    with dst.open("w", encoding="utf-8") as out:
        for record in _iter_records(src, limit=limit):
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1

    return count


def convert_many_to_file(sources: list[Path], dst: Path, limit: int | None = None) -> int:
    dst.parent.mkdir(parents=True, exist_ok=True)
    total = 0

    with dst.open("w", encoding="utf-8") as out:
        for src in sources:
            count = 0
            for record in _iter_records(src, limit=limit):
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
                count += 1
                total += 1
            print(f"{src} -> {dst} (+{count} rows)")

    return total


def _source_domain(src: Path) -> str:
    return src.name.split("__", 1)[0]


def _parse_domain_filter(values: list[str] | None) -> set[str] | None:
    if not values:
        return None

    domains = set()
    for value in values:
        domains.update(item.strip() for item in value.split(",") if item.strip())
    return domains


def filter_sources(
    sources: list[Path],
    *,
    include_domains: set[str] | None = None,
    exclude_domains: set[str] | None = None,
) -> list[Path]:
    selected = []
    for src in sources:
        domain = _source_domain(src)
        if include_domains is not None and domain not in include_domains:
            continue
        if exclude_domains is not None and domain in exclude_domains:
            continue
        selected.append(src)
    return selected


def iter_sources(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    return sorted(input_path.rglob("*.parquet"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="A parquet file or directory containing parquet files.")
    parser.add_argument("--output", required=True, help="Output JSONL file or directory.")
    parser.add_argument("--limit", type=int, default=None, help="Optional max rows per source file for smoke tests.")
    parser.add_argument(
        "--include-domain",
        action="append",
        default=None,
        help="Only convert these file domains, e.g. --include-domain math,codegen or --include-domain math --include-domain logic.",
    )
    parser.add_argument(
        "--exclude-domain",
        action="append",
        default=None,
        help="Skip these file domains, e.g. --exclude-domain stem.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    sources = iter_sources(input_path)
    if not sources:
        raise FileNotFoundError(f"No parquet files found under {input_path}")

    include_domains = _parse_domain_filter(args.include_domain)
    exclude_domains = _parse_domain_filter(args.exclude_domain)
    sources = filter_sources(sources, include_domains=include_domains, exclude_domains=exclude_domains)
    if not sources:
        raise ValueError(
            f"No parquet files left after domain filtering. include={include_domains}, exclude={exclude_domains}"
        )

    domains = sorted({_source_domain(src) for src in sources})
    print(f"selected {len(sources)} parquet files from domains: {', '.join(domains)}")

    if input_path.is_dir() and output_path.suffix == ".jsonl":
        total = convert_many_to_file(sources, output_path, limit=args.limit)
        print(f"wrote {total} rows from {len(sources)} files to {output_path}")
        return

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
