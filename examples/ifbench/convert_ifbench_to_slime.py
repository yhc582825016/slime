#!/usr/bin/env python3
# pyright: reportMissingImports=false

import argparse
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def convert_ifbench_to_slime(input_path: Path, output_path: Path) -> None:
    table = pq.read_table(input_path)
    rows = table.to_pylist()

    converted = []
    for row in rows:
        prompt = row.get("prompt", "")
        instruction_id_list = row.get("instruction_id_list") or []
        instruction_kwargs = row.get("kwargs") or []

        converted.append(
            {
                "messages": [{"role": "user", "content": str(prompt)}],
                "extra_info": {
                    "record_id": row.get("key"),
                    "prompt_text": str(prompt),
                    "instruction_id_list": instruction_id_list,
                    "instruction_kwargs": instruction_kwargs,
                    # Keep kwargs alias for compatibility with built-in ifbench rm_type.
                    "kwargs": instruction_kwargs,
                },
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(converted), output_path)
    print(f"Converted {len(converted)} samples -> {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert IFBench parquet to Slime IFBench format.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("examples/ifbench/IFBench_test/data/train-00000-of-00001.parquet"),
        help="Path to original IFBench parquet file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("examples/ifbench/IFBench_test/ifbench_test_slime.parquet"),
        help="Path to save converted Slime parquet file.",
    )
    args = parser.parse_args()

    convert_ifbench_to_slime(args.input, args.output)


if __name__ == "__main__":
    main()
