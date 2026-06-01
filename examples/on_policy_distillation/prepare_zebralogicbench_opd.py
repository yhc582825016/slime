#!/usr/bin/env python3
"""Convert ZebraLogicBench (ModelScope cache) to slime OPD eval jsonl.

Matches evalscope zebralogicbench: zero-shot prompt with example puzzle + JSON template.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import Dataset

DEFAULT_ARROW = (
    "/mnt/code/yehangcheng/cache_file/MODELSCOPE_CACHE/datasets/"
    "allenai___zebra_logic_bench-private/grid_mode-e41c5feb3e948ef8/"
    "0.0.0/master/zebra_logic_bench-private-test.arrow"
)
DEFAULT_OUTPUT = "/mnt/code/yehangcheng/all_data/rl_data/OPD/zebralogicbench_100_opd.jsonl"

PROMPT_TEMPLATE = """
# Example Puzzle

There are 3 houses, numbered 1 to 3 from left to right, as seen from across the street. \
Each house is occupied by a different person. \
Each house has a unique attribute for each of the following characteristics:
 - Each person has a unique name: `Peter`, `Eric`, `Arnold`.
 - Each person has a unique favorite drink: `tea`, `water`, `milk`

## Clues for the Example Puzzle

1. Peter is in the second house.
2. Arnold is directly left of the one who only drinks water.
3. The one who only drinks water is directly left of the person who likes milk.

## Answer to the Example Puzzle

{{
    "reasoning": "Given Clue 1, we know Peter is in House 2. According to Clue 2, \
Arnold is directly left of the one who only drinks water. \
The person in House 3 cannot be on the left of anyone, so Arnold must be in House 1. \
Thus, Peter drinks water, and Eric lives in House 3. \
Then, according to Clue 3, Eric drinks milk. Therefore, Arnold drinks tea.",
    "solution": {{
        "House 1": {{
            "Name": "Arnold",
            "Drink": "tea"
        }},
        "House 2": {{
            "Name": "Peter",
            "Drink": "water"
        }},
        "House 3": {{
            "Name": "Eric",
            "Drink": "milk"
        }}
    }}
}}

# Puzzle to Solve

{question}


# Instruction

Now please solve the above puzzle. Present your reasoning and solution in the following json format:

{json_template}

""".lstrip()


def build_json_template(solution: dict) -> str:
    num_houses = len(solution["rows"])
    columns = solution["header"]
    assert columns[0] == "House"
    json_template = {"reasoning": "___", "solution": {}}
    for i in range(num_houses):
        json_template["solution"][f"House {i + 1}"] = {columns[j]: "___" for j in range(1, len(columns))}
    return json.dumps(json_template, indent=4)


def format_prompt(puzzle: str, solution: dict) -> str:
    json_template = build_json_template(solution)
    return PROMPT_TEMPLATE.format(question=puzzle, json_template=json_template)


def process_record(record: dict) -> dict:
    solution = record["solution"]
    prompt = format_prompt(record["puzzle"], solution)
    return {
        "messages": [{"role": "user", "content": prompt}],
        "solution": json.dumps(solution),
        "metadata": {
            "rm_type": "zebralogicbench",
            "data_source": "logic__zebralogicbench",
            "record_id": record.get("id"),
            "size": record.get("size"),
            "created_at": record.get("created_at"),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arrow-path", type=str, default=DEFAULT_ARROW)
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT)
    parser.add_argument("--limit", type=int, default=100)
    args = parser.parse_args()

    ds = Dataset.from_file(args.arrow_path)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    count = min(args.limit, len(ds))
    with out_path.open("w", encoding="utf-8") as f:
        for index in range(count):
            row = process_record(ds[index])
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Wrote {count} rows to {out_path}")


if __name__ == "__main__":
    main()
