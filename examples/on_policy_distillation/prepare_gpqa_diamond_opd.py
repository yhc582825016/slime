#!/usr/bin/env python3
"""Convert GPQA Diamond (ModelScope cache) to slime OPD eval jsonl.

Matches evalscope gpqa_diamond: shuffled 4-way MCQ + CoT prompt ending with ANSWER: [LETTER].
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

from datasets import Dataset

DEFAULT_ARROW = (
    "/mnt/code/yehangcheng/cache_file/MODELSCOPE_CACHE/datasets/"
    "AI-ModelScope___gpqa_diamond/default-45d2f6f5c5b4a067/0.0.0/master/gpqa_diamond-train.arrow"
)
DEFAULT_OUTPUT = "/mnt/code/yehangcheng/all_data/rl_data/OPD/gpqa_diamond_opd.jsonl"

PROMPT_TEMPLATE = (
    "Answer the following multiple choice question. The last line of your response should be of the "
    "following format: 'ANSWER: [LETTER]' (without quotes) where [LETTER] is one of {letters}. "
    "Think step by step before answering.\n\n{question}\n\n{choices}"
)


def preprocess(text: str | None) -> str:
    if text is None:
        return " "
    text = text.strip()
    text = text.replace(" [title]", ". ")
    text = re.sub(r"\[.*?\]", "", text)
    text = text.replace("  ", " ")
    return text


def format_choices(choices: list[str]) -> str:
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    return "\n".join(f"{letters[i]}) {choice}" for i, choice in enumerate(choices))


def process_record(record: dict, index: int, seed: int) -> dict:
    choices = [
        preprocess(record["Incorrect Answer 1"]),
        preprocess(record["Incorrect Answer 2"]),
        preprocess(record["Incorrect Answer 3"]),
        preprocess(record["Correct Answer"]),
    ]
    rng = random.Random(seed + index)
    rng.shuffle(choices)
    correct = preprocess(record["Correct Answer"])
    correct_index = choices.index(correct)
    correct_letter = chr(65 + correct_index)
    letters = ",".join(chr(65 + i) for i in range(len(choices)))

    question = preprocess(record["Question"])
    choices_text = format_choices(choices)
    prompt = PROMPT_TEMPLATE.format(letters=letters, question=question, choices=choices_text)

    return {
        "messages": [{"role": "user", "content": prompt}],
        "solution": correct_letter,
        "metadata": {
            "rm_type": "gpqa",
            "data_source": "stem__gpqa_diamond",
            "record_id": record.get("Record ID"),
            "choices": choices,
            "correct_letter": correct_letter,
            "correct_answer": correct,
            "incorrect_answers": [
                preprocess(record["Incorrect Answer 1"]),
                preprocess(record["Incorrect Answer 2"]),
                preprocess(record["Incorrect Answer 3"]),
            ],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arrow-path", type=str, default=DEFAULT_ARROW)
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=42, help="Base seed; per-row shuffle uses seed + index")
    args = parser.parse_args()

    ds = Dataset.from_file(args.arrow_path)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as f:
        for index, record in enumerate(ds):
            row = process_record(record, index, args.seed)
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Wrote {len(ds)} rows to {out_path}")


if __name__ == "__main__":
    main()
