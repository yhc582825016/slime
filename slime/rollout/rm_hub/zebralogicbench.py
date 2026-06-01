import json
from typing import List


def extract_last_complete_json(text: str) -> str:
    stack: List[int] = []
    last_json_start: int | None = None
    last_json_str: str | None = None

    for i, char in enumerate(text):
        if char == "{":
            if not stack:
                last_json_start = i
            stack.append(i)
        elif char == "}":
            if stack:
                stack.pop()
                if not stack and last_json_start is not None:
                    last_json_str = text[last_json_start : i + 1]
                    last_json_start = None

    if last_json_str:
        try:
            return json.dumps(json.loads(last_json_str))
        except json.JSONDecodeError:
            return ""
    return ""


def process_results(prediction: str, reference: str) -> dict[str, int]:
    """Compare model prediction against reference; returns per-sample counts."""
    solved_puzzles = 0
    correct_cells = 0
    total_cells = 0
    no_answer = 0
    reason_lens: List[int] = []

    prediction_obj = json.loads(prediction)
    reference_obj = json.loads(reference)

    num_houses = len(reference_obj["rows"])
    columns = reference_obj["header"]
    this_total_cells = (len(columns) - 1) * num_houses
    total_cells += this_total_cells

    reference_table: dict[str, dict[str, str]] = {}
    for i in range(num_houses):
        reference_table[f"House {i + 1}"] = {
            columns[j]: reference_obj["rows"][i][j] for j in range(1, len(columns))
        }

    prediction_table = prediction_obj.get("solution", None)
    if prediction_table:
        reason = prediction_obj.get("reasoning", "")
        reason_lens.append(len(reason))

        this_correct_cells = 0
        for house, col_map in reference_table.items():
            for column, truth in col_map.items():
                if house not in prediction_table or column not in prediction_table[house]:
                    continue

                truth_cell = truth.lower().strip()
                pred_val = prediction_table[house][column]
                if pred_val is None:
                    continue

                if isinstance(pred_val, list):
                    if not pred_val:
                        continue
                    predicted_cell = str(pred_val[0]).lower().strip()
                elif isinstance(pred_val, str):
                    predicted_cell = pred_val.lower().strip()
                else:
                    raise ValueError(f"Unknown type: {type(pred_val)}")

                if truth_cell == predicted_cell:
                    this_correct_cells += 1

        correct_cells += this_correct_cells
        if this_correct_cells == this_total_cells:
            solved_puzzles += 1
    else:
        no_answer += 1

    return {
        "Solved Puzzle": solved_puzzles,
        "Solved Cell": correct_cells,
        "Cell Num": total_cells,
        "No answer": no_answer,
        "Reason Lens": sum(reason_lens),
    }


def compute_zebralogicbench_reward(response: str, label, metadata: dict | None = None) -> float:
    """Return puzzle_acc (1.0 if fully solved, else 0.0) for one sample."""
    if not response or not label:
        return 0.0

    filtered = extract_last_complete_json(response)
    if not filtered:
        return 0.0

    reference = label if isinstance(label, str) else json.dumps(label)
    try:
        results = process_results(filtered, reference)
        return float(results.get("Solved Puzzle", 0))
    except Exception:
        return 0.0
