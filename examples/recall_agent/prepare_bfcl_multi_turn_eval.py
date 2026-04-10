import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any


DEFAULT_BFCL_ROOT = Path("/dev/shm/ye/gorilla/berkeley-function-call-leaderboard")
MULTI_TURN_DATA_FILES = [
    "BFCL_v4_multi_turn_base.json",
    "BFCL_v4_multi_turn_long_context.json",
    "BFCL_v4_multi_turn_miss_func.json",
    "BFCL_v4_multi_turn_miss_param.json",
]


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fin:
        for line_no, line in enumerate(fin, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Failed to parse {path}:{line_no}: {exc}") from exc
            if not isinstance(payload, dict):
                raise TypeError(f"Expected dict records in {path}, got {type(payload).__name__}")
            records.append(payload)
    return records


def _load_func_docs(func_doc_root: Path, file_mapping: dict[str, str]) -> dict[str, list[dict[str, Any]]]:
    docs_by_class: dict[str, list[dict[str, Any]]] = {}
    for class_name, file_name in file_mapping.items():
        docs_by_class[class_name] = _load_jsonl(func_doc_root / file_name)
    return docs_by_class


def _render_tool_block(tool_schemas: list[dict[str, Any]]) -> str:
    tool_lines = [_json_dumps(tool) for tool in tool_schemas if isinstance(tool, dict)]
    return "<tools>\n" + "\n".join(tool_lines) + "\n</tools>" if tool_lines else "<tools>\n</tools>"


def _flatten_turn_text(turns: list[list[dict[str, Any]]]) -> str:
    rendered_turns: list[str] = []
    for idx, turn in enumerate(turns, start=1):
        lines: list[str] = []
        for message in turn:
            if not isinstance(message, dict):
                lines.append(str(message))
                continue
            role = str(message.get("role") or "user")
            content = message.get("content", "")
            if isinstance(content, list):
                content = "\n".join(
                    str(item.get("text", item)) if isinstance(item, dict) else str(item) for item in content
                )
            content = str(content).strip()
            if not content:
                continue
            if role == "user":
                lines.append(content)
            else:
                lines.append(f"[{role}] {content}")
        if lines:
            rendered_turns.append(f"Turn {idx}:\n" + "\n".join(lines))
    return "\n\n".join(rendered_turns)


def _render_system_prompt(initial_tools: list[dict[str, Any]]) -> str:
    return (
        "You are a helpful tool-using assistant solving a multi-turn function-calling task.\n\n"
        "Rules:\n"
        "1. For the current user turn, reason step by step and call tools when needed.\n"
        "2. Emit tool calls in the exact XML format below:\n"
        "<tool_call>\n"
        '{"name": "<function-name>", "arguments": {"arg": "value"}}\n'
        "</tool_call>\n"
        "3. After each tool call, you will receive execution results inside <tool_response></tool_response>.\n"
        "4. When the current user turn is complete, stop calling tools and wait. The environment may provide the next user turn.\n"
        "5. Additional tools may become available in later turns.\n"
        "6. After the final user turn is complete, output the full ordered list of tool calls across all turns in "
        '\\boxed{["func1(arg=...)", "func2(arg=...)", ...]}.\n'
        "7. Each item in the final boxed list must be a Python-style function-call string.\n"
        "8. Do not put the final answer anywhere except inside the last box.\n\n"
        "Available tools for the current turn are provided below:\n"
        f"{_render_tool_block(initial_tools)}"
    )


def _flatten_ground_truth(ground_truth_turns: Any) -> list[str]:
    flattened: list[str] = []
    if not isinstance(ground_truth_turns, list):
        return flattened
    for turn in ground_truth_turns:
        if not isinstance(turn, list):
            continue
        for call in turn:
            if call is None:
                continue
            call_text = str(call).strip()
            if call_text:
                flattened.append(call_text)
    return flattened


def _render_env_code(
    *,
    bfcl_root: Path,
    class_file_mapping: dict[str, str],
    initial_config: dict[str, Any],
    involved_classes: list[str],
    long_context: bool,
) -> str:
    return "\n".join(
        [
            "import copy",
            "import importlib",
            "import inspect",
            "import sys",
            "",
            f"BFCL_ROOT = {bfcl_root.as_posix()!r}",
            "if BFCL_ROOT not in sys.path:",
            "    sys.path.insert(0, BFCL_ROOT)",
            f"CLASS_FILE_PATH_MAPPING = {class_file_mapping!r}",
            f"INITIAL_CONFIG = {initial_config!r}",
            f"INVOLVED_CLASSES = {involved_classes!r}",
            f"LONG_CONTEXT = {long_context!r}",
            "",
            "for class_name in INVOLVED_CLASSES:",
            "    module = importlib.import_module(CLASS_FILE_PATH_MAPPING[class_name])",
            "    class_ = getattr(module, class_name)",
            "    instance = class_()",
            "    if hasattr(instance, '_load_scenario'):",
            "        class_initial_config = copy.deepcopy(INITIAL_CONFIG.get(class_name, {}))",
            "        instance._load_scenario(class_initial_config, long_context=LONG_CONTEXT)",
            "    globals()[f'__bfcl_{class_name.lower()}_instance'] = instance",
            "    for method_name, method in inspect.getmembers(instance, predicate=inspect.ismethod):",
            "        if method_name.startswith('_'):",
            "            continue",
            "        globals()[method_name] = method",
        ]
    )


def _prepare_tool_sets(
    entry: dict[str, Any],
    docs_by_class: dict[str, list[dict[str, Any]]],
) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    involved_classes = entry.get("involved_classes") or []
    all_tools: list[dict[str, Any]] = []
    for class_name in involved_classes:
        all_tools.extend(json.loads(json.dumps(docs_by_class[class_name], ensure_ascii=False)))

    additional_tools_by_turn: dict[str, list[dict[str, Any]]] = {}
    missed_function = entry.get("missed_function") or {}
    if not isinstance(missed_function, dict):
        return all_tools, additional_tools_by_turn

    remaining_tools: list[dict[str, Any]] = []
    holdout_names = {
        str(turn_idx): {str(name) for name in names if name is not None}
        for turn_idx, names in missed_function.items()
        if isinstance(names, list)
    }
    for tool in all_tools:
        tool_name = str(tool.get("name") or "").strip()
        matched_turn = None
        for turn_idx, names in holdout_names.items():
            if tool_name in names:
                matched_turn = turn_idx
                break
        if matched_turn is None:
            remaining_tools.append(tool)
            continue
        additional_tools_by_turn.setdefault(matched_turn, []).append(tool)

    return remaining_tools, additional_tools_by_turn


def _convert_entry(
    *,
    entry: dict[str, Any],
    ground_truth_entry: dict[str, Any],
    docs_by_class: dict[str, list[dict[str, Any]]],
    class_file_mapping: dict[str, str],
    bfcl_root: Path,
) -> dict[str, Any]:
    initial_tools, additional_tools_by_turn = _prepare_tool_sets(entry, docs_by_class)
    conversation_turns = entry.get("question") or []
    if not isinstance(conversation_turns, list) or not conversation_turns:
        raise ValueError(f"Entry {entry.get('id')} does not contain valid multi-turn messages.")

    first_turn = conversation_turns[0]
    if not isinstance(first_turn, list):
        raise ValueError(f"Entry {entry.get('id')} first turn is not a list.")

    flattened_ground_truth = _flatten_ground_truth(ground_truth_entry.get("ground_truth"))
    if not flattened_ground_truth:
        raise ValueError(f"Entry {entry.get('id')} does not contain non-empty ground truth.")

    long_context = "long_context" in str(entry.get("id") or "")
    env_code = _render_env_code(
        bfcl_root=bfcl_root,
        class_file_mapping=class_file_mapping,
        initial_config=entry.get("initial_config") or {},
        involved_classes=entry.get("involved_classes") or [],
        long_context=long_context,
    )

    prompt = [{"role": "system", "content": _render_system_prompt(initial_tools)}]
    prompt.extend(first_turn)

    return {
        "prompt": prompt,
        "label": {
            "ground_truth": [json.dumps(flattened_ground_truth, ensure_ascii=False)],
            "style": "rule",
        },
        "metadata": {
            "data_source": "bfcl_multi_turn",
            "ability": "re_call",
            "bfcl_eval_mode": True,
            "index": str(entry.get("id") or ""),
            "question": _flatten_turn_text(conversation_turns),
            "env": env_code,
            "tool_schemas": initial_tools,
            "tools": [],
            "conversation_turns": conversation_turns,
            "additional_tools_by_turn": additional_tools_by_turn,
            "expected_tool_sequence": flattened_ground_truth,
            "bfcl_category": str(entry.get("id") or "").rsplit("_", 1)[0],
            "bfcl_ground_truth_turns": ground_truth_entry.get("ground_truth") or [],
            "bfcl_test_entry": {
                "id": str(entry.get("id") or ""),
                "initial_config": entry.get("initial_config") or {},
                "involved_classes": entry.get("involved_classes") or [],
            },
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare a sampled BFCL multi-turn eval set for recall_agent.")
    parser.add_argument(
        "--bfcl-root",
        type=Path,
        default=DEFAULT_BFCL_ROOT,
        help="Path to the Berkeley Function Calling Leaderboard repository root.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        required=True,
        help="Output jsonl path for the converted evaluation dataset.",
    )
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=None,
        help="Optional manifest path. Defaults to <output-path>.manifest.json.",
    )
    parser.add_argument("--sample-size", type=int, default=200, help="Number of samples to draw from the 800 examples.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for deterministic sampling.")
    args = parser.parse_args()

    bfcl_root = args.bfcl_root.resolve()
    data_root = bfcl_root / "bfcl_eval" / "data"
    possible_answer_root = data_root / "possible_answer"
    func_doc_root = data_root / "multi_turn_func_doc"

    if str(bfcl_root) not in sys.path:
        sys.path.insert(0, str(bfcl_root))
    from bfcl_eval.constants.executable_backend_config import (  # pylint: disable=import-error
        CLASS_FILE_PATH_MAPPING,
        MULTI_TURN_FUNC_DOC_FILE_MAPPING,
    )

    docs_by_class = _load_func_docs(func_doc_root, MULTI_TURN_FUNC_DOC_FILE_MAPPING)

    records: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for file_name in MULTI_TURN_DATA_FILES:
        question_entries = _load_jsonl(data_root / file_name)
        answer_entries = _load_jsonl(possible_answer_root / file_name)
        answer_by_id = {entry["id"]: entry for entry in answer_entries}
        for entry in question_entries:
            entry_id = entry.get("id")
            if entry_id not in answer_by_id:
                raise KeyError(f"Missing possible answer for {entry_id} in {file_name}")
            records.append((entry, answer_by_id[entry_id]))

    if args.sample_size <= 0:
        raise ValueError("sample-size must be positive.")
    if args.sample_size > len(records):
        raise ValueError(f"sample-size {args.sample_size} exceeds total records {len(records)}.")

    rng = random.Random(args.seed)
    sampled_pairs = rng.sample(records, args.sample_size)
    sampled_pairs.sort(key=lambda pair: str(pair[0].get("id") or ""))

    converted = [
        _convert_entry(
            entry=entry,
            ground_truth_entry=ground_truth_entry,
            docs_by_class=docs_by_class,
            class_file_mapping=CLASS_FILE_PATH_MAPPING,
            bfcl_root=bfcl_root,
        )
        for entry, ground_truth_entry in sampled_pairs
    ]

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    with args.output_path.open("w", encoding="utf-8") as fout:
        for item in converted:
            fout.write(_json_dumps(item) + "\n")

    manifest_path = args.manifest_path or args.output_path.with_suffix(args.output_path.suffix + ".manifest.json")
    category_counter = Counter(item["metadata"]["bfcl_category"] for item in converted)
    manifest = {
        "bfcl_root": bfcl_root.as_posix(),
        "sample_size": args.sample_size,
        "seed": args.seed,
        "output_path": args.output_path.as_posix(),
        "categories": dict(sorted(category_counter.items())),
        "sampled_ids": [item["metadata"]["index"] for item in converted],
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print(
        f"prepared {len(converted)} BFCL multi-turn eval samples -> {args.output_path} "
        f"(seed={args.seed}, categories={dict(sorted(category_counter.items()))})"
    )


if __name__ == "__main__":
    main()
