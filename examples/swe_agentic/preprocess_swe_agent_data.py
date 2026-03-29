import argparse
import json
from pathlib import Path
from typing import Any


def _to_json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True)


def _extract_text_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        texts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") == "output_text":
                    texts.append(str(item.get("text", "")))
                elif "text" in item:
                    texts.append(str(item.get("text", "")))
            elif isinstance(item, str):
                texts.append(item)
        return "\n".join(x for x in texts if x.strip())
    return str(content)


def _normalize_tool(tool: dict[str, Any]) -> dict[str, Any]:
    # Dataset uses {"name","description","parameters"} style tools.
    # Convert to a stable OpenAI function-tool shape for prompt rendering.
    if "function" in tool:
        return tool
    return {
        "type": "function",
        "function": {
            "name": tool.get("name", ""),
            "description": tool.get("description", ""),
            "parameters": tool.get("parameters", {"type": "object", "properties": {}, "required": []}),
        },
    }


def _render_prompt(input_events: list[dict[str, Any]], tools: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    lines.append("You are an agent that must choose the next tool action.")
    lines.append("Return exactly ONE tool call in this format:")
    lines.append("<tool_call>{\"name\":\"...\",\"arguments\":{...}}</tool_call>")
    lines.append("")
    lines.append("<tools>")
    for tool in tools:
        lines.append(_to_json(_normalize_tool(tool)))
    lines.append("</tools>")
    lines.append("")
    lines.append("<conversation>")

    for ev in input_events:
        role = ev.get("role")
        ev_type = ev.get("type")

        if role in {"system", "user"}:
            text = _extract_text_content(ev.get("content", ""))
            lines.append(f"[{role}] {text}")
            continue

        if ev_type == "message":
            msg_role = ev.get("role", "assistant")
            text = _extract_text_content(ev.get("content", ""))
            lines.append(f"[{msg_role}] {text}")
        elif ev_type == "function_call":
            call = {"name": ev.get("name"), "arguments": ev.get("arguments")}
            lines.append(f"[assistant_tool_call] {_to_json(call)}")
        elif ev_type == "function_call_output":
            call_id = ev.get("call_id")
            output = _extract_text_content(ev.get("output", ""))
            lines.append(f"[tool_output:{call_id}] {output}")
        elif ev_type == "reasoning":
            # Keep concise thought summary (if present) as context.
            summary = ev.get("summary", [])
            text = _extract_text_content(summary)
            if text.strip():
                lines.append(f"[assistant_reasoning] {text}")

    lines.append("</conversation>")
    lines.append("")
    lines.append("Produce only the next tool call.")
    return "\n".join(lines)


def convert_file(input_path: Path, output_path: Path, limit: int | None = None) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    kept = 0

    with input_path.open("r", encoding="utf-8") as fin, output_path.open("w", encoding="utf-8") as fout:
        for raw in fin:
            total += 1
            if limit is not None and kept >= limit:
                break

            try:
                obj = json.loads(raw)
            except Exception:
                continue

            expected_action = obj.get("expected_action")
            rcp = obj.get("responses_create_params", {})
            input_events = rcp.get("input", [])
            tools = rcp.get("tools", [])

            if not expected_action or not isinstance(input_events, list):
                continue

            prompt_text = _render_prompt(input_events=input_events, tools=tools)
            # Keep prompt as OpenAI-style messages so slime can apply chat template
            # when processor is enabled (e.g. VLM checkpoints).
            prompt = [
                {"role": "system", "content": "You are a tool-calling code agent."},
                {"role": "user", "content": prompt_text},
            ]
            norm_tools = [_normalize_tool(t) for t in tools if isinstance(t, dict)]
            sample = {
                "prompt": prompt,
                "tools": norm_tools,
                "label": "",
                "metadata": {
                    "trajectory_id": obj.get("trajectory_id"),
                    "instance_id": (obj.get("metadata") or {}).get("instance_id"),
                    "tools": norm_tools,
                    "expected_action": expected_action,
                },
            }
            fout.write(_to_json(sample) + "\n")
            kept += 1

    print(f"done: read={total}, kept={kept}, output={output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert SWE Agent trajectory jsonl to slime prompt-data jsonl.")
    parser.add_argument("--input", required=True, help="Path to original trajectory jsonl")
    parser.add_argument("--output", required=True, help="Path to converted jsonl")
    parser.add_argument("--limit", type=int, default=None, help="Optional max kept samples")
    args = parser.parse_args()

    convert_file(Path(args.input), Path(args.output), args.limit)


if __name__ == "__main__":
    main()
