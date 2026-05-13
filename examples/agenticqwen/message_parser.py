from __future__ import annotations

import re
from typing import Any


def content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                parts.append(str(item.get("text") or item.get("content") or ""))
            else:
                parts.append(str(item))
        return "\n".join(part for part in parts if part)
    return str(content)


def extract_user_conversation(messages: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    round_id = 0
    for message in messages:
        role = message.get("role")
        content = content_to_text(message.get("content", ""))
        if role == "user":
            if "<tool_response>" in content:
                continue
            round_id += 1
            parts.append(f"[Round {round_id} - You]:\n{content.strip()}")
        elif role == "assistant":
            questions = re.findall(r"<question>(.*?)</question>", content, flags=re.DOTALL)
            if questions:
                parts.append(f"[Round {round_id} - Agent]:\n" + "\n".join(q.strip() for q in questions))
    return "\n\n".join(parts)


def extract_tool_history(messages: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    turn_number = 0
    for message in messages:
        role = message.get("role")
        content = content_to_text(message.get("content", ""))
        if role == "assistant":
            for tool_call in re.findall(r"<tool_call>(.*?)</tool_call>", content, flags=re.DOTALL):
                turn_number += 1
                parts.append(f"[Tool Call {turn_number}]:\n{tool_call.strip()}")
        elif role == "user":
            for tool_response in re.findall(r"<tool_response>(.*?)</tool_response>", content, flags=re.DOTALL):
                parts.append(f"[Tool Response {turn_number}]:\n{tool_response.strip()}")
    return "\n\n".join(parts)


def extract_solution_summary(messages: list[dict[str, Any]], response: str = "") -> str:
    all_messages = list(messages)
    if response:
        all_messages.append({"role": "assistant", "content": response})

    parts: list[str] = []
    round_id = 0
    for message in all_messages:
        role = message.get("role")
        content = content_to_text(message.get("content", ""))
        if role == "user":
            tool_responses = re.findall(r"<tool_response>(.*?)</tool_response>", content, flags=re.DOTALL)
            if tool_responses:
                for item in tool_responses:
                    parts.append(f"[Tool Response]:\n{item.strip()}")
            else:
                round_id += 1
                parts.append(f"[Round {round_id} - User]:\n{content.strip()}")
        elif role == "assistant":
            for item in re.findall(r"<question>(.*?)</question>", content, flags=re.DOTALL):
                parts.append(f"[Round {round_id} - Agent Question]:\n{item.strip()}")
            for item in re.findall(r"<tool_call>(.*?)</tool_call>", content, flags=re.DOTALL):
                parts.append(f"[Tool Call - Agent]:\n{item.strip()}")
            for item in re.findall(r"<answer>(.*?)</answer>", content, flags=re.DOTALL):
                parts.append(f"[Agent Answer]:\n{item.strip()}")
            if "###TRANSFER_TO_HUMAN" in content:
                parts.append("[Agent Escalated to Human]: ###TRANSFER_TO_HUMAN")
            if "###STOP" in content:
                parts.append("[Agent Stop]: ###STOP")
    return "\n\n".join(parts)

