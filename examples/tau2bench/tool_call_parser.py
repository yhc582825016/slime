from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any


@dataclass
class ParsedToolCall:
    name: str | None
    arguments: dict[str, Any]
    error: str | None = None


_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)


def _parse_xml_parameters(segment: str) -> dict[str, Any]:
    """
    Parse parameters from tau2 / chat-template XML tool format, e.g.:
      <parameter=patient_id>
      FPP5D10
      </parameter>
    and nested:
      <parameter>
      <key>customer_id</key>
      <value>Maria Lopez</value>
      </parameter>
    """
    args: dict[str, Any] = {}
    for m in re.finditer(
        r"<parameter\s*=\s*([a-zA-Z0-9_.-]+)\s*>(.*?)</parameter>",
        segment,
        re.DOTALL | re.IGNORECASE,
    ):
        args[m.group(1)] = m.group(2).strip()

    for m in re.finditer(
        r"<parameter>\s*(.*?)\s*</parameter>",
        segment,
        re.DOTALL | re.IGNORECASE,
    ):
        block = m.group(1)
        km = re.search(r"<key>\s*(.*?)\s*</key>", block, re.DOTALL | re.IGNORECASE)
        vm = re.search(r"<value>\s*(.*?)\s*</value>", block, re.DOTALL | re.IGNORECASE)
        if km and vm:
            args[km.group(1).strip()] = vm.group(1).strip()

    return args


def _parse_xml_style_tool_call(text: str) -> ParsedToolCall | None:
    """
    Parse <function=name>...</function> (and simple <name>...</name>) blocks
    inside a single <tool_call> inner blob. Returns None if no recognizable function.
    """
    t = (text or "").strip()
    if not t:
        return None

    name: str | None = None
    arg_segment = t

    m = re.search(r"<function\s*=\s*([a-zA-Z0-9_.-]+)\s*>", t, re.IGNORECASE)
    if m:
        name = m.group(1).strip()
        end = re.search(r"</function\s*>", t[m.end() :], re.IGNORECASE)
        if end:
            inner = t[m.end() : m.end() + end.start()]
            arg_segment = inner
        else:
            arg_segment = t[m.end() :]
    else:
        m2 = re.search(
            r"<function>\s*([a-zA-Z0-9_.-]+)\s*</function>",
            t,
            re.DOTALL | re.IGNORECASE,
        )
        if m2:
            name = m2.group(1).strip()
            arg_segment = t
        else:
            m3 = re.search(r"<name>\s*([a-zA-Z0-9_.-]+)\s*</name>", t, re.IGNORECASE)
            if m3:
                name = m3.group(1).strip()
                arg_segment = t

    if not name:
        return None

    arguments = _parse_xml_parameters(arg_segment)
    if not arguments:
        arguments = _parse_xml_parameters(t)

    return ParsedToolCall(name=name, arguments=arguments, error=None)


def _parse_single_tool_call(raw: str) -> ParsedToolCall:
    text = (raw or "").strip()
    if not text:
        return ParsedToolCall(name=None, arguments={}, error="empty tool call")

    json_err: str | None = None
    try:
        payload = json.loads(text)
        if isinstance(payload, dict):
            name = payload.get("name")
            arguments = payload.get("arguments", {})
            if isinstance(name, str) and name.strip():
                if isinstance(arguments, str):
                    try:
                        arguments = json.loads(arguments)
                    except json.JSONDecodeError:
                        arguments = {"value": arguments}
                if not isinstance(arguments, dict):
                    arguments = {"value": arguments}
                return ParsedToolCall(name=name.strip(), arguments=arguments, error=None)
    except json.JSONDecodeError as e:
        json_err = str(e)

    xml_tc = _parse_xml_style_tool_call(text)
    if xml_tc is not None:
        return xml_tc

    parts = ["could not parse tool_call as JSON or XML function format"]
    if json_err:
        parts.append(f"json: {json_err}")
    return ParsedToolCall(name=None, arguments={}, error="; ".join(parts))


def parse_all_tool_calls(response: str) -> list[ParsedToolCall]:
    calls = [_parse_single_tool_call(m.group(1)) for m in _TOOL_CALL_RE.finditer(response or "")]
    return calls


def parse_tool_call(response: str) -> ParsedToolCall | None:
    calls = parse_all_tool_calls(response)
    return calls[0] if calls else None
