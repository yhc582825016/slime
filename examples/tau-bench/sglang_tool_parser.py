import json
import re
from typing import Any

from sglang.srt.function_call.function_call_parser import FunctionCallParser
from sglang.srt.managers.io_struct import Function, Tool


def _parse_xml_tool_call(response: str) -> tuple[str, list[dict[str, Any]]]:
    """
    Fallback parser for Qwen-style XML tool calls such as:

    <tool_call>
    <function=get_order_details>
    <parameter=order_id>
    #W9077205
    </parameter>
    </function>
    </tool_call>
    """
    tool_block = re.search(r"<tool_call>\s*(.*?)\s*</tool_call>", response, flags=re.DOTALL)
    if not tool_block:
        return response, []

    body = tool_block.group(1)
    function_match = re.search(r"<function=([^\n>]+)>\s*(.*?)\s*</function>", body, flags=re.DOTALL)
    if not function_match:
        return response, []

    function_name = function_match.group(1).strip()
    function_body = function_match.group(2)
    parameters: dict[str, str] = {}

    for name, value in re.findall(
        r"<parameter=([^\n>]+)>\s*(.*?)\s*</parameter>",
        function_body,
        flags=re.DOTALL,
    ):
        parameters[name.strip()] = value.strip()

    normal_text = re.sub(r"<tool_call>\s*.*?\s*</tool_call>", "", response, flags=re.DOTALL).strip()
    calls = [{"name": function_name, "parameters": json.dumps(parameters, ensure_ascii=True)}]
    return normal_text, calls


def parse_tools(response: str, tools: list[dict[str, Any]], parser: str = "qwen3_coder"):
    """
    This function mimics the function call parser API from
    https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/entrypoints/http_server.py#L952
    But running locally
    """
    tools_list = [
        Tool(
            function=Function(
                name=tool["function"]["name"],
                description=tool["function"]["description"],
                parameters=tool["function"]["parameters"],
            ),
            type=tool["type"],
        )
        for tool in tools
    ]
    parser = FunctionCallParser(tools=tools_list, tool_call_parser=parser)

    try:
        normal_text, calls = parser.parse_non_stream(response)
    except Exception:
        normal_text, calls = _parse_xml_tool_call(response)
        if not calls:
            raise

    return {
        "normal_text": normal_text,
        "calls": [call.model_dump() if hasattr(call, "model_dump") else call for call in calls],
    }
