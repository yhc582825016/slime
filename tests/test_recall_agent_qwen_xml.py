import json
import importlib.util
import sys
import types
from pathlib import Path

from examples.recall_agent.env_recall_agent import RecallAgentEnv
from examples.recall_agent.preprocess_recall_agent_data import _convert_row


def _load_rm_helper():
    stub_pkg = types.ModuleType("slime.rollout.rm_hub")
    stub_pkg.__path__ = []
    stub_math_utils = types.ModuleType("slime.rollout.rm_hub.math_utils")
    stub_math_utils.extract_answer = lambda text: text
    stub_math_utils.grade_answer_mathd = lambda *_args, **_kwargs: False
    stub_math_utils.grade_answer_sympy = lambda *_args, **_kwargs: False
    sys.modules.setdefault("slime.rollout.rm_hub", stub_pkg)
    sys.modules["slime.rollout.rm_hub.math_utils"] = stub_math_utils

    module_path = Path(__file__).resolve().parents[1] / "slime" / "rollout" / "rm_hub" / "recall_agent.py"
    spec = importlib.util.spec_from_file_location("test_recall_agent_rm_helper", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


_rm_helper = _load_rm_helper()
_extract_bfcl_tool_calls_from_text = _rm_helper._extract_bfcl_tool_calls_from_text
_safe_normalize_final_answer = _rm_helper._safe_normalize_final_answer
_compact_normalized_answer = _rm_helper._compact_normalized_answer


def _function_tool(name: str, properties: dict[str, dict] | None = None) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": f"Tool {name}",
            "parameters": {
                "type": "object",
                "properties": properties or {},
                "required": sorted((properties or {}).keys()),
            },
        },
    }


def test_preprocess_uses_qwen_xml_prompt_and_top_level_tools():
    tool = _function_tool("search_books", {"title": {"type": "string"}})
    row = {
        "question": 'Find the book "The Great Adventure".',
        "extra_info": json.dumps({"tool_schemas": [tool], "index": 7}),
        "reward_model": {"ground_truth": ["The Great Adventure"]},
    }

    sample = _convert_row(row, row_idx=0)

    system_prompt = sample["prompt"][0]["content"]
    assert "<function=function_name>" in system_prompt
    assert "<parameter=arg_name>" in system_prompt
    assert '{"name": "<function-name>", "arguments": {"arg": "value"}}' not in system_prompt
    assert sample["tools"] == [tool]
    assert sample["metadata"]["tools"] == [tool]


def test_env_executes_qwen_xml_tool_calls():
    env = RecallAgentEnv(
        env_code="def add(x, y):\n    return int(x) + int(y)\n",
        tool_schemas=[_function_tool("add", {"x": {"type": "string"}, "y": {"type": "string"}})],
        max_turns=4,
    )
    try:
        _, reset_info = env.reset()
        assert reset_info["init_error"] is None

        response = (
            "Need a quick calculation.\n"
            "<tool_call>\n"
            "<function=add>\n"
            "<parameter=x>\n2\n</parameter>\n"
            "<parameter=y>\n3\n</parameter>\n"
            "</function>\n"
            "</tool_call>"
        )
        observation, done, info = env.step(response)

        assert done is False
        assert info["tool_executed"] is True
        assert info["tool_calls"] == [{"name": "add", "arguments": {"x": 2, "y": 3}}]
        assert "<tool_response>5</tool_response>" in observation["obs_str"]
    finally:
        env.close()


def test_bfcl_extractor_prefers_qwen_xml_but_keeps_legacy_json_compatibility():
    xml_response = (
        "<tool_call>\n"
        "<function=search_books>\n"
        "<parameter=title>\nThe Great Adventure\n</parameter>\n"
        "</function>\n"
        "</tool_call>"
    )
    legacy_json_response = (
        '<tool_call>{"function": "search_books", "parameters": {"title": "The Great Adventure"}}</tool_call>'
    )

    expected = ["search_books(title='The Great Adventure')"]
    assert _extract_bfcl_tool_calls_from_text(xml_response) == expected
    assert _extract_bfcl_tool_calls_from_text(legacy_json_response) == expected


def test_recall_answer_normalization_preserves_sentence_spacing_for_debugging():
    text = '\\boxed{\\text{"Advanced Python", due 2026-04-19}}'

    assert _safe_normalize_final_answer(text) == '"Advanced Python", due 2026-04-19'
    assert _compact_normalized_answer(text) == "AdvancedPython,due2026-04-19"
