from __future__ import annotations

import argparse
import copy
import json
import random
import re
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any


DEFAULT_BFCL_ROOT = Path("/dev/shm/ye/berkeley-function-call-leaderboard")
DEFAULT_HF_CKPT = Path("/dev/shm/Qwen3.5-4B")
DEFAULT_MODEL_NAME = "Qwen3.5-4B-slime-bfcl-official-external-FC"
MULTI_TURN_CATEGORY_GROUP = [
    "multi_turn_base",
    "multi_turn_long_context",
    "multi_turn_miss_func",
    "multi_turn_miss_param",
]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run official-BFCL-style multi-turn evaluation against an already running external server.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--bfcl-root", type=Path, default=DEFAULT_BFCL_ROOT)
    parser.add_argument("--hf-checkpoint", type=Path, default=DEFAULT_HF_CKPT)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument(
        "--test-category",
        default="multi_turn",
        help="Comma-separated test category names. `multi_turn` expands to all multi-turn categories.",
    )
    parser.add_argument("--sample-size", type=int, default=200)
    parser.add_argument(
        "--sample-size-mode",
        default="per_category",
        choices=["per_category", "total"],
        help=(
            "`per_category`: sample up to N examples for each requested category "
            "(matches official BFCL multi-turn reporting). "
            "`total`: sample N examples after combining all requested categories."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-threads", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--local-server-endpoint", default="127.0.0.1")
    parser.add_argument("--local-server-port", type=int, default=6031)
    parser.add_argument("--backend", default="sglang", choices=["sglang", "vllm"])
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    parser.add_argument("--result-dir", type=Path, default=None)
    parser.add_argument("--score-dir", type=Path, default=None)
    parser.add_argument("--summary-path", type=Path, default=None)
    parser.add_argument("--allow-overwrite", action="store_true")
    parser.add_argument("--include-input-log", action="store_true")
    parser.add_argument(
        "--keep-state-log",
        action="store_true",
        help="Keep verbose state logs in result metadata.",
    )
    return parser.parse_args()


def _expand_test_categories(raw_value: str) -> list[str]:
    items = [item.strip() for item in raw_value.split(",") if item.strip()]
    expanded: list[str] = []
    for item in items:
        if item == "multi_turn":
            expanded.extend(MULTI_TURN_CATEGORY_GROUP)
        else:
            expanded.append(item)
    deduped: list[str] = []
    seen: set[str] = set()
    for item in expanded:
        if item not in seen:
            deduped.append(item)
            seen.add(item)
    return deduped


def _json_dump(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _convert_to_function_call(function_call_list: list[dict[str, Any]] | dict[str, Any]) -> list[str]:
    if isinstance(function_call_list, dict):
        function_call_list = [function_call_list]

    execution_list: list[str] = []
    for function_call in function_call_list:
        for key, value in function_call.items():
            if isinstance(value, str):
                value = json.loads(value)
            execution_list.append(f"{key}({','.join([f'{k}={repr(v)}' for k, v in value.items()])})")
    return execution_list


def _normalize_tool_payload(payload: Any) -> dict[str, Any] | None:
    if not isinstance(payload, dict):
        return None

    name = payload.get("name")
    arguments = payload.get("arguments")
    if arguments is None:
        arguments = payload.get("parameters")

    fn_payload = payload.get("function")
    if isinstance(fn_payload, dict):
        if not name:
            name = fn_payload.get("name")
        if arguments is None:
            arguments = fn_payload.get("arguments")
        if arguments is None:
            arguments = fn_payload.get("parameters")
    elif (not name) and isinstance(fn_payload, str):
        name = fn_payload

    if not name:
        return None
    if isinstance(arguments, str):
        arguments = json.loads(arguments)
    if not isinstance(arguments, dict):
        arguments = {}
    return {"name": str(name).strip(), "arguments": arguments}


class OfficialQwenFCCompatHandler:
    def __init__(
        self,
        *,
        model_path_or_id: str,
        base_url: str,
        temperature: float,
        max_context_length: int,
        prompt_token_margin: int,
        tokenizer,
    ) -> None:
        from openai import OpenAI

        self.model_path_or_id = model_path_or_id
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature
        self.max_context_length = max_context_length
        self.prompt_token_margin = max(0, prompt_token_margin)
        self.tokenizer = tokenizer
        self.client = OpenAI(base_url=self.base_url, api_key="EMPTY")

    @staticmethod
    def _is_context_length_error(exc: Exception) -> bool:
        text = str(exc).lower()
        return "context length" in text or "longer than the model's context length" in text

    @staticmethod
    def _extract_tool_calls(input_string: str) -> list[dict[str, Any]]:
        pattern = r"<tool_call>\n(.*?)\n</tool_call>"
        matches = re.findall(pattern, input_string, re.DOTALL)
        result: list[dict[str, Any]] = []
        for match in matches:
            try:
                result.append(json.loads(match))
            except Exception:
                pass
        return result

    def decode_execute(self, result: str, has_tool_call_tag: bool = False) -> list[str]:
        del has_tool_call_tag
        tool_calls = self._extract_tool_calls(result)
        if not isinstance(tool_calls, list) or any(not isinstance(item, dict) for item in tool_calls):
            raise ValueError(f"Model did not return a list of function calls: {result}")
        decoded_result = []
        for item in tool_calls:
            normalized = _normalize_tool_payload(item)
            if normalized is None:
                raise ValueError(f"Invalid tool call payload: {item}")
            decoded_result.append({normalized["name"]: normalized["arguments"]})
        return _convert_to_function_call(decoded_result)

    def _format_prompt(self, messages: list[dict[str, Any]], function: list[dict[str, Any]]) -> str:
        formatted_prompt = ""

        if len(function) > 0:
            formatted_prompt += "<|im_start|>system\n"
            if messages and messages[0]["role"] == "system":
                formatted_prompt += messages[0]["content"] + "\n\n"
            formatted_prompt += (
                "# Tools\n\nYou may call one or more functions to assist with the user query.\n\n"
                "You are provided with function signatures within <tools></tools> XML tags:\n<tools>"
            )
            for tool in function:
                formatted_prompt += f"\n{json.dumps(tool)}"
            formatted_prompt += (
                '\n</tools>\n\nFor each function call, return a json object with function name and arguments '
                'within <tool_call></tool_call> XML tags:\n<tool_call>\n{"name": <function-name>, '
                '"arguments": <args-json-object>}\n</tool_call><|im_end|>\n'
            )
        elif messages and messages[0]["role"] == "system":
            formatted_prompt += f"<|im_start|>system\n{messages[0]['content']}<|im_end|>\n"

        last_query_index = len(messages) - 1
        for offset, message in enumerate(reversed(messages)):
            idx = len(messages) - 1 - offset
            if (
                message["role"] == "user"
                and isinstance(message["content"], str)
                and not (
                    message["content"].startswith("<tool_response>")
                    and message["content"].endswith("</tool_response>")
                )
            ):
                last_query_index = idx
                break

        for idx, message in enumerate(messages):
            role = message["role"]
            content = message["content"]

            if role == "user" or (role == "system" and idx != 0):
                formatted_prompt += f"<|im_start|>{role}\n{content}<|im_end|>\n"
            elif role == "assistant":
                reasoning_content = ""
                if message.get("reasoning_content"):
                    reasoning_content = message["reasoning_content"]
                elif isinstance(content, str) and "</think>" in content:
                    parts = content.split("</think>")
                    reasoning_content = parts[0].rstrip("\n").split("<think>")[-1].lstrip("\n")
                    content = parts[-1].lstrip("\n")

                if idx > last_query_index:
                    if idx == len(messages) - 1 or reasoning_content:
                        formatted_prompt += (
                            f"<|im_start|>{role}\n<think>\n"
                            + reasoning_content.strip("\n")
                            + f"\n</think>\n\n"
                            + str(content).lstrip("\n")
                        )
                    else:
                        formatted_prompt += f"<|im_start|>{role}\n{content}"
                else:
                    formatted_prompt += f"<|im_start|>{role}\n{content}"

                if "tool_calls" in message:
                    for tool_call in message["tool_calls"]:
                        if (tool_call == message["tool_calls"][0] and content) or tool_call != message["tool_calls"][0]:
                            formatted_prompt += "\n"
                        normalized = _normalize_tool_payload(tool_call)
                        if normalized is None:
                            continue
                        formatted_prompt += '<tool_call>\n{"name": "'
                        formatted_prompt += normalized["name"]
                        formatted_prompt += '", "arguments": '
                        formatted_prompt += json.dumps(normalized["arguments"])
                        formatted_prompt += "}\n</tool_call>"

                formatted_prompt += "<|im_end|>\n"
            elif role == "tool":
                prev_role = messages[idx - 1]["role"] if idx > 0 else None
                next_role = messages[idx + 1]["role"] if idx < len(messages) - 1 else None
                if idx == 0 or prev_role != "tool":
                    formatted_prompt += "<|im_start|>user"
                formatted_prompt += f"\n<tool_response>\n{content}\n</tool_response>"
                if idx == len(messages) - 1 or next_role != "tool":
                    formatted_prompt += "<|im_end|>\n"

        formatted_prompt += "<|im_start|>assistant\n"
        return formatted_prompt

    def inference_multi_turn(
        self,
        *,
        test_entry: dict[str, Any],
        model_name_for_execution: str,
        max_step_limit: int,
        execute_multi_turn_func_call,
        include_input_log: bool,
        exclude_state_log: bool,
        stateless_classes: set[str],
        omit_state_info_classes: set[str],
    ) -> tuple[list[list[str]], dict[str, Any]]:
        import time

        initial_config: dict[str, Any] = test_entry.get("initial_config", {})
        involved_classes: list[str] = test_entry["involved_classes"]
        test_entry_id: str = test_entry["id"]
        test_category: str = test_entry_id.rsplit("_", 1)[0]

        holdout_function: dict[str, list[dict[str, Any]]] = test_entry.get("missed_function", {})
        total_input_token_count: list[list[float]] = []
        total_output_token_count: list[list[float]] = []
        total_latency: list[list[float]] = []
        all_model_response: list[list[str]] = []
        all_inference_log: list[Any] = []
        all_reasoning_content: list[list[str]] = []
        force_quit = False

        _, involved_instances = execute_multi_turn_func_call(
            [],
            initial_config,
            involved_classes,
            model_name_for_execution,
            test_entry_id,
            long_context=("long_context" in test_category or "composite" in test_category),
            is_evaL_run=False,
        )

        if not exclude_state_log:
            state_log = []
            for class_name, class_instance in involved_instances.items():
                if class_name in stateless_classes or class_name in omit_state_info_classes:
                    continue
                class_instance = copy.deepcopy(class_instance)
                state_log.append(
                    {
                        "role": "state_info",
                        "class_name": class_name,
                        "content": {
                            key: value
                            for key, value in vars(class_instance).items()
                            if not key.startswith("_")
                        },
                    }
                )
            if state_log:
                all_inference_log.append(state_log)

        inference_data: dict[str, Any] = {"message": [], "function": test_entry["function"]}

        all_multi_turn_messages: list[list[dict[str, Any]]] = test_entry["question"]
        for turn_idx, current_turn_message in enumerate(all_multi_turn_messages):
            if str(turn_idx) in holdout_function:
                inference_data["function"].extend(holdout_function[str(turn_idx)])
                current_turn_message = [
                    {
                        "role": "user",
                        "content": "I have updated some more functions you can choose from. What about now?",
                    }
                ]

            inference_data["message"].extend(current_turn_message)

            current_turn_response: list[str] = []
            current_turn_inference_log: dict[str, Any] = {"begin_of_turn_query": current_turn_message}
            current_turn_input_token_count: list[float] = []
            current_turn_output_token_count: list[float] = []
            current_turn_latency: list[float] = []
            current_turn_reasoning_content: list[str] = []

            count = 0
            while True:
                step_key = f"step_{count}"
                current_step_inference_log: list[dict[str, Any]] = []
                current_turn_inference_log[step_key] = current_step_inference_log

                formatted_prompt = self._format_prompt(inference_data["message"], inference_data["function"])
                input_token_count = len(self.tokenizer.tokenize(formatted_prompt))
                leftover_tokens_count = 1000
                available_prompt_budget = self.max_context_length - self.prompt_token_margin
                if input_token_count + 2 > available_prompt_budget:
                    current_step_inference_log.append(
                        {
                            "role": "handler_log",
                            "content": (
                                "Prompt exceeds the safe context budget for official external eval. "
                                "Mark this sample as force-terminated instead of crashing the whole eval."
                            ),
                            "input_token_count": input_token_count,
                            "max_context_length": self.max_context_length,
                            "prompt_token_margin": self.prompt_token_margin,
                        }
                    )
                    force_quit = True
                    break
                if self.max_context_length >= input_token_count + 2:
                    leftover_tokens_count = min(4096, self.max_context_length - input_token_count - 2)

                if include_input_log:
                    current_step_inference_log.append(
                        {"role": "inference_input", "content": {"formatted_prompt": formatted_prompt}}
                    )

                start_time = time.time()
                try:
                    api_response = self.client.completions.create(
                        model=self.model_path_or_id,
                        temperature=self.temperature,
                        prompt=formatted_prompt,
                        max_tokens=leftover_tokens_count,
                        timeout=72000,
                    )
                except Exception as exc:
                    if not self._is_context_length_error(exc):
                        raise
                    current_step_inference_log.append(
                        {
                            "role": "handler_log",
                            "content": (
                                "The serving backend rejected this eval sample because the prompt exceeded "
                                "the model context window. Mark this sample as force-terminated."
                            ),
                            "error": str(exc),
                            "input_token_count": input_token_count,
                            "max_context_length": self.max_context_length,
                            "prompt_token_margin": self.prompt_token_margin,
                        }
                    )
                    force_quit = True
                    break
                query_latency = time.time() - start_time

                model_response = api_response.choices[0].text
                extracted_tool_calls = self._extract_tool_calls(model_response)
                reasoning_content = ""
                cleaned_response = model_response
                if "</think>" in model_response:
                    parts = model_response.split("</think>")
                    reasoning_content = parts[0].rstrip("\n").split("<think>")[-1].lstrip("\n")
                    cleaned_response = parts[-1].lstrip("\n")

                if extracted_tool_calls:
                    model_responses_message_for_chat_history = {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": extracted_tool_calls,
                    }
                else:
                    model_responses_message_for_chat_history = {
                        "role": "assistant",
                        "content": cleaned_response,
                    }
                model_responses_message_for_chat_history["reasoning_content"] = reasoning_content
                inference_data["message"].append(model_responses_message_for_chat_history)

                current_turn_input_token_count.append(api_response.usage.prompt_tokens)
                current_turn_output_token_count.append(api_response.usage.completion_tokens)
                current_turn_latency.append(query_latency)
                current_turn_response.append(cleaned_response)
                current_turn_reasoning_content.append(reasoning_content)

                log_entry = {"role": "assistant", "content": cleaned_response}
                if reasoning_content:
                    log_entry["reasoning_content"] = reasoning_content
                current_step_inference_log.append(log_entry)

                try:
                    decoded_model_responses = self.decode_execute(cleaned_response, has_tool_call_tag=False)
                    current_step_inference_log.append(
                        {
                            "role": "handler_log",
                            "content": "Successfully decoded model response.",
                            "model_response_decoded": decoded_model_responses,
                        }
                    )
                    if not decoded_model_responses:
                        current_step_inference_log.append(
                            {
                                "role": "handler_log",
                                "content": "Empty response from the model. Proceed to next turn.",
                                "model_response_decoded": decoded_model_responses,
                            }
                        )
                        break
                except Exception as exc:
                    current_step_inference_log.append(
                        {
                            "role": "handler_log",
                            "content": "Error decoding the model response. Proceed to next turn.",
                            "error": str(exc),
                        }
                    )
                    break

                execution_results, involved_instances = execute_multi_turn_func_call(
                    decoded_model_responses,
                    initial_config,
                    involved_classes,
                    model_name_for_execution,
                    test_entry_id,
                    long_context=("long_context" in test_category or "composite" in test_category),
                    is_evaL_run=False,
                )
                for execution_result, decoded_model_response in zip(
                    execution_results, extracted_tool_calls, strict=False
                ):
                    inference_data["message"].append(
                        {
                            "role": "tool",
                            "name": decoded_model_response,
                            "content": execution_result,
                        }
                    )
                    current_step_inference_log.append({"role": "tool", "content": execution_result})

                count += 1
                if count > max_step_limit:
                    force_quit = True
                    current_step_inference_log.append(
                        {
                            "role": "handler_log",
                            "content": f"Model has been forced to quit after {max_step_limit} steps.",
                        }
                    )
                    break

            all_model_response.append(current_turn_response)
            all_reasoning_content.append(current_turn_reasoning_content)
            all_inference_log.append(current_turn_inference_log)
            total_input_token_count.append(current_turn_input_token_count)
            total_output_token_count.append(current_turn_output_token_count)
            total_latency.append(current_turn_latency)

            if not exclude_state_log:
                state_log = []
                for class_name, class_instance in involved_instances.items():
                    if class_name in stateless_classes or class_name in omit_state_info_classes:
                        continue
                    class_instance = copy.deepcopy(class_instance)
                    state_log.append(
                        {
                            "role": "state_info",
                            "class_name": class_name,
                            "content": {
                                key: value
                                for key, value in vars(class_instance).items()
                                if not key.startswith("_")
                            },
                        }
                    )
                if state_log:
                    all_inference_log.append(state_log)

            if force_quit:
                break

        metadata: dict[str, Any] = {
            "input_token_count": total_input_token_count,
            "output_token_count": total_output_token_count,
            "latency": total_latency,
            "inference_log": all_inference_log,
        }
        if not all(
            all(content == "" for content in single_turn_reasoning_content)
            for single_turn_reasoning_content in all_reasoning_content
        ):
            metadata["reasoning_content"] = all_reasoning_content

        return all_model_response, metadata


def _build_sampled_records(
    load_dataset_entry,
    load_ground_truth_entry,
    categories: list[str],
    sample_size: int,
    seed: int,
    sample_size_mode: str,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    records_by_category: dict[str, list[dict[str, Any]]] = {}
    for category in categories:
        prompt_entries = load_dataset_entry(category)
        ground_truth_entries = load_ground_truth_entry(category)
        ground_truth_by_id = {entry["id"]: entry for entry in ground_truth_entries}
        category_records: list[dict[str, Any]] = []
        for prompt_entry in prompt_entries:
            entry_id = prompt_entry["id"]
            if entry_id not in ground_truth_by_id:
                raise KeyError(f"Missing ground truth for {entry_id}")
            category_records.append(
                {
                    "id": entry_id,
                    "category": category,
                    "prompt_entry": prompt_entry,
                    "ground_truth_entry": ground_truth_by_id[entry_id],
                }
            )
        records_by_category[category] = category_records

    if sample_size <= 0:
        raise ValueError("--sample-size must be positive.")

    if sample_size_mode == "per_category":
        sampled: list[dict[str, Any]] = []
        for category in categories:
            category_records = records_by_category[category]
            if sample_size > len(category_records):
                raise ValueError(
                    f"--sample-size {sample_size} exceeds available records {len(category_records)} "
                    f"for category {category}."
                )
            sampled.extend(rng.sample(category_records, sample_size))
    elif sample_size_mode == "total":
        records = [item for category in categories for item in records_by_category[category]]
        if sample_size > len(records):
            raise ValueError(f"--sample-size {sample_size} exceeds available records {len(records)}.")
        sampled = rng.sample(records, sample_size)
    else:
        raise ValueError(f"Unsupported sample_size_mode: {sample_size_mode}")

    sampled.sort(key=lambda item: item["id"])
    return sampled


def _clean_output_dirs(result_dir: Path, score_dir: Path, model_name: str, allow_overwrite: bool) -> None:
    model_result_dir = result_dir / model_name
    model_score_dir = score_dir / model_name
    if not allow_overwrite and (model_result_dir.exists() or model_score_dir.exists()):
        raise FileExistsError(
            f"Output for model `{model_name}` already exists. Pass --allow-overwrite to replace it."
        )
    if model_result_dir.exists():
        shutil.rmtree(model_result_dir)
    if model_score_dir.exists():
        shutil.rmtree(model_score_dir)


def _evaluate_single_multi_turn_entry(
    *,
    handler,
    test_entry_id: str,
    model_result_list: list[list[str]],
    ground_truth_list: list[list[str]],
    prompt_entry: dict[str, Any],
    model_name: str,
    test_category: str,
    multi_turn_checker,
    multi_turn_irrelevance_checker,
    is_empty_execute_response,
) -> dict[str, Any]:
    prompt_entry = copy.deepcopy(prompt_entry)
    if "function" in prompt_entry:
        del prompt_entry["function"]

    if not isinstance(model_result_list, list):
        return {
            "id": test_entry_id,
            "model_name": model_name,
            "test_category": test_category,
            "valid": False,
            "error": {
                "error_message": [
                    "Error during inference phase. Model did not output a list of model responses."
                ],
                "error_type": "multi_turn:inference_error",
            },
            "prompt": prompt_entry,
            "model_result": model_result_list,
            "possible_answer": ground_truth_list,
        }

    if len(model_result_list) != len(ground_truth_list):
        return {
            "id": test_entry_id,
            "model_name": model_name,
            "test_category": test_category,
            "valid": False,
            "error": {
                "error_message": [
                    "Model was force-terminated during inference phase. "
                    f"The length of the model result turns ({len(model_result_list)}) "
                    f"does not match the length of the ground truth turns ({len(ground_truth_list)})."
                ],
                "error_type": "multi_turn:force_terminated",
            },
            "prompt": prompt_entry,
            "model_result": model_result_list,
            "possible_answer": ground_truth_list,
        }

    multi_turn_model_result_list_decoded: list[list[list[str]]] = []
    for single_turn_model_result_list in model_result_list:
        single_turn_model_result_list_decoded: list[list[str]] = []
        for model_result_item in single_turn_model_result_list:
            try:
                decoded_result: list[str] = handler.decode_execute(model_result_item, has_tool_call_tag=False)
                if is_empty_execute_response(decoded_result):
                    continue
                single_turn_model_result_list_decoded.append(decoded_result)
            except Exception:
                continue
        multi_turn_model_result_list_decoded.append(single_turn_model_result_list_decoded)

    accuracy_checker_result = multi_turn_checker(
        multi_turn_model_result_list_decoded,
        ground_truth_list,
        prompt_entry,
        test_category,
        model_name,
    )
    if accuracy_checker_result.get("valid"):
        irrelevance_result = multi_turn_irrelevance_checker(
            multi_turn_model_result_list_decoded,
            ground_truth_list,
        )
        if not irrelevance_result.get("valid"):
            accuracy_checker_result = irrelevance_result

    if not accuracy_checker_result.get("valid"):
        return {
            "id": test_entry_id,
            "model_name": model_name,
            "test_category": test_category,
            "valid": False,
            "error": {k: v for k, v in accuracy_checker_result.items() if k != "valid"},
            "prompt": prompt_entry,
            "model_result_raw": model_result_list,
            "model_result_decoded": multi_turn_model_result_list_decoded,
            "possible_answer": ground_truth_list,
        }

    return {"valid": True}


def run_official_external_eval(
    *,
    bfcl_root: Path,
    hf_checkpoint: Path,
    model_name: str,
    test_category: str = "multi_turn",
    sample_size: int = 200,
    sample_size_mode: str = "per_category",
    seed: int = 42,
    num_threads: int = 16,
    temperature: float = 0.0,
    local_server_endpoint: str = "127.0.0.1",
    local_server_port: int = 6031,
    backend: str = "sglang",
    dtype: str = "bfloat16",
    gpu_memory_utilization: float = 0.8,
    result_dir: Path | None = None,
    score_dir: Path | None = None,
    summary_path: Path | None = None,
    allow_overwrite: bool = True,
    include_input_log: bool = False,
    keep_state_log: bool = False,
) -> dict[str, Any]:
    bfcl_root = bfcl_root.resolve()
    hf_checkpoint = hf_checkpoint.resolve()
    result_dir = (result_dir or (bfcl_root / "result")).resolve()
    score_dir = (score_dir or (bfcl_root / "score")).resolve()
    summary_path = (
        summary_path.resolve()
        if summary_path is not None
        else (score_dir / model_name / "multi_turn" / "slime_external_summary.json").resolve()
    )

    import os
    import sys

    if str(bfcl_root) not in sys.path:
        sys.path.insert(0, str(bfcl_root))

    os.environ["LOCAL_SERVER_ENDPOINT"] = local_server_endpoint
    os.environ["LOCAL_SERVER_PORT"] = str(local_server_port)
    os.environ.setdefault("OPENAI_API_KEY", "EMPTY")

    from bfcl_eval.constants.category_mapping import VERSION_PREFIX
    from bfcl_eval.constants.default_prompts import MAXIMUM_STEP_LIMIT
    from bfcl_eval.constants.executable_backend_config import OMIT_STATE_INFO_CLASSES, STATELESS_CLASSES
    from bfcl_eval.eval_checker.multi_turn_eval.multi_turn_checker import (
        multi_turn_checker,
        multi_turn_irrelevance_checker,
    )
    from bfcl_eval.eval_checker.multi_turn_eval.multi_turn_utils import (
        execute_multi_turn_func_call,
        is_empty_execute_response,
    )
    from bfcl_eval.utils import (
        get_directory_structure_by_category,
        get_file_name_by_category,
        load_dataset_entry,
        load_ground_truth_entry,
        write_list_of_dicts_to_file,
    )
    from openai import OpenAI
    from transformers import AutoTokenizer

    categories = _expand_test_categories(test_category)
    sampled_records = _build_sampled_records(
        load_dataset_entry=load_dataset_entry,
        load_ground_truth_entry=load_ground_truth_entry,
        categories=categories,
        sample_size=sample_size,
        seed=seed,
        sample_size_mode=sample_size_mode,
    )

    result_dir.mkdir(parents=True, exist_ok=True)
    score_dir.mkdir(parents=True, exist_ok=True)
    _clean_output_dirs(result_dir, score_dir, model_name, allow_overwrite)

    tokenizer = AutoTokenizer.from_pretrained(str(hf_checkpoint), trust_remote_code=True, local_files_only=True)
    config_path = hf_checkpoint / "config.json"
    config_data = json.loads(config_path.read_text(encoding="utf-8"))
    max_context_length = config_data.get("max_position_embeddings") or tokenizer.model_max_length
    prompt_token_margin = int(os.getenv("BFCL_EXTERNAL_PROMPT_TOKEN_MARGIN", "1024"))
    base_url = f"http://{local_server_endpoint}:{local_server_port}/v1"
    models_client = OpenAI(base_url=base_url, api_key="EMPTY")
    models_client.models.list()
    handler = OfficialQwenFCCompatHandler(
        model_path_or_id=str(hf_checkpoint),
        base_url=base_url,
        temperature=temperature,
        max_context_length=max_context_length,
        prompt_token_margin=prompt_token_margin,
        tokenizer=tokenizer,
    )

    include_input_log = bool(include_input_log)
    exclude_state_log = not bool(keep_state_log)

    def _run_one(record: dict[str, Any]) -> dict[str, Any]:
        prompt_entry = copy.deepcopy(record["prompt_entry"])
        result, metadata = handler.inference_multi_turn(
            test_entry=prompt_entry,
            model_name_for_execution=model_name.replace("/", "_"),
            max_step_limit=MAXIMUM_STEP_LIMIT,
            execute_multi_turn_func_call=execute_multi_turn_func_call,
            include_input_log=include_input_log,
            exclude_state_log=exclude_state_log,
            stateless_classes=set(STATELESS_CLASSES),
            omit_state_info_classes=set(OMIT_STATE_INFO_CLASSES),
        )
        return {
            "id": record["id"],
            "category": record["category"],
            "prompt_entry": record["prompt_entry"],
            "ground_truth_entry": record["ground_truth_entry"],
            "result_entry": {
                "id": record["id"],
                "result": result,
                **metadata,
            },
        }

    completed: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(_run_one, record) for record in sampled_records]
        total = len(futures)
        for idx, future in enumerate(as_completed(futures), start=1):
            completed.append(future.result())
            if idx % 20 == 0 or idx == total:
                print(f"progress {idx}/{total}", flush=True)

    completed.sort(key=lambda item: item["id"])
    grouped_result_entries: dict[str, list[dict[str, Any]]] = {}
    for item in completed:
        grouped_result_entries.setdefault(item["category"], []).append(item["result_entry"])
    for category, result_entries in grouped_result_entries.items():
        output_file_name = get_file_name_by_category(category, is_result_file=True)
        output_subdir = result_dir / model_name / get_directory_structure_by_category(category)
        write_list_of_dicts_to_file(output_file_name, result_entries, output_subdir)

    grouped_completed: dict[str, list[dict[str, Any]]] = {}
    for item in completed:
        grouped_completed.setdefault(item["category"], []).append(item)

    summary_categories: dict[str, dict[str, Any]] = {}
    total_correct = 0
    total_count = 0

    for category in categories:
        category_items = grouped_completed.get(category, [])
        invalid_entries: list[dict[str, Any]] = []
        correct_count = 0

        for item in category_items:
            evaluation_result = _evaluate_single_multi_turn_entry(
                handler=handler,
                test_entry_id=item["id"],
                model_result_list=item["result_entry"]["result"],
                ground_truth_list=item["ground_truth_entry"]["ground_truth"],
                prompt_entry=item["prompt_entry"],
                model_name=model_name,
                test_category=category,
                multi_turn_checker=multi_turn_checker,
                multi_turn_irrelevance_checker=multi_turn_irrelevance_checker,
                is_empty_execute_response=is_empty_execute_response,
            )
            if evaluation_result.get("valid"):
                correct_count += 1
            else:
                evaluation_result["inference_log"] = item["result_entry"].get("inference_log", "")
                invalid_entries.append(evaluation_result)

        total = len(category_items)
        accuracy = (correct_count / total) if total else 0.0
        total_correct += correct_count
        total_count += total
        summary_categories[category] = {
            "accuracy": accuracy,
            "correct_count": correct_count,
            "total_count": total,
        }

        score_payload = [
            {
                "accuracy": accuracy,
                "correct_count": correct_count,
                "total_count": total,
            },
            *invalid_entries,
        ]
        score_file_name = get_file_name_by_category(category, is_score_file=True)
        score_subdir = score_dir / model_name / get_directory_structure_by_category(category)
        write_list_of_dicts_to_file(score_file_name, score_payload, score_subdir)

    summary = {
        "version_prefix": VERSION_PREFIX,
        "model_name": model_name,
        "bfcl_root": str(bfcl_root),
        "hf_checkpoint": str(hf_checkpoint),
        "local_server_endpoint": local_server_endpoint,
        "local_server_port": local_server_port,
        "sample_size": sample_size,
        "sample_size_mode": sample_size_mode,
        "seed": seed,
        "temperature": temperature,
        "num_threads": num_threads,
        "max_step_limit": MAXIMUM_STEP_LIMIT,
        "categories": summary_categories,
        "overall_accuracy": (total_correct / total_count) if total_count else 0.0,
        "overall_correct_count": total_correct,
        "overall_total_count": total_count,
        "sampled_ids": [item["id"] for item in completed],
        "result_dir": str((result_dir / model_name).resolve()),
        "score_dir": str((score_dir / model_name).resolve()),
    }

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(_json_dump(summary) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    args = _parse_args()
    summary = run_official_external_eval(
        bfcl_root=args.bfcl_root,
        hf_checkpoint=args.hf_checkpoint,
        model_name=args.model_name,
        test_category=args.test_category,
        sample_size=args.sample_size,
        sample_size_mode=args.sample_size_mode,
        seed=args.seed,
        num_threads=args.num_threads,
        temperature=args.temperature,
        local_server_endpoint=args.local_server_endpoint,
        local_server_port=args.local_server_port,
        backend=args.backend,
        dtype=args.dtype,
        gpu_memory_utilization=args.gpu_memory_utilization,
        result_dir=args.result_dir,
        score_dir=args.score_dir,
        summary_path=args.summary_path,
        allow_overwrite=args.allow_overwrite,
        include_input_log=args.include_input_log,
        keep_state_log=args.keep_state_log,
    )
    print("SUMMARY_JSON=" + _json_dump(summary), flush=True)


if __name__ == "__main__":
    main()
