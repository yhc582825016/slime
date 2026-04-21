from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path
from typing import Any

import torch
from .base_env import BaseInteractionEnv

from slime.rollout.sglang_rollout import GenerateState
from slime.utils.http_utils import post
from slime.utils.processing_utils import encode_image_for_rollout_engine
from slime.utils.types import Sample

DEFAULT_ENV_MODULE = "examples.recall_agent.env_recall_agent"

DUMMY_MESSAGES = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "I am a user."},
]


def _load_env_module(env_path: str | None):
    target = env_path or DEFAULT_ENV_MODULE
    module_path = Path(target)
    if module_path.suffix == ".py" and module_path.exists():
        spec = importlib.util.spec_from_file_location(f"rollout_env_{module_path.stem}", module_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot import environment module from {module_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module
    return importlib.import_module(target)


def _build_env(env_module, sample: Sample, args: Any):
    build_fn = env_module.build_env
    if not callable(build_fn):
        raise ValueError("Environment module must expose a callable `build_env(sample, args)`.")
    try:
        return build_fn(sample=sample, args=args)
    except TypeError:
        return build_fn(sample, args)


def _encode_observation_for_generation(
    tokenizer,
    processor,
    message: dict,
    metadata: dict | None,
    apply_chat_template: bool,
    apply_chat_template_kwargs: dict | None,
):
    tools = metadata.get("tools") if metadata else None
    apply_kwargs = apply_chat_template_kwargs or {}

    trim_length = 0

    if apply_chat_template:
        dummy_prompt = tokenizer.apply_chat_template(
            DUMMY_MESSAGES,
            tools=tools,
            tokenize=False,
            add_generation_prompt=False,
            **apply_kwargs,
        )
        formatted_prompt = tokenizer.apply_chat_template(
            DUMMY_MESSAGES + [message],
            tools=tools,
            tokenize=False,
            add_generation_prompt=True,
            **apply_kwargs,
        )
        trim_length = len(tokenizer.encode(dummy_prompt, add_special_tokens=False))
    else:
        formatted_prompt = [message]

    multimodal_inputs = None
    multimodal_train_inputs = None
    if processor:
        from qwen_vl_utils import process_vision_info

        images, videos = process_vision_info([message])
        multimodal_inputs = {"images": images, "videos": videos}
        processor_output = processor(text=formatted_prompt, **multimodal_inputs)
        prompt_ids = processor_output["input_ids"][0]
        multimodal_train_inputs = {
            k: v for k, v in processor_output.items() if k not in ["input_ids", "attention_mask"]
        } or None
    else:
        prompt_ids = tokenizer.encode(formatted_prompt, add_special_tokens=False)

    if trim_length:
        prompt_ids = prompt_ids[trim_length:]

    image_data = []
    if multimodal_inputs and multimodal_inputs.get("images"):
        image_data = [encode_image_for_rollout_engine(img) for img in multimodal_inputs["images"]]
    return prompt_ids, image_data, multimodal_inputs, multimodal_train_inputs


def _merge_multimodal_train_inputs(chunks: list[dict | None]) -> dict | None:
    if not chunks:
        return None

    values_by_key = {}
    for chunk in chunks:
        if not chunk:
            continue
        for key, val in chunk.items():
            if val is None:
                continue
            values_by_key.setdefault(key, []).append(val)

    merged = {}
    for key, values in values_by_key.items():
        if all(isinstance(v, torch.Tensor) for v in values):
            merged[key] = torch.cat(values, dim=0)

    return merged


def _initialize_resources(args: Any, sample: Sample):
    env_module = _load_env_module(args.rollout_interaction_env_path)
    max_turns = args.max_turns
    if max_turns is None:
        raise ValueError("max_turns must be set via --custom-config-path in the custom config file.")
    state = GenerateState(args)
    url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"
    sample.metadata = sample.metadata or {}
    env = _build_env(env_module, sample, args)
    config = {"max_turns": max_turns}
    return env, config, state, url


def _prepare_initial_inputs(sample: Sample, processor, tokenizer):
    if processor:
        processor_output = processor(text=sample.prompt, **(sample.multimodal_inputs or {}))
        prompt_ids = processor_output["input_ids"][0]
        sample.multimodal_train_inputs = {
            k: v for k, v in processor_output.items() if k not in ["input_ids", "attention_mask"]
        } or None
    else:
        prompt_ids = tokenizer.encode(sample.prompt, add_special_tokens=False)

    image_data = []
    if sample.multimodal_inputs and sample.multimodal_inputs.get("images"):
        image_data = [encode_image_for_rollout_engine(img) for img in sample.multimodal_inputs["images"]]
    return prompt_ids, image_data, sample.multimodal_train_inputs


def _prepare_start_state(sample: Sample, state, args: Any, sampling_params: dict):
    prompt_ids, image_data, init_mm_train = _prepare_initial_inputs(sample, state.processor, state.tokenizer)
    current_image_data = image_data
    multimodal_train_inputs_buffer: list[dict | None] = []
    if init_mm_train:
        multimodal_train_inputs_buffer.append(init_mm_train)

    if not sample.tokens:
        sample.tokens = list(prompt_ids)
    response_tokens: list[int] = sample.tokens[len(prompt_ids) :] if len(sample.tokens) >= len(prompt_ids) else []
    sample.loss_mask = sample.loss_mask or []
    sample.rollout_log_probs = sample.rollout_log_probs or []
    sample.response_length = len(response_tokens)

    budget = None
    if args.rollout_max_context_len is not None:
        budget = args.rollout_max_context_len - len(sample.tokens)
    elif sampling_params.get("max_new_tokens") is not None:
        budget = sampling_params["max_new_tokens"] - len(sample.tokens)
    return current_image_data, response_tokens, budget, multimodal_train_inputs_buffer


async def _run_inference_step(url: str, tokens: list[int], sampling_params: dict, image_data, tokenizer):
    payload = {
        "input_ids": tokens,
        "sampling_params": sampling_params,
        "return_logprob": True,
    }
    if image_data:
        payload["image_data"] = image_data

    output = await post(url, payload)
    response_text = output["text"]
    if "output_token_logprobs" in output["meta_info"]:
        new_tokens = [item[1] for item in output["meta_info"]["output_token_logprobs"]]
        new_log_probs = [item[0] for item in output["meta_info"]["output_token_logprobs"]]
    else:
        new_tokens, new_log_probs = [], []
    finish_type = output["meta_info"]["finish_reason"]["type"]
    return response_text, new_tokens, new_log_probs, finish_type


def _process_env_step(env: BaseInteractionEnv, response_text: str, tokenizer, processor, args, sample_metadata):
    observation, done, _ = env.step(response_text)
    if done:
        return None, None, None, None, True

    next_user_message = env.format_observation(observation)
    obs_prompt_ids, obs_image_data, obs_multimodal_inputs, obs_multimodal_train_inputs = (
        _encode_observation_for_generation(
            tokenizer,
            processor,
            next_user_message,
            sample_metadata,
            args.apply_chat_template,
            args.apply_chat_template_kwargs,
        )
    )

    bos_id = tokenizer.bos_token_id
    if bos_id is not None and obs_prompt_ids and obs_prompt_ids[0] == bos_id:
        obs_prompt_ids = obs_prompt_ids[1:]

    return obs_prompt_ids, obs_image_data, obs_multimodal_inputs, obs_multimodal_train_inputs, False


def _append_to_sample(
    sample: Sample,
    response_tokens: list[int],
    tokens_to_add: list[int],
    logprobs: list[float],
    loss_mask_val: int,
) -> None:
    sample.tokens.extend(tokens_to_add)
    response_tokens.extend(tokens_to_add)
    sample.loss_mask.extend([loss_mask_val] * len(tokens_to_add))
    sample.rollout_log_probs.extend(logprobs)
    sample.response_length = len(response_tokens)


def _update_multimodal_state(
    sample: Sample,
    current_image_data,
    obs_image_data,
    obs_multimodal_inputs,
    obs_multimodal_train_inputs,
    multimodal_train_inputs_buffer: list[dict | None],
):
    if obs_image_data:
        current_image_data = (current_image_data or []) + obs_image_data

    if obs_multimodal_inputs:
        if not sample.multimodal_inputs:
            sample.multimodal_inputs = obs_multimodal_inputs
        elif isinstance(sample.multimodal_inputs, dict) and isinstance(obs_multimodal_inputs, dict):
            for key, val in obs_multimodal_inputs.items():
                if val is None:
                    continue
                if (
                    key in sample.multimodal_inputs
                    and isinstance(sample.multimodal_inputs[key], list)
                    and isinstance(val, list)
                ):
                    sample.multimodal_inputs[key].extend(val)
        else:
            sample.multimodal_inputs = obs_multimodal_inputs

    if obs_multimodal_train_inputs:
        multimodal_train_inputs_buffer.append(obs_multimodal_train_inputs)

    return current_image_data


def _contains_tool_call_markup(text: str) -> bool:
    text = text or ""
    return any(marker in text for marker in ("<tool_call>", "</tool_call>", "<function=", "</function>", "<parameter="))


def _mark_stop_reason(sample: Sample, reason: str, **details: Any) -> None:
    if not isinstance(sample.metadata, dict):
        sample.metadata = {}
    sample.metadata["stop_reason"] = reason
    for key, value in details.items():
        sample.metadata[f"stop_{key}"] = value


def _record_bfcl_response_step(sample: Sample, response_text: str, step_info: dict[str, Any] | None) -> None:
    metadata = sample.metadata if isinstance(sample.metadata, dict) else None
    if not metadata or not metadata.get("bfcl_eval_mode"):
        return

    trace = metadata.setdefault("bfcl_rollout_trace", {"turns": [[]], "current_turn": 0})
    turns = trace.setdefault("turns", [[]])
    current_turn = int(trace.get("current_turn", 0))
    while len(turns) <= current_turn:
        turns.append([])

    turns[current_turn].append(
        {
            "response_text": response_text,
            "step_info": dict(step_info or {}),
        }
    )

    if step_info and step_info.get("advanced_turn"):
        next_turn = current_turn + 1
        trace["current_turn"] = next_turn
        while len(turns) <= next_turn:
            turns.append([])


def _should_stop_on_finish(sample: Sample, finish_type: str, response_text: str) -> str | None:
    match finish_type:
        case "length":
            # If the model already emitted a complete tool_call block, give the env a chance
            # to execute it before stopping on length.
            if _contains_tool_call_markup(response_text):
                return None
            sample.status = Sample.Status.TRUNCATED
            return "finish_reason_length_without_complete_tool_call"
        case "abort":
            sample.status = Sample.Status.ABORTED
            return "finish_reason_abort"
    return None


def _update_budget(budget, consumed: int):
    if budget is None:
        return None
    return budget - consumed


def _finalize_sample(sample: Sample, tokenizer, response_tokens, multimodal_train_inputs_buffer):
    sample.multimodal_train_inputs = _merge_multimodal_train_inputs(multimodal_train_inputs_buffer)
    sample.response = tokenizer.decode(response_tokens, skip_special_tokens=False)
    sample.response_length = len(response_tokens)
    if sample.status is None:
        sample.status = Sample.Status.COMPLETED
    return sample


async def generate(args: Any, sample: Sample, sampling_params) -> Sample:
    assert not args.partial_rollout, "Partial rollout is not supported for interaction rollouts."

    env, config, state, url = _initialize_resources(args, sample)
    sampling_params = sampling_params.copy()
    current_image_data, response_tokens, budget, multimodal_train_inputs_buffer = _prepare_start_state(
        sample, state, args, sampling_params
    )
    try:
        _, reset_info = env.reset()
        if isinstance(reset_info, dict) and reset_info.get("init_error"):
            sample.status = Sample.Status.ABORTED
            _mark_stop_reason(
                sample,
                "env_init_error",
                turn=0,
                finish_type=None,
                budget=budget,
                has_complete_tool_call=False,
                status=str(sample.status),
            )
            return _finalize_sample(sample, state.tokenizer, response_tokens, multimodal_train_inputs_buffer)
        if budget is not None and budget <= 0:
            sample.status = Sample.Status.TRUNCATED
            _mark_stop_reason(
                sample,
                "budget_exhausted_before_generation",
                turn=0,
                finish_type=None,
                budget=budget,
                has_complete_tool_call=False,
                status=str(sample.status),
            )
            return sample

        cur_sampling_params = sampling_params
        for turn_idx in range(config["max_turns"]):
            if budget is not None:
                cur_sampling_params["max_new_tokens"] = budget

            response_text, new_response_tokens, new_response_log_probs, finish_type = await _run_inference_step(
                url, sample.tokens, cur_sampling_params, current_image_data, state.tokenizer
            )
            _append_to_sample(sample, response_tokens, new_response_tokens, new_response_log_probs, loss_mask_val=1)
            budget = _update_budget(budget, len(new_response_tokens))
            has_complete_tool_call = _contains_tool_call_markup(response_text)

            stop_reason = _should_stop_on_finish(sample, finish_type, response_text)
            if stop_reason is not None:
                _mark_stop_reason(
                    sample,
                    stop_reason,
                    turn=turn_idx + 1,
                    finish_type=finish_type,
                    budget=budget,
                    has_complete_tool_call=has_complete_tool_call,
                    status=str(sample.status),
                )
                break
            if budget is not None and budget <= 0 and not has_complete_tool_call:
                sample.status = Sample.Status.TRUNCATED
                _mark_stop_reason(
                    sample,
                    "budget_exhausted_without_complete_tool_call",
                    turn=turn_idx + 1,
                    finish_type=finish_type,
                    budget=budget,
                    has_complete_tool_call=has_complete_tool_call,
                    status=str(sample.status),
                )
                break

            env_observation, done, step_info = env.step(response_text)
            _record_bfcl_response_step(sample, response_text, step_info)
            if done:
                sample.status = Sample.Status.COMPLETED
                _mark_stop_reason(
                    sample,
                    "env_done",
                    turn=turn_idx + 1,
                    finish_type=finish_type,
                    budget=budget,
                    has_complete_tool_call=has_complete_tool_call,
                    status=str(sample.status),
                )
                break

            next_user_message = env.format_observation(env_observation)
            obs_prompt_ids, obs_image_data, obs_multimodal_inputs, obs_multimodal_train_inputs = (
                _encode_observation_for_generation(
                state.tokenizer,
                state.processor,
                next_user_message,
                sample.metadata,
                args.apply_chat_template,
                args.apply_chat_template_kwargs,
                )
            )

            obs_log_probs = [0.0] * len(obs_prompt_ids)
            _append_to_sample(sample, response_tokens, obs_prompt_ids, obs_log_probs, loss_mask_val=0)
            budget = _update_budget(budget, len(obs_prompt_ids))

            current_image_data = _update_multimodal_state(
                sample,
                current_image_data,
                obs_image_data,
                obs_multimodal_inputs,
                obs_multimodal_train_inputs,
                multimodal_train_inputs_buffer,
            )

            if budget is not None and budget <= 0:
                sample.status = Sample.Status.TRUNCATED
                _mark_stop_reason(
                    sample,
                    "budget_exhausted_after_observation",
                    turn=turn_idx + 1,
                    finish_type=finish_type,
                    budget=budget,
                    has_complete_tool_call=has_complete_tool_call,
                    status=str(sample.status),
                )
                break
            if turn_idx + 1 >= config["max_turns"]:
                sample.status = Sample.Status.COMPLETED
                _mark_stop_reason(
                    sample,
                    "max_turns_reached",
                    turn=turn_idx + 1,
                    finish_type=finish_type,
                    budget=budget,
                    has_complete_tool_call=has_complete_tool_call,
                    status=str(sample.status),
                )
                break

        if isinstance(sample.metadata, dict) and "stop_reason" not in sample.metadata:
            _mark_stop_reason(
                sample,
                "loop_exited_without_explicit_reason",
                turn=config["max_turns"],
                finish_type=None,
                budget=budget,
                has_complete_tool_call=False,
                status=str(sample.status) if sample.status is not None else None,
            )
        return _finalize_sample(sample, state.tokenizer, response_tokens, multimodal_train_inputs_buffer)
    finally:
        try:
            env.close()
        except Exception:
            pass
