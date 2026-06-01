import asyncio
import logging
import json
import os
import random
from typing import Any

import aiohttp
import torch

from slime.rollout.rm_hub.math_dapo_utils import compute_score as compute_score_dapo
from slime.utils import logging_utils
from slime.utils.metric_utils import compute_rollout_step
from slime.utils.processing_utils import encode_image_for_rollout_engine
from slime.utils.types import Sample

logger = logging.getLogger(__name__)
_IFBENCH_SCORER = None
_IFBENCH_SCORER_IMPORT_FAILED = False
_GPQA_SCORER = None
_GPQA_SCORER_IMPORT_FAILED = False
_ZEBRALOGICBENCH_SCORER = None
_ZEBRALOGICBENCH_SCORER_IMPORT_FAILED = False
_OPD_RM_SESSION: aiohttp.ClientSession | None = None
_OPD_RM_SEMAPHORE: asyncio.Semaphore | None = None


def _get_rm_concurrency() -> int:
    value = int(os.environ.get("OPD_RM_CONCURRENCY", "32"))
    return max(1, value)


def _get_rm_max_connections() -> int:
    value = int(os.environ.get("OPD_RM_MAX_CONNECTIONS", "64"))
    return max(1, value)


def _get_rm_retries() -> int:
    value = int(os.environ.get("OPD_RM_MAX_RETRIES", "6"))
    return max(1, value)


def _get_rm_session() -> aiohttp.ClientSession:
    global _OPD_RM_SESSION
    if _OPD_RM_SESSION is None or _OPD_RM_SESSION.closed:
        connector = aiohttp.TCPConnector(
            limit=_get_rm_max_connections(),
            enable_cleanup_closed=True,
        )
        timeout = aiohttp.ClientTimeout(total=600, connect=120, sock_connect=120, sock_read=600)
        _OPD_RM_SESSION = aiohttp.ClientSession(connector=connector, timeout=timeout)
    return _OPD_RM_SESSION


def _get_rm_semaphore() -> asyncio.Semaphore:
    global _OPD_RM_SEMAPHORE
    if _OPD_RM_SEMAPHORE is None:
        _OPD_RM_SEMAPHORE = asyncio.Semaphore(_get_rm_concurrency())
    return _OPD_RM_SEMAPHORE


async def reward_func(args, sample, **kwargs):
    payload = {
        # "text": sample.prompt + sample.response,
        "input_ids": sample.tokens,
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": 0,
            "skip_special_tokens": False,
        },
        "return_logprob": True,
        "logprob_start_len": 0,
    }

    if sample.multimodal_inputs and sample.multimodal_inputs.get("images"):
        image_data = sample.multimodal_inputs["images"]
        payload["image_data"] = [encode_image_for_rollout_engine(image) for image in image_data]

    session = _get_rm_session()
    semaphore = _get_rm_semaphore()
    max_retries = _get_rm_retries()

    async with semaphore:
        for attempt in range(max_retries):
            try:
                async with session.post(args.rm_url, json=payload) as resp:
                    resp.raise_for_status()
                    return await resp.json()
            except Exception as exc:
                if attempt + 1 >= max_retries:
                    raise
                backoff = min(2**attempt, 8) + random.random()
                logger.warning(
                    "OPD reward request failed (%s), retry in %.2fs (%d/%d), rm_url=%s",
                    type(exc).__name__,
                    backoff,
                    attempt + 1,
                    max_retries,
                    args.rm_url,
                )
                await asyncio.sleep(backoff)


def post_process_rewards(args, samples: list[Sample], **kwargs):
    """Process rewards from teacher model and extract teacher log probabilities.

    This function:
    1. Extracts teacher log-probs from the reward response (which contains sglang's logprob output)
    2. Trims them to match the response length
    3. Stores them in sample.teacher_log_probs for OPD KL penalty computation
    4. Returns scalar rewards (0.0 for pure distillation) compatible with GRPO/PPO

    Note: The reward_func calls the teacher server which returns token-level log-probs.
    For pure on-policy distillation without task rewards, we return 0.0 for each sample.
    The actual learning signal comes from the OPD KL penalty applied in compute_advantages_and_returns.
    """
    raw_rewards = [sample.get_reward_value(args) for sample in samples]
    response_lengths = [sample.response_length for sample in samples]

    # Extract teacher log-probs from the sglang response
    teacher_log_probs = [
        torch.tensor([item[0] for item in reward["meta_info"]["input_token_logprobs"][1:]], dtype=torch.float32)
        for reward in raw_rewards
    ]
    teacher_log_probs = [
        t_log_prob[-response_length:]
        for t_log_prob, response_length in zip(teacher_log_probs, response_lengths, strict=False)
    ]

    for sample, t_log_probs in zip(samples, teacher_log_probs, strict=False):
        sample.teacher_log_probs = t_log_probs

    # Return scalar rewards for GRPO/PPO advantage estimator
    # For pure on-policy distillation, we use 0.0 as the task reward.
    # The learning signal comes entirely from the OPD KL penalty.
    # If you have task rewards, you can add them here.
    scalar_rewards = [0.0] * len(samples)

    return scalar_rewards, scalar_rewards


def _strip_special_tokens(text: str) -> str:
    """Remove trailing special tokens like <|im_end|> that models sometimes emit."""
    # Strip any <|...|> style special tokens from the end
    import re
    return re.sub(r"(<\|[^|>]*\|>)+\s*$", "", text).strip()


def _extract_answer(response: str | None) -> str:
    response = response or ""
    end_think_idx = response.rfind("</think>")
    if end_think_idx != -1:
        return _strip_special_tokens(response[end_think_idx + len("</think>") :].strip())
    start_think_idx = response.rfind("<think>")
    if start_think_idx != -1:
        return _strip_special_tokens(response[start_think_idx + len("<think>") :].strip())
    return _strip_special_tokens(response.strip())


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _looks_like_ifbench(dataset_name: str, sample: Sample) -> bool:
    metadata = _as_dict(sample.metadata)
    label = sample.label
    data_source = metadata.get("data_source")

    if metadata.get("rm_type") == "ifbench" or data_source == "ood__ifbench":
        return True
    if "ifbench" in dataset_name.lower():
        return True
    if isinstance(label, list) and label and isinstance(label[0], dict):
        return "instruction_id" in label[0] or "instruction_id_list" in label[0]
    return False


def _parse_json_like(value: Any) -> Any:
    if isinstance(value, str):
        text = value.strip()
        if "</think>" in text:
            text = text.split("</think>")[-1].strip()
        try:
            return json.loads(text)
        except Exception:
            return text
    return value


def _looks_like_gpqa(dataset_name: str, sample: Sample) -> bool:
    name = dataset_name.lower()
    if "gpqa" in name:
        return True

    metadata = _as_dict(sample.metadata)
    if metadata.get("rm_type") == "gpqa":
        return True

    data_source = str(metadata.get("data_source", ""))
    if data_source.startswith("stem__gpqa"):
        return True

    label = sample.label
    if isinstance(label, str) and len(label.strip()) == 1 and label.strip().upper() in "ABCD":
        if metadata.get("choices") or metadata.get("correct_letter"):
            return True
    return False


def _looks_like_zebralogicbench(dataset_name: str, sample: Sample) -> bool:
    name = dataset_name.lower()
    if "zebralogicbench" in name or "zebra_logic" in name:
        return True

    metadata = _as_dict(sample.metadata)
    if metadata.get("rm_type") == "zebralogicbench":
        return True

    data_source = str(metadata.get("data_source", ""))
    if data_source.startswith("logic__zebralogicbench"):
        return True
    return False


def _looks_like_ic(dataset_name: str, sample: Sample) -> bool:
    name = dataset_name.lower()
    if name == "ic" or "ic_" in name:
        return True

    metadata = _as_dict(sample.metadata)
    data_source = metadata.get("data_source")
    if data_source == "ood__ic":
        return True

    label = _parse_json_like(sample.label)
    if isinstance(label, dict) and label.get("data_source") == "ood__ic":
        return True
    return False


def _score_math_sample(sample: Sample) -> dict[str, float]:
    result = compute_score_dapo(sample.response, sample.label or "")
    score = float(result.get("score", 0.0))
    acc = float(result.get("acc", score))
    return {"score": score, "acc": acc}


def _score_gpqa_sample(sample: Sample) -> dict[str, float]:
    global _GPQA_SCORER, _GPQA_SCORER_IMPORT_FAILED

    metadata = _as_dict(sample.metadata)
    if _GPQA_SCORER is None and not _GPQA_SCORER_IMPORT_FAILED:
        try:
            from slime.rollout.rm_hub.gpqa import compute_gpqa_reward

            _GPQA_SCORER = compute_gpqa_reward
        except Exception as exc:
            _GPQA_SCORER_IMPORT_FAILED = True
            logger.warning("GPQA scorer is unavailable; GPQA eval scores will be 0. Error: %s", exc)

    if _GPQA_SCORER is None:
        return {"score": 0.0, "acc": 0.0}

    try:
        score = float(_GPQA_SCORER(sample.response, sample.label, metadata=metadata))
        return {"score": score, "acc": score}
    except Exception as exc:
        logger.warning("Failed to score one GPQA sample; returning 0. Error: %s", exc)
        return {"score": 0.0, "acc": 0.0}


def _score_zebralogicbench_sample(sample: Sample) -> dict[str, float]:
    global _ZEBRALOGICBENCH_SCORER, _ZEBRALOGICBENCH_SCORER_IMPORT_FAILED

    if _ZEBRALOGICBENCH_SCORER is None and not _ZEBRALOGICBENCH_SCORER_IMPORT_FAILED:
        try:
            from slime.rollout.rm_hub.zebralogicbench import compute_zebralogicbench_reward

            _ZEBRALOGICBENCH_SCORER = compute_zebralogicbench_reward
        except Exception as exc:
            _ZEBRALOGICBENCH_SCORER_IMPORT_FAILED = True
            logger.warning(
                "ZebraLogicBench scorer is unavailable; ZebraLogicBench eval scores will be 0. Error: %s",
                exc,
            )

    if _ZEBRALOGICBENCH_SCORER is None:
        return {"score": 0.0, "acc": 0.0}

    try:
        score = float(_ZEBRALOGICBENCH_SCORER(sample.response, sample.label, metadata=_as_dict(sample.metadata)))
        return {"score": score, "acc": score}
    except Exception as exc:
        logger.warning("Failed to score one ZebraLogicBench sample; returning 0. Error: %s", exc)
        return {"score": 0.0, "acc": 0.0}


def _score_ifbench_sample(sample: Sample) -> dict[str, float]:
    global _IFBENCH_SCORER, _IFBENCH_SCORER_IMPORT_FAILED

    metadata = _as_dict(sample.metadata)
    answer = _extract_answer(sample.response)

    if _IFBENCH_SCORER is None and not _IFBENCH_SCORER_IMPORT_FAILED:
        try:
            from slime.rollout.rm_hub.ifbench import compute_ifbench_reward

            _IFBENCH_SCORER = compute_ifbench_reward
        except Exception as exc:
            _IFBENCH_SCORER_IMPORT_FAILED = True
            logger.warning("IFBench scorer is unavailable; IFBench eval scores will be 0. Error: %s", exc)

    if _IFBENCH_SCORER is None:
        return {"score": 0.0, "acc": 0.0}

    try:
        score = float(_IFBENCH_SCORER(answer, sample.label, metadata=metadata))
        return {"score": score, "acc": score}
    except Exception as exc:
        logger.warning("Failed to score one IFBench sample; returning 0. Error: %s", exc)
        return {"score": 0.0, "acc": 0.0}


def _normalize_ic_value(v: Any) -> Any:
    """Normalize a parsed IC value for comparison.

    Removes whitespace and unifies common variant characters so that formatting
    differences (e.g. "29 岁" vs "29岁", "2~3" vs "2-3") don't cause spurious
    mismatches.
    """
    if isinstance(v, str):
        # collapse all whitespace
        v = "".join(v.split())
        # normalize range/separator variants
        v = v.replace("~", "-").replace("～", "-").replace("—", "-").replace("–", "-")
        return v
    if isinstance(v, dict):
        return {k: _normalize_ic_value(val) for k, val in v.items()}
    if isinstance(v, list):
        return [_normalize_ic_value(i) for i in v]
    return v


def _score_ic_sample(sample: Sample) -> dict[str, float]:
    """Score one IC sample using the same logic as cal_metrics.py get_acc:
    try json.loads on both gt and pred together; if either fails both stay as
    strings. String values are normalized (spaces / separator variants) before
    comparison so that pure formatting differences don't count as wrong.
    """
    label = _parse_json_like(sample.label)
    if isinstance(label, dict):
        ground_truth = str(label.get("ground_truth", ""))
    else:
        ground_truth = str(label) if label is not None else ""

    pred_text = _extract_answer(sample.response)

    gt: Any = ground_truth
    pred: Any = pred_text
    try:
        gt = json.loads(ground_truth)
        pred = json.loads(pred_text)
    except Exception:
        pass

    gt_norm = _normalize_ic_value(gt)
    pred_norm = _normalize_ic_value(pred)

    score = float(gt_norm == pred_norm)
    # if gt_norm != pred_norm:
    logger.warning(
        "[IC mismatch] gt=%r | pred=%r | raw_response=%r",
        gt_norm,
        pred_norm,
        sample.response[:300],
    )
    return {"score": score, "acc": score}


def _score_eval_sample(dataset_name: str, sample: Sample) -> dict[str, float]:
    if _looks_like_ic(dataset_name, sample):
        return _score_ic_sample(sample)
    if _looks_like_ifbench(dataset_name, sample):
        return _score_ifbench_sample(sample)
    if _looks_like_gpqa(dataset_name, sample):
        return _score_gpqa_sample(sample)
    if _looks_like_zebralogicbench(dataset_name, sample):
        return _score_zebralogicbench_sample(sample)
    return _score_math_sample(sample)


def eval_log_function(rollout_id, args, data, extra_metrics):
    """Custom eval logging for OPD.

    Replaces the default logging which fails when rewards are dicts (OPD teacher response).
    Recomputes task scores for the two eval sets and logs per-dataset plus overall metrics.

    Returns True to skip the default logging.
    """
    log_dict = dict(extra_metrics) if extra_metrics else {}
    all_scores = []
    all_accs = []
    dataset_score_means = []
    dataset_acc_means = []

    for key, key_data in data.items():
        samples = key_data.get("samples", [])
        if not samples:
            continue

        results = [_score_eval_sample(key, sample) for sample in samples]
        scores = [float(r["score"]) for r in results]
        accs = [float(r["acc"]) for r in results]
        score_mean = sum(scores) / len(scores)
        acc_mean = sum(accs) / len(accs)

        log_dict[f"eval/{key}"] = score_mean
        log_dict[f"eval/{key}/score"] = score_mean
        log_dict[f"eval/{key}/acc"] = acc_mean
        if key.lower() == "ic":
            log_dict[f"eval/{key}"] = acc_mean

        all_scores.extend(scores)
        all_accs.extend(accs)
        dataset_score_means.append(score_mean)
        dataset_acc_means.append(acc_mean)

        # response length stats
        response_lengths = [sample.response_length for sample in samples]
        log_dict[f"eval/{key}/response_len_mean"] = sum(response_lengths) / len(response_lengths)

        if "truncated" in key_data:
            truncated = key_data["truncated"]
            log_dict[f"eval/{key}-truncated_ratio"] = sum(truncated) / len(truncated)

    if all_scores:
        log_dict["eval/summary/score"] = sum(all_scores) / len(all_scores)
        log_dict["eval/summary/acc"] = sum(all_accs) / len(all_accs)
        log_dict["eval/summary/macro_score"] = sum(dataset_score_means) / len(dataset_score_means)
        log_dict["eval/summary/macro_acc"] = sum(dataset_acc_means) / len(dataset_acc_means)

    step = compute_rollout_step(args, rollout_id)
    log_dict["eval/step"] = step

    logger.info(f"eval {rollout_id}: {log_dict}")
    logging_utils.log(args, log_dict, step_key="eval/step")
    return True
