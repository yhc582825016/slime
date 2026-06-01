#!/usr/bin/env python3
"""Standalone AIME eval against a pre-launched SGLang server (see sglang.sh).

Mirrors eval settings from run-qwen3-opd-4-4.sh (aime dataset, math_dapo scoring).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import httpx
from tqdm import tqdm

from slime.rollout.rm_hub.math_dapo_utils import compute_score as compute_score_dapo
from slime.utils.data import Dataset
from slime.utils.processing_utils import load_tokenizer

logger = logging.getLogger(__name__)

DEFAULT_DATA = "/mnt/code/yehangcheng/ms-swift/train_data/aime_2024_swift_.jsonl"
DEFAULT_MODEL = "/opt/users/ye/models/qwen3.5-opd-ori-35b-distill-qwen3.5-4b-509/iter_0000539"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 6035


@dataclass
class EvalResult:
    index: int
    prompt_index: int
    sample_index: int
    ground_truth: str
    response: str
    score: float
    acc: float
    pred: str | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate AIME on an external SGLang server.")
    parser.add_argument("--data-path", type=str, default=DEFAULT_DATA)
    parser.add_argument("--model-path", type=str, default=DEFAULT_MODEL, help="HF path for tokenizer/chat template")
    parser.add_argument("--host", type=str, default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--n-samples", type=int, default=4, help="Samples per prompt (pass@k style pool)")
    parser.add_argument("--max-new-tokens", type=int, default=32000)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=-1)
    parser.add_argument("--concurrency", type=int, default=50)
    parser.add_argument("--input-key", type=str, default="messages")
    parser.add_argument("--label-key", type=str, default="solution")
    parser.add_argument(
        "--apply-chat-template-kwargs",
        type=str,
        default='{"enable_thinking": false}',
        help="JSON kwargs passed to tokenizer.apply_chat_template",
    )
    parser.add_argument("--max-prompts", type=int, default=None, help="Limit number of prompts (debug)")
    parser.add_argument("--output", type=str, default=None, help="Optional path to save per-sample JSONL")
    parser.add_argument("--health-timeout-sec", type=int, default=600)
    parser.add_argument("--health-poll-sec", type=float, default=5.0)
    return parser.parse_args()


def wait_for_server(base_url: str, timeout_sec: int, poll_sec: float) -> None:
    health_url = f"{base_url}/health_generate"
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        try:
            resp = httpx.get(health_url, timeout=10.0)
            if resp.status_code == 200:
                info = httpx.get(f"{base_url}/get_model_info", timeout=10.0)
                info.raise_for_status()
                logger.info("SGLang server ready: %s", info.json())
                return
        except Exception as exc:
            logger.info("Waiting for SGLang at %s (%s)", health_url, exc)
        time.sleep(poll_sec)
    raise TimeoutError(f"SGLang server not ready at {health_url} after {timeout_sec}s")


async def generate_one(
    client: httpx.AsyncClient,
    generate_url: str,
    input_ids: list[int],
    sampling_params: dict,
    semaphore: asyncio.Semaphore,
) -> str:
    payload = {
        "input_ids": input_ids,
        "sampling_params": sampling_params,
        "return_logprob": False,
    }
    async with semaphore:
        resp = await client.post(generate_url, json=payload, timeout=httpx.Timeout(600.0, connect=30.0))
        resp.raise_for_status()
        data = resp.json()
        return data.get("text", "")


async def run_eval(args: argparse.Namespace) -> list[EvalResult]:
    base_url = f"http://{args.host}:{args.port}"
    generate_url = f"{base_url}/generate"
    wait_for_server(base_url, args.health_timeout_sec, args.health_poll_sec)

    apply_chat_template_kwargs = json.loads(args.apply_chat_template_kwargs)
    tokenizer = load_tokenizer(args.model_path, trust_remote_code=True)
    dataset = Dataset(
        path=args.data_path,
        tokenizer=tokenizer,
        processor=None,
        max_length=None,
        prompt_key=args.input_key,
        label_key=args.label_key,
        apply_chat_template=True,
        apply_chat_template_kwargs=apply_chat_template_kwargs,
    )

    sampling_params = {
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "max_new_tokens": args.max_new_tokens,
        "no_stop_trim": True,
        "spaces_between_special_tokens": False,
    }

    samples = dataset.samples
    if args.max_prompts is not None:
        samples = samples[: args.max_prompts]

    tasks: list[tuple[int, int, str, list[int]]] = []
    for prompt_index, sample in enumerate(samples):
        prompt_ids = tokenizer.encode(sample.prompt, add_special_tokens=False)
        for sample_index in range(args.n_samples):
            tasks.append((prompt_index, sample_index, sample.label or "", prompt_ids))

    semaphore = asyncio.Semaphore(args.concurrency)
    results: list[EvalResult] = []

    async with httpx.AsyncClient() as client:
        async def _run_one(task_index: int, prompt_index: int, sample_index: int, label: str, prompt_ids: list[int]):
            response = await generate_one(client, generate_url, prompt_ids, sampling_params, semaphore)
            scored = compute_score_dapo(response, label)
            return EvalResult(
                index=task_index,
                prompt_index=prompt_index,
                sample_index=sample_index,
                ground_truth=label,
                response=response,
                score=float(scored["score"]),
                acc=float(scored["acc"]),
                pred=scored.get("pred"),
            )

        coros = [
            asyncio.create_task(_run_one(i, prompt_index, sample_index, label, prompt_ids))
            for i, (prompt_index, sample_index, label, prompt_ids) in enumerate(tasks)
        ]
        results = []
        for coro in tqdm(asyncio.as_completed(coros), total=len(coros), desc="Eval aime"):
            results.append(await coro)

    results.sort(key=lambda r: (r.prompt_index, r.sample_index))
    return results


def summarize(results: list[EvalResult], n_samples: int) -> dict[str, float]:
    scores = [r.score for r in results]
    accs = [r.acc for r in results]
    summary = {
        "eval/aime": sum(scores) / len(scores) if scores else 0.0,
        "eval/aime/acc": sum(accs) / len(accs) if accs else 0.0,
        "eval/aime/score": sum(scores) / len(scores) if scores else 0.0,
        "num_prompts": len({r.prompt_index for r in results}),
        "num_samples": len(results),
    }

    # pass@k: at least one correct among n_samples per prompt
    by_prompt: dict[int, list[float]] = {}
    for r in results:
        by_prompt.setdefault(r.prompt_index, []).append(r.acc)
    pass_at_k = sum(1 for accs in by_prompt.values() if max(accs) > 0) / len(by_prompt) if by_prompt else 0.0
    summary[f"eval/aime/pass@{n_samples}"] = pass_at_k
    summary["eval/summary/acc"] = summary["eval/aime/acc"]
    summary["eval/summary/score"] = summary["eval/aime/score"]
    return summary


def save_results(path: str, results: list[EvalResult], summary: dict[str, float]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for r in results:
            f.write(
                json.dumps(
                    {
                        "prompt_index": r.prompt_index,
                        "sample_index": r.sample_index,
                        "ground_truth": r.ground_truth,
                        "pred": r.pred,
                        "score": r.score,
                        "acc": r.acc,
                        "response": r.response,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
        f.write(json.dumps({"summary": summary}, ensure_ascii=False) + "\n")
    logger.info("Wrote results to %s", out)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    results = asyncio.run(run_eval(args))
    summary = summarize(results, args.n_samples)

    print("\n=== AIME eval summary ===")
    for key in sorted(summary):
        print(f"  {key}: {summary[key]:.4f}" if isinstance(summary[key], float) else f"  {key}: {summary[key]}")

    if args.output:
        save_results(args.output, results, summary)

    return 0


if __name__ == "__main__":
    sys.exit(main())
