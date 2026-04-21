from __future__ import annotations

import os
from pathlib import Path
import logging

from slime.rollout.base_types import RolloutFnEvalOutput

from examples.recall_agent.eval_bfcl_multi_turn_official_external import run_official_external_eval

logger = logging.getLogger(__name__)


def generate_eval(args, rollout_id, data_source, evaluation=False):
    del data_source, evaluation

    bfcl_root = Path(os.getenv("BFCL_OFFICIAL_ROOT", "/dev/shm/ye/berkeley-function-call-leaderboard"))
    sample_size = int(os.getenv("BFCL_MULTI_TURN_SAMPLE_SIZE", "200"))
    sample_size_mode = os.getenv("BFCL_MULTI_TURN_SAMPLE_MODE", "per_category")
    seed = int(os.getenv("BFCL_MULTI_TURN_SAMPLE_SEED", "42"))
    num_threads = int(os.getenv("BFCL_EXTERNAL_NUM_THREADS", "16"))
    temperature = float(os.getenv("BFCL_EXTERNAL_TEMPERATURE", "0.0"))
    model_name_base = os.getenv("BFCL_EXTERNAL_MODEL_NAME", "slime-bfcl-official-external")
    model_name = f"{model_name_base}-eval{rollout_id}"
    router_ip = str(args.sglang_router_ip)
    router_port = int(args.sglang_router_port)

    logger.info(
        "Running official-compatible BFCL multi-turn eval via training router %s:%s "
        "(bfcl_root=%s, sample_size=%s, seed=%s, rollout_id=%s)",
        router_ip,
        router_port,
        bfcl_root,
        sample_size,
        seed,
        rollout_id,
    )

    summary = run_official_external_eval(
        bfcl_root=bfcl_root,
        hf_checkpoint=Path(args.hf_checkpoint),
        model_name=model_name,
        test_category="multi_turn",
        sample_size=sample_size,
        sample_size_mode=sample_size_mode,
        seed=seed,
        num_threads=num_threads,
        temperature=temperature,
        local_server_endpoint=router_ip,
        local_server_port=router_port,
        allow_overwrite=True,
    )

    metrics = {
        "bfcl_official/overall_accuracy": summary["overall_accuracy"],
        "bfcl_official/overall_correct_count": summary["overall_correct_count"],
        "bfcl_official/overall_total_count": summary["overall_total_count"],
    }
    for category, item in summary["categories"].items():
        metrics[f"bfcl_official/{category}"] = item["accuracy"]

    return RolloutFnEvalOutput(data={}, metrics=metrics)
