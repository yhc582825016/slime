#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." &>/dev/null && pwd)"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
cd "${ROOT_DIR}"

HF_CKPT="${HF_CKPT:-/dev/shm/ye/Qwen3-4B}"
MEGATRON_PATH="${MEGATRON_PATH:-/root/Megatron-LM}"
RAW_DATA="${RAW_DATA:-/dev/shm/ye/rl-data/AgenticQwen-Data/agenticqwen_synthetic_data.parquet}"
PROMPT_DATA_DIR="${PROMPT_DATA_DIR:-${SCRIPT_DIR}/data}"
PROMPT_TRAIN="${PROMPT_TRAIN:-${PROMPT_DATA_DIR}/train.jsonl}"
PROMPT_TEST="${PROMPT_TEST:-${PROMPT_DATA_DIR}/test.jsonl}"
SAVE_PATH="${SAVE_PATH:-/dev/shm/agenticqwen-slime}"

NUM_GPUS="${NUM_GPUS:-$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l || true)}"
if [[ -z "${NUM_GPUS}" || "${NUM_GPUS}" -le 0 ]]; then
  NUM_GPUS=1
fi

TP_SIZE="${TP_SIZE:-4}"
ROLLOUT_NUM_GPUS_PER_ENGINE="${ROLLOUT_NUM_GPUS_PER_ENGINE:-2}"
ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-32}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-128}"
N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-8}"
NUM_ROLLOUT="${NUM_ROLLOUT:-4000}"
MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU:-6144}"
ROLLOUT_MAX_RESPONSE_LEN="${ROLLOUT_MAX_RESPONSE_LEN:-20000}"

mkdir -p "${PROMPT_DATA_DIR}" "${SAVE_PATH}"

echo "[step] converting AgenticQwen parquet to slime jsonl..."
python "${SCRIPT_DIR}/preprocess_agenticqwen_data.py" \
  --input "${RAW_DATA}" \
  --output-dir "${PROMPT_DATA_DIR}"

echo "[step] starting AgenticQwen slime RL..."
PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}" \
python train.py \
  --hf-checkpoint "${HF_CKPT}" \
  --ref-load "${HF_CKPT}" \
  --load "${HF_CKPT}" \
  --save "${SAVE_PATH}" \
  --prompt-data "${PROMPT_TRAIN}" \
  --input-key prompt \
  --label-key label \
  --metadata-key metadata \
  --tool-key tools \
  --apply-chat-template \
  --rollout-global-dataset \
  --rollout-shuffle \
  --num-rollout "${NUM_ROLLOUT}" \
  --rollout-batch-size "${ROLLOUT_BATCH_SIZE}" \
  --n-samples-per-prompt "${N_SAMPLES_PER_PROMPT}" \
  --global-batch-size "${GLOBAL_BATCH_SIZE}" \
  --rollout-max-prompt-len 4096 \
  --rollout-max-response-len "${ROLLOUT_MAX_RESPONSE_LEN}" \
  --rollout-max-context-len "${ROLLOUT_MAX_RESPONSE_LEN}" \
  --max-tokens-per-gpu "${MAX_TOKENS_PER_GPU}" \
  --custom-generate-function-path examples.agenticqwen.rollout.generate \
  --custom-config-path examples/agenticqwen/agenticqwen_config.yaml \
  --custom-rm-path examples.agenticqwen.reward.reward_func \
  --advantage-estimator grpo \
  --use-kl-loss \
  --kl-loss-coef 0.0 \
  --kl-loss-type low_var_kl \
  --entropy-coef 0.0 \
  --eps-clip 0.2 \
  --eps-clip-high 0.28 \
  --optimizer adam \
  --lr 1e-6 \
  --tensor-model-parallel-size "${TP_SIZE}" \
  --rollout-num-gpus-per-engine "${ROLLOUT_NUM_GPUS_PER_ENGINE}" \
  --num-gpus "${NUM_GPUS}" \
  --sglang-mem-fraction-static 0.55 \
  --attention-backend flash \
  "$@"

