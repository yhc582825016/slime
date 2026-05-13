#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." &>/dev/null && pwd)"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
cd "${ROOT_DIR}"

export PYTHONUNBUFFERED=1

HF_CKPT="${HF_CKPT:-/dev/shm/Qwen3.5-4B}"
REF_LOAD="${REF_LOAD:-/dev/shm/Qwen3.5-4B-Thinking_torch_dist}"
LOAD_PATH="${LOAD_PATH:-${REF_LOAD}}"
SAVE_PATH="${SAVE_PATH:-/dev/shm/Qwen3.5-4B-mix-rl-slime}"
MEGATRON_PATH="${MEGATRON_PATH:-/root/Megatron-LM}"

PROMPT_DATA="${PROMPT_DATA:-/dev/shm/ye/rl-data/mix-rl-slime/mix_train.jsonl}"
RUN_PREPARE_DATA="${RUN_PREPARE_DATA:-0}"
NON_THINKING="${NON_THINKING:-1}"

# Codegen rewards default to unsafe_local in coder1. Set CODER1_EXEC=bwrap if
# bubblewrap is available and you want local isolation.
export CODER1_EXEC="${CODER1_EXEC:-unsafe_local}"

if command -v nvidia-smi >/dev/null 2>&1; then
  DETECTED_GPUS="$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')"
else
  DETECTED_GPUS=0
fi
NUM_GPUS="${NUM_GPUS:-${DETECTED_GPUS}}"
if [[ -z "${NUM_GPUS}" || "${NUM_GPUS}" -le 0 ]]; then
  NUM_GPUS=1
fi

TP_SIZE="${TP_SIZE:-4}"
ROLLOUT_NUM_GPUS_PER_ENGINE="${ROLLOUT_NUM_GPUS_PER_ENGINE:-1}"
SGLANG_MEM_FRACTION_STATIC="${SGLANG_MEM_FRACTION_STATIC:-0.4}"
DISABLE_OFFLOAD="${DISABLE_OFFLOAD:-1}"
MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU:-9216}"
ROLLOUT_MAX_RESPONSE_LEN="${ROLLOUT_MAX_RESPONSE_LEN:-8192}"
NUM_ROLLOUT="${NUM_ROLLOUT:-3000}"
ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-4}"
N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-8}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-32}"
ROLLOUT_TEMPERATURE="${ROLLOUT_TEMPERATURE:-1.0}"
ROLLOUT_TOP_P="${ROLLOUT_TOP_P:-0.95}"
ROLLOUT_TOP_K="${ROLLOUT_TOP_K:-20}"
SAVE_INTERVAL="${SAVE_INTERVAL:-100}"
LR="${LR:-1e-6}"

MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
RAY_DASHBOARD_PORT="${RAY_DASHBOARD_PORT:-8265}"

USE_EVAL="${USE_EVAL:-1}"
EVAL_INTERVAL="${EVAL_INTERVAL:-50}"
EVAL_DATA_NAME="${EVAL_DATA_NAME:-mix_rl_eval}"
EVAL_PROMPT_DATA="${EVAL_PROMPT_DATA:-${PROMPT_DATA}@[0:256]}"
N_SAMPLES_PER_EVAL_PROMPT="${N_SAMPLES_PER_EVAL_PROMPT:-1}"
EVAL_MAX_RESPONSE_LEN="${EVAL_MAX_RESPONSE_LEN:-3072}"
EVAL_TEMPERATURE="${EVAL_TEMPERATURE:-${ROLLOUT_TEMPERATURE}}"
EVAL_TOP_P="${EVAL_TOP_P:-${ROLLOUT_TOP_P}}"
EVAL_TOP_K="${EVAL_TOP_K:-${ROLLOUT_TOP_K}}"

USE_WANDB="${USE_WANDB:-1}"
WANDB_PROJECT="${WANDB_PROJECT:-slime}"
WANDB_GROUP="${WANDB_GROUP:-qwen3.5-4B-mix-rl}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-qwen3.5-4B-mix-rl}"
WANDB_KEY="${WANDB_KEY:-${WANDB_API_KEY:-}}"
WANDB_BASE_URL="${WANDB_BASE_URL:-https://api.wandb.ai}"

NOHUP="${NOHUP:-0}"
NOHUP_LOG="${NOHUP_LOG:-${SAVE_PATH}/run-nohup.log}"

echo "[config] ROOT_DIR=${ROOT_DIR}"
echo "[config] HF_CKPT=${HF_CKPT}"
echo "[config] REF_LOAD=${REF_LOAD}"
echo "[config] LOAD_PATH=${LOAD_PATH}"
echo "[config] SAVE_PATH=${SAVE_PATH}"
echo "[config] PROMPT_DATA=${PROMPT_DATA}"
echo "[config] RUN_PREPARE_DATA=${RUN_PREPARE_DATA}"
echo "[config] NON_THINKING=${NON_THINKING}"
echo "[config] CODER1_EXEC=${CODER1_EXEC}"
echo "[config] NUM_GPUS=${NUM_GPUS}, TP_SIZE=${TP_SIZE}, ROLLOUT_NUM_GPUS_PER_ENGINE=${ROLLOUT_NUM_GPUS_PER_ENGINE}"
echo "[config] SGLANG_MEM_FRACTION_STATIC=${SGLANG_MEM_FRACTION_STATIC}, DISABLE_OFFLOAD=${DISABLE_OFFLOAD}"

if [[ "${NOHUP}" == "1" && "${_NOHUP_LAUNCHED:-0}" != "1" ]]; then
  mkdir -p "$(dirname "${NOHUP_LOG}")"
  echo "[step] relaunch with nohup, log: ${NOHUP_LOG}"
  nohup env _NOHUP_LAUNCHED=1 NOHUP=0 bash "$0" "$@" >"${NOHUP_LOG}" 2>&1 &
  echo "[step] started in background, pid=$!"
  exit 0
fi

if [[ ! -d "${HF_CKPT}" ]]; then
  echo "[error] HF_CKPT not found: ${HF_CKPT}" >&2
  exit 1
fi
if [[ ! -d "${REF_LOAD}" ]]; then
  echo "[error] REF_LOAD not found: ${REF_LOAD}" >&2
  exit 1
fi
if [[ ! -d "${MEGATRON_PATH}" ]]; then
  echo "[error] MEGATRON_PATH not found: ${MEGATRON_PATH}" >&2
  exit 1
fi

if [[ "${RUN_PREPARE_DATA}" == "1" || ! -f "${PROMPT_DATA}" ]]; then
  echo "[step] preparing mixed Guru + ReasoningGym data..."
  MIX_OUTPUT="${PROMPT_DATA}" bash "${SCRIPT_DIR}/prepare_mix_rl_data.sh"
else
  echo "[step] skip data preparation (already exists): ${PROMPT_DATA}"
fi

mkdir -p "${SAVE_PATH}"

source "${ROOT_DIR}/scripts/models/qwen3.5-4B.sh"

CHAT_TEMPLATE_ARGS=()
if [[ "${NON_THINKING}" == "1" ]]; then
  CHAT_TEMPLATE_ARGS+=(--apply-chat-template-kwargs '{"enable_thinking": false}')
fi

CKPT_ARGS=(
  --hf-checkpoint "${HF_CKPT}"
  --ref-load "${REF_LOAD}"
  --load "${LOAD_PATH}"
  --save "${SAVE_PATH}"
  --save-interval "${SAVE_INTERVAL}"
)

ROLLOUT_ARGS=(
  --prompt-data "${PROMPT_DATA}"
  --input-key prompt
  --label-key reward_model
  --metadata-key metadata
  --apply-chat-template
  --rollout-shuffle
  --num-rollout "${NUM_ROLLOUT}"
  --rollout-batch-size "${ROLLOUT_BATCH_SIZE}"
  --n-samples-per-prompt "${N_SAMPLES_PER_PROMPT}"
  --rollout-max-response-len "${ROLLOUT_MAX_RESPONSE_LEN}"
  --rollout-temperature "${ROLLOUT_TEMPERATURE}"
  --rollout-top-p "${ROLLOUT_TOP_P}"
  --rollout-top-k "${ROLLOUT_TOP_K}"
  --global-batch-size "${GLOBAL_BATCH_SIZE}"
  --balance-data
)

PERF_ARGS=(
  --tensor-model-parallel-size "${TP_SIZE}"
  --sequence-parallel
  --pipeline-model-parallel-size 1
  --context-parallel-size 2
  --expert-model-parallel-size 1
  --expert-tensor-parallel-size 1
  --recompute-granularity full
  --recompute-method uniform
  --recompute-num-layers 1
  --use-dynamic-batch-size
  --max-tokens-per-gpu "${MAX_TOKENS_PER_GPU}"
)

GRPO_ARGS=(
  --advantage-estimator grpo
  --use-kl-loss
  --kl-loss-coef 0.00
  --kl-loss-type low_var_kl
  --entropy-coef 0.00
  --eps-clip 0.2
  --eps-clip-high 0.28
)

OPTIMIZER_ARGS=(
  --optimizer adam
  --lr "${LR}"
  --lr-decay-style constant
  --weight-decay 0.1
  --adam-beta1 0.9
  --adam-beta2 0.98
)

SGLANG_ARGS=(
  --rollout-num-gpus-per-engine "${ROLLOUT_NUM_GPUS_PER_ENGINE}"
  --sglang-mem-fraction-static "${SGLANG_MEM_FRACTION_STATIC}"
)

OFFLOAD_ARGS=()
if [[ "${DISABLE_OFFLOAD}" == "0" ]]; then
  OFFLOAD_ARGS+=(--no-offload-train --no-offload-rollout)
fi

MISC_ARGS=(
  --attention-dropout 0.0
  --hidden-dropout 0.0
  --accumulate-allreduce-grads-in-fp32
  --attention-softmax-in-fp32
  --attention-backend flash
)

CUSTOM_ARGS=(
  --custom-rm-path examples.mix_rl.reward.reward_func
  --reward-key score
)

EVAL_ARGS=()
if [[ "${USE_EVAL}" == "1" ]]; then
  EVAL_ARGS+=(
    --eval-interval "${EVAL_INTERVAL}"
    --eval-prompt-data "${EVAL_DATA_NAME}" "${EVAL_PROMPT_DATA}"
    --eval-input-key prompt
    --eval-label-key reward_model
    --n-samples-per-eval-prompt "${N_SAMPLES_PER_EVAL_PROMPT}"
    --eval-max-response-len "${EVAL_MAX_RESPONSE_LEN}"
    --eval-temperature "${EVAL_TEMPERATURE}"
    --eval-top-p "${EVAL_TOP_P}"
    --eval-top-k "${EVAL_TOP_K}"
  )
fi

WANDB_ARGS=()
if [[ "${USE_WANDB}" == "1" ]]; then
  export WANDB_API_KEY="${WANDB_KEY}"
  export WANDB_BASE_URL
  WANDB_ARGS+=(
    --use-wandb
    --wandb-project "${WANDB_PROJECT}"
    --wandb-group "${WANDB_GROUP}"
    --wandb-exp-name "${WANDB_RUN_NAME}"
  )
  if [[ -n "${WANDB_KEY}" ]]; then
    WANDB_ARGS+=(--wandb-key "${WANDB_KEY}")
  fi
fi

echo "[step] restarting ray..."
ray stop --force || true
pkill -f sglang || true

ray start --head \
  --node-ip-address "${MASTER_ADDR}" \
  --num-gpus "${NUM_GPUS}" \
  --disable-usage-stats \
  --dashboard-host=0.0.0.0 \
  --dashboard-port "${RAY_DASHBOARD_PORT}"

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"${MEGATRON_PATH}:${ROOT_DIR}\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"CODER1_EXEC\": \"${CODER1_EXEC}\",
    \"WANDB_API_KEY\": \"${WANDB_KEY}\",
    \"WANDB_BASE_URL\": \"${WANDB_BASE_URL}\"
  }
}"

echo "[step] submitting ray job..."
ray job submit --address="http://127.0.0.1:${RAY_DASHBOARD_PORT}" \
  --runtime-env-json="${RUNTIME_ENV_JSON}" \
  -- python3 train.py \
  --actor-num-nodes 1 \
  --actor-num-gpus-per-node "${NUM_GPUS}" \
  --colocate \
  "${MODEL_ARGS[@]}" \
  "${CKPT_ARGS[@]}" \
  "${CHAT_TEMPLATE_ARGS[@]}" \
  "${ROLLOUT_ARGS[@]}" \
  "${EVAL_ARGS[@]}" \
  "${OPTIMIZER_ARGS[@]}" \
  "${GRPO_ARGS[@]}" \
  "${WANDB_ARGS[@]}" \
  "${PERF_ARGS[@]}" \
  "${SGLANG_ARGS[@]}" \
  "${OFFLOAD_ARGS[@]}" \
  "${MISC_ARGS[@]}" \
  "${CUSTOM_ARGS[@]}"
