#!/bin/bash
# pkill -9 sglang
# sleep 3
# ray stop --force
# pkill -9 ray
# pkill -9 python
# sleep 3
# pkill -9 ray
# pkill -9 python

set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." &>/dev/null && pwd)"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
cd "${ROOT_DIR}"

export PYTHONBUFFERED=16

# NOTE:
# In colocate mode, slime enables rollout offload/memory_saver for SGLang.
# torch_memory_saver is incompatible with PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,
# and the rollout server will crash during initialization if both are enabled.
# Keep expandable_segments opt-in instead of enabling it by default here.
ENABLE_EXPANDABLE_SEGMENTS="${ENABLE_EXPANDABLE_SEGMENTS:-0}"
if [[ "${ENABLE_EXPANDABLE_SEGMENTS}" == "1" ]]; then
  export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
fi

HF_CKPT="${HF_CKPT:-/dev/shm/Qwen3.5-4B}"
MEGATRON_PATH="${MEGATRON_PATH:-/root/Megatron-LM}"
TAU_BENCH_ROOT="${TAU_BENCH_ROOT:-/dev/shm/ye/tau-bench}"
PROMPT_TRAIN="${PROMPT_TRAIN:-${SCRIPT_DIR}/retail_train_tasks.jsonl}"
PROMPT_DEV="${PROMPT_DEV:-${SCRIPT_DIR}/retail_dev_tasks.jsonl}"
REF_LOAD="${REF_LOAD:-/dev/shm/Qwen3.5-4B-Thinking_torch_dist}"
LOAD_PATH="${LOAD_PATH:-${REF_LOAD}}"
SAVE_PATH="${SAVE_PATH:-/dev/shm/Qwen3.5-4B_tau_bench}"
TAU_BENCH_JOB_LOG="${TAU_BENCH_JOB_LOG:-/dev/shm/ye/logs/tau_bench_qwen3_5_4b_$(date +%Y%m%d_%H%M%S).log}"

NUM_GPUS="${NUM_GPUS:-6}"
TP_SIZE="${TP_SIZE:-2}"
CONTEXT_PARALLEL_SIZE="${CONTEXT_PARALLEL_SIZE:-1}"
ROLLOUT_NUM_GPUS_PER_ENGINE="${ROLLOUT_NUM_GPUS_PER_ENGINE:-2}"

MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU:-3072}"
ROLLOUT_MAX_RESPONSE_LEN="${ROLLOUT_MAX_RESPONSE_LEN:-2048}"
NUM_ROLLOUT="${NUM_ROLLOUT:-500}"
ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-8}"
N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-4}"
# Keep this divisible by micro_batch_size * data_parallel_size.
# With 6 GPUs and TP=2, data_parallel_size is 3.
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-24}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-1}"
LOG_PROBS_CHUNK_SIZE="${LOG_PROBS_CHUNK_SIZE:-512}"
SAVE_INTERVAL="${SAVE_INTERVAL:-50}"

USE_EVAL="${USE_EVAL:-1}"
EVAL_INTERVAL="${EVAL_INTERVAL:-5}"
EVAL_DATA_NAME="${EVAL_DATA_NAME:-retail-dev}"
N_SAMPLES_PER_EVAL_PROMPT="${N_SAMPLES_PER_EVAL_PROMPT:-1}"
EVAL_MAX_RESPONSE_LEN="${EVAL_MAX_RESPONSE_LEN:-2048}"

USE_WANDB="${USE_WANDB:-1}"
WANDB_PROJECT="${WANDB_PROJECT:-slime}"
WANDB_GROUP="${WANDB_GROUP:-tau_bench_qwen3_5_4b}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-qwen3.5-4B-tau-bench-6gpu}"
WANDB_KEY="${WANDB_KEY:-${WANDB_API_KEY:-}}"
WANDB_BASE_URL="${WANDB_BASE_URL:-https://api.bandw.top}"

TAU_USER_MODEL="${TAU_USER_MODEL:-Qwen3.5-4B}"
TAU_USER_MODEL_PROVIDER="${TAU_USER_MODEL_PROVIDER:-openai}"
TAU_USER_STRATEGY="${TAU_USER_STRATEGY:-llm}"
TAU_USER_MAX_CONCURRENCY="${TAU_USER_MAX_CONCURRENCY:-16}"
TAU_MAX_NUM_STEPS="${TAU_MAX_NUM_STEPS:-32}"
OPENAI_BASE_URL="${OPENAI_BASE_URL:-http://127.0.0.1:4004/v1}"
OPENAI_API_KEY="${OPENAI_API_KEY:-sk-local}"
TAU_TOOL_CALL_PARSER="${TAU_TOOL_CALL_PARSER:-qwen3_coder}"

export TAU_TOOL_CALL_PARSER
export TAU_MAX_NUM_STEPS

USE_TENSORBOARD="${USE_TENSORBOARD:-0}"
TB_PROJECT_NAME="${TB_PROJECT_NAME:-slime}"
TB_EXPERIMENT_NAME="${TB_EXPERIMENT_NAME:-${WANDB_RUN_NAME}}"
TENSORBOARD_DIR="${TENSORBOARD_DIR:-$(dirname -- "${SAVE_PATH}")/tensorboard/${TB_PROJECT_NAME}/${TB_EXPERIMENT_NAME}}"

MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
RAY_DASHBOARD_PORT="${RAY_DASHBOARD_PORT:-8265}"

if [[ ! -d "${HF_CKPT}" ]]; then
  echo "[error] HF_CKPT not found: ${HF_CKPT}"
  exit 1
fi
if [[ ! -f "${PROMPT_TRAIN}" ]]; then
  echo "[error] PROMPT_TRAIN not found: ${PROMPT_TRAIN}"
  exit 1
fi
if [[ ! -f "${PROMPT_DEV}" ]]; then
  echo "[error] PROMPT_DEV not found: ${PROMPT_DEV}"
  exit 1
fi
if [[ ! -d "${MEGATRON_PATH}" ]]; then
  echo "[error] MEGATRON_PATH not found: ${MEGATRON_PATH}"
  exit 1
fi
if [[ ! -d "${TAU_BENCH_ROOT}" ]]; then
  echo "[error] TAU_BENCH_ROOT not found: ${TAU_BENCH_ROOT}"
  exit 1
fi

mkdir -p "${SAVE_PATH}" "$(dirname -- "${TAU_BENCH_JOB_LOG}")"

if [[ ! -d "${REF_LOAD}" ]]; then
  echo "[step] converting HF checkpoint -> torch_dist..."
  source "${ROOT_DIR}/scripts/models/qwen3.5-4B.sh"
  PYTHONPATH="${MEGATRON_PATH}:${ROOT_DIR}" \
    python "${ROOT_DIR}/tools/convert_hf_to_torch_dist.py" \
    "${MODEL_ARGS[@]}" \
    --hf-checkpoint "${HF_CKPT}" \
    --save "${REF_LOAD}"
fi

source "${ROOT_DIR}/scripts/models/qwen3.5-4B.sh"

CKPT_ARGS=(
  --hf-checkpoint "${HF_CKPT}"
  --ref-load "${REF_LOAD}"
  --load "${LOAD_PATH}"
  --save "${SAVE_PATH}"
  --save-interval "${SAVE_INTERVAL}"
)

ROLLOUT_ARGS=(
  --prompt-data "${PROMPT_TRAIN}"
  --input-key index
  --rollout-shuffle
  --num-rollout "${NUM_ROLLOUT}"
  --rollout-batch-size "${ROLLOUT_BATCH_SIZE}"
  --n-samples-per-prompt "${N_SAMPLES_PER_PROMPT}"
  --rollout-max-response-len "${ROLLOUT_MAX_RESPONSE_LEN}"
  --rollout-temperature 1.0
  --global-batch-size "${GLOBAL_BATCH_SIZE}"
  --dynamic-sampling-filter-path slime.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std
  --balance-data
)

EVAL_ARGS=()
if [[ "${USE_EVAL}" == "1" ]]; then
  EVAL_ARGS+=(
    --eval-interval "${EVAL_INTERVAL}"
    --eval-prompt-data "${EVAL_DATA_NAME}" "${PROMPT_DEV}"
    --n-samples-per-eval-prompt "${N_SAMPLES_PER_EVAL_PROMPT}"
    --eval-max-response-len "${EVAL_MAX_RESPONSE_LEN}"
    --eval-top-k 1
  )
fi

PERF_ARGS=(
  --tensor-model-parallel-size "${TP_SIZE}"
  --sequence-parallel
  --pipeline-model-parallel-size 1
  --context-parallel-size "${CONTEXT_PARALLEL_SIZE}"
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
  --kl-loss-coef 0.001
  --kl-loss-type low_var_kl
  --entropy-coef 0.00
  --eps-clip 0.2
  --eps-clip-high 0.28
)

OPTIMIZER_ARGS=(
  --optimizer adam
  --lr 1e-6
  --lr-decay-style constant
  --weight-decay 0.1
  --adam-beta1 0.9
  --adam-beta2 0.98
)

SGLANG_ARGS=(
  --rollout-num-gpus-per-engine "${ROLLOUT_NUM_GPUS_PER_ENGINE}"
  --sglang-mem-fraction-static 0.35
)

MISC_ARGS=(
  --attention-dropout 0.0
  --hidden-dropout 0.0
  --accumulate-allreduce-grads-in-fp32
  --attention-softmax-in-fp32
  --attention-backend flash
)

TRAIN_BATCH_ARGS=(
  --micro-batch-size "${MICRO_BATCH_SIZE}"
  --log-probs-chunk-size "${LOG_PROBS_CHUNK_SIZE}"
)

CUSTOM_ARGS=(
  --custom-generate-function-path generate_with_tau.generate
)

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

TENSORBOARD_ARGS=()
if [[ "${USE_TENSORBOARD}" == "1" ]]; then
  mkdir -p "${TENSORBOARD_DIR}"
  export TENSORBOARD_DIR
  TENSORBOARD_ARGS+=(
    --use-tensorboard
    --tb-project-name "${TB_PROJECT_NAME}"
    --tb-experiment-name "${TB_EXPERIMENT_NAME}"
  )
fi

echo "[step] restarting ray..."
ray stop --force || true
pkill -f sglang || true
sleep 2

ray start --head \
  --node-ip-address "${MASTER_ADDR}" \
  --num-gpus "${NUM_GPUS}" \
  --disable-usage-stats \
  --dashboard-host=0.0.0.0 \
  --dashboard-port "${RAY_DASHBOARD_PORT}"

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"${MEGATRON_PATH}:${TAU_BENCH_ROOT}:${SCRIPT_DIR}\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"TENSORBOARD_DIR\": \"${TENSORBOARD_DIR}\",
    \"WANDB_API_KEY\": \"${WANDB_API_KEY:-}\",
    \"WANDB_BASE_URL\": \"${WANDB_BASE_URL}\",
    \"TAU_USER_MODEL\": \"${TAU_USER_MODEL}\",
    \"TAU_USER_MODEL_PROVIDER\": \"${TAU_USER_MODEL_PROVIDER}\",
    \"TAU_USER_STRATEGY\": \"${TAU_USER_STRATEGY}\",
    \"TAU_USER_MAX_CONCURRENCY\": \"${TAU_USER_MAX_CONCURRENCY}\",
    \"TAU_MAX_NUM_STEPS\": \"${TAU_MAX_NUM_STEPS}\",
    \"OPENAI_BASE_URL\": \"${OPENAI_BASE_URL}\",
    \"OPENAI_API_BASE\": \"${OPENAI_BASE_URL}\",
    \"OPENAI_API_KEY\": \"${OPENAI_API_KEY}\",
    \"LITELLM_API_BASE\": \"${OPENAI_BASE_URL}\",
    \"LITELLM_API_KEY\": \"${OPENAI_API_KEY}\"
  }
}"

echo "[step] submitting ray job (nohup, log: ${TAU_BENCH_JOB_LOG})..."
TRAIN_CMD=(
  python3 train.py
  --actor-num-nodes 1
  --actor-num-gpus-per-node "${NUM_GPUS}"
  --rollout-num-gpus "${NUM_GPUS}"
  --colocate
  "${MODEL_ARGS[@]}"
  "${CKPT_ARGS[@]}"
  "${ROLLOUT_ARGS[@]}"
  "${EVAL_ARGS[@]}"
  "${OPTIMIZER_ARGS[@]}"
  "${GRPO_ARGS[@]}"
  "${WANDB_ARGS[@]}"
  "${TENSORBOARD_ARGS[@]}"
  "${TRAIN_BATCH_ARGS[@]}"
  "${PERF_ARGS[@]}"
  "${SGLANG_ARGS[@]}"
  "${MISC_ARGS[@]}"
  "${CUSTOM_ARGS[@]}"
)

printf -v TRAIN_CMD_STR '%q ' "${TRAIN_CMD[@]}"

nohup ray job submit --address="http://127.0.0.1:${RAY_DASHBOARD_PORT}" \
  --runtime-env-json="${RUNTIME_ENV_JSON}" \
  -- bash -lc "${TRAIN_CMD_STR}" \
  >> "${TAU_BENCH_JOB_LOG}" 2>&1 &
echo "[info] ray job submit pid: $!  (output: ${TAU_BENCH_JOB_LOG})"
if [[ "${USE_TENSORBOARD}" == "1" ]]; then
  echo "[info] tensorboard dir: ${TENSORBOARD_DIR}"
fi
