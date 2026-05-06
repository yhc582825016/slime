#!/bin/bash
pkill -9 sglang
sleep 3
ray stop --force
pkill -9 ray
pkill -9 python
sleep 3
pkill -9 ray
pkill -9 python

set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." &>/dev/null && pwd)"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
cd "${ROOT_DIR}"

export PYTHONBUFFERED=16

HF_CKPT="${HF_CKPT:-/dev/shm/Qwen3.5-4B}"
MEGATRON_PATH="${MEGATRON_PATH:-/root/Megatron-LM}"
# https://api.wandb.ai.
RAW_TRAIN="${RAW_TRAIN:-/dev/shm/ye/rl-data/agent_world/agent_world_recall_train.parquet}"
RAW_TEST="${RAW_TEST:-/dev/shm/ye/rl-data/agent_syn_data/test_filtered.parquet}"
PROMPT_DATA_DIR="${PROMPT_DATA_DIR:-${SCRIPT_DIR}/data}"
USE_BFCL_MULTI_TURN_EVAL="${USE_BFCL_MULTI_TURN_EVAL:-1}"
BFCL_ROOT="${BFCL_ROOT:-/dev/shm/ye/gorilla/berkeley-function-call-leaderboard}"
BFCL_MULTI_TURN_SAMPLE_SIZE="${BFCL_MULTI_TURN_SAMPLE_SIZE:-200}"
BFCL_MULTI_TURN_SAMPLE_MODE="${BFCL_MULTI_TURN_SAMPLE_MODE:-per_category}"
BFCL_MULTI_TURN_SAMPLE_SEED="${BFCL_MULTI_TURN_SAMPLE_SEED:-42}"
BFCL_MULTI_TURN_EVAL_JSONL="${BFCL_MULTI_TURN_EVAL_JSONL:-${PROMPT_DATA_DIR}/bfcl_multi_turn_test_${BFCL_MULTI_TURN_SAMPLE_SIZE}.jsonl}"
BFCL_EVAL_BACKEND="${BFCL_EVAL_BACKEND:-official_external}"
BFCL_OFFICIAL_ROOT="${BFCL_OFFICIAL_ROOT:-/dev/shm/ye/berkeley-function-call-leaderboard}"
BFCL_EXTERNAL_MODEL_NAME="${BFCL_EXTERNAL_MODEL_NAME:-qwen3.5-4B-recall-agent-bfcl-official-external}"
BFCL_EXTERNAL_NUM_THREADS="${BFCL_EXTERNAL_NUM_THREADS:-16}"
BFCL_EXTERNAL_TEMPERATURE="${BFCL_EXTERNAL_TEMPERATURE:-0.0}"
PROMPT_TRAIN="${PROMPT_TRAIN:-${PROMPT_DATA_DIR}/train.jsonl}"
PROMPT_TEST="${PROMPT_TEST:-${PROMPT_DATA_DIR}/test.jsonl}"
echo "PROMPT_TRAIN: ${PROMPT_TRAIN}"
echo "PROMPT_TEST: ${PROMPT_TEST}"
REF_LOAD="${REF_LOAD:-/dev/shm/Qwen3.5-4B-Thinking_torch_dist}"
LOAD_PATH="${LOAD_PATH:-${REF_LOAD}}"
SAVE_PATH="${SAVE_PATH:-/dev/shm/Qwen3.5-4B-Thinking_recall_agent_430}"
RECALL_AGENT_JOB_LOG="${RECALL_AGENT_JOB_LOG:-/dev/shm/ye/logs/recall_agent_$(date +%Y%m%d_%H%M%S).log}"

NUM_GPUS="${NUM_GPUS:-$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)}"
if [[ -z "${NUM_GPUS}" || "${NUM_GPUS}" -le 0 ]]; then
  NUM_GPUS=1
fi

TP_SIZE="${TP_SIZE:-4}"
ROLLOUT_NUM_GPUS_PER_ENGINE="${ROLLOUT_NUM_GPUS_PER_ENGINE:-2}"

MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU:-6144}"
ROLLOUT_MAX_RESPONSE_LEN="${ROLLOUT_MAX_RESPONSE_LEN:-20000}"
NUM_ROLLOUT="${NUM_ROLLOUT:-4000}"
ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-32}"
N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-8}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-128}"
SAVE_INTERVAL="${SAVE_INTERVAL:-50}"
OVER_SAMPLING_BATCH_SIZE="${OVER_SAMPLING_BATCH_SIZE:-32}"

USE_EVAL="${USE_EVAL:-1}"
EVAL_INTERVAL="${EVAL_INTERVAL:-10}"
EVAL_DATA_NAME="${EVAL_DATA_NAME:-recall_agent_eval}"
N_SAMPLES_PER_EVAL_PROMPT="${N_SAMPLES_PER_EVAL_PROMPT:-1}"
EVAL_MAX_RESPONSE_LEN="${EVAL_MAX_RESPONSE_LEN:-20000}"

USE_WANDB="${USE_WANDB:-1}"
WANDB_PROJECT="${WANDB_PROJECT:-slime}"
WANDB_GROUP="${WANDB_GROUP:-recall_agent_qwen3.5_4b}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-qwen3.5-4B-recall-agent}"
WANDB_KEY="${WANDB_KEY:-${WANDB_API_KEY:-}}"
WANDB_BASE_URL="${WANDB_BASE_URL:-https://api.bandw.top}"

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
if [[ ! -f "${RAW_TRAIN}" ]]; then
  echo "[error] RAW_TRAIN not found: ${RAW_TRAIN}"
  echo "[hint] expected a recall parquet file for preprocess_recall_agent_data.py"
  exit 1
fi
if [[ ! -f "${RAW_TEST}" ]]; then
  echo "[error] RAW_TEST not found: ${RAW_TEST}"
  exit 1
fi
if [[ ! -d "${MEGATRON_PATH}" ]]; then
  echo "[error] MEGATRON_PATH not found: ${MEGATRON_PATH}"
  exit 1
fi

mkdir -p "${PROMPT_DATA_DIR}" "${SAVE_PATH}" "$(dirname -- "${RECALL_AGENT_JOB_LOG}")"

echo "[step] converting recall-agent parquet to slime jsonl..."
python "${SCRIPT_DIR}/preprocess_recall_agent_data.py" \
  --train-input "${RAW_TRAIN}" \
  --test-input "${RAW_TEST}" \
  --output-dir "${PROMPT_DATA_DIR}"

if [[ "${USE_BFCL_MULTI_TURN_EVAL}" == "1" ]]; then
  if [[ "${BFCL_EVAL_BACKEND}" == "official_external" ]]; then
    echo "[step] BFCL multi-turn eval is set to official_external backend."
    echo "[step] switching eval_function_path to official-compatible BFCL eval that reuses the training router."
  else
    echo "[step] preparing BFCL multi-turn eval set (${BFCL_MULTI_TURN_SAMPLE_SIZE} samples, seed=${BFCL_MULTI_TURN_SAMPLE_SEED})..."
    python "${SCRIPT_DIR}/prepare_bfcl_multi_turn_eval.py" \
      --bfcl-root "${BFCL_ROOT}" \
      --output-path "${BFCL_MULTI_TURN_EVAL_JSONL}" \
      --sample-size "${BFCL_MULTI_TURN_SAMPLE_SIZE}" \
      --seed "${BFCL_MULTI_TURN_SAMPLE_SEED}"
    PROMPT_TEST="${BFCL_MULTI_TURN_EVAL_JSONL}"
  fi
fi

echo "PROMPT_TRAIN (resolved): ${PROMPT_TRAIN}"
echo "PROMPT_TEST (resolved): ${PROMPT_TEST}"

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
  --input-key prompt
  --label-key label
  --metadata-key metadata
  --tool-key tools
  --apply-chat-template
  --rollout-shuffle
  --num-rollout "${NUM_ROLLOUT}"
  --rollout-batch-size "${ROLLOUT_BATCH_SIZE}"
  --n-samples-per-prompt "${N_SAMPLES_PER_PROMPT}"
  --over-sampling-batch-size "${OVER_SAMPLING_BATCH_SIZE}"
  --dynamic-sampling-filter-path slime.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std
  --rollout-max-prompt-len 6000
  --rollout-max-response-len "${ROLLOUT_MAX_RESPONSE_LEN}"
  --rollout-temperature 1.0
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
  --sglang-mem-fraction-static 0.55
)

MISC_ARGS=(
  --attention-dropout 0.0
  --hidden-dropout 0.0
  --accumulate-allreduce-grads-in-fp32
  --attention-softmax-in-fp32
  --attention-backend flash
)

CUSTOM_ARGS=(
  --custom-generate-function-path examples.recall_agent.rollout.generate
  --custom-config-path examples/recall_agent/recall_agent_config.yaml
  --custom-rm-path slime.rollout.rm_hub.recall_agent.custom_rm
)

EVAL_ARGS=()
if [[ "${USE_EVAL}" == "1" ]]; then
  EVAL_ARGS+=(
    --eval-interval "${EVAL_INTERVAL}"
    --eval-prompt-data "${EVAL_DATA_NAME}" "${PROMPT_TEST}"
    --eval-input-key prompt
    --eval-label-key label
    --eval-max-response-len "${EVAL_MAX_RESPONSE_LEN}"
    --n-samples-per-eval-prompt "${N_SAMPLES_PER_EVAL_PROMPT}"
    --eval-top-p 1
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
    \"TENSORBOARD_DIR\": \"${TENSORBOARD_DIR}\",
    \"WANDB_API_KEY\": \"${WANDB_API_KEY:-}\",
    \"WANDB_BASE_URL\": \"${WANDB_BASE_URL}\",
    \"BFCL_OFFICIAL_ROOT\": \"${BFCL_OFFICIAL_ROOT}\",
    \"BFCL_MULTI_TURN_SAMPLE_SIZE\": \"${BFCL_MULTI_TURN_SAMPLE_SIZE}\",
    \"BFCL_MULTI_TURN_SAMPLE_MODE\": \"${BFCL_MULTI_TURN_SAMPLE_MODE}\",
    \"BFCL_MULTI_TURN_SAMPLE_SEED\": \"${BFCL_MULTI_TURN_SAMPLE_SEED}\",
    \"BFCL_EXTERNAL_MODEL_NAME\": \"${BFCL_EXTERNAL_MODEL_NAME}\",
    \"BFCL_EXTERNAL_NUM_THREADS\": \"${BFCL_EXTERNAL_NUM_THREADS}\",
    \"BFCL_EXTERNAL_TEMPERATURE\": \"${BFCL_EXTERNAL_TEMPERATURE}\"
  }
}"

if [[ "${USE_BFCL_MULTI_TURN_EVAL}" == "1" && "${BFCL_EVAL_BACKEND}" == "official_external" ]]; then
  EVAL_ARGS+=(--eval-function-path examples.recall_agent.bfcl_official_eval.generate_eval)
fi

echo "[step] submitting ray job (nohup, log: ${RECALL_AGENT_JOB_LOG})..."
TRAIN_CMD=(
  python3 train.py
  --actor-num-nodes 1
  --actor-num-gpus-per-node "${NUM_GPUS}"
  --colocate
  "${MODEL_ARGS[@]}"
  "${CKPT_ARGS[@]}"
  "${ROLLOUT_ARGS[@]}"
  "${EVAL_ARGS[@]}"
  "${OPTIMIZER_ARGS[@]}"
  "${GRPO_ARGS[@]}"
  "${WANDB_ARGS[@]}"
  "${TENSORBOARD_ARGS[@]}"
  "${PERF_ARGS[@]}"
  "${SGLANG_ARGS[@]}"
  "${MISC_ARGS[@]}"
  "${CUSTOM_ARGS[@]}"
)

printf -v TRAIN_CMD_STR '%q ' "${TRAIN_CMD[@]}"

nohup ray job submit --address="http://127.0.0.1:${RAY_DASHBOARD_PORT}" \
  --runtime-env-json="${RUNTIME_ENV_JSON}" \
  -- bash -lc "${TRAIN_CMD_STR}" \
  >> "${RECALL_AGENT_JOB_LOG}" 2>&1 &
echo "[info] ray job submit pid: $!  (output: ${RECALL_AGENT_JOB_LOG})"
if [[ "${USE_TENSORBOARD}" == "1" ]]; then
  echo "[info] tensorboard dir: ${TENSORBOARD_DIR}"
fi
if [[ "${USE_BFCL_MULTI_TURN_EVAL}" == "1" && "${BFCL_EVAL_BACKEND}" == "official_external" ]]; then
  echo "[info] post-train BFCL eval backend: ${BFCL_EVAL_BACKEND}"
  echo "[info] post-train BFCL official root: ${BFCL_OFFICIAL_ROOT}"
  echo "[info] in-training BFCL eval will reuse the slime training router endpoint chosen at runtime"
fi
