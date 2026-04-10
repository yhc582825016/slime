#!/bin/bash

set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." &>/dev/null && pwd)"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
cd "${ROOT_DIR}"

export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export RECALL_AGENT_DEBUG_REWARD=1
HF_CKPT="${HF_CKPT:-/opt/users/models/Qwen3.5-4B}"
MEGATRON_PATH="${MEGATRON_PATH:-/root/Megatron-LM}"

RAW_TRAIN="${RAW_TRAIN:-/mnt/code/yehangcheng/slime/examples/recall_agent/train_filtered.parquet}"
RAW_TEST="${RAW_TEST:-/mnt/code/yehangcheng/slime/examples/recall_agent/test_filtered.parquet}"
PROMPT_DATA_DIR="${PROMPT_DATA_DIR:-${SCRIPT_DIR}/data}"
PROMPT_TRAIN="${PROMPT_TRAIN:-${PROMPT_DATA_DIR}/train.jsonl}"
PROMPT_TEST="${PROMPT_TEST:-${PROMPT_DATA_DIR}/test.jsonl}"
echo "PROMPT_TRAIN: ${PROMPT_TRAIN}"
echo "PROMPT_TEST: ${PROMPT_TEST}"
REF_LOAD="${REF_LOAD:-/mnt/code/yehangcheng/checkpoint/Qwen3.5-4B-Thinking_torch_dist}"
LOAD_PATH="${LOAD_PATH:-${REF_LOAD}}"
SAVE_PATH="${SAVE_PATH:-/mnt/code/yehangcheng/checkpoint/General_model/slime/Qwen3.5-4B-Thinking_recall_agent}"
RECALL_AGENT_JOB_LOG="${RECALL_AGENT_JOB_LOG:-/mnt/code/yehangcheng/logs/recall_agent_$(date +%Y%m%d_%H%M%S).log}"

# 训练使用 4 张 GPU；若机器多于 4 张且未手动设置 CUDA_VISIBLE_DEVICES，则自动选用物理编号最大的后 4 张卡。
NUM_GPUS="${NUM_GPUS:-8}"
LAST_N_GPUS="${LAST_N_GPUS:-${NUM_GPUS}}"
TP_SIZE="${TP_SIZE:-4}"
ROLLOUT_NUM_GPUS_PER_ENGINE="${ROLLOUT_NUM_GPUS_PER_ENGINE:-1}"
DISABLE_JIT_FUSER="${DISABLE_JIT_FUSER:-0}"
LOG_PROBS_CHUNK_SIZE="${LOG_PROBS_CHUNK_SIZE:-1024}"
TORCHINDUCTOR_FORCE_DISABLE_CACHES="${TORCHINDUCTOR_FORCE_DISABLE_CACHES:-0}"
CUDA_LAUNCH_BLOCKING="${CUDA_LAUNCH_BLOCKING:-0}"

# 显存友好默认（仍可通过环境变量覆盖）：更长上下文/更大 batch 易 OOM，可按卡容量调高。
MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU:-4096}"
ROLLOUT_MAX_RESPONSE_LEN="${ROLLOUT_MAX_RESPONSE_LEN:-24000}"
NUM_ROLLOUT="${NUM_ROLLOUT:-4000}"
ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-32}"
N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-8}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-256}"
SAVE_INTERVAL="${SAVE_INTERVAL:-200}"
OVER_SAMPLING_BATCH_SIZE="${OVER_SAMPLING_BATCH_SIZE:-32}"
SGLANG_MEM_FRACTION_STATIC="${SGLANG_MEM_FRACTION_STATIC:-0.60}"

USE_EVAL="${USE_EVAL:-1}"
EVAL_INTERVAL="${EVAL_INTERVAL:-50}"
EVAL_DATA_NAME="${EVAL_DATA_NAME:-recall_agent_eval}"
N_SAMPLES_PER_EVAL_PROMPT="${N_SAMPLES_PER_EVAL_PROMPT:-1}"
EVAL_MAX_RESPONSE_LEN="${EVAL_MAX_RESPONSE_LEN:-24000}"

USE_WANDB="${USE_WANDB:-1}"
WANDB_PROJECT="${WANDB_PROJECT:-slime}"
WANDB_GROUP="${WANDB_GROUP:-recall_agent_qwen3.5_4b}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-qwen3.5-4B-recall-agent-gspo}"
WANDB_KEY="${WANDB_KEY:-${WANDB_API_KEY:-04b01529fb630482bdf2f363456479f197ac5694}}"
WANDB_BASE_URL="${WANDB_BASE_URL:-https://api.bandw.top}"

MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
RAY_DASHBOARD_PORT="${RAY_DASHBOARD_PORT:-8265}"


if [[ ! -d "${HF_CKPT}" ]]; then
  echo "[error] HF_CKPT not found: ${HF_CKPT}"
  exit 1
fi
if [[ ! -f "${RAW_TRAIN}" ]]; then
  echo "[error] RAW_TRAIN not found: ${RAW_TRAIN}"
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
  --apply-chat-template
  --rollout-shuffle
  --num-rollout "${NUM_ROLLOUT}"
  --rollout-batch-size "${ROLLOUT_BATCH_SIZE}"
  --n-samples-per-prompt "${N_SAMPLES_PER_PROMPT}"
  --over-sampling-batch-size "${OVER_SAMPLING_BATCH_SIZE}"
  --dynamic-sampling-filter-path slime.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std
  --rollout-max-response-len "${ROLLOUT_MAX_RESPONSE_LEN}"
  --rollout-temperature 1.0
  --global-batch-size "${GLOBAL_BATCH_SIZE}"
  --balance-data
)

PERF_ARGS=(
  --tensor-model-parallel-size "${TP_SIZE}"
  --sequence-parallel
  --pipeline-model-parallel-size 1
  --context-parallel-size 1
  --expert-model-parallel-size 1
  --expert-tensor-parallel-size 1
  --recompute-granularity full
  --recompute-method uniform
  --recompute-num-layers 1
  --use-dynamic-batch-size
  --max-tokens-per-gpu "${MAX_TOKENS_PER_GPU}"
  --log-probs-chunk-size "${LOG_PROBS_CHUNK_SIZE}"
)

if [[ "${DISABLE_JIT_FUSER}" == "1" ]]; then
  PERF_ARGS+=(--disable-jit-fuser)
fi

# GSPO（https://arxiv.org/abs/2507.18071）：与 GRPO 不同，clip 量级更小且通常关闭 --use-kl-loss；参见 slime/tests/test_gspo.sh
GSPO_ARGS=(
  --advantage-estimator gspo
  --kl-loss-coef 0.00
  --kl-loss-type low_var_kl
  --kl-coef 0.00
  --entropy-coef 0.00
  --eps-clip 0.0003
  --eps-clip-high 0.0004
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
  --sglang-mem-fraction-static "${SGLANG_MEM_FRACTION_STATIC}"
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

OFFLOAD_ARGS+=(--offload)

echo "[step] restarting ray..."
ray stop --force || true
pkill -f sglang || true

ray start --head \
  --node-ip-address "${MASTER_ADDR}" \
  --num-gpus "${NUM_GPUS}" \
  --disable-usage-stats \
  --dashboard-host=0.0.0.0 \
  --dashboard-port "${RAY_DASHBOARD_PORT}"

RUNTIME_CUDA_JSON=""
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  RUNTIME_CUDA_JSON=",\"CUDA_VISIBLE_DEVICES\": \"${CUDA_VISIBLE_DEVICES}\""
fi
RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"${MEGATRON_PATH}:${ROOT_DIR}\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"TORCHINDUCTOR_FORCE_DISABLE_CACHES\": \"${TORCHINDUCTOR_FORCE_DISABLE_CACHES}\",
    \"CUDA_LAUNCH_BLOCKING\": \"${CUDA_LAUNCH_BLOCKING}\",
    \"WANDB_API_KEY\": \"${WANDB_API_KEY:-}\",
    \"WANDB_BASE_URL\": \"${WANDB_BASE_URL}\"${RUNTIME_CUDA_JSON}
  }
}"

echo "[step] submitting ray job (nohup, log: ${RECALL_AGENT_JOB_LOG})..."
nohup ray job submit --address="http://127.0.0.1:${RAY_DASHBOARD_PORT}" \
  --runtime-env-json="${RUNTIME_ENV_JSON}" \
  -- python3 train.py \
  --actor-num-nodes 1 \
  --actor-num-gpus-per-node "${NUM_GPUS}" \
  --colocate \
  "${OFFLOAD_ARGS[@]}" \
  "${MODEL_ARGS[@]}" \
  "${CKPT_ARGS[@]}" \
  "${ROLLOUT_ARGS[@]}" \
  "${EVAL_ARGS[@]}" \
  "${OPTIMIZER_ARGS[@]}" \
  "${GSPO_ARGS[@]}" \
  "${WANDB_ARGS[@]}" \
  "${PERF_ARGS[@]}" \
  "${SGLANG_ARGS[@]}" \
  "${MISC_ARGS[@]}" \
  "${CUSTOM_ARGS[@]}" \
  >> "${RECALL_AGENT_JOB_LOG}" 2>&1 &
echo "[info] ray job submit pid: $!  (output: ${RECALL_AGENT_JOB_LOG})"
