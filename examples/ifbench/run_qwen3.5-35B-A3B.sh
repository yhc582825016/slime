#!/bin/bash
# IFBench GRPO with Qwen3.5-35B-A3B (MoE). Parallel layout matches slime MoE examples (TP=2, EP=8 on 8 GPUs).

set -ex
pkill -9 sglang || true
sleep 3
ray stop --force
pkill -9 ray || true
pkill -9 python || true
sleep 3
pkill -9 ray || true
pkill -9 python || true
export PYTHONBUFFERED=1

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
if [[ "${NVLINK_COUNT}" -gt 0 ]]; then
  HAS_NVLINK=1
else
  HAS_NVLINK=0
fi
echo "HAS_NVLINK: ${HAS_NVLINK} (detected ${NVLINK_COUNT} NVLink references)"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." &>/dev/null && pwd)"
MS_SWIFT_ROOT="${MS_SWIFT_ROOT:-${REPO_ROOT}/../ms-swift}"

source "${REPO_ROOT}/scripts/models/qwen3.5-35B-A3B.sh"

NUM_GPUS="${NUM_GPUS:-$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l)}"
if [[ -z "${NUM_GPUS}" || "${NUM_GPUS}" -le 0 ]]; then
  NUM_GPUS=8
fi
CP_SIZE="${CP_SIZE:-2}"
IFBENCH_USE_DEEPEP="${IFBENCH_USE_DEEPEP:-0}"

IFBENCH_TRAIN_DATA="${IFBENCH_TRAIN_DATA:-/mnt/code/yehangcheng/all_data/rl_data_repo/IF/IFbench/if_multi_constraints_upto5_train_swift_12k.parquet}"
IFBENCH_EVAL_DATA="${IFBENCH_EVAL_DATA:-/mnt/code/yehangcheng/slime/examples/ifbench/IFBench_test/ifbench_test_slime.parquet}"
MODEL_BASE="${MODEL_BASE:-/opt/users/models/Qwen3.5-35B-A3B-417}"
MODEL_HF_PATH="${MODEL_HF_PATH:-${MODEL_BASE}/checkpoint-17152-fused}"
MODEL_REF_PATH="${MODEL_REF_PATH:-${MODEL_BASE}/checkpoint-17152-fused_torch_dist}"
SAVE_PATH="${SAVE_PATH:-/mnt/code/yehangcheng/checkpoint/Agent_model/qwen3.5-35B-A3B-ifbench}"

IFBENCH_JOB_LOG="${IFBENCH_JOB_LOG:-/mnt/code/yehangcheng/logs/ifbench_qwen3.5-35B-A3B_$(date +%Y%m%d_%H%M%S).log}"

USE_WANDB="${USE_WANDB:-1}"
WANDB_PROJECT="${WANDB_PROJECT:-slime}"
WANDB_GROUP="${WANDB_GROUP:-ifbench_qwen3.5_35b_a3b}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-qwen3.5-35B-A3B-ifbench}"
WANDB_KEY="${WANDB_KEY:-${WANDB_API_KEY:-}}"
WANDB_BASE_URL="${WANDB_BASE_URL:-https://api.bandw.top}"

USE_TENSORBOARD="${USE_TENSORBOARD:-0}"
TB_PROJECT_NAME="${TB_PROJECT_NAME:-slime}"
TB_EXPERIMENT_NAME="${TB_EXPERIMENT_NAME:-${WANDB_RUN_NAME}}"
TENSORBOARD_DIR="${TENSORBOARD_DIR:-$(dirname -- "${SAVE_PATH:-${REPO_ROOT}}")/tensorboard/${TB_PROJECT_NAME}/${TB_EXPERIMENT_NAME}}"

RAY_DASHBOARD_PORT="${RAY_DASHBOARD_PORT:-8265}"
# Seconds to poll /api/version after ray start (avoids 502 from dashboard not ready yet).
RAY_DASHBOARD_READY_TIMEOUT_SEC="${RAY_DASHBOARD_READY_TIMEOUT_SEC:-180}"
# Set to 1 to `ray stop --force` before start (cleans stale cluster on same port).
IFBENCH_RAY_STOP_BEFORE_START="${IFBENCH_RAY_STOP_BEFORE_START:-1}"

CKPT_ARGS=(
  --hf-checkpoint "${MODEL_HF_PATH}"
  --ref-load "${MODEL_REF_PATH}"
  --save "${SAVE_PATH}"
  --save-interval 100
  --no-save-optim
  --no-save-rng
)

ROLLOUT_ARGS=(
  --prompt-data "${IFBENCH_TRAIN_DATA}"
  --input-key messages
  --metadata-key extra_info
  --apply-chat-template
  --apply-chat-template-kwargs '{"enable_thinking": false}'
  --rollout-shuffle
  --num-rollout 3000
  --rollout-batch-size 32
  --n-samples-per-prompt 8
  --rollout-max-response-len "${ROLLOUT_MAX_RESPONSE_LEN:-12000}"
  --rollout-temperature 1.0
  --global-batch-size 256
  --balance-data
  --dynamic-sampling-filter-path slime.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std
)

EVAL_ARGS=()
if [[ -n "${IFBENCH_EVAL_DATA}" ]]; then
  EVAL_INTERVAL="${EVAL_INTERVAL:-10}"
  EVAL_NAME="${EVAL_NAME:-ifbench}"
  N_SAMPLES_PER_EVAL_PROMPT="${N_SAMPLES_PER_EVAL_PROMPT:-1}"
  EVAL_MAX_RESPONSE_LEN="${EVAL_MAX_RESPONSE_LEN:-${ROLLOUT_MAX_RESPONSE_LEN:-12000}}"
  EVAL_ARGS=(
    --eval-interval "${EVAL_INTERVAL}"
    --eval-prompt-data "${EVAL_NAME}" "${IFBENCH_EVAL_DATA}"
    --n-samples-per-eval-prompt "${N_SAMPLES_PER_EVAL_PROMPT}"
    --eval-max-response-len "${EVAL_MAX_RESPONSE_LEN}"
  )
fi

GRPO_ARGS=(
  --advantage-estimator grpo
  --calculate-per-token-loss
  # --use-kl-loss
  # --kl-loss-coef 0.001
  # --kl-loss-type low_var_kl
  --entropy-coef 0.0
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
  --use-precision-aware-optimizer
)

PERF_ARGS=(
  --tensor-model-parallel-size 2
  --sequence-parallel
  --pipeline-model-parallel-size 1
  --context-parallel-size "${CP_SIZE}"
  --expert-model-parallel-size 8
  --expert-tensor-parallel-size 1
  --recompute-granularity full
  --recompute-method uniform
  --recompute-num-layers 1
  --use-dynamic-batch-size
  --max-tokens-per-gpu "${MAX_TOKENS_PER_GPU:-8192}"
)

SGLANG_ARGS=(
  --rollout-num-gpus-per-engine "${NUM_GPUS}"
  --sglang-mem-fraction-static 0.6
  --sglang-ep-size 8
  --sglang-cuda-graph-bs 1 2 4 8 $(seq 16 8 128)
)

MISC_ARGS=(
  --attention-dropout 0.0
  --hidden-dropout 0.0
  --accumulate-allreduce-grads-in-fp32
  --attention-softmax-in-fp32
  --attention-backend flash
)

if [[ "${IFBENCH_USE_DEEPEP}" == "1" ]]; then
  MISC_ARGS+=(
    --moe-token-dispatcher-type flex
    --moe-enable-deepep
  )
fi

CUSTOM_ARGS=(
  --custom-rm-path examples.ifbench.reward_ms_swift.reward_func
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

mkdir -p "$(dirname -- "${IFBENCH_JOB_LOG}")"
[[ -n "${SAVE_PATH}" ]] && mkdir -p "${SAVE_PATH}"

export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
# Avoid corporate HTTP(S)_PROXY sending localhost dashboard traffic to a proxy (often -> 502).
export NO_PROXY="127.0.0.1,localhost,${MASTER_ADDR}${NO_PROXY:+,${NO_PROXY}}"
export no_proxy="127.0.0.1,localhost,${MASTER_ADDR}${no_proxy:+,${no_proxy}}"

if [[ "${IFBENCH_RAY_STOP_BEFORE_START}" == "1" ]]; then
  echo "[step] ray stop --force (IFBENCH_RAY_STOP_BEFORE_START=1)..."
  ray stop --force || true
  sleep 2
fi

echo "[step] ray start head (dashboard ${RAY_DASHBOARD_PORT})..."
ray start --head --node-ip-address "${MASTER_ADDR}" --num-gpus "${NUM_GPUS}" --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port="${RAY_DASHBOARD_PORT}"

RAY_JOB_HTTP="http://127.0.0.1:${RAY_DASHBOARD_PORT}"
echo "[step] waiting for Ray dashboard ${RAY_JOB_HTTP} (timeout ${RAY_DASHBOARD_READY_TIMEOUT_SEC}s)..."
_ready=0
for ((_i = 1; _i <= RAY_DASHBOARD_READY_TIMEOUT_SEC; _i++)); do
  if curl --noproxy '*' -sf "${RAY_JOB_HTTP}/api/version" >/dev/null 2>&1; then
    _ready=1
    echo "[info] dashboard ready after ${_i}s"
    break
  fi
  sleep 1
done
if [[ "${_ready}" -ne 1 ]]; then
  echo "[error] dashboard not healthy at ${RAY_JOB_HTTP}/api/version after ${RAY_DASHBOARD_READY_TIMEOUT_SEC}s."
  echo "[hint] ray status; lsof -i :${RAY_DASHBOARD_PORT}; unset HTTP_PROXY HTTPS_PROXY or keep NO_PROXY including 127.0.0.1"
  exit 1
fi

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"${REPO_ROOT}:${MS_SWIFT_ROOT}:/root/Megatron-LM/\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"MS_SWIFT_IFBENCH_DIR\": \"${MS_SWIFT_ROOT}/plugin/IFbench\",
    \"TENSORBOARD_DIR\": \"${TENSORBOARD_DIR}\",
    \"WANDB_API_KEY\": \"${WANDB_API_KEY:-}\",
    \"WANDB_BASE_URL\": \"${WANDB_BASE_URL}\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\",
    \"TORCH_MEMORY_SAVER_STRICT\": \"0\",
    \"NCCL_TIMEOUT\": \"1800\",
    \"TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC\": \"300\",
    \"NCCL_SOCKET_IFNAME\": \"bond0\"
  }
}"

echo "[step] submitting ray job (nohup, log: ${IFBENCH_JOB_LOG})..."
TRAIN_CMD=(
  python3 "${REPO_ROOT}/train.py"
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

nohup ray job submit --address="${RAY_JOB_HTTP}" \
  --runtime-env-json="${RUNTIME_ENV_JSON}" \
  -- bash -lc "${TRAIN_CMD_STR}" \
  >> "${IFBENCH_JOB_LOG}" 2>&1 &
echo "[info] ray job submit pid: $!  (output: ${IFBENCH_JOB_LOG})"
if [[ "${USE_TENSORBOARD}" == "1" ]]; then
  echo "[info] tensorboard dir: ${TENSORBOARD_DIR}"
fi
if [[ "${USE_WANDB}" == "1" ]]; then
  echo "[info] wandb project: ${WANDB_PROJECT}  group: ${WANDB_GROUP}  run: ${WANDB_RUN_NAME}"
fi
