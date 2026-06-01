#!/bin/bash

# usage: bash examples/on_policy_distillation/run-qwen3-opd-rlvr.sh
# nohup: NOHUP=1 bash examples/on_policy_distillation/run-qwen3-opd-rlvr.sh
#
# On-Policy Distillation + RLVR with mix_rl rule-based rewards.
# Disaggregated mode: GPU 0-3 Megatron train (TP=4), GPU 4-7 SGLang rollout.
#
# OPD teacher modes (set OPD_TYPE):
#   sglang   - remote teacher server (--rm-url), recommended for 4+4 layout
#   megatron - teacher loaded in Megatron training (--opd-teacher-load)
#
# Training data: mix_train.jsonl (prompt / reward_model / metadata)
# RLVR scoring: examples.mix_rl.reward_score via reward.py adapters
#
# Usage (override via env vars):
#   HF_CKPT=...  REF_LOAD=...  TEACHER_LOAD=...  PROMPT_DATA=...  ./run-qwen3-opd-rlvr.sh
#
# Data format expected by the multi-domain reward:
#   { "prompt": [...], "reward_model": {"ground_truth": ..., "style": "rule"},
#     "metadata": {"data_source": "math__...", "extra_info": {...}} }
#
# Supported data_source prefixes: math, codegen, stem__gpqa, stem__supergpqa,
#   stem_web, ood__ifeval, ood__livebench, ood__ifbench, reasoning_gym,
#   simulation__codeio, simulation__cruxeval, logic__arcagi, logic__zebra_puzzle,
#   logic__ordering_puzzle, logic__graph, table, openai/gsm8k, aime, and more.
#   See examples/mix_rl/reward_score/__init__.py for the full list.

set -ex

export PYTHONUNBUFFERED=1
export CUDA_LAUNCH_BLOCKING="${CUDA_LAUNCH_BLOCKING:-0}"
# Single-node: keep NCCL/Gloo on loopback to avoid cluster NIC cross-talk
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-lo}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-lo}"
export TORCH_COMPILE_DISABLE="${TORCH_COMPILE_DISABLE:-1}"
# OPD reward HTTP protection knobs (used in slime.rollout.on_policy_distillation.reward_func)
export OPD_RM_CONCURRENCY="${OPD_RM_CONCURRENCY:-16}"
export OPD_RM_MAX_CONNECTIONS="${OPD_RM_MAX_CONNECTIONS:-32}"
export OPD_RM_MAX_RETRIES="${OPD_RM_MAX_RETRIES:-8}"
# slime rollout / SGLang HTTP: use loopback so workers connect reliably on single-node
export SLIME_HOST_IP="${SLIME_HOST_IP:-127.0.0.1}"

# Wandb config (set USE_WANDB=0 to disable)
USE_WANDB="${USE_WANDB:-1}"
WANDB_PROJECT="${WANDB_PROJECT:-slime}"
WANDB_GROUP="${WANDB_GROUP:-qwen3.5-4B-opd-rlvr}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-qwen3.5-4B-opd-rlvr}"
WANDB_KEY="${WANDB_KEY:-${WANDB_API_KEY:-04b01529fb630482bdf2f363456479f197ac5694}}"
WANDB_BASE_URL="${WANDB_BASE_URL:-https://api.wandb.ai}"

# Nohup config
NOHUP="${NOHUP:-1}"
NOHUP_LOG="${NOHUP_LOG:-/mnt/code/yehangcheng/logs/qwen3.5-4B-opd-rlvr-$(date +%Y%m%d_%H%M%S).log}"

if [[ "${NOHUP}" == "1" && "${_NOHUP_LAUNCHED:-0}" != "1" ]]; then
  mkdir -p /mnt/code/yehangcheng/logs
  echo "[step] relaunch with nohup, log: ${NOHUP_LOG}"
  nohup env _NOHUP_LAUNCHED=1 NOHUP=0 bash "$0" "$@" >"${NOHUP_LOG}" 2>&1 &
  echo "[step] started in background, pid=$!"
  exit 0
fi

# ---------------------------------------------------------------------------
# Configurable parameters (override via environment variables)
# ---------------------------------------------------------------------------
HF_CKPT="${HF_CKPT:-/opt/users/ye/checkpoints/qwen3.5-4b-509-9061/checkpoint-9061}"
REF_LOAD="${REF_LOAD:-/opt/users/ye/checkpoints/qwen3.5-4b-509-9061_torch_dist}"
LOAD_PATH="${LOAD_PATH:-/mnt/code/yehangcheng/checkpoint/General_model/qwen3.6-35b-distill-qwen3.5-4b-529-rlvr}"
SAVE_PATH="${SAVE_PATH:-/mnt/code/yehangcheng/checkpoint/General_model/qwen3.6-35b-distill-qwen3.5-4b-529-rlvr}"
MEGATRON_PATH="${MEGATRON_PATH:-/root/Megatron-LM}"

# Multi-domain prompt data (mix_rl format)
PROMPT_DATA="${PROMPT_DATA:-/mnt/code/yehangcheng/all_data/rl_data_repo/mix-rl-slime/mix_train.jsonl}"

# OPD teacher (sglang remote server)
OPD_TYPE="${OPD_TYPE:-sglang}"
TEACHER_IP="${TEACHER_IP:-10.16.80.9}"
TEACHER_PORT="${TEACHER_PORT:-13141}"
TEACHER_LOAD="${TEACHER_LOAD:-/opt/users/models/Qwen3.6-35B-A3B_torch_dist}"

NON_THINKING="${NON_THINKING:-1}"

# Codegen rewards default to unsafe_local. Set to bwrap if bubblewrap is available.
export CODER1_EXEC="${CODER1_EXEC:-unsafe_local}"

# Rollout / training hyper-parameters
NUM_ROLLOUT="${NUM_ROLLOUT:-1000}"
ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-16}"
N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-4}"
ROLLOUT_MAX_RESPONSE_LEN="${ROLLOUT_MAX_RESPONSE_LEN:-32000}"
ROLLOUT_TEMPERATURE="${ROLLOUT_TEMPERATURE:-1.0}"
ROLLOUT_TOP_P="${ROLLOUT_TOP_P:-1.0}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-64}"
SAVE_INTERVAL="${SAVE_INTERVAL:-20}"
LR="${LR:-1e-6}"
MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU:-16384}"

# OPD hyper-parameters
OPD_KL_COEF="${OPD_KL_COEF:-1.0}"

# Parallelism (4 train + 4 rollout on 8 GPUs)
ACTOR_NUM_GPUS_PER_NODE="${ACTOR_NUM_GPUS_PER_NODE:-4}"
ROLLOUT_NUM_GPUS="${ROLLOUT_NUM_GPUS:-4}"
NUM_GPUS_PER_NODE="${NUM_GPUS_PER_NODE:-8}"
TP_SIZE="${TP_SIZE:-4}"
ROLLOUT_NUM_GPUS_PER_ENGINE="${ROLLOUT_NUM_GPUS_PER_ENGINE:-4}"
SGLANG_MEM_FRACTION_STATIC="${SGLANG_MEM_FRACTION_STATIC:-0.8}"

# Ray
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
RAY_PORT="${RAY_PORT:-6379}"
RAY_DASHBOARD_PORT="${RAY_DASHBOARD_PORT:-8265}"

# ---------------------------------------------------------------------------

ray stop --force || true
unset RAY_ADDRESS

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SLIME_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${SLIME_ROOT}"
echo "Using SLIME_ROOT=${SLIME_ROOT} as cwd for ray job submit"

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
if [ "$NVLINK_COUNT" -gt 0 ]; then
    HAS_NVLINK=1
else
    HAS_NVLINK=0
fi
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"

NUM_GPUS="${NUM_GPUS:-${NUM_GPUS_PER_NODE}}"
if [[ "${NUM_GPUS}" -ne $((ACTOR_NUM_GPUS_PER_NODE + ROLLOUT_NUM_GPUS)) ]]; then
    echo "[warn] NUM_GPUS=${NUM_GPUS} != ACTOR(${ACTOR_NUM_GPUS_PER_NODE}) + ROLLOUT(${ROLLOUT_NUM_GPUS})"
fi

source "${SLIME_ROOT}/scripts/models/qwen3.5-4B.sh"

echo "[config] HF_CKPT=${HF_CKPT}"
echo "[config] REF_LOAD=${REF_LOAD}"
echo "[config] LOAD_PATH=${LOAD_PATH}"
echo "[config] SAVE_PATH=${SAVE_PATH}"
echo "[config] PROMPT_DATA=${PROMPT_DATA}"
echo "[config] OPD_TYPE=${OPD_TYPE}, OPD_KL_COEF=${OPD_KL_COEF}"
if [[ "${OPD_TYPE}" == "sglang" ]]; then
    echo "[config] TEACHER_URL=http://${TEACHER_IP}:${TEACHER_PORT}/generate"
else
    echo "[config] TEACHER_LOAD=${TEACHER_LOAD}"
fi
echo "[config] NON_THINKING=${NON_THINKING}"
echo "[config] CODER1_EXEC=${CODER1_EXEC}"
echo "[config] NUM_GPUS=${NUM_GPUS} (train=${ACTOR_NUM_GPUS_PER_NODE}, rollout=${ROLLOUT_NUM_GPUS}), TP_SIZE=${TP_SIZE}"

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
    --global-batch-size "${GLOBAL_BATCH_SIZE}"
    --balance-data
)

# OPD+RLVR reward wiring:
# - sglang: teacher server returns log-probs; post_process adds mix_rl RLVR scores
# - megatron: mix_rl reward_func at rollout; teacher forward at train time
if [[ "${OPD_TYPE}" == "sglang" ]]; then
    CUSTOM_RM_ARGS=(
        --custom-rm-path slime.rollout.on_policy_distillation.reward_func
        --custom-reward-post-process-path examples.on_policy_distillation.reward.post_process_opd_rlvr_rewards
        --rm-url "http://${TEACHER_IP}:${TEACHER_PORT}/generate"
    )
else
    CUSTOM_RM_ARGS=(
        --custom-rm-path examples.mix_rl.reward.reward_func
        --reward-key score
    )
fi

EVAL_ARGS=(
    --eval-interval 20
    --eval-prompt-data \
       aime /mnt/code/yehangcheng/all_data/rl_data/OPD/aime_2024_swift_4x_opd.jsonl  \
       gpqa /mnt/code/yehangcheng/all_data/rl_data/OPD/gpqa_diamond_opd.jsonl \
       ifbench /mnt/code/yehangcheng/all_data/rl_data/OPD/IFBench_test_opd.jsonl \
       ic /mnt/code/yehangcheng/all_data/rl_data/OPD/ic_inner_test5_500_opd.jsonl \
       zebralogicbench /mnt/code/yehangcheng/all_data/rl_data/OPD/zebralogicbench_100_opd.jsonl \
    --eval-input-key messages
    --eval-label-key solution
    --n-samples-per-eval-prompt 1
    --eval-max-response-len 32000
    --eval-top-p 1
    --custom-eval-rollout-log-function-path slime.rollout.on_policy_distillation.eval_log_function
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

    --micro-batch-size 2
    --use-dynamic-batch-size
    --max-tokens-per-gpu "${MAX_TOKENS_PER_GPU}"
)

GRPO_ARGS=(
    --advantage-estimator grpo
    --use-opd
    --opd-type "${OPD_TYPE}"
    --opd-kl-coef "${OPD_KL_COEF}"
    --use-kl-loss
    --kl-loss-coef 0.00
    --kl-loss-type low_var_kl
    --entropy-coef 0.00
)

if [[ "${OPD_TYPE}" == "megatron" ]]; then
    GRPO_ARGS+=(--opd-teacher-load "${TEACHER_LOAD}")
fi

OPTIMIZER_ARGS=(
    --optimizer adam
    --lr "${LR}"
    --lr-decay-style constant
    --weight-decay 0.1
    --adam-beta1 0.9
    --adam-beta2 0.98
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

# ---------------------------------------------------------------------------
# Launch Ray
# ---------------------------------------------------------------------------
if [[ "${OPD_TYPE}" == "sglang" && "${SKIP_TEACHER_WAIT:-0}" != "1" ]]; then
    echo "Waiting for remote teacher model server at ${TEACHER_IP}:${TEACHER_PORT}..."
    until curl -sf "http://${TEACHER_IP}:${TEACHER_PORT}/health_generate" > /dev/null; do
        echo "Waiting for the teacher model server to start..."
        sleep 5
    done
    curl "http://${TEACHER_IP}:${TEACHER_PORT}/get_model_info"
    echo "Teacher model server is up and running at ${TEACHER_IP}:${TEACHER_PORT}."
    sleep 10
fi

if [ "${RAY_CLEAN_START:-0}" = "1" ]; then
    ray stop --force || true
    sleep 2
fi
ray start --head --node-ip-address "${MASTER_ADDR}" --port "${RAY_PORT}" --num-gpus "${NUM_GPUS}" \
    --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port="${RAY_DASHBOARD_PORT}"

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"${MEGATRON_PATH}:${SLIME_ROOT}\",
    \"CUDA_VISIBLE_DEVICES\": \"${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"CUDA_LAUNCH_BLOCKING\": \"${CUDA_LAUNCH_BLOCKING}\",
    \"SGLANG_DISABLE_CUDNN_CHECK\": \"1\",
    \"SLIME_HOST_IP\": \"${SLIME_HOST_IP}\",
    \"NCCL_SOCKET_IFNAME\": \"${NCCL_SOCKET_IFNAME}\",
    \"GLOO_SOCKET_IFNAME\": \"${GLOO_SOCKET_IFNAME}\",
    \"TORCH_COMPILE_DISABLE\": \"${TORCH_COMPILE_DISABLE}\",
    \"OPD_RM_CONCURRENCY\": \"${OPD_RM_CONCURRENCY}\",
    \"OPD_RM_MAX_CONNECTIONS\": \"${OPD_RM_MAX_CONNECTIONS}\",
    \"OPD_RM_MAX_RETRIES\": \"${OPD_RM_MAX_RETRIES}\",
    \"CODER1_EXEC\": \"${CODER1_EXEC}\",
    \"WANDB_API_KEY\": \"${WANDB_KEY}\",
    \"WANDB_BASE_URL\": \"${WANDB_BASE_URL}\"
  }
}"

ray job submit --address="http://127.0.0.1:${RAY_DASHBOARD_PORT}" \
    --runtime-env-json="${RUNTIME_ENV_JSON}" \
    -- python3 "${SLIME_ROOT}/train.py" \
    --actor-num-nodes 1 \
    --actor-num-gpus-per-node "${ACTOR_NUM_GPUS_PER_NODE}" \
    --rollout-num-gpus "${ROLLOUT_NUM_GPUS}" \
    --num-gpus-per-node "${NUM_GPUS_PER_NODE}" \
    "${MODEL_ARGS[@]}" \
    "${CKPT_ARGS[@]}" \
    "${CHAT_TEMPLATE_ARGS[@]}" \
    "${ROLLOUT_ARGS[@]}" \
    "${OPTIMIZER_ARGS[@]}" \
    "${GRPO_ARGS[@]}" \
    "${WANDB_ARGS[@]}" \
    "${PERF_ARGS[@]}" \
    "${EVAL_ARGS[@]}" \
    "${SGLANG_ARGS[@]}" \
    "${MISC_ARGS[@]}" \
    "${CUSTOM_RM_ARGS[@]}"

####clear after training
sleep 3
ray stop --force
pkill -9 ray
pkill -9 python
sleep 3
pkill -9 ray
pkill -9 python
