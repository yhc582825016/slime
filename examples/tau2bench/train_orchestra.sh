#!/bin/bash
# ToolOrchestra RL 训练脚本
# 基于 slime + GRPO，训练 Qwen3.5-4B 作为 Orchestrator
#
# 使用方式：
#   cd /dev/shm/ye/slime
#   bash examples/tau2bench/train_orchestra.sh
#
# 前置条件：
#   1. 检索服务已启动（retrieval_general_thought.py --port 8000）
#   2. Expert 模型服务已启动（可与训练共用同一批 GPU，训练时暂停 expert 服务）

# 若由 launch.sh 统一管理，则跳过此处的进程清理
PRESERVE_USER_AGENT="${PRESERVE_USER_AGENT:-1}"
if [ "${SKIP_PROCESS_KILL:-0}" != "1" ]; then
    ray stop --force
    pkill -9 ray 2>/dev/null || true
    if [ "${PRESERVE_USER_AGENT}" != "1" ]; then
        pkill -9 sglang 2>/dev/null || true
        pkill -9 python 2>/dev/null || true
    else
        echo "Preserving external user-agent processes. Set PRESERVE_USER_AGENT=0 to kill all python/sglang processes."
    fi
    sleep 3
fi

# 清理上一轮 tau2 残留的临时文件
rm -rf /tmp/tau2_orch_* /tmp/tau2_transfer_* /tmp/tau2_output* 2>/dev/null
echo "Cleaned up tau2 temp files."

# 清理上一轮 rollout 日志（重新开始记录）
rm -rf /data/rollout_logs/train /data/rollout_logs/eval 2>/dev/null
echo "Cleaned up old rollout logs."

ulimit -n 65536
set -ex

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
SLIME_DIR="$(cd -- "${SCRIPT_DIR}/../.." &>/dev/null && pwd)"

# NVLink 检测
NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
HAS_NVLINK=$([ "$NVLINK_COUNT" -gt 0 ] && echo 1 || echo 0)
echo "HAS_NVLINK: $HAS_NVLINK"

# ── 模型架构（Qwen3.5-4B）───────────────────────────────────────────────────
source "${SLIME_DIR}/scripts/models/qwen3.5-4B.sh"
# CP=6 时 Megatron 要求 seq_length % (2*context_parallel_size) == 0，即 12 的倍数；8192 不满足。
MODEL_ARGS+=(--seq-length 8184)

# ── 路径配置 ─────────────────────────────────────────────────────────────────
HF_CKPT="${HF_CKPT:-/dev/shm/Qwen3.5-4B}"                       # HuggingFace 格式原始权重
REF_CKPT="${REF_CKPT:-/dev/shm/Qwen3.5-4B-Thinking_torch_dist}"  # Megatron 分布式格式（ref model）
SAVE_DIR="${SAVE_DIR:-/data/checkpoints/orchestra_qwen3.5_4b_rl}"
LOG_DIR="${LOG_DIR:-/dev/shm/ye/logs}"
TRAIN_LOG_DATE="${TRAIN_LOG_DATE:-$(date +%Y%m%d)}"
TRAIN_JOB_LOG="${TRAIN_JOB_LOG:-${LOG_DIR}/orchestra_train_${TRAIN_LOG_DATE}.log}"
mkdir -p "${LOG_DIR}"

USE_WANDB="${USE_WANDB:-1}"
WANDB_PROJECT="${WANDB_PROJECT:-ToolOrchestra}"
WANDB_GROUP="${WANDB_GROUP:-orchestra_qwen3.5_4b_rl}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-qwen3.5-4B-orchestra}"
WANDB_KEY="${WANDB_KEY:-${WANDB_API_KEY:-}}"
WANDB_BASE_URL="${WANDB_BASE_URL:-https://api.bandw.top}"

# DATA_CATEGORY 控制使用哪类数据：
#   all       — 全量（qa + func_call）[默认]
#   func_call — 仅 func_call（本地无专家服务时推荐）
#   qa        — 仅 qa（需要专家服务 + 检索服务）
DATA_CATEGORY="${DATA_CATEGORY:-func_call}"

case "${DATA_CATEGORY}" in
    func_call)
        DATA_SUFFIX="func_call"
        ;;
    qa)
        DATA_SUFFIX="qa"
        ;;
    *)
        DATA_SUFFIX="full"
        ;;
esac

SHARED_DATA_PATH="/dev/shm/ye/rl-data/data_slime_${DATA_SUFFIX}.jsonl"
LOCAL_DATA_PATH="${SCRIPT_DIR}/data/data_slime_${DATA_SUFFIX}.jsonl"
LEGACY_DATA_PATH="/dev/shm/ye/slime-agentic/agentic/ToolOrchestra/data/data_slime_${DATA_SUFFIX}.jsonl"
if [ -z "${DATA_PATH:-}" ]; then
    if [ -f "${SHARED_DATA_PATH}" ]; then
        DATA_PATH="${SHARED_DATA_PATH}"
    elif [ -f "${LOCAL_DATA_PATH}" ]; then
        DATA_PATH="${LOCAL_DATA_PATH}"
    else
        DATA_PATH="${LEGACY_DATA_PATH}"
    fi
fi
echo "DATA_CATEGORY=${DATA_CATEGORY}, DATA_PATH=${DATA_PATH}"

# ── Checkpoint ───────────────────────────────────────────────────────────────
CKPT_ARGS=(
    --hf-checkpoint "${HF_CKPT}"
    --ref-load       "${REF_CKPT}"
    --save           "${SAVE_DIR}"
    --save-interval  50
)

# ── Rollout（数据 + 采样）────────────────────────────────────────────────────
ROLLOUT_ARGS=(
    --prompt-data    "${DATA_PATH}"
    --input-key      problem
    --label-key      answer
    --metadata-key   metadata
    --tool-key       tools
    --rollout-shuffle
    --reward-key     score

    --num-epoch              2
    --rollout-batch-size     32
    --n-samples-per-prompt   8
    --rollout-max-response-len 4096
    --rollout-max-context-len 10384
    --rollout-temperature    0.7
    --global-batch-size      126
    --balance-data
)

# ── 评估 ─────────────────────────────────────────────────────────────────────
EVAL_DATA_PATH="${EVAL_DATA_PATH:-${SCRIPT_DIR}/data/eval_tau2.jsonl}"
EVAL_ARGS=(
    --eval-interval             "${EVAL_INTERVAL:-10}"
    --eval-prompt-data          orchestra_tau2 "${EVAL_DATA_PATH}"
    --eval-input-key            problem
    --eval-label-key            answer
    --eval-tool-key             tools
    --n-samples-per-eval-prompt 1
    --eval-max-response-len     8184
    --eval-top-p                0.95
)

# ── 并行 & 性能 ───────────────────────────────────────────────────────────────
# 4B 模型，6× GPU（GPU 2-7），TP=2（DP=3），降低单卡显存
PERF_ARGS=(
    --tensor-model-parallel-size   1
    --pipeline-model-parallel-size 1
    --context-parallel-size        6
    --expert-model-parallel-size   1
    --expert-tensor-parallel-size  1

    --use-dynamic-batch-size
    --max-tokens-per-gpu         8184
    --log-probs-max-tokens-per-gpu 16368

    --train-memory-margin-bytes  8589934592
    --recompute-granularity      full
    --recompute-method           uniform
    --recompute-num-layers       1
    --recompute-loss-function
    --log-probs-chunk-size       256
)

# ── GRPO 算法 ─────────────────────────────────────────────────────────────────
GRPO_ARGS=(
    --advantage-estimator grpo
    --use-kl-loss
    --kl-loss-coef        0.001
    --kl-loss-type        low_var_kl
    --entropy-coef        0.0
    --eps-clip            0.2
    --eps-clip-high       0.3
)

# ── 优化器 ───────────────────────────────────────────────────────────────────
OPTIMIZER_ARGS=(
    --optimizer      adam
    --lr             1e-6
    --lr-decay-style constant
    --weight-decay   0.1
    --adam-beta1     0.9
    --adam-beta2     0.98
)

# ── SGLang 推理引擎 ───────────────────────────────────────────────────────────
# 与 Megatron TP=2 对齐：rollout 引擎跨 2 卡
SGLANG_ARGS=(
    --rollout-num-gpus-per-engine 2
    --sglang-mem-fraction-static  0.50
    --sglang-context-length       131072
    --sglang-server-concurrency   512
)

# ── 其他 ─────────────────────────────────────────────────────────────────────
MISC_ARGS=(
    --attention-dropout 0.0
    --hidden-dropout    0.0
    --accumulate-allreduce-grads-in-fp32
    --attention-softmax-in-fp32
    --attention-backend flash
)

# ── 自定义函数路径 ─────────────────────────────────────────────────────────────
CUSTOM_ARGS=(
    --custom-generate-function-path              rollout.generate
    --custom-rm-path                             rollout.reward_func
    --custom-convert-samples-to-train-data-path  custom_convert.custom_convert
)

# ── WandB（按需开启）─────────────────────────────────────────────────────────
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

# ── 启动 Ray ──────────────────────────────────────────────────────────────────
# 禁用 OpenTelemetry gRPC metrics 导出，避免 Ray InfoActor SIGSEGV（gRPC getenv 竞态）
export OTEL_SDK_DISABLED=true
export RAY_DISABLE_EXPORT_METRICS=1
export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
ray start --head \
    --node-ip-address "${MASTER_ADDR}" \
    --num-gpus 6 \
    --disable-usage-stats \
    --dashboard-host=0.0.0.0 \
    --dashboard-port=8265

# PYTHONPATH:
#   - tau2bench 目录：rollout.py / reward.py / custom_convert.py / orchestra_solver.py
#   - tau2bench/agentflow：SGLangEngine / GenerationOutput
#   - slime 根目录
#   - Megatron-LM
# 是否启用专家模型（本地无专家服务时设为 0，主模型将直接调用领域工具）
USE_EXPERT="${USE_EXPERT:-1}"
# 单条 tau2 任务内 Orchestrator 与环境的最大交互轮次
ORCHESTRA_MAX_TURNS="${ORCHESTRA_MAX_TURNS:-8}"

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"${SCRIPT_DIR}:/root/Megatron-LM:${SLIME_DIR}\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"PYTORCH_ALLOC_CONF\": \"max_split_size_mb:128\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\",
    \"CUDA_VISIBLE_DEVICES\": \"${CUDA_VISIBLE_DEVICES:-2,3,4,5,6,7}\",
    \"SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN\": \"1\",
    \"OTEL_SDK_DISABLED\": \"true\",
    \"RAY_DISABLE_EXPORT_METRICS\": \"1\",
    \"WANDB_API_KEY\": \"${WANDB_API_KEY:-}\",
    \"WANDB_BASE_URL\": \"${WANDB_BASE_URL}\",
    \"DASHSCOPE_API_KEY\": \"${DASHSCOPE_API_KEY:-dummy}\",
    \"DASHSCOPE_BASE_URL\": \"${DASHSCOPE_BASE_URL:-http://127.0.0.1:6032/v1}\",
    \"DASHSCOPE_MODEL\": \"${DASHSCOPE_MODEL:-Qwen3.5-4B}\",
    \"DASHSCOPE_ENABLE_THINKING\": \"${DASHSCOPE_ENABLE_THINKING:-0}\",
    \"ORCHESTRA_MAX_TURNS\": \"${ORCHESTRA_MAX_TURNS}\",
    \"USE_EXPERT\": \"${USE_EXPERT}\"
  }
}"
# ── 提交训练任务 ──────────────────────────────────────────────────────────────
TRAIN_CMD=(
    ray job submit --address="http://127.0.0.1:8265"
    --runtime-env-json="${RUNTIME_ENV_JSON}"
    -- python3 "${SLIME_DIR}/train.py"
    --actor-num-nodes         1
    --actor-num-gpus-per-node 6
    --colocate
    "${MODEL_ARGS[@]}"
    "${CKPT_ARGS[@]}"
    "${ROLLOUT_ARGS[@]}"
    "${OPTIMIZER_ARGS[@]}"
    "${GRPO_ARGS[@]}"
    "${EVAL_ARGS[@]}"
    "${WANDB_ARGS[@]}"
    "${PERF_ARGS[@]}"
    "${SGLANG_ARGS[@]}"
    "${MISC_ARGS[@]}"
    "${CUSTOM_ARGS[@]}"
)

echo "Submitting training job with nohup log: ${TRAIN_JOB_LOG}"
nohup "${TRAIN_CMD[@]}" > "${TRAIN_JOB_LOG}" 2>&1 &
TRAIN_PID=$!
echo "Training job submitted in background, pid=${TRAIN_PID}"
