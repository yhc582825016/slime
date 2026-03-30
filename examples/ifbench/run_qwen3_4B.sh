#!/bin/bash

set -ex

export PYTHONBUFFERED=1

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." &>/dev/null && pwd)"
MS_SWIFT_ROOT="${MS_SWIFT_ROOT:-${REPO_ROOT}/../ms-swift}"

source "${REPO_ROOT}/scripts/models/qwen3-4B.sh"

IFBENCH_TRAIN_DATA="${IFBENCH_TRAIN_DATA:-/dev/shm/ye/rl-data/IF/IFbench/if_multi_constraints_upto5_train_swift_12k.parquet}"
IFBENCH_EVAL_DATA="${IFBENCH_EVAL_DATA:-${REPO_ROOT}/examples/ifbench/IFBench_test/ifbench_test_slime.parquet}"
MODEL_HF_PATH="${MODEL_HF_PATH:-/dev/shm/Qwen3-4B}"
MODEL_REF_PATH="${MODEL_REF_PATH:-}"
SAVE_PATH="${SAVE_PATH:-}"

CKPT_ARGS=(
  --hf-checkpoint "${MODEL_HF_PATH}"
  --ref-load "${MODEL_REF_PATH}"
  --save "${SAVE_PATH}"
  --save-interval 20
)

ROLLOUT_ARGS=(
  --prompt-data "${IFBENCH_TRAIN_DATA}"
  --input-key messages
  --metadata-key extra_info
  --apply-chat-template
  --rollout-shuffle
  --num-rollout 3000
  --rollout-batch-size 32
  --n-samples-per-prompt 8
  --rollout-max-response-len 4096
  --rollout-temperature 1.0
  --global-batch-size 256
  --balance-data
)

EVAL_ARGS=()
if [[ -n "${IFBENCH_EVAL_DATA}" ]]; then
  EVAL_INTERVAL="${EVAL_INTERVAL:-20}"
  EVAL_NAME="${EVAL_NAME:-ifbench}"
  N_SAMPLES_PER_EVAL_PROMPT="${N_SAMPLES_PER_EVAL_PROMPT:-1}"
  EVAL_MAX_RESPONSE_LEN="${EVAL_MAX_RESPONSE_LEN:-4096}"
  EVAL_ARGS=(
    --eval-interval "${EVAL_INTERVAL}"
    --eval-prompt-data "${EVAL_NAME}" "${IFBENCH_EVAL_DATA}"
    --n-samples-per-eval-prompt "${N_SAMPLES_PER_EVAL_PROMPT}"
    --eval-max-response-len "${EVAL_MAX_RESPONSE_LEN}"
  )
fi

GRPO_ARGS=(
  --advantage-estimator grpo
  --use-kl-loss
  --kl-loss-coef 0.001
  --kl-loss-type low_var_kl
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
)

PERF_ARGS=(
  --tensor-model-parallel-size 2
  --sequence-parallel
  --pipeline-model-parallel-size 1
  --context-parallel-size 1
  --expert-model-parallel-size 1
  --expert-tensor-parallel-size 1
  --recompute-granularity full
  --recompute-method uniform
  --recompute-num-layers 1
  --use-dynamic-batch-size
  --max-tokens-per-gpu 9216
)

SGLANG_ARGS=(
  --rollout-num-gpus-per-engine 2
  --sglang-mem-fraction-static 0.7
)

MISC_ARGS=(
  --attention-dropout 0.0
  --hidden-dropout 0.0
  --accumulate-allreduce-grads-in-fp32
  --attention-softmax-in-fp32
  --attention-backend flash
)

CUSTOM_ARGS=(
  --custom-rm-path examples.ifbench.reward_ms_swift.reward_func
)

export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
ray start --head --node-ip-address "${MASTER_ADDR}" --num-gpus 8 --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"${REPO_ROOT}:${MS_SWIFT_ROOT}:/root/Megatron-LM/\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"MS_SWIFT_IFBENCH_DIR\": \"${MS_SWIFT_ROOT}/plugin/IFbench\"
  }
}"

ray job submit --address="http://127.0.0.1:8265" \
  --runtime-env-json="${RUNTIME_ENV_JSON}" \
  -- python3 "${REPO_ROOT}/train.py" \
  --actor-num-nodes 1 \
  --actor-num-gpus-per-node 8 \
  --colocate \
  ${MODEL_ARGS[@]} \
  ${CKPT_ARGS[@]} \
  ${ROLLOUT_ARGS[@]} \
  ${EVAL_ARGS[@]} \
  ${OPTIMIZER_ARGS[@]} \
  ${GRPO_ARGS[@]} \
  ${PERF_ARGS[@]} \
  ${SGLANG_ARGS[@]} \
  ${MISC_ARGS[@]} \
  ${CUSTOM_ARGS[@]}
