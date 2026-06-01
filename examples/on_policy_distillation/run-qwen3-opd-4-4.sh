#!/bin/bash

# usage: bash examples/on_policy_distillation/run-qwen3-opd-collocate.sh
# nohup: NOHUP=1 bash examples/on_policy_distillation/run-qwen3-opd-collocate.sh
#
# Disaggregated mode: GPU 0-3 Megatron train (TP=2), GPU 4-7 SGLang rollout; teacher is remote.

set -ex

export PYTHONUNBUFFERED=1
export CUDA_LAUNCH_BLOCKING="${CUDA_LAUNCH_BLOCKING:-0}"
# Single-node: keep NCCL/Gloo on loopback to avoid cluster NIC cross-talk (10.78.x noise)
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
WANDB_GROUP="${WANDB_GROUP:-qwen3.5-4B-opd}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-qwen3.5-4B-opd}"
WANDB_KEY="${WANDB_KEY:-${WANDB_API_KEY:-04b01529fb630482bdf2f363456479f197ac5694}}"
WANDB_BASE_URL="${WANDB_BASE_URL:-https://api.wandb.ai}"

# Nohup config
NOHUP="${NOHUP:-1}"
NOHUP_LOG="${NOHUP_LOG:-/mnt/code/yehangcheng/logs/qwen3.5-4B-opd-$(date +%Y%m%d_%H%M%S).log}"

if [[ "${NOHUP}" == "1" && "${_NOHUP_LAUNCHED:-0}" != "1" ]]; then
  mkdir -p /mnt/code/yehangcheng/logs
  echo "[step] relaunch with nohup, log: ${NOHUP_LOG}"
  nohup env _NOHUP_LAUNCHED=1 NOHUP=0 bash "$0" "$@" >"${NOHUP_LOG}" 2>&1 &
  echo "[step] started in background, pid=$!"
  exit 0
fi

ray stop --force || true
unset RAY_ADDRESS
# Remote teacher model server
TEACHER_IP="10.16.80.9"
TEACHER_PORT=13141

echo "Waiting for remote teacher model server at ${TEACHER_IP}:${TEACHER_PORT}..."

## Wait for the remote teacher model server to be ready
until curl -sf http://$TEACHER_IP:$TEACHER_PORT/health_generate > /dev/null; do
    echo "Waiting for the teacher model server to start..."
    sleep 5
done

curl http://$TEACHER_IP:$TEACHER_PORT/get_model_info
echo "Teacher model server is up and running at $TEACHER_IP:$TEACHER_PORT."
sleep 10


NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
if [ "$NVLINK_COUNT" -gt 0 ]; then
    HAS_NVLINK=1
else
    HAS_NVLINK=0
fi
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"

source "/mnt/code/yehangcheng/slime/scripts/models/qwen3.5-4B.sh"


CKPT_ARGS=(
   --hf-checkpoint /opt/users/ye/checkpoints/qwen3.5-4b-509-9061/checkpoint-9061
   --ref-load /opt/users/ye/checkpoints/qwen3.5-4b-509-9061_torch_dist
   --load /mnt/code/yehangcheng/checkpoint/General_model/qwen3.6-35b-distill-qwen3.5-4b-528
   --save /mnt/code/yehangcheng/checkpoint/General_model/qwen3.6-35b-distill-qwen3.5-4b-528
   --save-interval 20
)

ROLLOUT_ARGS=(
   --prompt-data /mnt/code/yehangcheng/all_data/rl_data/OPD/reason_gym_guru_math-gt8-2w-ifbench5k-nemotron-if-5k_sft_457w_two_sources_30k_opd_merged.jsonl
   --input-key messages
   --label-key solution
   --apply-chat-template
   --apply-chat-template-kwargs '{"enable_thinking":false}'
   --rollout-shuffle
   --num-rollout 1000
   --rollout-batch-size 16
   --n-samples-per-prompt 4
   --rollout-max-response-len 32000
   --rollout-temperature 1
   --global-batch-size 64
   --balance-data
)

RM_ARGS=(
   --custom-rm-path slime.rollout.on_policy_distillation.reward_func
   --custom-reward-post-process-path slime.rollout.on_policy_distillation.post_process_rewards
   --rm-url http://$TEACHER_IP:$TEACHER_PORT/generate
)
      # ic /mnt/code/yehangcheng/all_data/rl_data/OPD/ic_inner_test5_500_opd.jsonl
EVAL_ARGS=(
   --eval-interval 20
   --eval-prompt-data \
      aime /mnt/code/yehangcheng/all_data/rl_data/OPD/aime_2024_swift_4x_opd.jsonl \
      gpqa /mnt/code/yehangcheng/all_data/rl_data/OPD/gpqa_diamond_opd.jsonl \
      ifbench /mnt/code/yehangcheng/all_data/rl_data/OPD/IFBench_test_opd.jsonl \
      ic /mnt/code/yehangcheng/all_data/rl_data/OPD/ic_inner_test5_500_opd.jsonl \
      zebralogicbench /mnt/code/yehangcheng/all_data/rl_data/OPD/zebralogicbench_100_opd.jsonl \
   --n-samples-per-eval-prompt 1
   --eval-max-response-len 32000
   --eval-top-p 1
   --custom-eval-rollout-log-function-path slime.rollout.on_policy_distillation.eval_log_function
)

PERF_ARGS=(
   --tensor-model-parallel-size 4
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
   --max-tokens-per-gpu 16384
)

GRPO_ARGS=(
   --advantage-estimator grpo
   --use-opd
   --opd-type sglang
   --opd-kl-coef 1.0
   --use-kl-loss
   --kl-loss-coef 0.00
   --kl-loss-type low_var_kl
   --entropy-coef 0.00
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-6
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
   --rollout-num-gpus-per-engine 4
   --sglang-mem-fraction-static 0.8
)

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
)

# launch Ray on 8 GPUs: 0-3 train, 4-7 rollout (teacher is remote, no local GPU)
export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
export RAY_PORT=${RAY_PORT:-6379}
export RAY_DASHBOARD_PORT=${RAY_DASHBOARD_PORT:-8265}
ray start --head --node-ip-address ${MASTER_ADDR} --port "${RAY_PORT}" --num-gpus 8 \
  --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port="${RAY_DASHBOARD_PORT}"


SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SLIME_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${SLIME_ROOT}"

ray job submit --address="http://127.0.0.1:${RAY_DASHBOARD_PORT}" \
   --runtime-env-json="{
     \"env_vars\": {
        \"PYTHONPATH\": \"/root/Megatron-LM/\",
        \"CUDA_VISIBLE_DEVICES\": \"0,1,2,3,4,5,6,7\",
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
        \"WANDB_API_KEY\": \"${WANDB_KEY}\",
        \"WANDB_BASE_URL\": \"${WANDB_BASE_URL}\"
     }
   }" \
   -- python3 "${SLIME_ROOT}/train.py" \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node 4 \
   --rollout-num-gpus 4 \
   --num-gpus-per-node 8 \
   ${MODEL_ARGS[@]} \
   ${CKPT_ARGS[@]} \
   ${ROLLOUT_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${GRPO_ARGS[@]} \
   ${WANDB_ARGS[@]} \
   ${PERF_ARGS[@]} \
   ${EVAL_ARGS[@]} \
   ${SGLANG_ARGS[@]} \
   ${MISC_ARGS[@]} \
   ${RM_ARGS[@]}



####clear after training
sleep 3
ray stop --force
pkill -9 ray
pkill -9 python
sleep 3
pkill -9 ray
pkill -9 python
