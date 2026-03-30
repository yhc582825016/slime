#!/bin/bash

set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." &>/dev/null && pwd)"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
cd "${ROOT_DIR}"

export PYTHONUNBUFFERED=1

# -------------------------------
# User-configurable variables
# -------------------------------
HF_CKPT="${HF_CKPT:-/dev/shm/Qwen3.5-4B}"
MEGATRON_PATH="${MEGATRON_PATH:-/root/Megatron-LM}"

RAW_DATA="${RAW_DATA:-/dev/shm/gym_data/swe_agent/Nemotron-RL-Agentic-SWE-Pivot-v1/train.jsonl}"
PROMPT_DATA="${PROMPT_DATA:-/dev/shm/gym_data/swe_agent/Nemotron-RL-Agentic-SWE-Pivot-v1/train_slime.jsonl}"

REF_LOAD="${REF_LOAD:-/dev/shm/Qwen3.5-4B_torch_dist}"
LOAD_PATH="${LOAD_PATH:-${REF_LOAD}}"
SAVE_PATH="${SAVE_PATH:-/dev/shm/Qwen3.5-4B_slime_swe_agentic}"

NUM_GPUS="${NUM_GPUS:-$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)}"
if [[ -z "${NUM_GPUS}" || "${NUM_GPUS}" -le 0 ]]; then
  NUM_GPUS=1
fi
TP_SIZE="${TP_SIZE:-4}"
ROLLOUT_NUM_GPUS_PER_ENGINE="${ROLLOUT_NUM_GPUS_PER_ENGINE:-1}"

MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU:-6144}"
ROLLOUT_MAX_RESPONSE_LEN="${ROLLOUT_MAX_RESPONSE_LEN:-4096}"
NUM_ROLLOUT="${NUM_ROLLOUT:-10000}"
ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-8}"
N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-8}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-32}"
SAVE_INTERVAL="${SAVE_INTERVAL:-200}"

MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
RAY_DASHBOARD_PORT="${RAY_DASHBOARD_PORT:-8265}"

# -------------------------------
# Optional background mode
# -------------------------------
NOHUP="${NOHUP:-1}"
NOHUP_LOG="${NOHUP_LOG:-${SAVE_PATH}/run-nohup.log}"

# -------------------------------
# Optional eval config
# -------------------------------
USE_EVAL="${USE_EVAL:-1}"
EVAL_INTERVAL="${EVAL_INTERVAL:-50}"
EVAL_DATA_NAME="${EVAL_DATA_NAME:-swe_agentic_eval}"
EVAL_PROMPT_DATA="${EVAL_PROMPT_DATA:-${PROMPT_DATA}@[0:200]}"
N_SAMPLES_PER_EVAL_PROMPT="${N_SAMPLES_PER_EVAL_PROMPT:-1}"
EVAL_MAX_RESPONSE_LEN="${EVAL_MAX_RESPONSE_LEN:-1024}"

# -------------------------------
# Optional logging config (W&B)
# -------------------------------
USE_WANDB="${USE_WANDB:-1}"
WANDB_PROJECT="${WANDB_PROJECT:-slime}"
WANDB_GROUP="${WANDB_GROUP:-qwen3.5-4B}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-qwen3.5-4B-swe-agentic}"
WANDB_KEY="${WANDB_KEY:-${WANDB_API_KEY:-}}"
WANDB_BASE_URL="${WANDB_BASE_URL:-https://api.bandw.top}"

export WANDB_API_KEY="${WANDB_KEY:-04b01529fb630482bdf2f363456479f197ac5694}"
export WANDB_BASE_URL

echo "[config] ROOT_DIR=${ROOT_DIR}"
echo "[config] HF_CKPT=${HF_CKPT}"
echo "[config] RAW_DATA=${RAW_DATA}"
echo "[config] PROMPT_DATA=${PROMPT_DATA}"
echo "[config] REF_LOAD=${REF_LOAD}"
echo "[config] SAVE_PATH=${SAVE_PATH}"
echo "[config] NUM_GPUS=${NUM_GPUS}, TP_SIZE=${TP_SIZE}"
echo "[config] USE_EVAL=${USE_EVAL}, EVAL_INTERVAL=${EVAL_INTERVAL}, EVAL_PROMPT_DATA=${EVAL_PROMPT_DATA}"
echo "[config] USE_WANDB=${USE_WANDB}, WANDB_PROJECT=${WANDB_PROJECT}, WANDB_GROUP=${WANDB_GROUP}"

if [[ "${NOHUP}" == "1" && "${_NOHUP_LAUNCHED:-0}" != "1" ]]; then
  mkdir -p "$(dirname "${NOHUP_LOG}")"
  echo "[step] relaunch with nohup, log: ${NOHUP_LOG}"
  nohup env _NOHUP_LAUNCHED=1 NOHUP=0 bash "$0" "$@" >"${NOHUP_LOG}" 2>&1 &
  echo "[step] started in background, pid=$!"
  exit 0
fi

# sglang 0.5.9 in slime image requires transformers 4.57.1.
TF_VER="$(python - <<'PY'
import transformers
print(transformers.__version__)
PY
)"
if [[ "${TF_VER}" != "4.57.1" ]]; then
  echo "[error] incompatible transformers version: ${TF_VER}"
  echo "[hint] please run: pip install -U \"transformers==4.57.1\" \"tokenizers<0.23,>=0.22\""
  exit 1
fi

if [[ ! -d "${HF_CKPT}" ]]; then
  echo "[error] HF_CKPT not found: ${HF_CKPT}"
  exit 1
fi
if [[ ! -f "${RAW_DATA}" ]]; then
  echo "[error] RAW_DATA not found: ${RAW_DATA}"
  exit 1
fi
if [[ ! -d "${MEGATRON_PATH}" ]]; then
  echo "[error] MEGATRON_PATH not found: ${MEGATRON_PATH}"
  exit 1
fi

# -------------------------------
# 1) Convert raw SWE-agent trajectories to slime prompt-data
# -------------------------------
NEED_REBUILD_PROMPT_DATA=0
if [[ ! -f "${PROMPT_DATA}" ]]; then
  NEED_REBUILD_PROMPT_DATA=1
else
  # The dataset format for VLM checkpoints requires prompt to be list[messages].
  if ! python - <<PY
import json
ok = False
with open("${PROMPT_DATA}", "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        ok = isinstance(obj.get("prompt"), list) and isinstance(obj.get("tools"), list)
        break
print("ok" if ok else "bad")
raise SystemExit(0 if ok else 1)
PY
  then
    echo "[step] detected old/incompatible prompt-data format, will rebuild: ${PROMPT_DATA}"
    NEED_REBUILD_PROMPT_DATA=1
  fi
fi

if [[ "${NEED_REBUILD_PROMPT_DATA}" -eq 1 ]]; then
  echo "[step] converting dataset to slime format..."
  python "${SCRIPT_DIR}/preprocess_swe_agent_data.py" \
    --input "${RAW_DATA}" \
    --output "${PROMPT_DATA}"
else
  echo "[step] skip dataset conversion (already exists): ${PROMPT_DATA}"
fi

# -------------------------------
# 2) Convert HF checkpoint to torch_dist if needed
# -------------------------------
if [[ ! -d "${REF_LOAD}" ]]; then
  echo "[step] converting HF checkpoint -> torch_dist..."
  source "${ROOT_DIR}/scripts/models/qwen3.5-4B.sh"
  PYTHONPATH="${MEGATRON_PATH}:${ROOT_DIR}" \
    python "${ROOT_DIR}/tools/convert_hf_to_torch_dist.py" \
    "${MODEL_ARGS[@]}" \
    --hf-checkpoint "${HF_CKPT}" \
    --save "${REF_LOAD}"
else
  echo "[step] skip checkpoint conversion (already exists): ${REF_LOAD}"
fi

mkdir -p "${SAVE_PATH}"

# -------------------------------
# 3) Build training args
# -------------------------------
source "${ROOT_DIR}/scripts/models/qwen3.5-4B.sh"

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
  --tool-key tools
  --label-key label
  --metadata-key metadata
  --apply-chat-template
  --rollout-shuffle
  --num-rollout "${NUM_ROLLOUT}"
  --rollout-batch-size "${ROLLOUT_BATCH_SIZE}"
  --n-samples-per-prompt "${N_SAMPLES_PER_PROMPT}"
  --rollout-max-response-len "${ROLLOUT_MAX_RESPONSE_LEN}"
  --rollout-temperature 0.8
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
  --lr 1e-6
  --lr-decay-style constant
  --weight-decay 0.1
  --adam-beta1 0.9
  --adam-beta2 0.98
)

SGLANG_ARGS=(
  --rollout-num-gpus-per-engine "${ROLLOUT_NUM_GPUS_PER_ENGINE}"
  --sglang-mem-fraction-static 0.5
)

MISC_ARGS=(
  --attention-dropout 0.0
  --hidden-dropout 0.0
  --accumulate-allreduce-grads-in-fp32
  --attention-softmax-in-fp32
  --attention-backend flash
)

CUSTOM_ARGS=(
  --custom-rm-path examples.swe_agentic.reward_expected_action.reward_func
)

EVAL_ARGS=()
if [[ "${USE_EVAL}" == "1" ]]; then
  EVAL_ARGS+=(
    --eval-interval "${EVAL_INTERVAL}"
    --eval-prompt-data "${EVAL_DATA_NAME}" "${EVAL_PROMPT_DATA}"
    --eval-input-key prompt
    --eval-label-key label
    --eval-tool-key tools
    --n-samples-per-eval-prompt "${N_SAMPLES_PER_EVAL_PROMPT}"
    --eval-max-response-len "${EVAL_MAX_RESPONSE_LEN}"
    --eval-top-p 1
  )
fi

WANDB_ARGS=()
if [[ "${USE_WANDB}" == "1" ]]; then
  WANDB_ARGS+=(
    --use-wandb
    --wandb-project "${WANDB_PROJECT}"
    --wandb-group "${WANDB_GROUP}"
    --wandb-exp-name "${WANDB_RUN_NAME}"
  )
  if [[ -n "${WANDB_KEY}" ]]; then
    WANDB_ARGS+=(--wandb-key "${WANDB_KEY}")
  else
    echo "[warn] USE_WANDB=1 but WANDB_KEY is empty; relying on existing wandb login/session."
  fi
fi

# -------------------------------
# 4) Start ray and launch training job
# -------------------------------
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
    \"WANDB_API_KEY\": \"${WANDB_API_KEY}\",
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
  "${ROLLOUT_ARGS[@]}" \
  "${EVAL_ARGS[@]}" \
  "${OPTIMIZER_ARGS[@]}" \
  "${GRPO_ARGS[@]}" \
  "${WANDB_ARGS[@]}" \
  "${PERF_ARGS[@]}" \
  "${SGLANG_ARGS[@]}" \
  "${MISC_ARGS[@]}" \
  "${CUSTOM_ARGS[@]}"
