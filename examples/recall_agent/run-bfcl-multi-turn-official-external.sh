#!/bin/bash

set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." &>/dev/null && pwd)"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
cd "${ROOT_DIR}"

source /datas/miniconda3/etc/profile.d/conda.sh
conda activate ye

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy

BFCL_ROOT="${BFCL_ROOT:-/dev/shm/ye/berkeley-function-call-leaderboard}"
HF_CKPT="${HF_CKPT:-/dev/shm/Qwen3.5-4B}"
MODEL_NAME="${MODEL_NAME:-Qwen3.5-4B-slime-bfcl-official-external-FC}"
LOCAL_SERVER_ENDPOINT="${LOCAL_SERVER_ENDPOINT:-127.0.0.1}"
LOCAL_SERVER_PORT="${LOCAL_SERVER_PORT:-6031}"
SAMPLE_SIZE="${SAMPLE_SIZE:-200}"
SAMPLE_SIZE_MODE="${SAMPLE_SIZE_MODE:-per_category}"
SEED="${SEED:-42}"
NUM_THREADS="${NUM_THREADS:-16}"
TEMPERATURE="${TEMPERATURE:-0.0}"
ALLOW_OVERWRITE="${ALLOW_OVERWRITE:-1}"

ARGS=(
  --bfcl-root "${BFCL_ROOT}"
  --hf-checkpoint "${HF_CKPT}"
  --model-name "${MODEL_NAME}"
  --local-server-endpoint "${LOCAL_SERVER_ENDPOINT}"
  --local-server-port "${LOCAL_SERVER_PORT}"
  --sample-size "${SAMPLE_SIZE}"
  --sample-size-mode "${SAMPLE_SIZE_MODE}"
  --seed "${SEED}"
  --num-threads "${NUM_THREADS}"
  --temperature "${TEMPERATURE}"
)

if [[ "${ALLOW_OVERWRITE}" == "1" ]]; then
  ARGS+=(--allow-overwrite)
fi

python "${SCRIPT_DIR}/eval_bfcl_multi_turn_official_external.py" "${ARGS[@]}"
