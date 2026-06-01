#!/bin/bash
# Standalone AIME eval on a pre-launched SGLang server (sglang.sh, port 6035).
#
# Usage:
#   bash examples/on_policy_distillation/run-eval-aime-sglang.sh
#
# Prerequisites:
#   1. Start SGLang: bash /mnt/code/yehangcheng/sglang.sh
#   2. Wait until http://127.0.0.1:6035/health_generate is ready

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SLIME_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

export PYTHONUNBUFFERED=1
export http_proxy=""
export https_proxy=""
export PYTHONPATH="${PYTHONPATH:-}:${SLIME_ROOT}"

# Match run-qwen3-opd-4-4.sh EVAL_ARGS for aime
SGLANG_HOST="${SGLANG_HOST:-127.0.0.1}"
SGLANG_PORT="${SGLANG_PORT:-6035}"
MODEL_PATH="${MODEL_PATH:-/opt/users/ye/models/qwen3.5-opd-ori-9b-distill-qwen3.5-4b-509/iter_0000099}"
DATA_PATH="${DATA_PATH:-/mnt/code/yehangcheng/ms-swift/train_data/aime_2024_swift_.jsonl}"
N_SAMPLES="${N_SAMPLES:-4}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-32000}"
TOP_P="${TOP_P:-1}"
TEMPERATURE="${TEMPERATURE:-1}"
CONCURRENCY="${CONCURRENCY:-16}"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="${OUTPUT_DIR:-/mnt/code/yehangcheng/logs/opd_eval}"
mkdir -p "${OUTPUT_DIR}"
OUTPUT_JSONL="${OUTPUT_JSONL:-${OUTPUT_DIR}/aime_eval_${TIMESTAMP}.jsonl}"

echo "[eval] SGLang: http://${SGLANG_HOST}:${SGLANG_PORT}"
echo "[eval] model: ${MODEL_PATH}"
echo "[eval] data:  ${DATA_PATH}"
echo "[eval] output: ${OUTPUT_JSONL}"

python3 "${SCRIPT_DIR}/eval_aime_sglang.py" \
  --host "${SGLANG_HOST}" \
  --port "${SGLANG_PORT}" \
  --model-path "${MODEL_PATH}" \
  --data-path "${DATA_PATH}" \
  --n-samples "${N_SAMPLES}" \
  --max-new-tokens "${MAX_NEW_TOKENS}" \
  --top-p "${TOP_P}" \
  --temperature "${TEMPERATURE}" \
  --concurrency "${CONCURRENCY}" \
  --output "${OUTPUT_JSONL}"
