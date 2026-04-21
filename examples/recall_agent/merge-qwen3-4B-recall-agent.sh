#!/bin/bash

set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." &>/dev/null && pwd)"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
cd "${ROOT_DIR}"

HF_CKPT="${HF_CKPT:-/dev/shm/ye/Qwen3-4B}"
MEGATRON_PATH="${MEGATRON_PATH:-/root/Megatron-LM}"
SAVE_PATH="${SAVE_PATH:-/dev/shm/Qwen3-4B-Thinking_recall_agent}"
INPUT_ITER="${INPUT_ITER:-iter_0000199}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/dev/shm/Qwen3-4B-Thinking_recall_agent_hf}"
VOCAB_SIZE="${VOCAB_SIZE:-151936}"
LOAD_MAX_WORKERS="${LOAD_MAX_WORKERS:-4}"
SAVE_MAX_WORKERS="${SAVE_MAX_WORKERS:-16}"
FORCE_OVERWRITE="${FORCE_OVERWRITE:-0}"

if [[ ! -d "${HF_CKPT}" ]]; then
  echo "[error] HF_CKPT not found: ${HF_CKPT}"
  exit 1
fi

if [[ ! -d "${MEGATRON_PATH}" ]]; then
  echo "[error] MEGATRON_PATH not found: ${MEGATRON_PATH}"
  exit 1
fi

if [[ ! -d "${SAVE_PATH}" ]]; then
  echo "[error] SAVE_PATH not found: ${SAVE_PATH}"
  exit 1
fi

if [[ -n "${INPUT_ITER}" ]]; then
  INPUT_DIR="${SAVE_PATH%/}/${INPUT_ITER}"
  if [[ ! -d "${INPUT_DIR}" ]]; then
    echo "[error] INPUT_ITER directory not found: ${INPUT_DIR}"
    exit 1
  fi
else
  INPUT_DIR="$(find "${SAVE_PATH}" -maxdepth 1 -mindepth 1 -type d -name 'iter_*' | sort | tail -n 1)"
  if [[ -z "${INPUT_DIR}" ]]; then
    echo "[error] no iter_* checkpoints found under ${SAVE_PATH}"
    exit 1
  fi
fi

ITER_NAME="$(basename "${INPUT_DIR}")"
OUTPUT_DIR="${OUTPUT_ROOT%/}/${ITER_NAME}"

FORCE_ARGS=()
if [[ "${FORCE_OVERWRITE}" == "1" ]]; then
  FORCE_ARGS+=(--force)
fi

echo "[info] ROOT_DIR: ${ROOT_DIR}"
echo "[info] HF_CKPT: ${HF_CKPT}"
echo "[info] SAVE_PATH: ${SAVE_PATH}"
echo "[info] INPUT_DIR: ${INPUT_DIR}"
echo "[info] OUTPUT_DIR: ${OUTPUT_DIR}"

PYTHONPATH="${MEGATRON_PATH}:${ROOT_DIR}" \
python "${ROOT_DIR}/tools/convert_torch_dist_to_hf_parallel.py" \
  --input-dir "${INPUT_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  --origin-hf-dir "${HF_CKPT}" \
  --vocab-size "${VOCAB_SIZE}" \
  --load-max-workers "${LOAD_MAX_WORKERS}" \
  --save-max-workers "${SAVE_MAX_WORKERS}" \
  "${FORCE_ARGS[@]}"

echo "[done] merged checkpoint exported to ${OUTPUT_DIR}"
