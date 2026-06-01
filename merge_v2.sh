#!/usr/bin/env bash
set -euo pipefail

cd /mnt/code/yehangcheng/slime

INPUT_DIR=${INPUT_DIR:-/mnt/code/yehangcheng/checkpoint/General_model/qwen3.5-opd-513/iter_0000299}
ORIGIN_HF_DIR=${ORIGIN_HF_DIR:-/opt/users/models/Qwen3.5-4B}
OUTPUT_DIR=${OUTPUT_DIR:-/opt/users/ye/models/qwen3.5-opd-513/iter_0000299}

force_args=()
if [[ "${FORCE:-0}" == "1" ]]; then
  force_args+=(--force)
fi

PYTHONPATH="/root/Megatron-LM:$(pwd)" \
python tools/convert_torch_dist_to_hf.py \
  --input-dir "${INPUT_DIR}" \
  --origin-hf-dir "${ORIGIN_HF_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  "${force_args[@]}"
