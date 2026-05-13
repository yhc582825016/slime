#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." &>/dev/null && pwd)"
cd "${ROOT_DIR}"

export PYTHONUNBUFFERED=1

GURU_INPUT="${GURU_INPUT:-/dev/shm/ye/rl-data/guru-RL-92k/train}"
GYM_INPUT="${GYM_INPUT:-/dev/shm/ye/rl-data/Nemotron-RL-ReasoningGym-v1/data/train.jsonl}"
OUT_DIR="${OUT_DIR:-/dev/shm/ye/rl-data/mix-rl-slime}"

GURU_OUTPUT="${GURU_OUTPUT:-${OUT_DIR}/guru.jsonl}"
GYM_OUTPUT="${GYM_OUTPUT:-${OUT_DIR}/reasoning_gym.jsonl}"
MIX_OUTPUT="${MIX_OUTPUT:-${OUT_DIR}/mix_train.jsonl}"

# Guru domains are inferred from the parquet filename prefix before "__":
# codegen, logic, math, simulation, stem, table.
#
# By default we skip stem because stem__web needs an external LLM judge.
# Override with GURU_EXCLUDE_DOMAIN="" to include everything, or set
# GURU_INCLUDE_DOMAIN="math,logic,table" to keep only selected domains.
GURU_INCLUDE_DOMAIN="${GURU_INCLUDE_DOMAIN:-}"
GURU_EXCLUDE_DOMAIN="${GURU_EXCLUDE_DOMAIN:-stem}"

# Optional smoke-test limits. Empty means full data.
GURU_LIMIT="${GURU_LIMIT:-}"
GYM_LIMIT="${GYM_LIMIT:-}"

mkdir -p "${OUT_DIR}" "$(dirname "${GURU_OUTPUT}")" "$(dirname "${GYM_OUTPUT}")" "$(dirname "${MIX_OUTPUT}")"

echo "[config] ROOT_DIR=${ROOT_DIR}"
echo "[config] GURU_INPUT=${GURU_INPUT}"
echo "[config] GYM_INPUT=${GYM_INPUT}"
echo "[config] OUT_DIR=${OUT_DIR}"
echo "[config] GURU_INCLUDE_DOMAIN=${GURU_INCLUDE_DOMAIN:-<all>}"
echo "[config] GURU_EXCLUDE_DOMAIN=${GURU_EXCLUDE_DOMAIN:-<none>}"
echo "[config] GURU_OUTPUT=${GURU_OUTPUT}"
echo "[config] GYM_OUTPUT=${GYM_OUTPUT}"
echo "[config] MIX_OUTPUT=${MIX_OUTPUT}"

if [[ ! -e "${GURU_INPUT}" ]]; then
  echo "[error] GURU_INPUT not found: ${GURU_INPUT}" >&2
  exit 1
fi
if [[ ! -f "${GYM_INPUT}" ]]; then
  echo "[error] GYM_INPUT not found: ${GYM_INPUT}" >&2
  exit 1
fi

GURU_ARGS=(
  --input "${GURU_INPUT}"
  --output "${GURU_OUTPUT}"
)
if [[ -n "${GURU_INCLUDE_DOMAIN}" ]]; then
  GURU_ARGS+=(--include-domain "${GURU_INCLUDE_DOMAIN}")
fi
if [[ -n "${GURU_EXCLUDE_DOMAIN}" ]]; then
  GURU_ARGS+=(--exclude-domain "${GURU_EXCLUDE_DOMAIN}")
fi
if [[ -n "${GURU_LIMIT}" ]]; then
  GURU_ARGS+=(--limit "${GURU_LIMIT}")
fi

echo "[step] converting Guru data..."
python -m examples.mix_rl.convert_guru_to_slime "${GURU_ARGS[@]}"

echo "[step] converting ReasoningGym data..."
GYM_INPUT="${GYM_INPUT}" GYM_OUTPUT="${GYM_OUTPUT}" GYM_LIMIT="${GYM_LIMIT}" python - <<'PY'
import json
import os
from pathlib import Path

src = Path(os.environ["GYM_INPUT"])
dst = Path(os.environ["GYM_OUTPUT"])
limit_raw = os.environ.get("GYM_LIMIT", "")
limit = int(limit_raw) if limit_raw else None

dst.parent.mkdir(parents=True, exist_ok=True)
count = 0
with src.open("r", encoding="utf-8") as f, dst.open("w", encoding="utf-8") as out:
    for line in f:
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        metadata = row.get("metadata") or {}
        extra_info = dict(metadata)
        extra_info.setdefault("question", row.get("question"))
        record = {
            "prompt": row.get("question"),
            "reward_model": {"ground_truth": row.get("answer")},
            "metadata": {
                "data_source": "reasoning_gym",
                "ability": "reasoning_gym",
                "extra_info": extra_info,
            },
        }
        out.write(json.dumps(record, ensure_ascii=False) + "\n")
        count += 1
        if limit is not None and count >= limit:
            break

print(f"{src} -> {dst} ({count} rows)")
PY

echo "[step] merging Guru + ReasoningGym..."
cat "${GURU_OUTPUT}" "${GYM_OUTPUT}" > "${MIX_OUTPUT}"

echo "[result] line counts:"
wc -l "${GURU_OUTPUT}" "${GYM_OUTPUT}" "${MIX_OUTPUT}"

echo "[result] data_source counts:"
MIX_OUTPUT="${MIX_OUTPUT}" python - <<'PY'
import json
import os
from collections import Counter

counter = Counter()
with open(os.environ["MIX_OUTPUT"], "r", encoding="utf-8") as f:
    for line in f:
        row = json.loads(line)
        metadata = row.get("metadata") or {}
        counter[metadata.get("data_source")] += 1

for key, value in sorted(counter.items(), key=lambda item: str(item[0])):
    print(f"{key}\t{value}")
PY

echo "[done] mixed prompt data: ${MIX_OUTPUT}"
