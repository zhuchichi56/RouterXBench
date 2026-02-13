#!/bin/bash
set -euo pipefail

# Prepare artifacts with logits + embeddings together:
# - compute logits + embeddings
# - patches config.yaml temporarily (restored on exit)
#
# Usage:
#   bash src/scripts/prepare_logits_embeddings.sh [datasets...]
#
# Optional env vars:
#   PREPARE_TEXT_FIELD=instruction
#   PREPARE_EMBED_BATCH_SIZE=64
#   PREPARE_EMBED_SOURCE=longformer|weak_model_hs
#   PREPARE_HS_EMBED_MODE=first|last|mean|layer:<idx>

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
SRC_DIR="${ROOT_DIR}/src"

CONFIG_PATH="${ROOT_DIR}/config.yaml"

PREPARE_STEPS_JSON='["logits","embeddings"]'
PREPARE_TEXT_FIELD="${PREPARE_TEXT_FIELD:-instruction}"
PREPARE_EMBED_BATCH_SIZE="${PREPARE_EMBED_BATCH_SIZE:-64}"
PREPARE_EMBED_SOURCE="${PREPARE_EMBED_SOURCE:-longformer}"
PREPARE_HS_EMBED_MODE="${PREPARE_HS_EMBED_MODE:-first}"

TEST_DATASETS_DEFAULT="math mmlu_test"

if [ "$#" -gt 0 ]; then
  DATASETS=("$@")
else
  # shellcheck disable=SC2206
  DATASETS=(${TEST_DATASETS_DEFAULT})
fi

if [ ! -f "${CONFIG_PATH}" ]; then
  echo "error: config not found: ${CONFIG_PATH}"
  exit 1
fi

TS="$(date +%Y%m%d_%H%M%S)"
BACKUP_PATH="${CONFIG_PATH}.bak.${TS}"
cp "${CONFIG_PATH}" "${BACKUP_PATH}"
trap 'mv -f "${BACKUP_PATH}" "${CONFIG_PATH}"' EXIT

export CONFIG_PATH
export PREPARE_STEPS_JSON
export PREPARE_TEXT_FIELD
export PREPARE_EMBED_BATCH_SIZE
export PREPARE_EMBED_SOURCE
export PREPARE_HS_EMBED_MODE

python - <<'PY'
import os

config_path = os.environ["CONFIG_PATH"]
steps_json = os.environ["PREPARE_STEPS_JSON"]
text_field = os.environ["PREPARE_TEXT_FIELD"]
embed_bs = os.environ["PREPARE_EMBED_BATCH_SIZE"]
embed_source = os.environ["PREPARE_EMBED_SOURCE"]
hs_embed_mode = os.environ["PREPARE_HS_EMBED_MODE"]

lines = open(config_path, "r", encoding="utf-8").read().splitlines(True)

def set_top_level(key: str, value_yaml: str) -> None:
    global lines
    for i, line in enumerate(lines):
        if line.startswith(f"{key}:"):
            lines[i] = f"{key}: {value_yaml}\n"
            return
    lines.append(f"{key}: {value_yaml}\n")

set_top_level("prepare_steps", steps_json)
set_top_level("prepare_text_field", f"\"{text_field}\"")
set_top_level("prepare_embed_batch_size", str(int(embed_bs)))
set_top_level("prepare_embedding_source", f"\"{embed_source}\"")
set_top_level("prepare_hs_embedding_mode", f"\"{hs_embed_mode}\"")

open(config_path, "w", encoding="utf-8").writelines(lines)
PY

cd "${SRC_DIR}"
python main.py --mode prepare --datasets "${DATASETS[@]}"
