#!/bin/bash
set -euo pipefail

# Prepare artifacts: logits only (no embeddings, no scores)
# - patches config.yaml temporarily (restored on exit)
#
# Usage:
#   bash src/scripts/prepare_logits.sh [datasets...]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
SRC_DIR="${ROOT_DIR}/src"

CONFIG_PATH="${ROOT_DIR}/config.yaml"

PREPARE_STEPS_JSON='["logits"]'
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

python - <<'PY'
import os

config_path = os.environ["CONFIG_PATH"]
steps_json = os.environ["PREPARE_STEPS_JSON"]

lines = open(config_path, "r", encoding="utf-8").read().splitlines(True)

def set_top_level(key: str, value_yaml: str) -> None:
    global lines
    for i, line in enumerate(lines):
        if line.startswith(f"{key}:"):
            lines[i] = f"{key}: {value_yaml}\n"
            return
    lines.append(f"{key}: {value_yaml}\n")

set_top_level("prepare_steps", steps_json)
open(config_path, "w", encoding="utf-8").writelines(lines)
PY

cd "${SRC_DIR}"
python main.py --mode prepare --datasets "${DATASETS[@]}"

