#!/bin/bash
set -euo pipefail

# Part 3/3: evaluation
# - patches config.yaml temporarily (restored on exit)
# - runs one or more router types on the full TEST_DATASETS list
#
# Usage:
#   bash src/scripts/eval.sh
#   bash src/scripts/eval.sh probe
#   bash src/scripts/eval.sh probe --probe-type dynamic_dirichlet
#   bash src/scripts/eval.sh embedding_mlp probe

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
SRC_DIR="${ROOT_DIR}/src"

CONFIG_PATH="${ROOT_DIR}/config.yaml"

TEST_DATASETS_DEFAULT="math mmlu_pro_biology mmlu_pro_business mmlu_pro_chemistry mmlu_pro_computer_science mmlu_pro_economics mmlu_pro_engineering mmlu_pro_health mmlu_pro_history mmlu_pro_law mmlu_pro_math mmlu_pro_other mmlu_pro_philosophy mmlu_pro_physics mmlu_pro_psychology magpie_5k_test alpaca_5k_test big_math_5k_test mmlu_test"
AVAILABLE_ROUTERS=(
  "embedding_mlp"
  "probe"
  "self_based"
  "logits_based_routers"
  "trained_deberta"
)

print_usage() {
  cat <<'EOF'
Usage:
  bash src/scripts/eval.sh [router ...] [--probe-type PROBE_TYPE]

Examples:
  bash src/scripts/eval.sh
  bash src/scripts/eval.sh probe
  bash src/scripts/eval.sh probe --probe-type dynamic_dirichlet
  bash src/scripts/eval.sh embedding_mlp probe

Notes:
  - All model/probe/dataset parameters still come from config.yaml.
  - This script selects which router_type(s) to run.
  - --probe-type only affects router_type=probe.
EOF
}

contains_router() {
  local target="$1"
  for r in "${AVAILABLE_ROUTERS[@]}"; do
    if [[ "${r}" == "${target}" ]]; then
      return 0
    fi
  done
  return 1
}

# Always run on the full test set.
# shellcheck disable=SC2206
DATASETS=(${TEST_DATASETS_DEFAULT})

if [[ $# -gt 0 && ( "$1" == "-h" || "$1" == "--help" ) ]]; then
  print_usage
  exit 0
fi

if [[ $# -gt 0 && "$1" == "--" ]]; then
  shift
fi

ROUTERS=()
PROBE_TYPE=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --probe-type)
      if [[ $# -lt 2 ]]; then
        echo "error: --probe-type requires a value"
        exit 1
      fi
      PROBE_TYPE="$2"
      shift 2
      ;;
    *)
      if ! contains_router "$1"; then
        echo "error: unknown router '$1'"
        echo "available: ${AVAILABLE_ROUTERS[*]}"
        exit 1
      fi
      ROUTERS+=("$1")
      shift
      ;;
  esac
done

if [[ ${#ROUTERS[@]} -eq 0 ]]; then
  ROUTERS=("${AVAILABLE_ROUTERS[@]}")
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

patch_config_for_router() {
  local router="$1"

  # Reset config to the original contents before patching.
  cp "${BACKUP_PATH}" "${CONFIG_PATH}"

  export ROUTER_TYPE="${router}"
  export PROBE_TYPE

  python - <<'PY'
import os
import json

config_path = os.environ["CONFIG_PATH"]
router_type = os.environ["ROUTER_TYPE"]
probe_type = os.environ.get("PROBE_TYPE", "").strip()

lines = open(config_path, "r", encoding="utf-8").read().splitlines(True)

def find_block(block: str):
    start = None
    for i, line in enumerate(lines):
        if line.startswith(f"{block}:"):
            start = i
            break
    if start is None:
        lines.append(f"\n{block}:\n")
        start = len(lines) - 1
    end = len(lines)
    for j in range(start + 1, len(lines)):
        if lines[j] and not lines[j].startswith(" ") and lines[j].strip().endswith(":"):
            end = j
            break
    return start, end

def set_in_block(block: str, key: str, value_yaml: str):
    start, end = find_block(block)
    prefix = f"  {key}:"
    for i in range(start + 1, end):
        if lines[i].startswith(prefix):
            lines[i] = f"  {key}: {value_yaml}\n"
            return
    lines.insert(start + 1, f"  {key}: {value_yaml}\n")

set_in_block("router", "router_type", f"\"{router_type}\"")
if router_type == "probe" and probe_type:
    set_in_block("router", "probe_type", f"\"{probe_type}\"")
    set_in_block("training", "probe_types", json.dumps([probe_type]))

open(config_path, "w", encoding="utf-8").writelines(lines)
PY
}

cd "${SRC_DIR}"
for r in "${ROUTERS[@]}"; do
  echo "=== eval router=${r} datasets=${#DATASETS[@]} ==="
  patch_config_for_router "${r}"
  python main.py --mode eval --datasets "${DATASETS[@]}"
done
