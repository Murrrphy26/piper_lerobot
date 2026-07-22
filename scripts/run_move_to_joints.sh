#!/usr/bin/env bash

set -euo pipefail

LEFT_CAN="${LEFT_CAN:-can2}"
RIGHT_CAN="${RIGHT_CAN:-can0}"
PRESET="${PRESET:-demo}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

echo "Moving arms to joint targets..."
echo "  left  -> ${LEFT_CAN}"
echo "  right -> ${RIGHT_CAN}"
echo "  preset/args -> ${PRESET} $*"
echo

# 默认 demo；可覆盖：
#   PRESET=home bash scripts/run_move_to_joints.sh
#   bash scripts/run_move_to_joints.sh --preset home
#   bash scripts/run_move_to_joints.sh --preset sweep
#   bash scripts/run_move_to_joints.sh --preset sweep --duration 10 --amp-deg 5 --freq-hz 0.3
if [[ $# -eq 0 ]]; then
  set -- --preset "${PRESET}"
fi

python tools/move_to_joints.py \
  --left-can "${LEFT_CAN}" \
  --right-can "${RIGHT_CAN}" \
  "$@"
