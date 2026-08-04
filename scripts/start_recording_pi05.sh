#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="${1:-configs/record_pick_cube_pi05.json}"
SKIP_BRINGUP="${SKIP_BRINGUP:-false}"
SKIP_RESET="${SKIP_RESET:-true}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${REPO_ROOT}"

if [[ "${SKIP_BRINGUP}" != "true" ]]; then
  "${SCRIPT_DIR}/bringup_can.sh"
fi

if [[ "${SKIP_RESET}" != "true" ]]; then
  "${SCRIPT_DIR}/reset_arms.sh"
fi

export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

echo
echo "Starting recording."
echo "  Move the leader arms to teleoperate the followers (same CAN)."
echo "  Follower joint states are recorded as observation and action."
echo "  Press Ctrl+C once after each episode to save."
echo

python -m piper_train.start_recording --config "${CONFIG_PATH}"
