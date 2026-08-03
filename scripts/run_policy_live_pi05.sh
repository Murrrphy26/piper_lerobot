#!/usr/bin/env bash
set -euo pipefail

# PI05 真机推理：先按同一 config 回到数据集 episode0 初始位姿，再跑 policy live。
#
# 用法：
#   bash scripts/run_policy_live_pi05.sh configs/fyx_pi05.json
#   SKIP_MOVE_TO_START=true bash scripts/run_policy_live_pi05.sh configs/fyx_pi05.json

CONFIG_PATH="${1:-configs/record_pick_cube_pi05.json}"
SKIP_BRINGUP="${SKIP_BRINGUP:-true}"
SKIP_RESET="${SKIP_RESET:-true}"
SKIP_MOVE_TO_START="${SKIP_MOVE_TO_START:-false}"
if [[ $# -gt 0 ]]; then
  shift
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True,max_split_size_mb:128}"

cd "${REPO_ROOT}"

if [[ "${SKIP_BRINGUP}" != "true" ]]; then
  "${SCRIPT_DIR}/bringup_can.sh"
fi

if [[ "${SKIP_RESET}" != "true" ]]; then
  "${SCRIPT_DIR}/reset_arms.sh"
fi

export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

if [[ "${SKIP_MOVE_TO_START}" != "true" ]]; then
  echo "=== [1/2] Move to episode start (config=${CONFIG_PATH}) ==="
  bash "${SCRIPT_DIR}/run_move_to_episode_start.sh" "${CONFIG_PATH}"
  echo
  echo "=== [2/2] Start policy live ==="
else
  echo "SKIP_MOVE_TO_START=true：跳过归位，直接推理。"
fi

python -m piper_towel_fold.start_policy_live --config "${CONFIG_PATH}" "$@"
