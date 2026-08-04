#!/usr/bin/env bash
set -euo pipefail

# 毛巾折叠真机推理：归位 → SDK 打开夹爪并交互合拢 → policy live。
#
# 用法：
#   bash scripts/run_policy_live_towel.sh configs/record_towel_fold_pi05.json
#   SKIP_MOVE_TO_START=true bash scripts/run_policy_live_towel.sh configs/record_towel_fold_pi05.json
#   SKIP_GRIPPER_PREP=true bash scripts/run_policy_live_towel.sh configs/record_towel_fold_pi05.json
#
# 环境变量：
#   SKIP_MOVE_TO_START  默认 false；true 时跳过 move_to_episode_start
#   SKIP_GRIPPER_PREP   默认 false；true 时跳过夹爪交互准备
#   SKIP_BRINGUP / SKIP_RESET  默认 true（与 pi05/xvla 脚本一致）

CONFIG_PATH="${1:-configs/record_towel_fold_pi05.json}"
SKIP_BRINGUP="${SKIP_BRINGUP:-true}"
SKIP_RESET="${SKIP_RESET:-true}"
SKIP_MOVE_TO_START="${SKIP_MOVE_TO_START:-false}"
SKIP_GRIPPER_PREP="${SKIP_GRIPPER_PREP:-false}"
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

STEP=1
TOTAL=3
if [[ "${SKIP_MOVE_TO_START}" == "true" ]]; then
  TOTAL=$((TOTAL - 1))
fi
if [[ "${SKIP_GRIPPER_PREP}" == "true" ]]; then
  TOTAL=$((TOTAL - 1))
fi

if [[ "${SKIP_MOVE_TO_START}" != "true" ]]; then
  echo "=== [${STEP}/${TOTAL}] Move to episode start (config=${CONFIG_PATH}) ==="
  bash "${SCRIPT_DIR}/run_move_to_episode_start.sh" "${CONFIG_PATH}"
  echo
  STEP=$((STEP + 1))
else
  echo "SKIP_MOVE_TO_START=true：跳过归位。"
fi

if [[ "${SKIP_GRIPPER_PREP}" != "true" ]]; then
  echo "=== [${STEP}/${TOTAL}] Interactive gripper prep (SDK) ==="
  python "${REPO_ROOT}/tools/interactive_gripper_prep_towel.py" --config "${CONFIG_PATH}"
  echo
  STEP=$((STEP + 1))
else
  echo "SKIP_GRIPPER_PREP=true：跳过夹爪交互准备。"
fi

echo "=== [${STEP}/${TOTAL}] Start policy live ==="
python -m piper_train.start_policy_live --config "${CONFIG_PATH}" "$@"
