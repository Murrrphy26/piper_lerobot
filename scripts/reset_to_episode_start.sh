#!/usr/bin/env bash

set -euo pipefail

# 先松开双夹爪，再移动到配置所指数据集的第 0 条 episode 初始关节位姿。
# 移动过程中保持夹爪打开，避免使用数据首帧中的夹爪目标重新夹紧。
#
# 用法：
#   bash scripts/run_open_grippers_and_move_to_episode_start.sh configs/record_towel_fold_pi05.json
#   bash scripts/run_open_grippers_and_move_to_episode_start.sh configs/record_towel_fold_pi05.json --dry-run
#   OPEN_GRIPPER_M=0.065 bash scripts/run_open_grippers_and_move_to_episode_start.sh <config>

CONFIG_PATH="${1:-configs/record_towel_fold_pi05.json}"
if [[ $# -gt 0 ]]; then
  shift
fi

OPEN_GRIPPER_M="${OPEN_GRIPPER_M:-0.07}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

exec "${SCRIPT_DIR}/run_move_to_episode_start.sh" \
  "${CONFIG_PATH}" \
  --episode-index 0 \
  --open-grippers-first \
  --keep-grippers-open \
  --open-gripper-m "${OPEN_GRIPPER_M}" \
  "$@"
