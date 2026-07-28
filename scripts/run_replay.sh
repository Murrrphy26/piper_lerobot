#!/usr/bin/env bash

set -euo pipefail

LEFT_CAN="${LEFT_CAN:-can2}"
RIGHT_CAN="${RIGHT_CAN:-can0}"
DATASET_ROOT="${DATASET_ROOT:-data/lerobot/local/cube_cam_adjustment_ojag_cams-right-front-side}"
#cube_v727_yjw_ojag, cube_cam_adjustment, cube_small_dataset
EPISODE_INDEX="${EPISODE_INDEX:-0}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

echo "Replaying recorded episode on Piper arms..."
echo "  left  -> ${LEFT_CAN}"
echo "  right -> ${RIGHT_CAN}"
echo "  dataset -> ${DATASET_ROOT}"
echo "  episode -> ${EPISODE_INDEX}"
echo "  extra args -> $*"
echo

# 用法示例：
#   bash scripts/run_replay.sh
#   bash scripts/run_replay.sh --episode-index 3
#   # 默认已是 1:1 不限速；若要软件限速：
#   bash scripts/run_replay.sh --rate-limit --replay-max-joint-step-rad 0.2
#   bash scripts/run_replay.sh --source obs_joints_action_gripper
#   bash scripts/run_replay.sh --speed 0.5   # 半速（改的是时间，不是软件限速）

python tools/replay_episode.py \
  --left-can "${LEFT_CAN}" \
  --right-can "${RIGHT_CAN}" \
  --dataset-root "${DATASET_ROOT}" \
  --episode-index "${EPISODE_INDEX}" \
  "$@"
