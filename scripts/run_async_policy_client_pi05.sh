#!/usr/bin/env bash
# 运行在：工控机
# 作用：先松开夹爪并回到固定 home 位姿，等待人工确认后，
#      再启动远程 pi05 异步推理客户端。
#
# 用法：
#   bash scripts/run_async_policy_client_pi05.sh configs/record_towel_fold_pi05.json
#
# 环境变量：
#   SKIP_BRINGUP     默认 true；false 时先初始化 CAN
#   SKIP_RESET       默认 true；false 时先退出 teach/drag 模式
#   SKIP_RESET_POSE  默认 false；true 时跳过“松夹爪 + 固定 home 归位”
#   SKIP_CONFIRM     默认 false；true 时归位后不等待回车（不建议真机使用）
#   OPEN_GRIPPER_M   默认 0.07；归位过程中左右夹爪保持该开度（米）
set -euo pipefail

CONFIG_PATH="${1:-configs/record_towel_fold_pi05.json}"
if [[ $# -gt 0 ]]; then
  shift
fi

SKIP_BRINGUP="${SKIP_BRINGUP:-true}"
SKIP_RESET="${SKIP_RESET:-true}"
SKIP_RESET_POSE="${SKIP_RESET_POSE:-false}"
SKIP_CONFIRM="${SKIP_CONFIRM:-false}"
OPEN_GRIPPER_M="${OPEN_GRIPPER_M:-0.07}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# shellcheck source=lib/remote_gpu_config.sh
source "${SCRIPT_DIR}/lib/remote_gpu_config.sh"
remote_gpu_load_config "${CONFIG_PATH}" "${REPO_ROOT}"

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

echo "Piper async client (remote GPU mode)."
echo "  config: ${CONFIG_PATH}"
echo "  policy server: ${ASYNC_SERVER_ADDRESS}"
echo "  ssh host: ${REMOTE_SSH_HOST}"
echo

if [[ "${SKIP_BRINGUP}" != "true" ]]; then
  "${SCRIPT_DIR}/bringup_can.sh"
fi

if [[ "${SKIP_RESET}" != "true" ]]; then
  "${SCRIPT_DIR}/reset_arms.sh"
fi

if [[ "${SKIP_RESET_POSE}" != "true" ]]; then
  echo "=== [1/2] 松开夹爪并移动到固定 home 位姿 ==="
  "${SCRIPT_DIR}/run_move_to_joints.sh" \
    --preset home \
    --left-gripper "${OPEN_GRIPPER_M}" \
    --right-gripper "${OPEN_GRIPPER_M}"
else
  echo "SKIP_RESET_POSE=true：跳过松夹爪和固定 home 归位。"
fi

echo
echo "=== [2/2] 准备启动异步推理 ==="
if [[ "${SKIP_CONFIRM}" != "true" ]]; then
  echo "请确认：人员已离开机械臂工作区、急停可用、SSH 隧道和 Policy Server 正常。"
  read -r -p "按回车开始推理；按 Ctrl+C 取消：" </dev/tty
fi

exec python -m piper_train.start_async_policy_client --config "${CONFIG_PATH}" "$@"
