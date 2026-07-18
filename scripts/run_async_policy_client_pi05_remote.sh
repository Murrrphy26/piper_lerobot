#!/usr/bin/env bash
# 运行在：工控机
# 作用：启动前打印远程推理前置条件，然后运行 Robot Client
set -euo pipefail
CONFIG_PATH="${1:-configs/record_pick_cube_pi05.json}"
if [[ $# -gt 0 ]]; then
  shift
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# shellcheck source=lib/remote_gpu_config.sh
source "${SCRIPT_DIR}/lib/remote_gpu_config.sh"
remote_gpu_load_config "${CONFIG_PATH}" "${REPO_ROOT}"

echo "Piper async client (remote GPU mode)."
echo "  machine: robot IPC"
echo "  policy server: ${ASYNC_SERVER_ADDRESS}"
echo "  ssh host: ${REMOTE_SSH_HOST}"
echo
if [[ "${REMOTE_USE_SSH_TUNNEL}" == "true" ]]; then
  echo "Prerequisites on robot IPC:"
  echo "  1) On allinai2: bash scripts/start_policy_server_pi05.sh"
  echo "  2) On robot IPC: bash scripts/ssh_tunnel_policy_server.sh  (keep open)"
  echo "  3) Then run this script"
  echo
fi
exec "${SCRIPT_DIR}/run_async_policy_client_pi05.sh" "${CONFIG_PATH}" "$@"
