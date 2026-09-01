#!/usr/bin/env bash
# 可选：运行在工控机，通过 SSH 远程启动 allinai2 上的 Policy Server
# 推荐做法：登录 allinai2 后直接运行 start_policy_server_pi05.sh
# 仅当你不想登录 GPU 服务器、且工控机能 ssh allinai2 时使用本脚本
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

echo "Starting remote policy server via SSH (optional convenience script)."
echo "  run from: robot IPC"
echo "  target: ${REMOTE_SSH_HOST}"
echo "  gpu repo: ${REMOTE_REPO_ROOT}"
echo
echo "Preferred: log into allinai2 and run scripts/start_policy_server_pi05.sh directly."
echo
ssh -t "${REMOTE_SSH_HOST}" bash -s -- "${REMOTE_REPO_ROOT}" "${CONFIG_PATH}" "$@" <<'REMOTE_EOF'
set -euo pipefail

REMOTE_REPO_ROOT="$1"
CONFIG_PATH="$2"
shift 2

cd "${REMOTE_REPO_ROOT}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True,max_split_size_mb:128}"
export PYTHONPATH="${PWD}/src${PYTHONPATH:+:${PYTHONPATH}}"

echo "[remote] repo: ${PWD}"
echo "[remote] launching policy server..."
python -m piper_train.start_async_policy_server --config "${CONFIG_PATH}" "$@"
REMOTE_EOF
