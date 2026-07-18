#!/usr/bin/env bash
# 运行在：工控机（本地已录制数据集的机器）
# 作用：从 config 读取 dataset root/repo_id，打包并通过 scp 上传到 allinai2，在远端仓库同路径解压
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

eval "$(
  CONFIG_PATH="${REMOTE_CONFIG_PATH}" python - <<'PY'
import json
import os
import shlex

path = os.environ["CONFIG_PATH"]
with open(path, encoding="utf-8") as f:
    cfg = json.load(f)

root = cfg["root"]
repo_id = cfg["repo_id"]
dataset_rel = f"{root}/{repo_id}"

def emit(name, value):
    print(f"export {name}={shlex.quote(str(value))}")

emit("DATASET_ROOT_REL", dataset_rel)
emit("DATASET_REPO_ID", repo_id)
PY
)"

LOCAL_DATASET="${REPO_ROOT}/${DATASET_ROOT_REL}"
REMOTE_DATASET="${REMOTE_REPO_ROOT}/${DATASET_ROOT_REL}"
REMOTE_ARCHIVE="/tmp/dataset_$(echo "${DATASET_REPO_ID}" | tr '/' '_')_$(date +%Y%m%d_%H%M%S).tar.gz"
LOCAL_ARCHIVE="$(mktemp "/tmp/dataset_${DATASET_REPO_ID//\//_}.XXXXXX.tar.gz")"

cleanup() {
  rm -f "${LOCAL_ARCHIVE}"
}
trap cleanup EXIT

if [[ ! -f "${LOCAL_DATASET}/meta/info.json" ]]; then
  echo "本地数据集不存在: ${LOCAL_DATASET}/meta/info.json" >&2
  echo "请先完成录制，或检查 config 中的 root / repo_id。" >&2
  exit 1
fi

echo "打包数据集（保持 piper 仓库内相对路径一致）"
echo "  config:       ${REMOTE_CONFIG_PATH}"
echo "  相对路径:     ${DATASET_ROOT_REL}"
echo "  本地 piper:   ${REPO_ROOT}"
echo "  本地数据集:   ${LOCAL_DATASET}"
echo "  服务器 piper: ${REMOTE_REPO_ROOT}  (来自 config remote_gpu.gpu_repo_root)"
echo "  服务器数据集: ${REMOTE_DATASET}"
echo "  ssh:          ${REMOTE_SSH_HOST}"
echo

echo "Creating archive (relative to repo root: ${DATASET_ROOT_REL}) ..."
tar -czf "${LOCAL_ARCHIVE}" -C "${REPO_ROOT}" "${DATASET_ROOT_REL}"

archive_size="$(du -h "${LOCAL_ARCHIVE}" | awk '{print $1}')"
echo "Archive size: ${archive_size}"
echo

echo "Uploading to ${REMOTE_SSH_HOST}:${REMOTE_ARCHIVE} ..."
scp "${LOCAL_ARCHIVE}" "${REMOTE_SSH_HOST}:${REMOTE_ARCHIVE}"

echo
echo "Extracting on remote ..."
ssh "${REMOTE_SSH_HOST}" bash -s -- \
  "${REMOTE_REPO_ROOT}" \
  "${REMOTE_ARCHIVE}" \
  "${REMOTE_DATASET}" <<'REMOTE_EOF'
set -euo pipefail

remote_repo_root="$1"
remote_archive="$2"
remote_dataset="$3"

mkdir -p "${remote_repo_root}"
cd "${remote_repo_root}"
tar -xzf "${remote_archive}"
rm -f "${remote_archive}"

if [[ ! -f "${remote_dataset}/meta/info.json" ]]; then
  echo "Remote extraction finished, but metadata is missing: ${remote_dataset}/meta/info.json" >&2
  exit 1
fi

echo "Remote dataset ready: ${remote_dataset}"
REMOTE_EOF

echo
echo "Done. You can now train on the server:"
echo "  ssh ${REMOTE_SSH_HOST}"
echo "  cd ${REMOTE_REPO_ROOT}"
echo "  bash scripts/start_training_pi05.sh ${CONFIG_PATH}"
