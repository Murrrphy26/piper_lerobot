#!/usr/bin/env bash
# 工控机：pi05 async 客户端（当前改动版 / smooth）
# 默认配置：fps=20、hold 空队列上一拍、以及你后续手调的 control 参数
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${SCRIPT_DIR}/run_async_policy_client_pi05.sh" \
  configs/record_pick_cube_pi05_yjw.json \
  "$@"
