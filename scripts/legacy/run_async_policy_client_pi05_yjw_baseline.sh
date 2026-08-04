#!/usr/bin/env bash
# 工控机：pi05 async 客户端（改动前 baseline / 一键恢复入口）
# 对应备份：backups/pi05_smooth_20260729_102129
# 行为：fps=10、无 hold、原始 smoothing/speed/threshold
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${SCRIPT_DIR}/run_async_policy_client_pi05.sh" \
  configs/record_pick_cube_pi05_yjw_baseline.json \
  "$@"
