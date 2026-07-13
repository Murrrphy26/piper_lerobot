#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="${1:-configs/record_pick_cube_pi05.json}"
SKIP_BRINGUP="${SKIP_BRINGUP:-false}"
SKIP_RESET="${SKIP_RESET:-false}"
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
python -m piper_towel_fold.start_policy_live --config "${CONFIG_PATH}" "$@"
