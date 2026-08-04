#!/usr/bin/env bash
set -euo pipefail

# PI05 训练入口；与 start_training.sh 相同，默认会自动 *_ojag 改写 action。
#   bash scripts/start_training_pi05.sh configs/record_pick_cube_pi05.json
#   bash scripts/start_training_pi05.sh configs/foo.json --no-action-compose

CONFIG_PATH="${1:-configs/record_pick_cube_pi05.json}"
if [[ $# -gt 0 ]]; then
  shift
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

python -m piper_train.start_training --config "${CONFIG_PATH}" "$@"
