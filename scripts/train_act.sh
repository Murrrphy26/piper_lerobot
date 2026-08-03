#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

REPO_ID="${REPO_ID:-local/piper_pick_cube}"
ROOT="${ROOT:-data/lerobot}"
POLICY_TYPE="${POLICY_TYPE:-act}"
JOB_NAME="${JOB_NAME:-${POLICY_TYPE}_piper_pick_cube}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/train/${JOB_NAME}}"
POLICY_REPO_ID="${POLICY_REPO_ID:-local/${JOB_NAME}}"

DEVICE="${DEVICE:-cuda}"
STEPS="${STEPS:-5000}"
BATCH_SIZE="${BATCH_SIZE:-4}"
LOG_FREQ="${LOG_FREQ:-100}"
SAVE_FREQ="${SAVE_FREQ:-1000}"
WANDB_ENABLE="${WANDB_ENABLE:-false}"
VIDEO_BACKEND="${VIDEO_BACKEND:-pyav}"
PYTORCH_ALLOC_CONF_DEFAULT="${PYTORCH_ALLOC_CONF_DEFAULT:-expandable_segments:True}"
OUTCOME_FILTER="${OUTCOME_FILTER:-exclude-failures}"
INCLUDE_UNKNOWN="${INCLUDE_UNKNOWN:-false}"
EXCLUDE_UNLABELED="${EXCLUDE_UNLABELED:-true}"
EPISODES="${EPISODES:-}"

DATASET_ROOT="${DATASET_ROOT:-${ROOT}/${REPO_ID}}"

if [[ ! -f "${DATASET_ROOT}/meta/info.json" ]]; then
  echo "Dataset metadata not found: ${DATASET_ROOT}/meta/info.json" >&2
  echo "Set ROOT and REPO_ID, or set DATASET_ROOT to the directory that contains meta/info.json." >&2
  exit 1
fi

python - "${DATASET_ROOT}/meta/info.json" <<'PY'
import json
import sys
from pathlib import Path

info_path = Path(sys.argv[1])
info = json.loads(info_path.read_text(encoding="utf-8"))
features = info.get("features", {})

updated = False
for feature_name, feature in features.items():
    if not isinstance(feature, dict):
        continue
    if not feature_name.startswith("observation.images."):
        continue
    if feature.get("dtype") not in {"image", "video"}:
        continue
    if "names" in feature:
        continue

    shape = feature.get("shape")
    if isinstance(shape, list) and len(shape) == 3:
        feature["names"] = ["height", "width", "channels"]
        updated = True

if updated:
    info_path.write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    print(f"Patched dataset metadata: {info_path}")
PY

if ! find "${DATASET_ROOT}/data" -name "*.parquet" -print -quit >/dev/null 2>&1; then
  echo "No parquet files found under ${DATASET_ROOT}/data." >&2
  echo "LeRobot stores joint states/actions in parquet files; record or finalize the dataset first." >&2
  exit 1
fi

if ! command -v lerobot-train >/dev/null 2>&1; then
  echo "lerobot-train was not found in PATH." >&2
  echo "Activate your LeRobot Python environment first, then rerun this script." >&2
  exit 1
fi

echo "Training ${POLICY_TYPE} policy"
echo "  dataset: ${REPO_ID}"
echo "  dataset root: ${DATASET_ROOT}"
echo "  output: ${OUTPUT_DIR}"
echo "  device: ${DEVICE}"
echo "  steps: ${STEPS}"
echo "  batch size: ${BATCH_SIZE}"
echo "  video backend: ${VIDEO_BACKEND}"
echo
echo "If POLICY_TYPE=${POLICY_TYPE} uses a pretrained checkpoint, LeRobot may download it the first time."
echo "For ACT from scratch, the dataset videos/parquet are enough and no robot model is pulled."
echo "Training is finished when the command reaches STEPS=${STEPS} and writes checkpoints under ${OUTPUT_DIR}."
echo

if [[ -z "${PYTORCH_CUDA_ALLOC_CONF:-}" ]]; then
  export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_ALLOC_CONF_DEFAULT}"
fi

EPISODE_ARGS=()
if [[ -n "${EPISODES}" ]]; then
  EPISODE_ARGS=(--dataset.episodes="${EPISODES}")
else
  TRAINING_EPISODES="$(
    PYTHONPATH="${REPO_ROOT:-.}/src${PYTHONPATH:+:${PYTHONPATH}}" python - "${DATASET_ROOT}" "${OUTCOME_FILTER}" "${INCLUDE_UNKNOWN}" "${EXCLUDE_UNLABELED}" <<'PY'
import json
import sys
from pathlib import Path

from piper_towel_fold.episode_outcomes import resolve_training_episodes

dataset_root = Path(sys.argv[1])
training = {
    "outcome_filter": sys.argv[2],
    "include_unknown": sys.argv[3].lower() in {"1", "true", "yes"},
    "exclude_unlabeled": sys.argv[4].lower() not in {"0", "false", "no"},
}
episodes = resolve_training_episodes(dataset_root, training)
if episodes is not None:
    print(json.dumps(episodes))
PY
  )"
  if [[ -n "${TRAINING_EPISODES}" ]]; then
    EPISODE_ARGS=(--dataset.episodes="${TRAINING_EPISODES}")
    echo "Outcome filter: ${OUTCOME_FILTER}"
    echo "Training episodes: ${TRAINING_EPISODES}"
    echo
  fi
fi

lerobot-train \
  --dataset.repo_id="${REPO_ID}" \
  --dataset.root="${DATASET_ROOT}" \
  --policy.type="${POLICY_TYPE}" \
  --output_dir="${OUTPUT_DIR}" \
  --job_name="${JOB_NAME}" \
  --policy.device="${DEVICE}" \
  --dataset.video_backend="${VIDEO_BACKEND}" \
  --steps="${STEPS}" \
  --batch_size="${BATCH_SIZE}" \
  --log_freq="${LOG_FREQ}" \
  --save_freq="${SAVE_FREQ}" \
  --wandb.enable="${WANDB_ENABLE}" \
  --policy.repo_id="${POLICY_REPO_ID}" \
  --policy.push_to_hub=false \
  "${EPISODE_ARGS[@]}"
