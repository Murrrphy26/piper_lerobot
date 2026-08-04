#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from piper_train.offline_infer import get_column, get_episode_bounds, import_lerobot_dataset, load_dataset
from piper_train.preprocessing import (
    FramePreprocessor,
    load_preprocessing_config,
    numpy_image_from_sample,
    state_dict_to_vector,
    vector_to_state_dict,
)
from piper_train.recorder import ARM_STATE_KEYS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply offline preprocessing to a LeRobot dataset and write a new dataset.",
    )
    parser.add_argument(
        "--config",
        default="configs/record_pick_cube.json",
        help="JSON config that contains a preprocessing section.",
    )
    parser.add_argument("--source-repo-id", default="", help="Override source dataset repo id.")
    parser.add_argument("--source-root", default="", help="Override source dataset root directory.")
    parser.add_argument("--target-repo-id", default="", help="Target dataset repo id.")
    parser.add_argument("--target-root", default="", help="Target dataset root directory.")
    parser.add_argument(
        "--episodes",
        default="",
        help="Optional comma-separated episode indices to process. Default: all episodes.",
    )
    parser.add_argument("--video-backend", default="pyav", choices=("pyav", "torchcodec"))
    parser.add_argument(
        "--smoothing-method",
        default="savgol",
        choices=("savgol", "ema"),
        help="Smoothing method for offline batch processing.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print the plan without writing data.")
    return parser.parse_args()


def load_json_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as config_file:
        data = json.load(config_file)
    if not isinstance(data, dict):
        raise ValueError("Config file must contain a JSON object.")
    return data


def parse_episode_list(value: str) -> list[int] | None:
    if not value.strip():
        return None
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def tensor_to_list(values: Any) -> list[float]:
    try:
        import torch
    except ImportError:
        torch = None

    if torch is not None and isinstance(values, torch.Tensor):
        return [float(item) for item in values.detach().cpu().reshape(-1).tolist()]
    return [float(item) for item in values]


def image_keys_from_features(features: dict[str, Any]) -> list[str]:
    keys: list[str] = []
    for feature_name in features:
        if feature_name.startswith("observation.images."):
            keys.append(feature_name.removeprefix("observation.images."))
    return keys


def sample_to_frame(
    sample: dict[str, Any],
    camera_names: list[str],
    task: str,
) -> dict[str, Any]:
    frame: dict[str, Any] = {
        "observation.state": np.asarray(
            tensor_to_list(sample["observation.state"]),
            dtype=np.float32,
        ),
        "action": np.asarray(tensor_to_list(sample["action"]), dtype=np.float32),
        "task": str(sample.get("task", task)),
    }
    for camera_name in camera_names:
        image_key = f"observation.images.{camera_name}"
        frame[image_key] = numpy_image_from_sample(sample[image_key])
    return frame


def sample_to_observation_action(
    sample: dict[str, Any],
    camera_names: list[str],
) -> tuple[dict[str, Any], dict[str, float]]:
    observation = vector_to_state_dict(
        np.asarray(tensor_to_list(sample["observation.state"]), dtype=np.float32),
    )
    action = vector_to_state_dict(
        np.asarray(tensor_to_list(sample["action"]), dtype=np.float32),
    )
    for camera_name in camera_names:
        observation[camera_name] = numpy_image_from_sample(sample[f"observation.images.{camera_name}"])
    return observation, action


def observation_action_to_frame(
    observation: dict[str, Any],
    action: dict[str, float],
    camera_names: list[str],
    task: str,
) -> dict[str, Any]:
    frame: dict[str, Any] = {
        "observation.state": state_dict_to_vector(observation),
        "action": state_dict_to_vector(action),
        "task": task,
    }
    for camera_name in camera_names:
        frame[f"observation.images.{camera_name}"] = observation[camera_name]
    return frame


def create_target_dataset(
    source_dataset: Any,
    target_repo_id: str,
    target_root: Path,
    use_videos: bool,
) -> Any:
    dataset_cls = import_lerobot_dataset()
    info = source_dataset.meta.info
    features = dict(info["features"])
    if not use_videos:
        for feature_name, feature in features.items():
            if feature_name.startswith("observation.images.") and feature.get("dtype") == "video":
                feature = dict(feature)
                feature["dtype"] = "image"
                features[feature_name] = feature

    target_dataset_root = target_root / target_repo_id
    if target_dataset_root.exists():
        raise FileExistsError(
            f"Target dataset already exists: {target_dataset_root}. "
            "Choose a new --target-repo-id or remove the directory first."
        )

    create_kwargs = {
        "repo_id": target_repo_id,
        "fps": int(info["fps"]),
        "features": features,
        "robot_type": info.get("robot_type", "piper"),
        "root": target_dataset_root,
        "use_videos": use_videos,
    }
    try:
        return dataset_cls.create(**create_kwargs)
    except TypeError:
        create_kwargs = dict(create_kwargs)
        create_kwargs.pop("use_videos", None)
        return dataset_cls.create(**create_kwargs)


def copy_episode_outcomes(source_root: Path, target_root: Path, episode_indices: list[int] | None) -> None:
    source_path = source_root / "episode_outcomes.jsonl"
    if not source_path.exists():
        return

    if episode_indices is None:
        shutil.copy2(source_path, target_root / "episode_outcomes.jsonl")
        return

    selected = set(episode_indices)
    lines: list[str] = []
    for line in source_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        episode_index = record.get("episode_index")
        if episode_index is None or int(episode_index) in selected:
            lines.append(json.dumps(record, ensure_ascii=False))

    if lines:
        (target_root / "episode_outcomes.jsonl").write_text(
            "\n".join(lines) + "\n",
            encoding="utf-8",
        )


def main() -> None:
    args = parse_args()
    config = load_json_config(REPO_ROOT / args.config)
    preprocessing_config = load_preprocessing_config(config.get("preprocessing"))
    if not preprocessing_config.active:
        raise ValueError("Config preprocessing.enabled is false. Enable it before running this tool.")

    preprocessing_config.smoothing.method = args.smoothing_method

    source_repo_id = args.source_repo_id or str(config["repo_id"])
    source_root = Path(args.source_root or config.get("root", "data/lerobot"))
    target_repo_id = args.target_repo_id or f"{source_repo_id.split('/')[-1]}_pp"
    if "/" not in target_repo_id:
        target_repo_id = f"local/{target_repo_id}"
    target_root = Path(args.target_root or str(source_root))

    source_dataset_root = source_root / source_repo_id
    source_dataset = load_dataset(source_repo_id, str(source_dataset_root), args.video_backend)
    camera_names = image_keys_from_features(source_dataset.meta.features)
    episode_indices = parse_episode_list(args.episodes)
    if episode_indices is None:
        from_indices = get_column(source_dataset.meta.episodes, "dataset_from_index")
        if from_indices is None:
            raise KeyError("dataset.meta.episodes does not contain dataset_from_index.")
        episode_indices = list(range(len(from_indices)))

    print("Offline dataset preprocessing")
    print(f"  source: {source_dataset_root}")
    print(f"  target: {target_root / target_repo_id}")
    print(f"  episodes: {episode_indices}")
    print(f"  smoothing: {args.smoothing_method}")
    print(f"  images: {preprocessing_config.images.enabled}")
    print(f"  smoothing enabled: {preprocessing_config.smoothing.enabled}")
    print()

    if args.dry_run:
        return

    use_videos = any(
        feature.get("dtype") == "video"
        for key, feature in source_dataset.meta.features.items()
        if key.startswith("observation.images.")
    )
    target_dataset = create_target_dataset(
        source_dataset,
        target_repo_id=target_repo_id,
        target_root=target_root,
        use_videos=use_videos,
    )
    frame_preprocessor = FramePreprocessor(preprocessing_config, mode="batch")
    default_task = str(config.get("task", ""))

    try:
        for episode_index in episode_indices:
            start, stop = get_episode_bounds(source_dataset, episode_index)
            frame_preprocessor.reset()
            task = default_task

            for frame_index in range(start, stop):
                sample = dict(source_dataset[frame_index])
                if sample.get("task"):
                    task = str(sample["task"])
                observation, action = sample_to_observation_action(sample, camera_names)
                frame_preprocessor.process_frame(observation, action, camera_names)

            processed_frames = frame_preprocessor.flush_episode()
            if not processed_frames:
                continue

            for observation, action in processed_frames:
                frame = observation_action_to_frame(observation, action, camera_names, task)
                target_dataset.add_frame(frame)
            target_dataset.save_episode()
            print(f"processed episode {episode_index}: {len(processed_frames)} frames")

        finalize = getattr(target_dataset, "finalize", None)
        if callable(finalize):
            finalize()
    except Exception:
        target_dataset_root = target_root / target_repo_id
        if target_dataset_root.exists():
            shutil.rmtree(target_dataset_root)
        raise

    copy_episode_outcomes(source_dataset_root, target_root / target_repo_id, episode_indices)
    print()
    print(f"Wrote preprocessed dataset to {target_root / target_repo_id}")


if __name__ == "__main__":
    main()
