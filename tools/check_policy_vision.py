import argparse
from copy import deepcopy
from typing import Any

import torch

from piper_train.offline_infer import (
    action_tensor_to_dict,
    get_episode_bounds,
    load_dataset,
    load_policy,
    select_policy_inputs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check whether a trained policy reacts to image inputs.")
    parser.add_argument(
        "--policy-path",
        default="outputs/train/act_piper_pick_cube_fixed_v1/checkpoints/last/pretrained_model",
    )
    parser.add_argument("--repo-id", default="local/pick-fixed-v1")
    parser.add_argument("--dataset-root", default="data/lerobot/local/pick-fixed-v1")
    parser.add_argument("--video-backend", default="pyav", choices=("pyav", "torchcodec"))
    parser.add_argument("--episode-index", type=int, default=0)
    parser.add_argument("--frame-offset", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def feature_names(features: Any) -> list[str]:
    if isinstance(features, dict):
        return list(features.keys())
    return list(getattr(features, "keys", lambda: [])())


def image_keys(batch: dict[str, Any]) -> list[str]:
    return sorted(key for key in batch if key.startswith("observation.images."))


def zero_images(batch: dict[str, Any]) -> dict[str, Any]:
    changed = dict(batch)
    for key in image_keys(changed):
        changed[key] = torch.zeros_like(changed[key])
    return changed


def swap_top_left(batch: dict[str, Any]) -> dict[str, Any]:
    changed = dict(batch)
    top_key = "observation.images.cam_top"
    left_key = "observation.images.cam_left"
    if top_key in changed and left_key in changed:
        changed[top_key], changed[left_key] = changed[left_key], changed[top_key]
    return changed


def run_action(policy: Any, preprocessor: Any, postprocessor: Any, batch: dict[str, Any], action_names: list[str]) -> dict[str, float]:
    reset = getattr(policy, "reset", None)
    if callable(reset):
        reset()
    processed = preprocessor(batch)
    with torch.inference_mode():
        action = policy.select_action(processed)
        action = postprocessor(action)
    return action_tensor_to_dict(action, action_names)


def max_abs_delta(left: dict[str, float], right: dict[str, float]) -> float:
    return max(abs(left[key] - right[key]) for key in left.keys() & right.keys())


def print_action(label: str, action: dict[str, float]) -> None:
    right = ", ".join(f"rj{index}={action[f'right_joint_{index}.pos']:.4f}" for index in range(1, 7))
    print(f"{label}: {right}, rg={action['right_gripper.pos']:.5f}")


def main() -> None:
    args = parse_args()
    dataset = load_dataset(args.repo_id, args.dataset_root, args.video_backend)
    config, policy, make_pre_post_processors = load_policy(args.policy_path, args.device)
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=config,
        pretrained_path=args.policy_path,
        dataset_stats=getattr(dataset.meta, "stats", None),
        preprocessor_overrides={"device_processor": {"device": args.device}},
    )

    input_features = getattr(config, "input_features", {})
    output_features = getattr(config, "output_features", {})
    print(f"policy path: {args.policy_path}")
    print(f"dataset root: {args.dataset_root}")
    print(f"policy type: {getattr(config, 'type', '<unknown>')}")
    print(f"input features: {feature_names(input_features)}")
    print(f"output features: {feature_names(output_features)}")
    print()

    action_names = list(dataset.meta.features["action"]["names"])
    episode_start, _ = get_episode_bounds(dataset, args.episode_index)
    frame_index = episode_start + args.frame_offset
    sample = dict(dataset[frame_index])
    batch = select_policy_inputs(sample)
    images = image_keys(batch)
    print(f"frame index: {frame_index}")
    print(f"batch keys: {sorted(batch.keys())}")
    print(f"image keys: {images}")
    for key in images:
        value = batch[key]
        if isinstance(value, torch.Tensor):
            print(
                f"  {key}: shape={tuple(value.shape)} dtype={value.dtype} "
                f"min={float(value.min()):.4f} max={float(value.max()):.4f}"
            )
    print()

    original = run_action(policy, preprocessor, postprocessor, deepcopy(batch), action_names)
    black = run_action(policy, preprocessor, postprocessor, zero_images(deepcopy(batch)), action_names)
    swapped = run_action(policy, preprocessor, postprocessor, swap_top_left(deepcopy(batch)), action_names)

    print_action("original", original)
    print_action("black images", black)
    print_action("top/left swapped", swapped)
    print()
    print(f"max abs delta, original vs black: {max_abs_delta(original, black):.8f}")
    print(f"max abs delta, original vs swapped: {max_abs_delta(original, swapped):.8f}")
    print()
    print("Interpretation:")
    print("  near 0.0 means this action is almost image-invariant at this frame.")
    print("  a noticeable value means the policy is using image content or camera placement.")


if __name__ == "__main__":
    main()
