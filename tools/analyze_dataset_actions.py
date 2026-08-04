import argparse
from typing import Any

import torch

from piper_train.offline_infer import get_episode_bounds, load_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize action ranges in a LeRobot dataset.")
    parser.add_argument("--repo-id", default="local/piper_pick_cube")
    parser.add_argument("--dataset-root", default="data/lerobot/local/piper_pick_cube")
    parser.add_argument("--video-backend", default="pyav", choices=("pyav", "torchcodec"))
    parser.add_argument("--episode-index", type=int, default=None)
    parser.add_argument("--stride", type=int, default=1, help="Read every Nth frame.")
    return parser.parse_args()


def action_values(action: Any) -> list[float]:
    if isinstance(action, torch.Tensor):
        return [float(value) for value in action.detach().cpu().reshape(-1).tolist()]
    return [float(value) for value in action]


def main() -> None:
    args = parse_args()
    if args.stride <= 0:
        raise ValueError("--stride must be greater than 0.")

    dataset = load_dataset(args.repo_id, args.dataset_root, args.video_backend)
    action_feature = dataset.meta.features["action"]
    action_names = list(action_feature["names"])

    if args.episode_index is None:
        start, stop = 0, len(dataset)
        label = "all episodes"
    else:
        start, stop = get_episode_bounds(dataset, args.episode_index)
        label = f"episode {args.episode_index}"

    mins = [float("inf")] * len(action_names)
    maxs = [float("-inf")] * len(action_names)
    first = None
    last = None
    count = 0

    for frame_index in range(start, stop, args.stride):
        values = action_values(dataset[frame_index]["action"])
        values = values[: len(action_names)]
        if first is None:
            first = values
        last = values
        for index, value in enumerate(values):
            mins[index] = min(mins[index], value)
            maxs[index] = max(maxs[index], value)
        count += 1

    if count == 0:
        raise RuntimeError("No frames were read.")

    print(f"dataset: {args.dataset_root}")
    print(f"range: {label}, frames read: {count}, stride: {args.stride}")
    print()
    print(f"{'name':28s} {'min':>11s} {'max':>11s} {'range':>11s} {'first':>11s} {'last':>11s}")
    print("-" * 80)
    for index, name in enumerate(action_names):
        min_value = mins[index]
        max_value = maxs[index]
        first_value = first[index] if first is not None else 0.0
        last_value = last[index] if last is not None else 0.0
        print(
            f"{name:28s} "
            f"{min_value:11.6f} "
            f"{max_value:11.6f} "
            f"{(max_value - min_value):11.6f} "
            f"{first_value:11.6f} "
            f"{last_value:11.6f}"
        )


if __name__ == "__main__":
    main()
