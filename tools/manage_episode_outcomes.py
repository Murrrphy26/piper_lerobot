#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from piper_towel_fold.episode_outcomes import (  # noqa: E402
    DEFAULT_OUTCOME_FILTER,
    select_episodes_for_training,
    summarize_outcomes,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect and export episode outcome labels.")
    parser.add_argument(
        "--dataset-root",
        required=True,
        help="Path to the dataset directory that contains meta/info.json.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    summary_parser = subparsers.add_parser("summary", help="Print outcome counts.")
    summary_parser.set_defaults(command="summary")

    list_parser = subparsers.add_parser("list", help="List episode index and outcome pairs.")
    list_parser.add_argument(
        "--outcome",
        choices=("success", "failure", "unknown", "unlabeled"),
        default=None,
        help="Only show episodes with the given outcome.",
    )
    list_parser.set_defaults(command="list")

    export_parser = subparsers.add_parser(
        "export-train-episodes",
        help="Print episode indices suitable for lerobot-train --dataset.episodes.",
    )
    export_parser.add_argument(
        "--mode",
        default=DEFAULT_OUTCOME_FILTER,
        choices=("exclude-failures", "success-only", "failures-only", "none"),
        help="Episode selection mode.",
    )
    export_parser.add_argument(
        "--include-unknown",
        action="store_true",
        help="Keep unknown episodes when mode=exclude-failures.",
    )
    export_parser.add_argument(
        "--include-unlabeled",
        action="store_true",
        help="Keep episodes without a label.",
    )
    export_parser.add_argument(
        "--format",
        choices=("json", "list"),
        default="json",
        help="Output format.",
    )
    export_parser.set_defaults(command="export-train-episodes")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_root = Path(args.dataset_root)

    if args.command == "summary":
        summary = summarize_outcomes(dataset_root)
        counts = summary["counts"]
        print(f"dataset: {summary['dataset_root']}")
        print(f"episodes: {summary['num_episodes']}")
        print(f"records: {summary['num_records']}")
        print(
            "counts: "
            f"success={counts['success']} "
            f"failure={counts['failure']} "
            f"unknown={counts['unknown']} "
            f"unlabeled={counts['unlabeled']}"
        )
        if summary["unlabeled"]:
            print(f"unlabeled indices: {summary['unlabeled']}")
        return

    if args.command == "list":
        summary = summarize_outcomes(dataset_root)
        for episode_index in range(summary["num_episodes"]):
            outcome = summary["aligned"].get(episode_index, "unlabeled")
            if args.outcome is not None and outcome != args.outcome:
                continue
            print(f"{episode_index}\t{outcome}")
        return

    if args.command == "export-train-episodes":
        episodes = select_episodes_for_training(
            dataset_root,
            outcome_filter=args.mode,
            include_unknown=args.include_unknown,
            exclude_unlabeled=not args.include_unlabeled,
        )
        if episodes is None:
            print("[]")
            return
        if args.format == "list":
            print(",".join(str(index) for index in episodes))
        else:
            print(json.dumps(episodes))
        return

    raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
