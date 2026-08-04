#!/usr/bin/env python3
"""CLI wrapper: parse LeRobot training logs and judge loss convergence.

Usage:
  python tools/plot_train_loss.py --log train.log --plot loss.png
  python tools/plot_train_loss.py --log outputs/train/pi05_xxx/train.log
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from piper_train.train_loss import (
    analyze,
    parse_log_text,
    print_report,
    save_csv,
    save_plot,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot / diagnose LeRobot training loss from logs.")
    parser.add_argument(
        "--log",
        type=Path,
        help="Training log file (stdout from lerobot-train / tee). Reads stdin if omitted and piped.",
    )
    parser.add_argument("--log-freq", type=int, default=None, help="Override inferred log frequency.")
    parser.add_argument("--plot", type=Path, default=None, help="Optional PNG path for the loss curve.")
    parser.add_argument("--csv", type=Path, default=None, help="Optional CSV export path.")
    parser.add_argument("--smooth", type=int, default=5, help="Moving-average window for the plot.")
    parser.add_argument(
        "--tail-frac",
        type=float,
        default=0.3,
        help="Fraction of points treated as the late training window.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.log is not None:
        if not args.log.exists():
            raise FileNotFoundError(f"Log not found: {args.log}")
        text = args.log.read_text(encoding="utf-8", errors="replace")
    elif not sys.stdin.isatty():
        text = sys.stdin.read()
    else:
        raise SystemExit(
            "请提供 --log <训练日志文件>，或把日志通过管道传入。\n"
            "例: python tools/plot_train_loss.py --log train.log --plot loss.png"
        )

    points = parse_log_text(text, log_freq=args.log_freq)
    if len(points) < 3:
        raise SystemExit(
            f"只解析到 {len(points)} 个 loss 点，日志可能还不完整。"
            "确认日志里有类似 `loss:1.234` 的 LeRobot 训练行。"
        )

    summary = analyze(points, tail_frac=args.tail_frac)
    print_report(points, summary)

    if args.csv is not None:
        save_csv(points, args.csv)
        print(f"Wrote CSV: {args.csv}")
    if args.plot is not None:
        if save_plot(points, args.plot, smooth=max(1, args.smooth)):
            print(f"Wrote plot: {args.plot}")
        else:
            raise SystemExit("matplotlib 未安装。可: pip install matplotlib；或去掉 --plot。")


if __name__ == "__main__":
    main()
