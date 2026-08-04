#!/usr/bin/env python3
"""读取 LeRobot parquet 数据中夹爪（left/right_gripper.pos）的数值范围。

数据集中夹爪单位为米（Piper 官方行程约 [0, 0.07] m），同时换算成毫米打印。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# observation.state / action 的 14 维布局（见 meta/info.json）
LEFT_GRIPPER_IDX = 6
RIGHT_GRIPPER_IDX = 13


def load_gripper_arrays(chunk_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """返回 (state_left, state_right, action_left, action_right)。"""
    parquet_files = sorted(chunk_dir.glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"未找到 parquet 文件: {chunk_dir}")

    state_left, state_right = [], []
    action_left, action_right = [], []

    for path in parquet_files:
        df = pd.read_parquet(path, columns=["observation.state", "action"])
        state = np.stack(df["observation.state"].to_numpy())
        action = np.stack(df["action"].to_numpy())
        state_left.append(state[:, LEFT_GRIPPER_IDX])
        state_right.append(state[:, RIGHT_GRIPPER_IDX])
        action_left.append(action[:, LEFT_GRIPPER_IDX])
        action_right.append(action[:, RIGHT_GRIPPER_IDX])
        print(f"  {path.name}: frames={len(df)}")

    return (
        np.concatenate(state_left),
        np.concatenate(state_right),
        np.concatenate(action_left),
        np.concatenate(action_right),
    )


def print_range(name: str, values: np.ndarray) -> None:
    vmin, vmax = float(values.min()), float(values.max())
    print(f"{name}:")
    print(f"  min = {vmin:.6f} m  ({vmin * 1000:.3f} mm)")
    print(f"  max = {vmax:.6f} m  ({vmax * 1000:.3f} mm)")
    print(f"  range = [{vmin:.6f}, {vmax:.6f}] m")
    print(f"  mean = {float(values.mean()):.6f} m, std = {float(values.std()):.6f} m")
    print(f"  frames = {len(values)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="读取 chunk 目录下夹爪开合范围")
    parser.add_argument(
        "--chunk-dir",
        type=Path,
        default=Path(
            "/home/agilex/code/yjw/piper/data/lerobot/local/cube_pi05_v2/data/chunk-000"
        ),
        help="包含 file-*.parquet 的 chunk 目录",
    )
    args = parser.parse_args()

    chunk_dir = args.chunk_dir.resolve()
    if not chunk_dir.is_dir():
        raise SystemExit(f"目录不存在: {chunk_dir}")

    print(f"读取目录: {chunk_dir}")
    state_l, state_r, action_l, action_r = load_gripper_arrays(chunk_dir)

    print("\n" + "=" * 60)
    print("夹爪开合范围（单位：米 / 毫米）")
    print("=" * 60)
    print_range("observation.state  left_gripper.pos", state_l)
    print_range("observation.state  right_gripper.pos", state_r)
    print_range("action            left_gripper.pos", action_l)
    print_range("action            right_gripper.pos", action_r)

    # 合并 state+action、左右侧的全局范围
    all_vals = np.concatenate([state_l, state_r, action_l, action_r])
    print("-" * 60)
    print_range("全局（左右 + state/action）", all_vals)


if __name__ == "__main__":
    main()
