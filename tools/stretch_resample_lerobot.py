#!/usr/bin/env python3
"""LeRobot 数据集批量离线时序拉伸 + 重采样预处理。

【流水线（禁止简化/合并）】
  1) 以原始 timestamp 为唯一对齐基准
  2) 墙钟时长按 STRETCH_K 拉伸（等效慢速示教）
  3) CubicSpline 插值连续关节 (qpos/action)；夹爪 ZOH (kind="previous")
  4) 先生成稠密 source_fps(默认30) 中间序列，再均匀重采样至 target_fps(默认20)
  5) 图像禁止光流/RIFE；对每个输出时间戳映射回原始墙钟后就近取真实帧

【ACT 训练配套参数说明】（新数据集 fps=20，维持约 10s 时域视野）
  chunk_size: 200          # 200 / 20fps = 10s
  n_action_steps: 100      # 100 / 20fps = 5s 执行窗口
  dataset_fps: 20

默认输出仓库：local/cube_v723_stretch15_20fps
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from scipy.interpolate import CubicSpline, interp1d

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from piper_train.offline_infer import (  # noqa: E402
    get_column,
    get_episode_bounds,
    import_lerobot_dataset,
    load_dataset,
)
from piper_train.preprocessing import numpy_image_from_sample  # noqa: E402

# ---------------------------------------------------------------------------
# ACT 训练配套（脚本内注释标注；请在训练配置中同步设置）
# ---------------------------------------------------------------------------
# chunk_size: 200
# n_action_steps: 100
# dataset_fps: 20


# =============================================================================
# CLI
# =============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "批量时序拉伸+两段式插值重采样：稠密30fps→目标20fps，"
            "图像就近真实帧匹配，写出全新 LeRobot 数据集。"
        ),
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=REPO_ROOT / "data/lerobot/local/cube_v723",
        help="原始数据集路径（含 meta/info.json）。",
    )
    parser.add_argument(
        "--target-root",
        type=Path,
        default=REPO_ROOT / "data/lerobot/local/cube_v723_stretch15_20fps",
        help="输出数据集路径（全新仓库，不覆盖原始数据）。",
    )
    parser.add_argument(
        "--source-repo-id",
        default="local/cube_v723",
        help="加载 LeRobotDataset 用的 repo_id。",
    )
    parser.add_argument(
        "--target-repo-id",
        default="local/cube_v723_stretch15_20fps",
        help="新数据集 repo_id。",
    )
    parser.add_argument(
        "--stretch-k",
        type=float,
        default=1.5,
        help="时序拉伸系数 STRETCH_K（墙钟时长 ×K）。",
    )
    parser.add_argument(
        "--source-fps",
        type=float,
        default=30.0,
        help="稠密中间序列帧率（必须先插值到该帧率）。",
    )
    parser.add_argument(
        "--target-fps",
        type=float,
        default=20.0,
        help="最终输出帧率 TARGET_FPS。",
    )
    parser.add_argument(
        "--gripper-indices",
        default="6,13",
        help="夹爪离散通道下标（逗号分隔），对这些通道强制 ZOH。",
    )
    parser.add_argument(
        "--video-backend",
        default="pyav",
        choices=("pyav", "torchcodec"),
        help="读取原始视频帧的后端。",
    )
    parser.add_argument(
        "--export-vis",
        action="store_true",
        help="随机导出少量处理后 episode 的可视化校验视频。",
    )
    parser.add_argument(
        "--vis-num",
        type=int,
        default=3,
        help="可视化导出条数（需配合 --export-vis）。",
    )
    parser.add_argument(
        "--vis-dir",
        type=Path,
        default=None,
        help="可视化视频输出目录（默认 <target-root>/_vis_preview）。",
    )
    parser.add_argument("--seed", type=int, default=0, help="可视化抽样随机种子。")
    parser.add_argument(
        "--episodes",
        default="",
        help="可选：逗号分隔的 episode 下标；默认处理全部。",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只打印计划，不写数据。",
    )
    return parser.parse_args()


def parse_int_list(value: str) -> list[int]:
    if not value.strip():
        return []
    return [int(item.strip()) for item in value.split(",") if item.strip()]


# =============================================================================
# 基础工具
# =============================================================================


def to_numpy_1d(value: Any) -> np.ndarray:
    """将 sample 中的向量转为 float64 一维数组。"""
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    arr = np.asarray(value, dtype=np.float64).reshape(-1)
    return arr


def to_float_scalar(value: Any) -> float:
    arr = to_numpy_1d(value)
    if arr.size == 0:
        raise ValueError("Empty scalar value.")
    return float(arr[0])


def continuous_mask(dim: int, gripper_indices: list[int]) -> np.ndarray:
    mask = np.ones(dim, dtype=bool)
    for idx in gripper_indices:
        if idx < 0 or idx >= dim:
            raise ValueError(f"gripper index {idx} out of range for dim={dim}")
        mask[idx] = False
    return mask


def image_keys_from_features(features: dict[str, Any]) -> list[str]:
    keys: list[str] = []
    for feature_name in features:
        if feature_name.startswith("observation.images."):
            keys.append(feature_name.removeprefix("observation.images."))
    return keys


# =============================================================================
# 异常校验
# =============================================================================


class EpisodeValidationError(ValueError):
    """单条 episode 时序/数值异常，应跳过该集。"""


def validate_episode_timeseries(
    timestamps: np.ndarray,
    qpos: np.ndarray,
    action: np.ndarray,
) -> None:
    """过滤 NaN、长度不一致、timestamp 非严格递增等异常 episode。"""
    if timestamps.ndim != 1 or timestamps.size < 2:
        raise EpisodeValidationError(f"timestamp 长度不足: {timestamps.size}")
    if qpos.ndim != 2 or action.ndim != 2:
        raise EpisodeValidationError("qpos/action 必须是二维数组 [T, D]")
    if not (len(timestamps) == len(qpos) == len(action)):
        raise EpisodeValidationError(
            f"长度不一致: t={len(timestamps)}, qpos={len(qpos)}, action={len(action)}"
        )
    if np.isnan(timestamps).any() or np.isnan(qpos).any() or np.isnan(action).any():
        raise EpisodeValidationError("存在 NaN")
    if np.isinf(timestamps).any() or np.isinf(qpos).any() or np.isinf(action).any():
        raise EpisodeValidationError("存在 Inf")
    dt = np.diff(timestamps)
    if not np.all(dt > 0):
        raise EpisodeValidationError("timestamp 未严格递增（时序异常）")


# =============================================================================
# 核心时序层：拉伸 → 稠密30fps插值 → 均匀重采样至20fps
# =============================================================================


def stretch_timestamps(timestamps: np.ndarray, stretch_k: float) -> np.ndarray:
    """墙钟拉伸：t' = t * STRETCH_K。

    风险点：只改时间轴、不插值信号，会把同一关节位移摊到更长时间，
    但中间时刻没有合法状态；后续必须对 qpos/action 做插值。
    """
    if stretch_k <= 0:
        raise ValueError(f"stretch_k must be > 0, got {stretch_k}")
    return timestamps.astype(np.float64) * float(stretch_k)


def build_uniform_timeline(t_start: float, t_end: float, fps: float) -> np.ndarray:
    """在 [t_start, t_end] 上生成均匀 fps 时间轴（含两端）。"""
    if t_end < t_start:
        raise ValueError(f"invalid timeline range: [{t_start}, {t_end}]")
    if fps <= 0:
        raise ValueError(f"fps must be > 0, got {fps}")
    duration = t_end - t_start
    n_steps = int(np.floor(duration * fps + 1e-9))
    n = n_steps + 1
    if n < 2:
        return np.asarray([t_start, t_end], dtype=np.float64)
    return t_start + np.arange(n, dtype=np.float64) / float(fps)


def interpolate_continuous(t_src: np.ndarray, y_src: np.ndarray, t_query: np.ndarray) -> np.ndarray:
    """连续关节：CubicSpline 插值。

    风险点：端点外禁止外推炸值，查询时间先 clip 到 [t_src[0], t_src[-1]]。
    """
    if y_src.ndim == 1:
        y_src = y_src[:, None]
    t_clip = np.clip(t_query, t_src[0], t_src[-1])
    out = np.empty((len(t_query), y_src.shape[1]), dtype=np.float64)
    for d in range(y_src.shape[1]):
        spline = CubicSpline(t_src, y_src[:, d], bc_type="natural")
        out[:, d] = spline(t_clip)
    return out


def interpolate_gripper_zoh(t_src: np.ndarray, g_src: np.ndarray, t_query: np.ndarray) -> np.ndarray:
    """夹爪离散/保持段：零阶保持 interp1d(kind='previous')。

    风险点：禁止对夹爪做 CubicSpline，否则保持不动阶段会出现非法中间开合值。
    遥操作中夹爪常有长保持段，必须 ZOH。
    """
    if g_src.ndim == 1:
        g_src = g_src[:, None]
    t_clip = np.clip(t_query, t_src[0], t_src[-1])
    out = np.empty((len(t_query), g_src.shape[1]), dtype=np.float64)
    for d in range(g_src.shape[1]):
        # kind="previous"：查询时刻取「不晚于该时刻的最近源样本」
        fn = interp1d(
            t_src,
            g_src[:, d],
            kind="previous",
            bounds_error=False,
            fill_value=(g_src[0, d], g_src[-1, d]),
            assume_sorted=True,
        )
        out[:, d] = fn(t_clip)
    return out


def interpolate_vector_series(
    t_src: np.ndarray,
    y_src: np.ndarray,
    t_query: np.ndarray,
    gripper_indices: list[int],
) -> np.ndarray:
    """按通道类型分别插值：连续关节样条，夹爪 ZOH，再拼回完整向量。"""
    dim = y_src.shape[1]
    mask = continuous_mask(dim, gripper_indices)
    out = np.zeros((len(t_query), dim), dtype=np.float64)

    cont_idx = np.where(mask)[0]
    grip_idx = np.where(~mask)[0]
    if cont_idx.size:
        out[:, cont_idx] = interpolate_continuous(t_src, y_src[:, cont_idx], t_query)
    if grip_idx.size:
        out[:, grip_idx] = interpolate_gripper_zoh(t_src, y_src[:, grip_idx], t_query)
    return out


@dataclass
class ResampledEpisode:
    """处理后的 episode 时序结果（最终 target_fps）。"""

    timestamps: np.ndarray  # 拉伸后墙钟上的输出时间轴
    qpos: np.ndarray
    action: np.ndarray
    dense_timestamps: np.ndarray  # 中间稠密 30fps 时间轴（质检/调试）
    dense_qpos: np.ndarray
    dense_action: np.ndarray


def process_episode_signals(
    timestamps_raw: np.ndarray,
    qpos_raw: np.ndarray,
    action_raw: np.ndarray,
    stretch_k: float,
    source_fps: float,
    target_fps: float,
    gripper_indices: list[int],
) -> ResampledEpisode:
    """严格两段式时序处理（不可合并为一步）。

    段1：在拉伸后时间轴上插值 → 稠密 source_fps 序列
    段2：仅从稠密序列均匀重采样 → target_fps（不再回看原始样本）
    """
    validate_episode_timeseries(timestamps_raw, qpos_raw, action_raw)

    # --- 时间轴拉伸 ---
    t_stretched = stretch_timestamps(timestamps_raw, stretch_k)

    # ========== 段1：插值生成稠密 source_fps 中间序列 ==========
    # 风险点：此步必须以 t_stretched 为源时间；禁止用帧号抽帧代替插值。
    t_dense = build_uniform_timeline(float(t_stretched[0]), float(t_stretched[-1]), source_fps)
    qpos_dense = interpolate_vector_series(t_stretched, qpos_raw, t_dense, gripper_indices)
    action_dense = interpolate_vector_series(t_stretched, action_raw, t_dense, gripper_indices)

    # ========== 段2：从稠密序列均匀重采样至 target_fps ==========
    # 风险点：段2的插值源必须是 dense_*，禁止直接对原始 (t,y) 一步插到 target_fps。
    t_out = build_uniform_timeline(float(t_dense[0]), float(t_dense[-1]), target_fps)
    qpos_out = interpolate_vector_series(t_dense, qpos_dense, t_out, gripper_indices)
    action_out = interpolate_vector_series(t_dense, action_dense, t_out, gripper_indices)

    return ResampledEpisode(
        timestamps=t_out,
        qpos=qpos_out,
        action=action_out,
        dense_timestamps=t_dense,
        dense_qpos=qpos_dense,
        dense_action=action_dense,
    )


# =============================================================================
# 图像匹配：就近真实帧（禁止光流/RIFE）
# =============================================================================


def map_output_time_to_original(t_out: float, stretch_k: float) -> float:
    """将拉伸后输出时刻映射回原始墙钟。

    风险点：若误用 t_out 直接在原始视频上查找，会系统性错位；
    正确关系为 t_lookup = t_out / STRETCH_K。
    """
    return float(t_out) / float(stretch_k)


def nearest_frame_index(t_lookup: float, frame_timestamps: np.ndarray) -> int:
    """基于 timestamp 就近匹配；禁止用输出帧号去对齐原始帧号。"""
    return int(np.argmin(np.abs(frame_timestamps - t_lookup)))


def build_output_to_source_indices(
    t_out: np.ndarray,
    timestamps_raw: np.ndarray,
    stretch_k: float,
) -> np.ndarray:
    """为每个输出时刻计算原始 episode 内帧下标。"""
    indices = np.empty(len(t_out), dtype=np.int64)
    for i, t in enumerate(t_out):
        t_lookup = map_output_time_to_original(float(t), stretch_k)
        indices[i] = nearest_frame_index(t_lookup, timestamps_raw)
    return indices


# =============================================================================
# 质检：关节速度统计
# =============================================================================


def joint_angular_speeds(timestamps: np.ndarray, qpos: np.ndarray, joint_mask: np.ndarray) -> np.ndarray:
    """计算每步、每关节的 |Δq/Δt|，返回展平速度样本。"""
    if len(timestamps) < 2:
        return np.asarray([], dtype=np.float64)
    dt = np.diff(timestamps)
    dq = np.diff(qpos[:, joint_mask], axis=0)
    # 避免除零
    dt = np.maximum(dt, 1e-9)[:, None]
    speeds = np.abs(dq / dt)
    return speeds.reshape(-1)


def compute_joint_speed_stats(
    timestamps: np.ndarray,
    qpos: np.ndarray,
    gripper_indices: list[int],
) -> dict[str, float]:
    mask = continuous_mask(qpos.shape[1], gripper_indices)
    speeds = joint_angular_speeds(timestamps, qpos, mask)
    if speeds.size == 0:
        return {"mean_abs_speed": float("nan"), "p95_abs_speed": float("nan")}
    return {
        "mean_abs_speed": float(np.mean(speeds)),
        "p95_abs_speed": float(np.percentile(speeds, 95)),
    }


def log_episode_qc(ep_idx: int, before: dict[str, float], after: dict[str, float], n_out: int) -> None:
    print(
        f"[QC] episode {ep_idx}: "
        f"raw_mean_joint_speed={before['mean_abs_speed']:.6f} rad/s, "
        f"out_p95_joint_speed={after['p95_abs_speed']:.6f} rad/s, "
        f"out_frames={n_out}"
    )


# =============================================================================
# Episode 读写
# =============================================================================


@dataclass
class RawEpisode:
    episode_index: int
    global_start: int
    global_stop: int
    timestamps: np.ndarray
    qpos: np.ndarray
    action: np.ndarray
    task: str


def load_raw_episode(dataset: Any, episode_index: int) -> RawEpisode:
    """从 LeRobot 数据集加载单集 parquet 时序（图像按需稍后取）。"""
    start, stop = get_episode_bounds(dataset, episode_index)
    n = stop - start
    if n <= 0:
        raise EpisodeValidationError(f"empty episode {episode_index}")

    timestamps = np.empty(n, dtype=np.float64)
    qpos = np.empty((n, 0), dtype=np.float64)
    action = np.empty((n, 0), dtype=np.float64)
    task = ""

    for i, frame_index in enumerate(range(start, stop)):
        sample = dict(dataset[frame_index])
        timestamps[i] = to_float_scalar(sample["timestamp"])
        q = to_numpy_1d(sample["observation.state"])
        a = to_numpy_1d(sample["action"])
        if i == 0:
            qpos = np.empty((n, q.size), dtype=np.float64)
            action = np.empty((n, a.size), dtype=np.float64)
        qpos[i] = q
        action[i] = a
        if sample.get("task"):
            task = str(sample["task"])

    validate_episode_timeseries(timestamps, qpos, action)
    return RawEpisode(
        episode_index=episode_index,
        global_start=start,
        global_stop=stop,
        timestamps=timestamps,
        qpos=qpos,
        action=action,
        task=task,
    )


def fetch_camera_images(
    dataset: Any,
    global_frame_index: int,
    camera_names: list[str],
    cache: dict[int, dict[str, np.ndarray]],
) -> dict[str, np.ndarray]:
    """按全局帧下标取真实图像；带缓存避免重复解码。"""
    if global_frame_index in cache:
        return cache[global_frame_index]
    sample = dict(dataset[global_frame_index])
    images: dict[str, np.ndarray] = {}
    for name in camera_names:
        key = f"observation.images.{name}"
        images[name] = numpy_image_from_sample(sample[key])
    cache[global_frame_index] = images
    return images


def create_target_dataset(
    source_dataset: Any,
    target_repo_id: str,
    target_root: Path,
    target_fps: float,
) -> Any:
    """创建全新输出仓库；强制 fps=target_fps，并同步修正视频元信息中的 fps。"""
    dataset_cls = import_lerobot_dataset()
    info = source_dataset.meta.info
    features = dict(info["features"])

    # 同步修正各相机 video.fps，避免 info 仍写 30
    for feature_name, feature in list(features.items()):
        if not feature_name.startswith("observation.images."):
            continue
        if not isinstance(feature, dict):
            continue
        feature = dict(feature)
        meta_info = feature.get("info")
        if isinstance(meta_info, dict):
            meta_info = dict(meta_info)
            meta_info["video.fps"] = float(target_fps)
            feature["info"] = meta_info
        features[feature_name] = feature

    if target_root.exists():
        raise FileExistsError(
            f"Target dataset already exists: {target_root}. "
            "请更换 --target-root，或先删除该目录。"
        )
    target_root.parent.mkdir(parents=True, exist_ok=True)

    create_kwargs = {
        "repo_id": target_repo_id,
        "fps": int(round(target_fps)),
        "features": features,
        "robot_type": info.get("robot_type", "piper"),
        "root": target_root,
        "use_videos": True,
    }
    try:
        return dataset_cls.create(**create_kwargs)
    except TypeError:
        create_kwargs = dict(create_kwargs)
        create_kwargs.pop("use_videos", None)
        return dataset_cls.create(**create_kwargs)


def finalize_info_json(target_root: Path, target_fps: float) -> None:
    """写完数据后再次确认 info.json 的 fps / 视频时长相关字段。"""
    info_path = target_root / "meta" / "info.json"
    if not info_path.exists():
        return
    info = json.loads(info_path.read_text(encoding="utf-8"))
    info["fps"] = int(round(target_fps))
    features = info.get("features", {})
    if isinstance(features, dict):
        for feature_name, feature in features.items():
            if not feature_name.startswith("observation.images."):
                continue
            if not isinstance(feature, dict):
                continue
            meta_info = feature.get("info")
            if isinstance(meta_info, dict):
                meta_info["video.fps"] = float(target_fps)
    info_path.write_text(json.dumps(info, indent=4, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[meta] updated {info_path} -> fps={info['fps']}")


def copy_episode_outcomes(
    source_root: Path,
    target_root: Path,
    old_to_new: dict[int, int],
) -> None:
    """按成功写出的 episode 映射拷贝 outcome 标签。"""
    source_path = source_root / "episode_outcomes.jsonl"
    if not source_path.exists() or not old_to_new:
        return
    lines: list[str] = []
    for line in source_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        old_idx = record.get("episode_index")
        if old_idx is None:
            continue
        old_idx = int(old_idx)
        if old_idx not in old_to_new:
            continue
        record = dict(record)
        record["episode_index"] = old_to_new[old_idx]
        record["source_episode_index"] = old_idx
        lines.append(json.dumps(record, ensure_ascii=False))
    if lines:
        (target_root / "episode_outcomes.jsonl").write_text(
            "\n".join(lines) + "\n",
            encoding="utf-8",
        )


# =============================================================================
# 可视化校验视频
# =============================================================================


def _tile_cameras(images: dict[str, np.ndarray], camera_names: list[str]) -> np.ndarray:
    panels: list[np.ndarray] = []
    for name in camera_names:
        img = images[name]
        if img.ndim == 3 and img.shape[2] == 3:
            # LeRobot 图像为 RGB；OpenCV 写视频要 BGR
            bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        else:
            bgr = img
        captioned = bgr.copy()
        cv2.putText(
            captioned,
            name,
            (8, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        panels.append(captioned)
    if not panels:
        raise ValueError("no camera images to tile")
    # 横向拼接；高度不一致时统一到最小高
    h = min(p.shape[0] for p in panels)
    resized = [cv2.resize(p, (int(p.shape[1] * h / p.shape[0]), h)) for p in panels]
    return np.concatenate(resized, axis=1)


def export_sync_preview_video(
    frames_rgb_by_cam: list[dict[str, np.ndarray]],
    qpos: np.ndarray,
    action: np.ndarray,
    camera_names: list[str],
    out_path: Path,
    fps: float,
) -> None:
    """导出多相机拼图 + 底部动作条，供人工检查图像/动作是否错位。"""
    if not frames_rgb_by_cam:
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    first = _tile_cameras(frames_rgb_by_cam[0], camera_names)
    # 底部预留曲线条带
    band_h = 120
    height = first.shape[0] + band_h
    width = first.shape[1]
    writer = cv2.VideoWriter(
        str(out_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps),
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"failed to open VideoWriter: {out_path}")

    # 用右臂关节 0 与右夹爪做简易曲线
    joint_trace = qpos[:, 7] if qpos.shape[1] > 7 else qpos[:, 0]
    grip_trace = qpos[:, 13] if qpos.shape[1] > 13 else qpos[:, -1]
    j_min, j_max = float(joint_trace.min()), float(joint_trace.max())
    g_min, g_max = float(grip_trace.min()), float(grip_trace.max())
    j_span = max(j_max - j_min, 1e-6)
    g_span = max(g_max - g_min, 1e-6)

    for i, images in enumerate(frames_rgb_by_cam):
        canvas = np.zeros((height, width, 3), dtype=np.uint8)
        tile = _tile_cameras(images, camera_names)
        canvas[: tile.shape[0], : tile.shape[1]] = tile
        # 绘制历史曲线
        band = canvas[tile.shape[0] :, :]
        band[:] = (30, 30, 30)
        xs = np.linspace(0, width - 1, num=i + 1).astype(int)
        for x, jv, gv in zip(xs, joint_trace[: i + 1], grip_trace[: i + 1]):
            yj = int((1.0 - (jv - j_min) / j_span) * (band_h - 10)) + 5
            yg = int((1.0 - (gv - g_min) / g_span) * (band_h - 10)) + 5
            cv2.circle(band, (int(x), yj), 1, (255, 180, 0), -1)
            cv2.circle(band, (int(x), yg), 1, (0, 200, 255), -1)
        cv2.putText(
            band,
            f"frame={i}  joint(orange) gripper(cyan)  |action0|={abs(float(action[i, 0])):.3f}",
            (8, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (220, 220, 220),
            1,
            cv2.LINE_AA,
        )
        writer.write(canvas)
    writer.release()
    print(f"[vis] wrote {out_path}")


# =============================================================================
# 单集处理 + 批量编排
# =============================================================================


@dataclass
class ProcessedEpisodeBundle:
    source_episode_index: int
    resampled: ResampledEpisode
    source_frame_indices: np.ndarray  # episode 内下标
    task: str
    qc_before: dict[str, float]
    qc_after: dict[str, float]


def safe_process_episode(
    dataset: Any,
    episode_index: int,
    stretch_k: float,
    source_fps: float,
    target_fps: float,
    gripper_indices: list[int],
) -> ProcessedEpisodeBundle | None:
    """异常捕获：失败则返回 None 并打印原因。"""
    try:
        raw = load_raw_episode(dataset, episode_index)
        qc_before = compute_joint_speed_stats(raw.timestamps, raw.qpos, gripper_indices)
        resampled = process_episode_signals(
            timestamps_raw=raw.timestamps,
            qpos_raw=raw.qpos,
            action_raw=raw.action,
            stretch_k=stretch_k,
            source_fps=source_fps,
            target_fps=target_fps,
            gripper_indices=gripper_indices,
        )
        # 二次 NaN 检查（插值后）
        if (
            np.isnan(resampled.qpos).any()
            or np.isnan(resampled.action).any()
            or np.isnan(resampled.timestamps).any()
        ):
            raise EpisodeValidationError("插值后出现 NaN")

        src_indices = build_output_to_source_indices(
            resampled.timestamps,
            raw.timestamps,
            stretch_k,
        )
        qc_after = compute_joint_speed_stats(
            resampled.timestamps,
            resampled.qpos,
            gripper_indices,
        )
        log_episode_qc(episode_index, qc_before, qc_after, len(resampled.timestamps))
        return ProcessedEpisodeBundle(
            source_episode_index=episode_index,
            resampled=resampled,
            source_frame_indices=src_indices,
            task=raw.task,
            qc_before=qc_before,
            qc_after=qc_after,
        )
    except EpisodeValidationError as exc:
        print(f"[SKIP] episode {episode_index}: {exc}")
        return None
    except Exception as exc:  # noqa: BLE001 - 批量任务需吞掉单集异常
        print(f"[SKIP] episode {episode_index}: unexpected error: {exc}")
        traceback.print_exc()
        return None


def write_processed_episode(
    target_dataset: Any,
    source_dataset: Any,
    bundle: ProcessedEpisodeBundle,
    camera_names: list[str],
    collect_vis: bool,
) -> list[dict[str, np.ndarray]] | None:
    """把处理后的帧写入目标数据集；可选收集可视化帧。"""
    start, _ = get_episode_bounds(source_dataset, bundle.source_episode_index)
    cache: dict[int, dict[str, np.ndarray]] = {}
    vis_frames: list[dict[str, np.ndarray]] | None = [] if collect_vis else None

    for i in range(len(bundle.resampled.timestamps)):
        local_idx = int(bundle.source_frame_indices[i])
        global_idx = start + local_idx
        images = fetch_camera_images(source_dataset, global_idx, camera_names, cache)

        frame: dict[str, Any] = {
            "observation.state": bundle.resampled.qpos[i].astype(np.float32),
            "action": bundle.resampled.action[i].astype(np.float32),
            "task": bundle.task,
        }
        for name in camera_names:
            frame[f"observation.images.{name}"] = images[name]
        target_dataset.add_frame(frame)

        if vis_frames is not None:
            vis_frames.append({name: images[name].copy() for name in camera_names})

    target_dataset.save_episode()
    return vis_frames


def process_all_episodes(args: argparse.Namespace) -> None:
    source_root = args.source_root.resolve()
    target_root = args.target_root.resolve()
    gripper_indices = parse_int_list(args.gripper_indices)
    episode_filter = parse_int_list(args.episodes)

    if not (source_root / "meta" / "info.json").exists():
        raise FileNotFoundError(f"找不到原始数据集: {source_root / 'meta' / 'info.json'}")

    source_dataset = load_dataset(
        repo_id=args.source_repo_id,
        dataset_root=str(source_root),
        video_backend=args.video_backend,
    )
    camera_names = image_keys_from_features(source_dataset.meta.features)
    from_indices = get_column(source_dataset.meta.episodes, "dataset_from_index")
    if from_indices is None:
        raise KeyError("dataset.meta.episodes 缺少 dataset_from_index")
    all_episodes = list(range(len(from_indices)))
    episode_indices = episode_filter if episode_filter else all_episodes

    print("=" * 72)
    print("LeRobot 时序拉伸 + 两段式重采样")
    print(f"  source:      {source_root}")
    print(f"  target:      {target_root}")
    print(f"  stretch_k:   {args.stretch_k}")
    print(f"  dense_fps:   {args.source_fps}  (中间序列)")
    print(f"  target_fps:  {args.target_fps}")
    print(f"  cameras:     {camera_names}")
    print(f"  gripper_idx: {gripper_indices}")
    print(f"  episodes:    {episode_indices}")
    print("  ACT hint:    chunk_size=200, n_action_steps=100, dataset_fps=20")
    print("=" * 72)

    if args.dry_run:
        print("dry-run: 退出（未写数据）")
        return

    # 可视化抽样集合
    vis_candidates = list(episode_indices)
    random.Random(args.seed).shuffle(vis_candidates)
    vis_selected = set(vis_candidates[: max(0, args.vis_num)]) if args.export_vis else set()
    vis_dir = args.vis_dir or (target_root / "_vis_preview")

    target_dataset = create_target_dataset(
        source_dataset=source_dataset,
        target_repo_id=args.target_repo_id,
        target_root=target_root,
        target_fps=args.target_fps,
    )

    old_to_new: dict[int, int] = {}
    n_ok = 0
    n_skip = 0

    try:
        for ep_idx in episode_indices:
            bundle = safe_process_episode(
                dataset=source_dataset,
                episode_index=ep_idx,
                stretch_k=args.stretch_k,
                source_fps=args.source_fps,
                target_fps=args.target_fps,
                gripper_indices=gripper_indices,
            )
            if bundle is None:
                n_skip += 1
                continue

            collect_vis = args.export_vis and ep_idx in vis_selected
            vis_frames = write_processed_episode(
                target_dataset=target_dataset,
                source_dataset=source_dataset,
                bundle=bundle,
                camera_names=camera_names,
                collect_vis=collect_vis,
            )
            new_idx = n_ok
            old_to_new[ep_idx] = new_idx
            n_ok += 1
            print(
                f"[OK] source_ep={ep_idx} -> new_ep={new_idx}, "
                f"frames={len(bundle.resampled.timestamps)} "
                f"(dense={len(bundle.resampled.dense_timestamps)})"
            )

            if vis_frames:
                export_sync_preview_video(
                    frames_rgb_by_cam=vis_frames,
                    qpos=bundle.resampled.qpos,
                    action=bundle.resampled.action,
                    camera_names=camera_names,
                    out_path=vis_dir / f"preview_src{ep_idx:03d}_new{new_idx:03d}.mp4",
                    fps=args.target_fps,
                )

        finalize = getattr(target_dataset, "finalize", None)
        if callable(finalize):
            finalize()
    except Exception:
        if target_root.exists():
            shutil.rmtree(target_root)
        raise

    finalize_info_json(target_root, args.target_fps)
    copy_episode_outcomes(source_root, target_root, old_to_new)

    print()
    print(f"完成: 成功 {n_ok} 集, 跳过 {n_skip} 集")
    print(f"输出目录: {target_root}")
    if args.export_vis:
        print(f"可视化目录: {vis_dir}")


def main() -> None:
    args = parse_args()
    if args.stretch_k <= 0:
        raise SystemExit("--stretch-k must be > 0")
    if args.source_fps <= 0 or args.target_fps <= 0:
        raise SystemExit("--source-fps / --target-fps must be > 0")
    process_all_episodes(args)


if __name__ == "__main__":
    main()
