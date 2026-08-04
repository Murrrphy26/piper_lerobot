#!/usr/bin/env python3
"""Find RGB-like /dev/video devices and save one screenshot per device.

The robot PC may enumerate cameras differently after reboot. This script scans
video nodes, attempts to capture a frame from each one, and saves screenshots for
devices that look like usable RGB/color cameras.

It intentionally does not assume Orbbec node offsets such as "base + 6".
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import subprocess
import time
from typing import Any

import cv2
import numpy as np


def list_video_devices() -> list[Path]:
    devices: list[Path] = []
    for path in Path("/dev").glob("video*"):
        match = re.fullmatch(r"video(\d+)", path.name)
        if match:
            devices.append(path)
    return sorted(devices, key=lambda item: int(item.name.replace("video", "")))


def run_text(command: list[str]) -> str:
    try:
        return subprocess.check_output(command, text=True, stderr=subprocess.STDOUT, timeout=3)
    except Exception:
        return ""


def device_metadata(device: Path) -> dict[str, Any]:
    udev = run_text(["udevadm", "info", "-q", "property", "-n", str(device)])
    v4l2_all = run_text(["v4l2-ctl", "-d", str(device), "--all"])
    formats = run_text(["v4l2-ctl", "-d", str(device), "--list-formats-ext"])

    props: dict[str, str] = {}
    for line in udev.splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            props[key] = value

    card = ""
    bus = ""
    for line in v4l2_all.splitlines():
        stripped = line.strip()
        if stripped.startswith("Card type"):
            card = stripped.split(":", 1)[-1].strip()
        elif stripped.startswith("Bus info"):
            bus = stripped.split(":", 1)[-1].strip()

    format_hits = []
    for token in ("MJPG", "YUYV", "RGB", "BGR", "NV12", "YUY2"):
        if token.lower() in formats.lower():
            format_hits.append(token)

    return {
        "udev": props,
        "card": card,
        "bus": bus,
        "format_hits": format_hits,
        "formats_raw": formats,
    }


def capture_frame(
    device: Path,
    *,
    width: int,
    height: int,
    warmup_frames: int,
) -> tuple[bool, np.ndarray | None, dict[str, Any]]:
    cap = cv2.VideoCapture(str(device))
    if not cap.isOpened():
        return False, None, {"reason": "open_failed"}

    try:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        frame = None
        for _ in range(max(1, warmup_frames)):
            ok, candidate = cap.read()
            if ok and candidate is not None:
                frame = candidate
            else:
                time.sleep(0.03)

        if frame is None:
            return False, None, {
                "reason": "read_failed",
                "actual_width": actual_width,
                "actual_height": actual_height,
            }

        return True, frame, {
            "actual_width": actual_width,
            "actual_height": actual_height,
            "shape": list(frame.shape),
        }
    finally:
        cap.release()


def color_score(frame: np.ndarray) -> dict[str, Any]:
    """Return simple heuristics for whether a frame looks RGB/color-like."""
    if frame.ndim == 2:
        return {"rgb_like": False, "reason": "single_channel", "channel_std_mean": 0.0}
    if frame.ndim != 3 or frame.shape[2] < 3:
        return {"rgb_like": False, "reason": f"bad_shape_{frame.shape}", "channel_std_mean": 0.0}

    sample = frame[:, :, :3].astype(np.float32)
    channel_means = sample.reshape(-1, 3).mean(axis=0)
    channel_stds = sample.reshape(-1, 3).std(axis=0)
    spatial_std = float(sample.mean(axis=2).std())
    channel_std_mean = float(channel_stds.mean())
    channel_mean_spread = float(channel_means.max() - channel_means.min())

    # Depth/metadata streams often read as nearly black, nearly flat, or weird
    # single-channel-like images. This heuristic is deliberately permissive:
    # save anything that looks like a real color image, let the user inspect.
    rgb_like = bool(spatial_std > 5.0 and channel_std_mean > 3.0)
    return {
        "rgb_like": rgb_like,
        "spatial_std": spatial_std,
        "channel_std_mean": channel_std_mean,
        "channel_mean_spread": channel_mean_spread,
        "channel_means_bgr": [float(x) for x in channel_means.tolist()],
    }


def make_contact_sheet(images: list[tuple[str, np.ndarray]], output: Path, thumb_width: int = 320) -> None:
    if not images:
        return
    thumbs: list[np.ndarray] = []
    for label, image in images:
        h, w = image.shape[:2]
        scale = thumb_width / max(1, w)
        thumb = cv2.resize(image, (thumb_width, max(1, int(h * scale))))
        cv2.putText(thumb, label, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        thumbs.append(thumb)

    max_h = max(item.shape[0] for item in thumbs)
    padded = []
    for thumb in thumbs:
        if thumb.shape[0] < max_h:
            pad = np.zeros((max_h - thumb.shape[0], thumb.shape[1], 3), dtype=np.uint8)
            thumb = np.vstack([thumb, pad])
        padded.append(thumb)

    sheet = np.hstack(padded)
    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), sheet)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Find RGB-like video devices and save screenshots.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/camera_scan"))
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--warmup-frames", type=int, default=5)
    parser.add_argument(
        "--save-all-readable",
        action="store_true",
        help="Save frames from every readable node, not only RGB-like nodes.",
    )
    parser.add_argument(
        "--device",
        action="append",
        default=None,
        help="Only scan this device/index. Can be repeated, e.g. --device 16 --device /dev/video18.",
    )
    return parser


def normalize_device(value: str) -> Path:
    value = str(value).strip()
    if value.isdigit():
        return Path(f"/dev/video{value}")
    return Path(value)


def main() -> None:
    args = build_arg_parser().parse_args()
    devices = [normalize_device(item) for item in args.device] if args.device else list_video_devices()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    manifest: list[dict[str, Any]] = []
    contact_images: list[tuple[str, np.ndarray]] = []

    for device in devices:
        record: dict[str, Any] = {"device": str(device)}
        record.update(device_metadata(device))

        ok, frame, capture_info = capture_frame(
            device,
            width=args.width,
            height=args.height,
            warmup_frames=args.warmup_frames,
        )
        record["capture"] = capture_info
        record["readable"] = bool(ok)

        if ok and frame is not None:
            score = color_score(frame)
            record["color_score"] = score
            should_save = bool(score["rgb_like"] or args.save_all_readable)
            record["rgb_like"] = bool(score["rgb_like"])
            if should_save:
                stem = device.name
                image_path = args.output_dir / f"{stem}.jpg"
                cv2.imwrite(str(image_path), frame)
                record["image_path"] = str(image_path)
                contact_images.append((stem, frame))
        else:
            record["rgb_like"] = False

        manifest.append(record)
        print(
            f"{device}: readable={record['readable']} rgb_like={record['rgb_like']} "
            f"card={record.get('card') or record.get('udev', {}).get('ID_MODEL', '')}"
        )

    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    make_contact_sheet(contact_images, args.output_dir / "contact_sheet.jpg")

    print()
    print(f"manifest:      {manifest_path}")
    if contact_images:
        print(f"contact sheet: {args.output_dir / 'contact_sheet.jpg'}")
    print("RGB-like devices:")
    for item in manifest:
        if item.get("rgb_like"):
            print(f"  {item['device']} -> {item.get('image_path')}")


if __name__ == "__main__":
    main()
