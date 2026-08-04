#!/usr/bin/env python3
"""Save a low-rate live preview from one camera.

This is useful after rebooting the robot PC, when /dev/video indices may have
changed. The latest frame is always written to ``camera_temp/live.jpg``.

Examples:
  python tools/show.py 16
  python tools/show.py /dev/video16 --width 640 --height 480
"""

from __future__ import annotations

import argparse
from pathlib import Path
import time

import cv2


SAVE_FPS = 2.0


def normalize_device(value: str) -> str:
    value = str(value).strip()
    if value.isdigit():
        return f"/dev/video{value}"
    return value


def preview_to_file(
    dev_path: str,
    *,
    width: int,
    height: int,
    output: Path,
) -> None:
    cap = cv2.VideoCapture(dev_path)
    if not cap.isOpened():
        print(f"❌ 无法打开 {dev_path}")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    use_resize = actual_width != width or actual_height != height

    output.parent.mkdir(parents=True, exist_ok=True)

    print(f"✅ 相机已打开: {dev_path}")
    print(f"目标分辨率: {width}x{height}")
    print(f"实际分辨率: {actual_width}x{actual_height}")
    print(f"保存频率: {SAVE_FPS:g} fps")
    print(f"输出文件: {output}")
    if use_resize:
        print(f"⚠️ 相机不支持目标分辨率，将保存时缩放到 {width}x{height}")
    print("按 Ctrl+C 停止。")

    frame_count = 0
    saved_count = 0
    last_print = time.monotonic()
    period = 1.0 / SAVE_FPS

    try:
        while True:
            started = time.monotonic()
            ret, frame = cap.read()
            if not ret or frame is None:
                print("\n⚠️ 读取帧失败")
                break

            frame_count += 1
            if use_resize:
                frame = cv2.resize(frame, (width, height))

            cv2.imwrite(str(output), frame)
            saved_count += 1

            now = time.monotonic()
            if now - last_print >= 1.0:
                h, w = frame.shape[:2]
                print(
                    f"\r输出: {w}x{h} | 已保存: {saved_count} | "
                    f"读取帧: {frame_count} | 时间: {time.strftime('%H:%M:%S')}",
                    end="",
                    flush=True,
                )
                last_print = now

            elapsed = time.monotonic() - started
            time.sleep(max(0.0, period - elapsed))
    except KeyboardInterrupt:
        print("\n停止预览。")
    finally:
        cap.release()
        print(f"\n最新帧保存在: {output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Save one camera preview frame at 2 fps.")
    parser.add_argument(
        "camera",
        nargs="?",
        default="0",
        help="Camera index such as 16, or a device path such as /dev/video16.",
    )
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--output", type=Path, default=Path("camera_temp/live.jpg"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    preview_to_file(
        normalize_device(args.camera),
        width=args.width,
        height=args.height,
        output=args.output,
    )


if __name__ == "__main__":
    main()
