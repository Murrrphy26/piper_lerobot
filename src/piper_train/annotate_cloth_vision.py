"""Debug CLI for white-cloth vision state estimation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import cv2

from .cloth_vision import (
    ClothGeometryThresholds,
    ClothVisionAnnotator,
    WhiteClothVisionConfig,
    overlay_cloth_result,
)


def _read_rgb_image(path: Path):
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _capture_rgb(camera: str, *, width: int, height: int, warmup_frames: int):
    device = f"/dev/video{camera}" if camera.isdigit() else camera
    cap = cv2.VideoCapture(device)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera: {device}")
    try:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        frame = None
        for _ in range(max(1, warmup_frames)):
            ok, frame = cap.read()
            if not ok or frame is None:
                time.sleep(0.05)
                continue
        if frame is None:
            raise RuntimeError(f"Could not read frame from camera: {device}")
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    finally:
        cap.release()


def _open_camera(camera: str, *, width: int, height: int):
    device = f"/dev/video{camera}" if camera.isdigit() else camera
    cap = cv2.VideoCapture(device)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera: {device}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return device, cap


def _parse_roi(value: str | None) -> tuple[float, float, float, float] | None:
    if not value:
        return None
    parts = [float(item.strip()) for item in value.split(",")]
    if len(parts) != 4:
        raise ValueError("--roi must be x0,y0,x1,y1")
    return (parts[0], parts[1], parts[2], parts[3])


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Annotate white cloth state from an image or camera.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--image", type=Path, help="Input image path.")
    source.add_argument("--camera", help="Camera index or /dev/videoX path.")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--warmup-frames", type=int, default=5)
    parser.add_argument(
        "--watch",
        action="store_true",
        help="Continuously annotate camera frames until Ctrl+C. Only valid with --camera.",
    )
    parser.add_argument("--watch-fps", type=float, default=2.0)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/vision_debug"))
    parser.add_argument("--prefix", default=None)

    parser.add_argument("--max-saturation", type=int, default=80)
    parser.add_argument("--min-value", type=int, default=135)
    parser.add_argument("--min-rgb-mean", type=int, default=130)
    parser.add_argument("--max-rgb-channel-spread", type=int, default=70)
    parser.add_argument("--blur-kernel", type=int, default=5)
    parser.add_argument("--morph-kernel", type=int, default=7)
    parser.add_argument("--morph-iterations", type=int, default=2)
    parser.add_argument("--min-area-ratio", type=float, default=0.005)
    parser.add_argument("--min-area-ratio-for-flat", type=float, default=0.12)
    parser.add_argument("--min-extent-for-flat", type=float, default=0.35)
    parser.add_argument("--min-compactness-for-flat", type=float, default=0.55)
    parser.add_argument("--target-center", default="0.5,0.5")
    parser.add_argument("--max-center-error-norm", type=float, default=0.18)
    parser.add_argument("--target-orientation-deg", type=float, default=0.0)
    parser.add_argument("--max-orientation-error-deg", type=float, default=20.0)
    parser.add_argument(
        "--roi",
        default=None,
        help="Optional normalized ROI x0,y0,x1,y1, e.g. 0.05,0.05,0.95,0.95.",
    )
    return parser


def _parse_pair(value: str) -> tuple[float, float]:
    parts = [float(item.strip()) for item in value.split(",")]
    if len(parts) != 2:
        raise ValueError("Expected x,y")
    return (parts[0], parts[1])


def _build_annotator(args: argparse.Namespace) -> ClothVisionAnnotator:
    vision_config = WhiteClothVisionConfig(
        max_saturation=args.max_saturation,
        min_value=args.min_value,
        min_rgb_mean=args.min_rgb_mean,
        max_rgb_channel_spread=args.max_rgb_channel_spread,
        blur_kernel=args.blur_kernel,
        morph_kernel=args.morph_kernel,
        morph_iterations=args.morph_iterations,
        min_area_ratio=args.min_area_ratio,
        roi_norm=_parse_roi(args.roi),
    )
    thresholds = ClothGeometryThresholds(
        min_area_ratio_for_flat=args.min_area_ratio_for_flat,
        min_extent_for_flat=args.min_extent_for_flat,
        min_compactness_for_flat=args.min_compactness_for_flat,
        target_center_norm=_parse_pair(args.target_center),
        max_center_error_norm=args.max_center_error_norm,
        target_orientation_deg=args.target_orientation_deg,
        max_orientation_error_deg=args.max_orientation_error_deg,
    )
    return ClothVisionAnnotator(vision_config=vision_config, thresholds=thresholds)


def _write_outputs(
    *,
    rgb,
    annotator: ClothVisionAnnotator,
    output_dir: Path,
    prefix: str,
) -> dict:
    result = annotator.annotate(rgb)
    overlay = overlay_cloth_result(rgb, result)

    output_dir.mkdir(parents=True, exist_ok=True)
    state_path = output_dir / f"{prefix}_state.json"
    mask_path = output_dir / f"{prefix}_mask.png"
    overlay_path = output_dir / f"{prefix}_overlay.jpg"

    payload = result.to_dict()
    state_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    cv2.imwrite(str(mask_path), result.mask)
    cv2.imwrite(str(overlay_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    return {
        "payload": payload,
        "state_path": state_path,
        "mask_path": mask_path,
        "overlay_path": overlay_path,
    }


def _run_watch(args: argparse.Namespace, annotator: ClothVisionAnnotator) -> None:
    if args.camera is None:
        raise ValueError("--watch requires --camera")
    device, cap = _open_camera(args.camera, width=args.width, height=args.height)
    period = 1.0 / max(0.1, float(args.watch_fps))
    print(f"Watching {device} at {args.watch_fps:g} fps. Press Ctrl+C to stop.")
    try:
        index = 0
        while True:
            started = time.monotonic()
            ok, frame = cap.read()
            if not ok or frame is None:
                print("read failed")
                time.sleep(period)
                continue
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = _write_outputs(
                rgb=rgb,
                annotator=annotator,
                output_dir=args.output_dir,
                prefix=args.prefix or f"watch_{index:06d}",
            )
            state = result["payload"]["state"]
            checks = result["payload"]["checks"]
            print(
                f"\rvalid={state['valid']} area={state['area_ratio']:.3f} "
                f"angle={state['orientation_deg']} aligned={checks['aligned_enough']} "
                f"overlay={result['overlay_path']}",
                end="",
                flush=True,
            )
            index += 1
            elapsed = time.monotonic() - started
            time.sleep(max(0.0, period - elapsed))
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        cap.release()


def main() -> None:
    args = build_arg_parser().parse_args()
    annotator = _build_annotator(args)
    if args.watch:
        _run_watch(args, annotator)
        return

    if args.image is not None:
        rgb = _read_rgb_image(args.image)
        default_prefix = args.image.stem
    else:
        rgb = _capture_rgb(args.camera, width=args.width, height=args.height, warmup_frames=args.warmup_frames)
        default_prefix = f"camera_{str(args.camera).replace('/', '_')}"

    result = _write_outputs(
        rgb=rgb,
        annotator=annotator,
        output_dir=args.output_dir,
        prefix=args.prefix or default_prefix,
    )
    print(json.dumps(result["payload"], ensure_ascii=False, indent=2))
    print(f"mask:    {result['mask_path']}")
    print(f"overlay: {result['overlay_path']}")
    print(f"state:   {result['state_path']}")


if __name__ == "__main__":
    main()
