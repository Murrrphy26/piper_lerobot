#!/usr/bin/env python3
"""Scan V4L2 nodes, identify RGB camera nodes, and save screenshots.

Why this exists:
- One physical Orbbec depth camera exposes several /dev/video* nodes.
- Some depth/IR nodes can be opened by OpenCV, so "cap.read() works" is not
  enough to say it is RGB.
- Old test_cameras.py only scanned video0..video14 and missed video16/18/20.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import time
from multiprocessing import Pipe, Process
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

RGB_STRONG = {"MJPG", "YUYV", "RGB3", "BGR3", "YUY2"}
DEPTH = {"Z16", "Y16"}
IR = {"GREY", "Y8", "Y10", "Y12"}


@dataclass
class NodeInfo:
    device: str
    index: int
    kind: str
    readable: bool
    formats: list[str]
    card: str = ""
    bus: str = ""
    driver: str = ""
    vendor: str = ""
    model: str = ""
    serial: str = ""
    width: int | None = None
    height: int | None = None
    screenshot: str | None = None
    error: str | None = None
    frame_stats: dict[str, Any] | None = None


def run_text(cmd: list[str], timeout: float = 3.0) -> str:
    try:
        p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=timeout, check=False)
        return p.stdout or ""
    except Exception as exc:  # diagnostic script
        return f"{type(exc).__name__}: {exc}"


def list_video_nodes() -> list[Path]:
    nodes = []
    for p in Path('/dev').glob('video*'):
        if re.fullmatch(r'video\d+', p.name):
            nodes.append(p)
    return sorted(nodes, key=lambda p: int(p.name.replace('video', '')))


def parse_key(text: str, pattern: str) -> str:
    m = re.search(pattern, text)
    return m.group(1).strip() if m else ""


def v4l2_meta(dev: str) -> tuple[dict[str, str], list[str]]:
    all_text = run_text(['v4l2-ctl', '-d', dev, '--all'])
    fmt_text = run_text(['v4l2-ctl', '-d', dev, '--list-formats-ext'])
    meta = {
        'driver': parse_key(all_text, r'Driver name\s*:\s*(.+)'),
        'card': parse_key(all_text, r'Card type\s*:\s*(.+)'),
        'bus': parse_key(all_text, r'Bus info\s*:\s*(.+)'),
    }
    formats: list[str] = []
    for m in re.finditer(r"\[\d+\]:\s*'([^']+)'", fmt_text):
        fmt = m.group(1).strip().upper()
        if fmt and fmt not in formats:
            formats.append(fmt)
    return meta, formats


def udev_meta(dev: str) -> dict[str, str]:
    text = run_text(['udevadm', 'info', '-q', 'property', '-n', dev])
    out: dict[str, str] = {}
    for line in text.splitlines():
        if '=' not in line:
            continue
        k, v = line.split('=', 1)
        if k in {'ID_VENDOR', 'ID_MODEL', 'ID_SERIAL', 'ID_PATH'}:
            out[k] = v
    return out


def classify(formats: list[str], card: str) -> str:
    fmts = set(f.upper().strip() for f in formats)
    if not fmts:
        return 'unknown'
    if fmts & RGB_STRONG:
        return 'rgb'
    if fmts & DEPTH:
        return 'depth'
    if fmts & IR or 'NV12' in fmts:
        # On these Orbbec devices, GREY/NV12 nodes are not normal RGB nodes.
        return 'ir_or_aux'
    return 'unknown'


def capture(dev: str, width: int, height: int, warmup: int) -> tuple[bool, np.ndarray | None, str | None]:
    cap = cv2.VideoCapture(dev, cv2.CAP_V4L2)
    if not cap.isOpened():
        cap.release()
        cap = cv2.VideoCapture(dev)
    if not cap.isOpened():
        return False, None, 'open_failed'
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    frame = None
    ok = False
    for _ in range(max(1, warmup)):
        ok, frame = cap.read()
        time.sleep(0.03)
    cap.release()
    if not ok or frame is None:
        return False, None, 'read_failed'
    return True, frame, None


def _capture_worker(conn: Any, dev: str, width: int, height: int, warmup: int) -> None:
    try:
        conn.send(capture(dev, width, height, warmup))
    except Exception as exc:  # diagnostic script
        conn.send((False, None, f'{type(exc).__name__}: {exc}'))
    finally:
        conn.close()


def capture_with_timeout(
    dev: str,
    width: int,
    height: int,
    warmup: int,
    timeout_s: float,
) -> tuple[bool, np.ndarray | None, str | None]:
    parent, child = Pipe(duplex=False)
    proc = Process(target=_capture_worker, args=(child, dev, width, height, warmup))
    proc.start()
    child.close()
    try:
        if parent.poll(timeout_s):
            return parent.recv()
        proc.terminate()
        proc.join(timeout=1.0)
        if proc.is_alive():
            proc.kill()
            proc.join(timeout=1.0)
        return False, None, f'capture_timeout_{timeout_s:g}s'
    finally:
        parent.close()
        if proc.is_alive():
            proc.terminate()
            proc.join(timeout=1.0)


def stats(frame: np.ndarray) -> dict[str, Any]:
    arr = np.asarray(frame)
    out: dict[str, Any] = {'shape': list(arr.shape), 'mean': float(arr.mean()), 'std': float(arr.std())}
    if arr.ndim == 3 and arr.shape[2] >= 3:
        chans = arr[:, :, :3].astype(np.float32).reshape(-1, 3)
        out['channel_means_bgr'] = [float(x) for x in chans.mean(axis=0)]
        out['channel_stds_bgr'] = [float(x) for x in chans.std(axis=0)]
    return out


def put_label(img: np.ndarray, label: str) -> np.ndarray:
    out = img.copy()
    if out.ndim == 2:
        out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)
    cv2.rectangle(out, (0, 0), (out.shape[1], 32), (0, 0, 0), -1)
    cv2.putText(out, label, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
    return out


def contact_sheet(items: list[NodeInfo], out: Path, thumb_w: int = 320) -> None:
    imgs = []
    for item in items:
        if not item.screenshot:
            continue
        img = cv2.imread(item.screenshot)
        if img is None:
            continue
        h, w = img.shape[:2]
        img = cv2.resize(img, (thumb_w, max(1, int(h * thumb_w / max(1, w)))))
        imgs.append(put_label(img, f"{Path(item.device).name} {item.kind}"))
    if not imgs:
        return
    cols = min(3, len(imgs))
    rows = (len(imgs) + cols - 1) // cols
    cell_h = max(i.shape[0] for i in imgs)
    pad = 8
    sheet = np.full((rows * cell_h + (rows + 1) * pad, cols * thumb_w + (cols + 1) * pad, 3), 245, np.uint8)
    for n, img in enumerate(imgs):
        r, c = divmod(n, cols)
        y = pad + r * (cell_h + pad)
        x = pad + c * (thumb_w + pad)
        sheet[y:y + img.shape[0], x:x + img.shape[1]] = img
    cv2.imwrite(str(out), sheet)


def main() -> int:
    ap = argparse.ArgumentParser(description='Find RGB camera nodes and save screenshots.')
    ap.add_argument('--output', default='camera_test/rgb_scan')
    ap.add_argument('--width', type=int, default=640)
    ap.add_argument('--height', type=int, default=480)
    ap.add_argument('--warmup', type=int, default=2)
    ap.add_argument('--capture-timeout', type=float, default=6.0, help='seconds allowed for one device screenshot before marking timeout')
    ap.add_argument('--save-non-rgb', action='store_true', help='save screenshots from non-RGB nodes when --probe-non-rgb is enabled')
    ap.add_argument('--probe-non-rgb', action='store_true', help='also try to read depth/IR/aux nodes; default only captures RGB candidates')
    ap.add_argument('--device', action='append', help='scan only N or /dev/videoN; can repeat')
    args = ap.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.device:
        nodes = [Path(f'/dev/video{x}') if str(x).isdigit() else Path(str(x)) for x in args.device]
    else:
        nodes = list_video_nodes()

    results: list[NodeInfo] = []
    for node in nodes:
        idx_m = re.search(r'video(\d+)$', node.name)
        idx = int(idx_m.group(1)) if idx_m else -1
        dev = str(node)
        meta, formats = v4l2_meta(dev)
        udev = udev_meta(dev)
        kind = classify(formats, meta.get('card', ''))
        info = NodeInfo(
            device=dev,
            index=idx,
            kind=kind,
            readable=False,
            formats=formats,
            card=meta.get('card', ''),
            bus=meta.get('bus', ''),
            driver=meta.get('driver', ''),
            vendor=udev.get('ID_VENDOR', ''),
            model=udev.get('ID_MODEL', ''),
            serial=udev.get('ID_SERIAL', ''),
        )
        should_try = kind == 'rgb' or args.probe_non_rgb
        if should_try:
            ok, frame, err = capture_with_timeout(dev, args.width, args.height, args.warmup, args.capture_timeout)
            info.readable = ok
            info.error = err
            if ok and frame is not None:
                h, w = frame.shape[:2]
                info.width = int(w)
                info.height = int(h)
                info.frame_stats = stats(frame)
                if kind == 'rgb' or args.save_non_rgb:
                    shot = out_dir / f'video{idx:02d}_{kind}_{w}x{h}.jpg'
                    cv2.imwrite(str(shot), put_label(frame, f'{node.name} {kind}'))
                    info.screenshot = str(shot)
        results.append(info)

    rgb_candidates = [x for x in results if x.kind == 'rgb']
    rgb = [x for x in rgb_candidates if x.readable]
    unreadable_rgb = [x for x in rgb_candidates if not x.readable]
    readable_non_rgb = [x for x in results if x.kind != 'rgb' and x.readable]
    (out_dir / 'manifest.json').write_text(json.dumps([asdict(x) for x in results], ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    contact_sheet(rgb, out_dir / 'contact_sheet_rgb.jpg')
    contact_sheet([x for x in results if x.screenshot], out_dir / 'contact_sheet_all_saved.jpg')

    print('\nRGB camera nodes:')
    if not rgb_candidates:
        print('  (none)')
    for x in rgb:
        print(f'  ✅ {x.device:<12} {x.width}x{x.height} formats={",".join(x.formats)} bus={x.bus} shot={x.screenshot}')
    for x in unreadable_rgb:
        print(f'  ⚠️ {x.device:<12} unreadable error={x.error} formats={",".join(x.formats)} bus={x.bus}')
    if readable_non_rgb:
        print('\nReadable but NOT RGB:')
        for x in readable_non_rgb:
            print(f'  {x.kind:<9} {x.device:<12} {x.width}x{x.height} formats={",".join(x.formats)}')
    print(f'\nRGB metadata candidates: {len(rgb_candidates)}')
    print(f'RGB readable screenshots: {len(rgb)}')
    print(f'Output dir: {out_dir}')
    print(f'Manifest: {out_dir / "manifest.json"}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
