"""Lightweight cloth vision utilities for the folding harness.

The first target setup is simple and deliberately transparent:

- yellow/brown wooden table
- white garment/towel
- top camera observation

The module extracts a white/low-saturation mask and reports geometric state for
the largest cloth-like component. It does not decide robot actions; it only
produces state that a harness can log or use for stage checks.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
from typing import Any

import cv2
import numpy as np


@dataclass(frozen=True)
class WhiteClothVisionConfig:
    """Parameters for white cloth segmentation on a wooden table."""

    # HSV thresholds for white fabric: low saturation, high value.
    max_saturation: int = 80
    min_value: int = 135

    # Extra RGB guard: white cloth should be bright across channels. This helps
    # reject yellow wood highlights that may have high V but non-white color.
    min_rgb_mean: int = 130
    max_rgb_channel_spread: int = 70

    # Morphology / component filtering.
    blur_kernel: int = 5
    morph_kernel: int = 7
    morph_iterations: int = 2
    min_area_ratio: float = 0.005

    # Optional region of interest in normalized image coordinates.
    # Format: [x0, y0, x1, y1], values in [0, 1].
    roi_norm: tuple[float, float, float, float] | None = None


@dataclass
class ClothVisionState:
    valid: bool
    image_width: int
    image_height: int
    area_px: int = 0
    area_ratio: float = 0.0
    bbox_xywh: list[int] | None = None
    center_xy: list[float] | None = None
    orientation_deg: float | None = None
    compactness: float | None = None
    extent: float | None = None
    contour_area_px: float | None = None
    hull_area_px: float | None = None
    reason: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ClothGeometryThresholds:
    """Loose geometric thresholds for harness-level checks.

    These thresholds are intentionally camera/image-space based. They are not
    meant to replace policy perception; they give the harness a simple progress
    signal until a stronger VLM/keypoint layer exists.
    """

    min_area_ratio_for_visible: float = 0.01
    min_area_ratio_for_flat: float = 0.12
    min_extent_for_flat: float = 0.35
    min_compactness_for_flat: float = 0.55
    target_center_norm: tuple[float, float] = (0.5, 0.5)
    max_center_error_norm: float = 0.18
    target_orientation_deg: float = 0.0
    max_orientation_error_deg: float = 20.0


@dataclass
class ClothGeometryChecks:
    visible: bool
    flat_enough: bool
    centered_enough: bool
    oriented_enough: bool
    aligned_enough: bool
    center_error_norm: float | None = None
    orientation_error_deg: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ClothVisionResult:
    state: ClothVisionState
    checks: ClothGeometryChecks
    mask: np.ndarray

    def to_dict(self) -> dict[str, Any]:
        return {
            "state": self.state.to_dict(),
            "checks": self.checks.to_dict(),
        }


class ClothVisionAnnotator:
    """Callable wrapper used by future harness code."""

    def __init__(
        self,
        vision_config: WhiteClothVisionConfig | None = None,
        thresholds: ClothGeometryThresholds | None = None,
    ) -> None:
        self.vision_config = vision_config or WhiteClothVisionConfig()
        self.thresholds = thresholds or ClothGeometryThresholds()

    def annotate(self, rgb: np.ndarray) -> ClothVisionResult:
        state, mask = estimate_cloth_state(rgb, self.vision_config)
        checks = evaluate_cloth_geometry(state, self.thresholds)
        return ClothVisionResult(state=state, checks=checks, mask=mask)


def _ensure_odd_kernel(value: int) -> int:
    value = max(1, int(value))
    if value % 2 == 0:
        value += 1
    return value


def _roi_mask(height: int, width: int, roi_norm: tuple[float, float, float, float] | None) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.uint8)
    if roi_norm is None:
        mask[:, :] = 255
        return mask
    x0, y0, x1, y1 = roi_norm
    x0_i = int(np.clip(x0, 0.0, 1.0) * width)
    y0_i = int(np.clip(y0, 0.0, 1.0) * height)
    x1_i = int(np.clip(x1, 0.0, 1.0) * width)
    y1_i = int(np.clip(y1, 0.0, 1.0) * height)
    if x1_i <= x0_i or y1_i <= y0_i:
        return mask
    mask[y0_i:y1_i, x0_i:x1_i] = 255
    return mask


def white_cloth_mask(rgb: np.ndarray, config: WhiteClothVisionConfig | None = None) -> np.ndarray:
    """Return a uint8 mask for white fabric pixels.

    Input is expected to be RGB with shape ``H x W x 3``.
    """
    config = config or WhiteClothVisionConfig()
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"Expected RGB image HxWx3, got shape {rgb.shape}")

    image = rgb.astype(np.uint8, copy=False)
    blur_kernel = _ensure_odd_kernel(config.blur_kernel)
    if blur_kernel > 1:
        image = cv2.GaussianBlur(image, (blur_kernel, blur_kernel), 0)

    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]

    rgb_mean = image.astype(np.int16).mean(axis=2)
    rgb_spread = image.astype(np.int16).max(axis=2) - image.astype(np.int16).min(axis=2)

    mask_bool = (
        (saturation <= int(config.max_saturation))
        & (value >= int(config.min_value))
        & (rgb_mean >= int(config.min_rgb_mean))
        & (rgb_spread <= int(config.max_rgb_channel_spread))
    )
    mask = (mask_bool.astype(np.uint8)) * 255

    roi = _roi_mask(mask.shape[0], mask.shape[1], config.roi_norm)
    mask = cv2.bitwise_and(mask, roi)

    kernel_size = _ensure_odd_kernel(config.morph_kernel)
    if kernel_size > 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=max(0, config.morph_iterations))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=max(0, config.morph_iterations))

    return mask


def _largest_component(mask: np.ndarray, min_area_px: int) -> tuple[np.ndarray, int]:
    num_labels, labels, stats, _centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:
        return np.zeros_like(mask), 0

    best_label = 0
    best_area = 0
    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area > best_area:
            best_label = label
            best_area = area

    if best_area < min_area_px:
        return np.zeros_like(mask), best_area
    return ((labels == best_label).astype(np.uint8) * 255), best_area


def _orientation_from_points(points_xy: np.ndarray) -> float | None:
    if len(points_xy) < 2:
        return None
    centered = points_xy.astype(np.float32) - points_xy.astype(np.float32).mean(axis=0, keepdims=True)
    cov = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    principal = eigvecs[:, int(np.argmax(eigvals))]
    angle = math.degrees(math.atan2(float(principal[1]), float(principal[0])))
    # Normalize to [-90, 90), because garment principal axis has no arrow.
    while angle >= 90.0:
        angle -= 180.0
    while angle < -90.0:
        angle += 180.0
    return float(angle)


def estimate_cloth_state(
    rgb: np.ndarray,
    config: WhiteClothVisionConfig | None = None,
) -> tuple[ClothVisionState, np.ndarray]:
    """Estimate cloth state and return ``(state, largest_component_mask)``."""
    config = config or WhiteClothVisionConfig()
    height, width = rgb.shape[:2]
    min_area_px = int(config.min_area_ratio * width * height)
    raw_mask = white_cloth_mask(rgb, config)
    mask, component_area = _largest_component(raw_mask, min_area_px)
    if component_area < min_area_px:
        return (
            ClothVisionState(
                valid=False,
                image_width=width,
                image_height=height,
                area_px=int(component_area),
                area_ratio=float(component_area) / float(width * height),
                reason="component_too_small",
                extra={"min_area_px": min_area_px},
            ),
            mask,
        )

    contours, _hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return (
            ClothVisionState(
                valid=False,
                image_width=width,
                image_height=height,
                area_px=int(component_area),
                area_ratio=float(component_area) / float(width * height),
                reason="no_contour",
            ),
            mask,
        )

    contour = max(contours, key=cv2.contourArea)
    contour_area = float(cv2.contourArea(contour))
    x, y, w, h = cv2.boundingRect(contour)
    moments = cv2.moments(contour)
    if abs(moments["m00"]) > 1e-6:
        center_x = float(moments["m10"] / moments["m00"])
        center_y = float(moments["m01"] / moments["m00"])
    else:
        center_x = float(x + w / 2.0)
        center_y = float(y + h / 2.0)

    points_yx = np.column_stack(np.nonzero(mask))
    points_xy = points_yx[:, ::-1]
    orientation = _orientation_from_points(points_xy)

    hull = cv2.convexHull(contour)
    hull_area = float(cv2.contourArea(hull))
    compactness = float(contour_area / hull_area) if hull_area > 1e-6 else None
    bbox_area = float(max(1, w * h))
    extent = float(contour_area / bbox_area)

    return (
        ClothVisionState(
            valid=True,
            image_width=width,
            image_height=height,
            area_px=int(component_area),
            area_ratio=float(component_area) / float(width * height),
            bbox_xywh=[int(x), int(y), int(w), int(h)],
            center_xy=[center_x, center_y],
            orientation_deg=orientation,
            compactness=compactness,
            extent=extent,
            contour_area_px=contour_area,
            hull_area_px=hull_area,
        ),
        mask,
    )


def _axis_angle_error_deg(angle: float, target: float) -> float:
    """Return orientation error for an undirected principal axis."""
    diff = float(angle) - float(target)
    while diff >= 90.0:
        diff -= 180.0
    while diff < -90.0:
        diff += 180.0
    return abs(diff)


def evaluate_cloth_geometry(
    state: ClothVisionState,
    thresholds: ClothGeometryThresholds | None = None,
) -> ClothGeometryChecks:
    thresholds = thresholds or ClothGeometryThresholds()
    visible = bool(state.valid and state.area_ratio >= thresholds.min_area_ratio_for_visible)

    flat_enough = bool(
        visible
        and state.area_ratio >= thresholds.min_area_ratio_for_flat
        and state.extent is not None
        and state.extent >= thresholds.min_extent_for_flat
        and state.compactness is not None
        and state.compactness >= thresholds.min_compactness_for_flat
    )

    center_error_norm: float | None = None
    centered_enough = False
    if visible and state.center_xy is not None and state.image_width > 0 and state.image_height > 0:
        cx_norm = state.center_xy[0] / float(state.image_width)
        cy_norm = state.center_xy[1] / float(state.image_height)
        tx, ty = thresholds.target_center_norm
        center_error_norm = float(math.hypot(cx_norm - tx, cy_norm - ty))
        centered_enough = center_error_norm <= thresholds.max_center_error_norm

    orientation_error_deg: float | None = None
    oriented_enough = False
    if visible and state.orientation_deg is not None:
        orientation_error_deg = _axis_angle_error_deg(
            state.orientation_deg,
            thresholds.target_orientation_deg,
        )
        oriented_enough = orientation_error_deg <= thresholds.max_orientation_error_deg

    return ClothGeometryChecks(
        visible=visible,
        flat_enough=flat_enough,
        centered_enough=centered_enough,
        oriented_enough=oriented_enough,
        aligned_enough=bool(flat_enough and centered_enough and oriented_enough),
        center_error_norm=center_error_norm,
        orientation_error_deg=orientation_error_deg,
    )


def overlay_cloth_state(rgb: np.ndarray, mask: np.ndarray, state: ClothVisionState) -> np.ndarray:
    """Draw mask, bbox, center, and orientation on an RGB image."""
    overlay = rgb.astype(np.uint8, copy=True)
    color_layer = np.zeros_like(overlay)
    color_layer[:, :, 1] = mask
    overlay = cv2.addWeighted(overlay, 0.75, color_layer, 0.25, 0)

    if state.valid and state.bbox_xywh is not None:
        x, y, w, h = state.bbox_xywh
        cv2.rectangle(overlay, (x, y), (x + w, y + h), (255, 0, 0), 2)
    if state.valid and state.center_xy is not None:
        cx, cy = state.center_xy
        cv2.circle(overlay, (int(round(cx)), int(round(cy))), 5, (255, 0, 255), -1)
        if state.orientation_deg is not None:
            length = 60
            theta = math.radians(state.orientation_deg)
            dx = math.cos(theta) * length
            dy = math.sin(theta) * length
            p1 = (int(round(cx - dx)), int(round(cy - dy)))
            p2 = (int(round(cx + dx)), int(round(cy + dy)))
            cv2.line(overlay, p1, p2, (0, 0, 255), 2)

    label = (
        f"valid={state.valid} area={state.area_ratio:.3f} "
        f"angle={state.orientation_deg if state.orientation_deg is not None else 'NA'}"
    )
    cv2.putText(overlay, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    return overlay


def overlay_cloth_result(rgb: np.ndarray, result: ClothVisionResult) -> np.ndarray:
    overlay = overlay_cloth_state(rgb, result.mask, result.state)
    checks = result.checks
    label = (
        f"visible={checks.visible} flat={checks.flat_enough} "
        f"centered={checks.centered_enough} oriented={checks.oriented_enough} "
        f"aligned={checks.aligned_enough}"
    )
    cv2.putText(overlay, label, (10, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
    return overlay
