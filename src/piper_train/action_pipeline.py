"""Shared action post-processing and safety filters for live robot control.

This module is intentionally policy-agnostic: ACT, async pi05, and future task
harnesses can all route predicted actions through the same small pipeline before
calling ``PiperRobot.send_action``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Protocol


ActionDict = dict[str, float]


def smooth_action(
    action: ActionDict,
    previous_action: ActionDict | None,
    alpha: float,
) -> ActionDict:
    """EMA smoothing used online before sending robot commands."""
    if previous_action is None:
        return dict(action)

    return {
        key: previous_action.get(key, value) * (1.0 - alpha) + value * alpha
        for key, value in action.items()
    }


@dataclass(frozen=True)
class EEPose:
    """Minimal end-effector pose used by workspace safety checks."""

    x: float
    y: float
    z: float


class FKProvider(Protocol):
    """Forward-kinematics provider interface.

    Implementations may use a URDF, DH parameters, SDK FK, or a calibrated model.
    Returning ``None`` means FK is unavailable for that side/action.
    """

    def ee_pose(self, side: str, joints_rad: list[float]) -> EEPose | None:
        ...


class PiperSDKFKProvider:
    """FK provider backed by ``piper_sdk.C_PiperForwardKinematics``.

    The SDK FK function takes six joint positions in radians and returns one
    ``[x, y, z, r, p, y]`` pose per link. Its xyz output is in millimeters, so
    this provider converts the final link position to meters.
    """

    def __init__(self, dh_is_offset: int = 1) -> None:
        try:
            from piper_sdk import C_PiperForwardKinematics
        except ImportError as exc:
            raise ImportError("piper_sdk is required for safety_fk_provider='piper_sdk'.") from exc

        self._fk = C_PiperForwardKinematics(dh_is_offset=dh_is_offset)

    def ee_pose(self, side: str, joints_rad: list[float]) -> EEPose | None:
        del side  # The SDK FK is for one Piper arm; side only selects joints upstream.
        if len(joints_rad) != 6:
            return None
        link_poses = self._fk.CalFK([float(value) for value in joints_rad])
        if not link_poses:
            return None
        ee = link_poses[-1]
        return EEPose(x=float(ee[0]) / 1000.0, y=float(ee[1]) / 1000.0, z=float(ee[2]) / 1000.0)


@dataclass(frozen=True)
class WorkspaceSideLimit:
    """Workspace constraints for one arm side.

    ``allowed_below_min_m`` is the configurable tolerance: if ``min_z_m`` is the
    calibrated near-table height, commands are allowed down to
    ``min_z_m - allowed_below_min_m``.
    """

    min_z_m: float | None = None
    allowed_below_min_m: float = 0.0
    x_min_m: float | None = None
    x_max_m: float | None = None
    y_min_m: float | None = None
    y_max_m: float | None = None


@dataclass(frozen=True)
class ActionSafetyConfig:
    enabled: bool = False
    on_violation: str = "hold_previous"  # hold_previous | warn | stop
    left: WorkspaceSideLimit = field(default_factory=WorkspaceSideLimit)
    right: WorkspaceSideLimit = field(default_factory=WorkspaceSideLimit)
    finite_check: bool = True

    @staticmethod
    def from_namespace(args: Any) -> "ActionSafetyConfig":
        allowed = float(getattr(args, "safety_allowed_below_min_m", 0.0) or 0.0)
        left_min = getattr(args, "safety_left_min_z_m", None)
        right_min = getattr(args, "safety_right_min_z_m", None)
        return ActionSafetyConfig(
            enabled=bool(getattr(args, "safety_enabled", False)),
            on_violation=str(getattr(args, "safety_on_violation", "hold_previous")),
            left=WorkspaceSideLimit(
                min_z_m=float(left_min) if left_min is not None else None,
                allowed_below_min_m=allowed,
            ),
            right=WorkspaceSideLimit(
                min_z_m=float(right_min) if right_min is not None else None,
                allowed_below_min_m=allowed,
            ),
            finite_check=bool(getattr(args, "safety_finite_check", True)),
        )


@dataclass
class ActionPipelineResult:
    smoothed_action: ActionDict
    filtered_action: ActionDict
    events: list[dict[str, Any]] = field(default_factory=list)
    blocked: bool = False


class ActionPipeline:
    """Apply smoothing and optional safety filtering to policy actions."""

    def __init__(
        self,
        safety: ActionSafetyConfig | None = None,
        fk_provider: FKProvider | None = None,
    ) -> None:
        self.safety = safety or ActionSafetyConfig()
        self.fk_provider = fk_provider
        self._last_safe_action: ActionDict | None = None

    def reset(self, current_action: ActionDict | None = None) -> None:
        self._last_safe_action = dict(current_action) if current_action is not None else None

    def process(
        self,
        predicted_action: ActionDict,
        previous_action: ActionDict | None,
        alpha: float,
    ) -> ActionPipelineResult:
        smoothed = smooth_action(predicted_action, previous_action, alpha)
        filtered, events, blocked = self._apply_safety(smoothed)
        if not blocked or self.safety.on_violation == "warn":
            self._last_safe_action = dict(filtered)
        return ActionPipelineResult(
            smoothed_action=smoothed,
            filtered_action=filtered,
            events=events,
            blocked=blocked,
        )

    def _apply_safety(self, action: ActionDict) -> tuple[ActionDict, list[dict[str, Any]], bool]:
        events: list[dict[str, Any]] = []
        blocked = False

        if not self.safety.enabled:
            return dict(action), events, blocked

        if self.safety.finite_check:
            bad_keys = [key for key, value in action.items() if not math.isfinite(float(value))]
            if bad_keys:
                events.append(
                    {"type": "non_finite_action", "keys": bad_keys, "severity": "violation"}
                )
                blocked = True

        events.extend(self._workspace_events(action))
        blocked = blocked or any(event.get("severity") == "violation" for event in events)

        if not blocked or self.safety.on_violation == "warn":
            return dict(action), events, blocked

        if self.safety.on_violation == "stop":
            raise RuntimeError(f"Action safety violation: {events}")

        hold = dict(self._last_safe_action) if self._last_safe_action is not None else dict(action)
        events.append({"type": "held_previous_safe_action", "severity": "info"})
        return hold, events, True

    def _workspace_events(self, action: ActionDict) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []
        for side, limits in (("left", self.safety.left), ("right", self.safety.right)):
            if not self._side_has_limits(limits):
                continue
            joints = [float(action.get(f"{side}_joint_{index}.pos", 0.0)) for index in range(1, 7)]
            if self.fk_provider is None:
                events.append(
                    {"type": "workspace_fk_unavailable", "side": side, "severity": "info"}
                )
                continue
            pose = self.fk_provider.ee_pose(side, joints)
            if pose is None:
                events.append(
                    {"type": "workspace_fk_unavailable", "side": side, "severity": "info"}
                )
                continue
            if limits.min_z_m is not None:
                floor = limits.min_z_m - max(0.0, limits.allowed_below_min_m)
                if pose.z < floor:
                    events.append(
                        {
                            "type": "workspace_z_min",
                            "side": side,
                            "severity": "violation",
                            "z": pose.z,
                            "min_z_m": limits.min_z_m,
                            "allowed_below_min_m": limits.allowed_below_min_m,
                            "effective_min_z_m": floor,
                        }
                    )
        return events

    @staticmethod
    def _side_has_limits(limits: WorkspaceSideLimit) -> bool:
        return any(
            value is not None
            for value in (
                limits.min_z_m,
                limits.x_min_m,
                limits.x_max_m,
                limits.y_min_m,
                limits.y_max_m,
            )
        )


def make_fk_provider_from_namespace(args: Any) -> FKProvider | None:
    provider = str(getattr(args, "safety_fk_provider", "piper_sdk") or "none").lower()
    if provider in {"", "none", "off", "disabled"}:
        return None
    if provider == "piper_sdk":
        return PiperSDKFKProvider(dh_is_offset=int(getattr(args, "safety_dh_is_offset", 1)))
    raise ValueError(f"Unsupported safety_fk_provider: {provider}")
