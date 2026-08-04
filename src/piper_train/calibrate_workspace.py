"""Calibrate simple Piper workspace safety limits from live arm poses.

Usage example:

    python -m piper_train.calibrate_workspace configs/record_towel_fold_act.json \\
      --side right \\
      --output configs/calibration/workspace_towel.json \\
      --allowed-below-min-m 0.005

Move the selected follower arm close to the table before pressing Enter. The
script reads the current follower joints, runs SDK FK, and writes a safety block
that can be copied into a config file.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import math
from pathlib import Path
from typing import Any

from .action_pipeline import PiperSDKFKProvider
from .start_policy_live import load_config


SIDES = ("left", "right")


def _load_sdk_interface() -> type:
    try:
        from piper_sdk import C_PiperInterface_V2
    except ImportError:
        try:
            from piper_sdk import C_PiperInterface as C_PiperInterface_V2
        except ImportError as exc:
            raise ImportError("piper_sdk is required to calibrate workspace safety.") from exc
    return C_PiperInterface_V2


def _sdk_create_kwargs(can_name: str) -> dict[str, Any]:
    kwargs: dict[str, Any] = {"can_name": can_name}
    try:
        from piper_sdk import LogLevel

        kwargs["logger_level"] = LogLevel.SILENT
    except Exception:
        pass
    return kwargs


def _connect_arm(can_name: str) -> object:
    interface_cls = _load_sdk_interface()
    arm = interface_cls(**_sdk_create_kwargs(can_name))
    arm.ConnectPort()
    return arm


def _disconnect_arm(arm: object) -> None:
    disconnect = getattr(arm, "DisconnectPort", None)
    if callable(disconnect):
        disconnect()


def _joints_mdeg_to_rad(values: list[float | int]) -> list[float]:
    return [math.radians(float(value) / 1000.0) for value in values]


def _read_follower_joints_rad(arm: object) -> list[float]:
    joint_state = arm.GetArmJointMsgs().joint_state
    return _joints_mdeg_to_rad(
        [
            joint_state.joint_1,
            joint_state.joint_2,
            joint_state.joint_3,
            joint_state.joint_4,
            joint_state.joint_5,
            joint_state.joint_6,
        ]
    )


def _resolve_can_port(config: dict[str, Any], side: str, override: str | None) -> str:
    if override:
        return override
    key = f"follower_{side}_can"
    value = config.get(key)
    if value is None or str(value).strip() == "":
        raise ValueError(f"Config does not define {key}; pass --can to override.")
    return str(value)


def _round_list(values: list[float], digits: int = 6) -> list[float]:
    return [round(float(value), digits) for value in values]


def calibrate_side(
    *,
    config: dict[str, Any],
    config_path: Path,
    side: str,
    can_override: str | None,
    dh_is_offset: int,
    z_margin_m: float,
    prompt: bool,
) -> dict[str, Any]:
    can_name = _resolve_can_port(config, side, can_override)
    if prompt:
        print()
        print(f"[{side}] CAN port: {can_name}")
        print("Move the follower gripper to the near-table safety reference pose.")
        print("Press Enter when the gripper is at the reference height; Ctrl+C to cancel.")
        input("> ")

    arm = _connect_arm(can_name)
    try:
        joints_rad = _read_follower_joints_rad(arm)
    finally:
        _disconnect_arm(arm)

    fk = PiperSDKFKProvider(dh_is_offset=dh_is_offset)
    pose = fk.ee_pose(side, joints_rad)
    if pose is None:
        raise RuntimeError(f"SDK FK returned no pose for {side}.")

    min_z_m = float(pose.z) + float(z_margin_m)
    return {
        "side": side,
        "can_port": can_name,
        "source_config": str(config_path),
        "dh_is_offset": int(dh_is_offset),
        "joints_rad": _round_list(joints_rad),
        "ee_xyz_m": _round_list([pose.x, pose.y, pose.z]),
        "z_margin_m": round(float(z_margin_m), 6),
        "min_z_m": round(min_z_m, 6),
    }


def build_output(
    *,
    config_path: Path,
    samples: dict[str, dict[str, Any]],
    allowed_below_min_m: float,
    on_violation: str,
    dh_is_offset: int,
) -> dict[str, Any]:
    safety: dict[str, Any] = {
        "enabled": True,
        "fk_provider": "piper_sdk",
        "dh_is_offset": int(dh_is_offset),
        "allowed_below_min_m": round(float(allowed_below_min_m), 6),
        "on_violation": on_violation,
    }
    for side, sample in samples.items():
        safety[f"{side}_min_z_m"] = sample["min_z_m"]

    return {
        "version": 1,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source_config": str(config_path),
        "samples": samples,
        "safety": safety,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Calibrate Piper workspace safety z limits from current follower arm poses."
    )
    parser.add_argument("config", help="Existing task config used to read follower CAN ports.")
    parser.add_argument(
        "--side",
        choices=("left", "right", "both"),
        default="right",
        help="Which follower arm to calibrate.",
    )
    parser.add_argument("--can", default=None, help="Override CAN port when calibrating one side.")
    parser.add_argument(
        "--output",
        default="configs/calibration/workspace_safety.json",
        help="Where to write the calibration JSON.",
    )
    parser.add_argument(
        "--allowed-below-min-m",
        type=float,
        default=0.005,
        help="How far the end-effector may go below the calibrated min z before blocking.",
    )
    parser.add_argument(
        "--z-margin-m",
        type=float,
        default=0.0,
        help="Extra positive margin added to the measured z when writing min_z_m.",
    )
    parser.add_argument(
        "--on-violation",
        choices=("hold_previous", "warn", "stop"),
        default="hold_previous",
        help="Safety behavior to write into the output safety block.",
    )
    parser.add_argument(
        "--dh-is-offset",
        type=int,
        choices=(0, 1),
        default=1,
        help="Passed to piper_sdk.C_PiperForwardKinematics(dh_is_offset=...).",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Do not wait for Enter before reading each pose.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    config_path = Path(args.config)
    config = load_config(config_path)

    sides = SIDES if args.side == "both" else (args.side,)
    if args.can and args.side == "both":
        raise ValueError("--can can only be used when --side is left or right.")

    samples: dict[str, dict[str, Any]] = {}
    for side in sides:
        samples[side] = calibrate_side(
            config=config,
            config_path=config_path,
            side=side,
            can_override=args.can,
            dh_is_offset=args.dh_is_offset,
            z_margin_m=args.z_margin_m,
            prompt=not args.yes,
        )
        print(f"[{side}] ee_xyz_m={samples[side]['ee_xyz_m']} -> min_z_m={samples[side]['min_z_m']}")

    output = build_output(
        config_path=config_path,
        samples=samples,
        allowed_below_min_m=args.allowed_below_min_m,
        on_violation=args.on_violation,
        dh_is_offset=args.dh_is_offset,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print()
    print(f"Wrote workspace safety calibration: {output_path}")
    print("Copy this block into your config under policy_live.safety / async_inference.safety / safety:")
    print(json.dumps(output["safety"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
