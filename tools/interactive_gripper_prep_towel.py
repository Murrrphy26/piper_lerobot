#!/usr/bin/env python3
"""毛巾折叠推理前：SDK 打开双臂夹爪，再交互确认逐侧合拢。

流程：
  1. 连接左右从臂，打开两侧夹爪
  2. 控制台提示「是否关闭左臂夹爪」，输入 y 后合拢左夹爪
  3. 同样对右臂
  4. 断开连接，交还给后续 policy live

用法：
  PYTHONPATH=src python tools/interactive_gripper_prep_towel.py --config configs/record_towel_fold_pi05.json
  PYTHONPATH=src python tools/interactive_gripper_prep_towel.py --left-can can2 --right-can can0
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any


def import_piper_interface() -> type:
    try:
        from piper_sdk import C_PiperInterface_V2

        return C_PiperInterface_V2
    except ImportError:
        from piper_sdk import C_PiperInterface

        return C_PiperInterface


def load_json_config(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("config must be a JSON object")
    return data


def resolve_can_ports(config: dict[str, Any] | None) -> tuple[str, str]:
    if not config:
        return "can2", "can0"

    policy_live = config.get("policy_live") or {}
    if not isinstance(policy_live, dict):
        policy_live = {}

    def pick(key: str, default: str) -> str:
        for block in (policy_live, config):
            if key in block and block[key] not in (None, ""):
                return str(block[key])
        return default

    left = pick("follower_left_can", pick("left_can", "can2"))
    right = pick("follower_right_can", pick("right_can", "can0"))
    return left, right


def call_connect_port(arm: Any, piper_init: bool) -> None:
    connect_port = getattr(arm, "ConnectPort")
    try:
        connect_port(piper_init=piper_init)
    except TypeError:
        connect_port()


def enable_sdk_control(arm: Any, control_speed: int) -> None:
    enable_arm = getattr(arm, "EnableArm", None)
    if callable(enable_arm):
        try:
            enable_arm(7)
        except TypeError:
            enable_arm()

    motion_ctrl_2 = getattr(arm, "MotionCtrl_2", None)
    if callable(motion_ctrl_2):
        try:
            motion_ctrl_2(0x01, 0x01, int(control_speed), 0x00)
        except TypeError:
            motion_ctrl_2(0x01, 0x01, int(control_speed))


def make_arm(interface_cls: type, can_name: str, *, piper_init: bool, control_speed: int) -> Any:
    create_kwargs: dict[str, object] = {"can_name": can_name}
    try:
        from piper_sdk import LogLevel

        create_kwargs["logger_level"] = LogLevel.SILENT
    except Exception:
        pass

    try:
        arm = interface_cls(**create_kwargs)
    except TypeError:
        arm = interface_cls(can_name=can_name)

    call_connect_port(arm, piper_init=piper_init)
    enable_sdk_control(arm, control_speed)
    return arm


def gripper_ctrl(arm: Any, position_m: float, effort: int) -> None:
    """下发夹爪目标。position_m：0=合拢，约 0.07=全开（米）。"""
    method = getattr(arm, "GripperCtrl", None)
    if not callable(method):
        raise RuntimeError("当前 piper_sdk 无 GripperCtrl")
    gripper_um = int(round(float(position_m) * 1_000_000.0))
    method(gripper_um, int(effort), 0x01, 0)


def read_gripper_m(arm: Any) -> float | None:
    getter = getattr(arm, "GetArmGripperMsgs", None)
    if not callable(getter):
        return None
    try:
        return float(getter().gripper_state.grippers_angle) / 1_000_000.0
    except Exception:
        return None


def disconnect_arm(arm: Any) -> None:
    disconnect = getattr(arm, "DisconnectPort", None)
    if callable(disconnect):
        disconnect()


def wait_for_yes(prompt: str) -> None:
    """阻塞直到用户输入 y / Y。"""
    while True:
        answer = input(prompt).strip().lower()
        if answer == "y":
            return
        print("  （请输入 y 确认，或 Ctrl+C 中止）")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="毛巾折叠：归位后 SDK 打开夹爪，交互确认逐侧合拢。",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="录制/推理 JSON，用于解析 follower_*_can（可被 --left/right-can 覆盖）",
    )
    parser.add_argument("--left-can", default=None)
    parser.add_argument("--right-can", default=None)
    parser.add_argument(
        "--open-m",
        type=float,
        default=0.007,
        help="打开夹爪目标位置（米），默认 0.07（全开）",
    )
    parser.add_argument(
        "--close-m",
        type=float,
        default=0.0,
        help="合拢夹爪目标位置（米），默认 0.0",
    )
    parser.add_argument("--gripper-effort", type=int, default=1000)
    parser.add_argument("--control-speed", type=int, default=15)
    parser.add_argument("--piper-init", action="store_true")
    parser.add_argument(
        "--settle",
        type=float,
        default=0.8,
        help="每次夹爪指令后等待秒数",
    )
    parser.add_argument(
        "--skip-prompt",
        action="store_true",
        help="跳过交互：打开后直接合拢左右夹爪（调试用）",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config: dict[str, Any] | None = None
    if args.config:
        config = load_json_config(Path(args.config))

    left_can, right_can = resolve_can_ports(config)
    if args.left_can:
        left_can = args.left_can
    if args.right_can:
        right_can = args.right_can

    print("=== 毛巾折叠：夹爪交互准备 ===")
    print(f"  left_can={left_can}  right_can={right_can}")
    print(f"  open={args.open_m:.4f} m  close={args.close_m:.4f} m  effort={args.gripper_effort}")
    print()

    interface_cls = import_piper_interface()
    left_arm = None
    right_arm = None
    try:
        print(f"连接左臂 {left_can} …")
        left_arm = make_arm(
            interface_cls, left_can, piper_init=args.piper_init, control_speed=args.control_speed
        )
        print(f"连接右臂 {right_can} …")
        right_arm = make_arm(
            interface_cls, right_can, piper_init=args.piper_init, control_speed=args.control_speed
        )
        time.sleep(0.3)

        print("打开左右夹爪 …")
        gripper_ctrl(left_arm, args.open_m, args.gripper_effort)
        gripper_ctrl(right_arm, args.open_m, args.gripper_effort)
        time.sleep(args.settle)
        lg = read_gripper_m(left_arm)
        rg = read_gripper_m(right_arm)
        if lg is not None and rg is not None:
            print(f"  当前反馈：left={lg:.4f} m  right={rg:.4f} m")
        print("夹爪已打开。请将毛巾/物体放好后再继续。")
        print()

        if args.skip_prompt:
            print("skip-prompt：直接合拢左右夹爪")
            gripper_ctrl(left_arm, args.close_m, args.gripper_effort)
            time.sleep(args.settle)
            gripper_ctrl(right_arm, args.close_m, args.gripper_effort)
            time.sleep(args.settle)
        else:
            wait_for_yes("是否关闭左臂夹爪？[y] ")
            print("合拢左臂夹爪 …")
            gripper_ctrl(left_arm, args.close_m, args.gripper_effort)
            time.sleep(args.settle)
            lg = read_gripper_m(left_arm)
            if lg is not None:
                print(f"  左夹爪反馈：{lg:.4f} m")
            print()

            wait_for_yes("是否关闭右臂夹爪？[y] ")
            print("合拢右臂夹爪 …")
            gripper_ctrl(right_arm, args.close_m, args.gripper_effort)
            time.sleep(args.settle)
            rg = read_gripper_m(right_arm)
            if rg is not None:
                print(f"  右夹爪反馈：{rg:.4f} m")
            print()

        print("夹爪准备完成，随后开始推理。")
    except KeyboardInterrupt:
        print("\n已中断夹爪准备。", file=sys.stderr)
        raise SystemExit(130) from None
    finally:
        if left_arm is not None:
            disconnect_arm(left_arm)
        if right_arm is not None:
            disconnect_arm(right_arm)


if __name__ == "__main__":
    main()
