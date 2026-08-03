#!/usr/bin/env python3
"""示教共 CAN：同一 CAN 上对比【从臂反馈】vs【主臂指令】。

用法（示教模式、主臂航空头保持连接）：
  python tools/sniff_leader_follower_can.py --can can2
  python tools/sniff_leader_follower_can.py --can can2,can0 --duration 30

从臂 = GetArm*Msgs（实际位置）
主臂 = GetArm*Ctrl（发给从臂的目标）
夹爪单位 mm（开合行程）；关节单位 deg。
"""

from __future__ import annotations

import argparse
import time

try:
    from piper_sdk import C_PiperInterface_V2 as PiperInterface
except ImportError:
    from piper_sdk import C_PiperInterface as PiperInterface

MODE_NAMES = {
    0x00: "待机",
    0x01: "CAN控制/跟随",
    0x02: "示教",
    0x03: "以太网",
    0x04: "WiFi",
    0x07: "离线轨迹",
}


def joints_to_deg(msg) -> list[float]:
    return [
        float(msg.joint_1) / 1000.0,
        float(msg.joint_2) / 1000.0,
        float(msg.joint_3) / 1000.0,
        float(msg.joint_4) / 1000.0,
        float(msg.joint_5) / 1000.0,
        float(msg.joint_6) / 1000.0,
    ]


def fmt_joints(values: list[float]) -> str:
    return "[" + ", ".join(f"{v:7.2f}" for v in values) + "]"


def sniff_one(can_name: str, hz: float, duration: float) -> None:
    arm = PiperInterface(can_name=can_name)
    arm.ConnectPort()
    period = 1.0 / hz
    end_at = time.monotonic() + duration

    print()
    print("=" * 88)
    print(f"  {can_name}  主臂 / 从臂 对比")
    print("  从臂 = 实际反馈 GetArm*Msgs    主臂 = 示教指令 GetArm*Ctrl")
    print("  关节单位 deg（J1..J6）        夹爪单位 mm（开合行程，越小越合拢）")
    print("=" * 88)

    frame = 0
    try:
        while time.monotonic() < end_at:
            t0 = time.monotonic()
            follower_joints = joints_to_deg(arm.GetArmJointMsgs().joint_state)
            leader_joints = joints_to_deg(arm.GetArmJointCtrl().joint_ctrl)
            # SDK grippers_angle 单位 0.001mm → 先转米再 *1000 得 mm
            follower_grip_mm = arm.GetArmGripperMsgs().gripper_state.grippers_angle / 1_000_000.0 * 1000.0
            leader_grip_mm = arm.GetArmGripperCtrl().gripper_ctrl.grippers_angle / 1_000_000.0 * 1000.0
            joint_diff = [abs(a - b) for a, b in zip(leader_joints, follower_joints, strict=True)]
            grip_diff = leader_grip_mm - follower_grip_mm

            try:
                mode_code = int(arm.GetArmStatus().arm_status.ctrl_mode)
                mode_name = MODE_NAMES.get(mode_code, f"未知(0x{mode_code:x})")
            except Exception:
                mode_code = -1
                mode_name = "?"

            frame += 1
            mode_text = f"{mode_name}(0x{mode_code:x})" if mode_code >= 0 else mode_name
            print(f"\n[{can_name}] #{frame:04d}  模式={mode_text}")
            print(f"  从臂关节 deg  {fmt_joints(follower_joints)}")
            print(f"  主臂关节 deg  {fmt_joints(leader_joints)}")
            print(f"  关节|主-从|max = {max(joint_diff):6.3f} deg")
            contact_hint = "  ← 主臂更合拢(可能夹到物体)" if grip_diff < -5 else ""
            print(
                f"  从臂夹爪 = {follower_grip_mm:6.1f} mm    "
                f"主臂夹爪 = {leader_grip_mm:6.1f} mm    "
                f"主-从 = {grip_diff:+6.1f} mm{contact_hint}"
            )

            sleep = period - (time.monotonic() - t0)
            if sleep > 0:
                time.sleep(sleep)
    finally:
        disconnect = getattr(arm, "DisconnectPort", None)
        if callable(disconnect):
            disconnect()


def main() -> None:
    parser = argparse.ArgumentParser(description="对比同一 CAN 上的主臂指令与从臂反馈")
    parser.add_argument("--can", default="can0", help="例如 can2 或 can2,can0")
    parser.add_argument("--hz", type=float, default=5.0)
    parser.add_argument("--duration", type=float, default=20.0)
    args = parser.parse_args()

    cans = [c.strip() for c in args.can.split(",") if c.strip()]
    print("请保持主臂航空头连接，并处于示教跟随。")
    print("晃主臂：主臂关节应先变；夹住物体时通常 主臂夹爪 < 从臂夹爪。")
    for can_name in cans:
        sniff_one(can_name, hz=args.hz, duration=args.duration)


if __name__ == "__main__":
    main()
