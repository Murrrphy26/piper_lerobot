#!/usr/bin/env python3
"""只读查询 Piper 主臂示教/阻抗相关参数（不改任何设置）。

说明：
  - `piper_sdk.ArmMsgParamEnquiryAndConfig(4)` 只是构造一条报文对象，
    不会连 CAN、也不会发查询。正确用法是：
      arm = C_PiperInterface_V2(can_name=...)
      arm.ConnectPort(...)
      arm.ArmParamEnquiryAndConfig(4)          # 接口方法：发出查询
      arm.GetGripperTeachingPendantParamFeedback()

用法:
  conda activate piper
  cd /home/agilex/code/yjw/piper
  python tools/test/query_master_impedance_params.py
  python tools/test/query_master_impedance_params.py --can can0,can2
"""

from __future__ import annotations

import argparse
import time
from typing import Any

try:
    from piper_sdk import C_PiperInterface_V2 as PiperInterface
except ImportError:
    from piper_sdk import C_PiperInterface as PiperInterface

CTRL_MODE = {
    0x00: "待机",
    0x01: "CAN控制",
    0x02: "示教",
    0x03: "以太网",
    0x04: "WiFi",
    0x07: "离线轨迹",
}

MOVE_MODE = {
    0x00: "MOVE_P",
    0x01: "MOVE_J",
    0x02: "MOVE_L",
    0x03: "MOVE_C",
    0x04: "MOVE_M(MIT)",
    0x05: "MOVE_CPV",
}

MIT_MODE = {
    0x00: "位置速度",
    0xAD: "MIT",
    0xFF: "无效",
}

INSTALLATION = {
    0x00: "未设置/无效",
    0x01: "水平正装",
    0x02: "侧装左",
    0x03: "侧装右",
}

END_LOAD = {
    0x00: "空载",
    0x01: "半载",
    0x02: "满载",
    0x03: "无效(未设置查询项)",
}


def _safe_int(value: Any, default: int = -1) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _getattr_path(obj: Any, *names: str, default: Any = None) -> Any:
    cur = obj
    for name in names:
        if cur is None:
            return default
        cur = getattr(cur, name, None)
    return default if cur is None else cur


def connect_arm(can_name: str, piper_init: bool) -> Any:
    arm = PiperInterface(can_name=can_name)
    connect = getattr(arm, "ConnectPort")
    try:
        connect(piper_init=piper_init)
    except TypeError:
        connect()
    return arm


def disconnect_arm(arm: Any) -> None:
    disconnect = getattr(arm, "DisconnectPort", None)
    if callable(disconnect):
        try:
            disconnect()
        except Exception as exc:
            print(f"  DisconnectPort failed: {exc!r}")


def query_firmware(arm: Any, wait_s: float) -> str:
    search = getattr(arm, "SearchPiperFirmwareVersion", None)
    getter = getattr(arm, "GetPiperFirmwareVersion", None)
    if not callable(search) or not callable(getter):
        return "<firmware API missing>"
    try:
        search()
        time.sleep(wait_s)
        return str(getter())
    except Exception as exc:
        return f"<firmware query failed: {exc!r}>"


def unwrap_teaching(feedback: Any) -> Any:
    if feedback is None:
        return None
    for name in (
        "arm_gripper_teaching_param_feedback",
        "gripper_teaching_pendant_param",
        "feedback",
    ):
        inner = getattr(feedback, name, None)
        if inner is not None:
            return inner
    return feedback


def query_teaching_params(arm: Any, wait_s: float, retries: int = 5) -> Any:
    enquiry = getattr(arm, "ArmParamEnquiryAndConfig", None)
    getter = getattr(arm, "GetGripperTeachingPendantParamFeedback", None)
    if not callable(enquiry) or not callable(getter):
        return None
    last = None
    for _ in range(max(1, retries)):
        # param_enquiry=4 → 查询夹爪/示教器参数（只读请求）
        enquiry(4)
        time.sleep(wait_s)
        last = getter()
        inner = unwrap_teaching(last)
        friction = getattr(inner, "teaching_friction", 0)
        range_per = getattr(inner, "teaching_range_per", 0)
        # non-zero means real feedback arrived (0/0/0 is default empty cache)
        if range_per or getattr(inner, "max_range_config", 0) or friction:
            return last
    return last


def can_link_summary(can_name: str) -> str:
    try:
        import subprocess

        out = subprocess.check_output(
            ["ip", "-details", "-statistics", "link", "show", can_name],
            text=True,
            stderr=subprocess.STDOUT,
        )
    except Exception as exc:
        return f"<ip link failed: {exc!r}>"

    state = "?"
    err_pass = "?"
    for line in out.splitlines():
        if "can state" in line:
            parts = line.strip().split()
            if len(parts) >= 3:
                state = parts[2]
        if "error-pass" in line and "bus-off" in line:
            # header line; next numeric line has counters
            continue
        nums = line.strip().split()
        if len(nums) >= 5 and nums[0].isdigit() and "error-pass" not in line:
            # counters line under can statistics
            pass
    # parse more robustly
    lines = out.splitlines()
    for i, line in enumerate(lines):
        if "can state" in line:
            toks = line.split()
            try:
                state = toks[toks.index("state") + 1]
            except Exception:
                pass
        if line.strip().startswith("re-started") and i + 1 < len(lines):
            vals = lines[i + 1].split()
            if len(vals) >= 5:
                err_pass = vals[4]
    return f"state={state} error-pass={err_pass}"


def print_arm_report(can_name: str, piper_init: bool, wait_s: float) -> None:
    print()
    print("=" * 72)
    print(f"  CAN: {can_name}  (read-only)")
    print(f"  link: {can_link_summary(can_name)}")
    print("=" * 72)

    try:
        arm = connect_arm(can_name, piper_init=piper_init)
    except Exception as exc:
        print(f"  Connect failed: {exc!r}")
        return

    try:
        time.sleep(0.3)

        sdk_ver = None
        for name in ("GetCurrentSDKVersion", "GetCurrentInterfaceVersion", "GetCurrentProtocolVersion"):
            fn = getattr(arm, name, None)
            if callable(fn):
                try:
                    print(f"  {name}: {fn()}")
                except Exception as exc:
                    print(f"  {name}: <failed: {exc!r}>")

        print(f"  firmware: {query_firmware(arm, wait_s)}")

        status = None
        get_status = getattr(arm, "GetArmStatus", None)
        if callable(get_status):
            try:
                status = get_status()
                ctrl_mode = _safe_int(_getattr_path(status, "arm_status", "ctrl_mode"))
                arm_status = _safe_int(_getattr_path(status, "arm_status", "arm_status"))
                err_code = _getattr_path(status, "arm_status", "err_code", default="?")
                print(
                    f"  ctrl_mode: 0x{ctrl_mode:02X} ({CTRL_MODE.get(ctrl_mode, '未知')})  "
                    f"arm_status={arm_status}  err_code={err_code}"
                )
            except Exception as exc:
                print(f"  GetArmStatus failed: {exc!r}")

        mode = None
        get_mode = getattr(arm, "GetArmModeCtrl", None) or getattr(arm, "GetArmCtrlCode151", None)
        if callable(get_mode):
            try:
                mode = get_mode()
                # feedback may be nested differently across SDK versions
                mode_obj = (
                    getattr(mode, "mode_ctrl", None)
                    or getattr(mode, "ctrl_151", None)
                    or getattr(mode, "arm_mode_ctrl", None)
                    or mode
                )
                move_mode = _safe_int(getattr(mode_obj, "move_mode", -1))
                mit_mode = _safe_int(
                    getattr(mode_obj, "mit_mode", None)
                    if getattr(mode_obj, "mit_mode", None) is not None
                    else getattr(mode_obj, "is_mit_mode", -1)
                )
                spd = getattr(mode_obj, "move_spd_rate_ctrl", "?")
                install = _safe_int(getattr(mode_obj, "installation_pos", 0), 0)
                print(
                    f"  mode_ctrl(0x151): move={MOVE_MODE.get(move_mode, move_mode)}  "
                    f"mit={MIT_MODE.get(mit_mode, mit_mode)}  "
                    f"spd%={spd}  install={INSTALLATION.get(install, install)}"
                )
                print(f"  raw mode_ctrl: {mode}")
            except Exception as exc:
                print(f"  GetArmModeCtrl failed: {exc!r}")

        teaching = query_teaching_params(arm, wait_s)
        if teaching is None:
            print("  teaching params: <API missing>")
        else:
            inner = unwrap_teaching(teaching)
            range_per = getattr(inner, "teaching_range_per", 0)
            max_range = getattr(inner, "max_range_config", 0)
            friction = getattr(inner, "teaching_friction", 0)
            stamp = getattr(teaching, "time_stamp", getattr(teaching, "timestamp", "?"))
            hz = getattr(teaching, "Hz", getattr(teaching, "hz", "?"))
            print("  teaching pendant (0x47E after enquiry=4):")
            print(f"    teaching_range_per : {range_per}  (主臂行程放大%, 100~200)")
            print(f"    max_range_config   : {max_range}  (夹爪行程 mm: 0/70/100)")
            print(f"    teaching_friction  : {friction}  (示教摩擦/阻尼感, 1~10; 0=未反馈)")
            print(f"    feedback stamp/Hz  : {stamp} / {hz}")
            if not (range_per or max_range or friction):
                print("    WARNING: 全 0 —— 多半没收到 0x47E 回包（臂未上电/未进主从/固件过旧/CAN无数据）")
            print(f"  raw teaching feedback: {teaching}")

        # Note: end-load has no dedicated getter in public SDK; enquiry fields default
        # set_end_load=0x03 means "invalid / do not change" when used as a set command.
        print("  end_load note: SDK 无只读 getter；设负载需 ArmParamEnquiryAndConfig(...,0xAE,load)")
        print("                 当前脚本未改负载。上下不对称时优先核对空载/半载/满载是否贴合实际。")

        # Quick joint snapshot (read only)
        get_joints = getattr(arm, "GetArmJointMsgs", None)
        get_ctrl = getattr(arm, "GetArmJointCtrl", None)
        if callable(get_joints):
            try:
                js = get_joints().joint_state
                jdeg = [getattr(js, f"joint_{i}") / 1000.0 for i in range(1, 7)]
                print("  follower joints deg:", "[" + ", ".join(f"{v:7.2f}" for v in jdeg) + "]")
            except Exception as exc:
                print(f"  GetArmJointMsgs failed: {exc!r}")
        if callable(get_ctrl):
            try:
                jc = get_ctrl().joint_ctrl
                jdeg = [getattr(jc, f"joint_{i}") / 1000.0 for i in range(1, 7)]
                print("  leader   joints deg:", "[" + ", ".join(f"{v:7.2f}" for v in jdeg) + "]")
            except Exception as exc:
                print(f"  GetArmJointCtrl failed: {exc!r}")

    finally:
        disconnect_arm(arm)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read-only query of Piper teaching/impedance-related parameters."
    )
    parser.add_argument(
        "--can",
        default="can0,can2",
        help="Comma-separated CAN interfaces (default: can0,can2)",
    )
    parser.add_argument(
        "--wait",
        type=float,
        default=0.2,
        help="Seconds to wait after enquiry/firmware request for feedback",
    )
    parser.add_argument(
        "--piper-init",
        action="store_true",
        help="Allow ConnectPort piper_init (default off for safer read-only)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cans = [c.strip() for c in args.can.split(",") if c.strip()]
    print("Piper master-impedance / teaching param probe (READ-ONLY)")
    print(f"cans={cans}  piper_init={args.piper_init}  wait={args.wait}s")
    print()
    print("关于你在 REPL 里看到的输出:")
    print("  piper.ArmMsgParamEnquiryAndConfig(4)  → 只是本地构造消息对象")
    print("  其中 set_end_load=3(0x03) 表示「无效/不改负载」，并非臂上当前负载读数。")
    print("  要查示教摩擦，应对接口实例调用 ArmParamEnquiryAndConfig(4) 后再 Get...Feedback。")

    for can_name in cans:
        print_arm_report(can_name, piper_init=args.piper_init, wait_s=args.wait)

    print()
    print("Done.")


if __name__ == "__main__":
    main()
