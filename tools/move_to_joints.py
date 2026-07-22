"""Move Piper dual-arm followers to given joint angles (degrees).

Starting pose reference (from read_state):
  left:  j1=0.287, j2=0.000, j3=0.000, j4=-0.634, j5=25.124, j6=3.356, gripper=0.000280
  right: j1=0.000, j2=0.000, j3=0.000, j4=1.580, j5=26.510, j6=2.634, gripper=0.000140

Examples:
  # 先回到上述起始位姿，再在真实当前位姿上 j5 +5°
  PYTHONPATH=src python tools/move_to_joints.py --preset home
  PYTHONPATH=src python tools/move_to_joints.py --preset demo

  # 连续平滑扫动约 10s（默认左右 j5 正弦），用于观察抖动
  PYTHONPATH=src python tools/move_to_joints.py --preset sweep
  PYTHONPATH=src python tools/move_to_joints.py --preset sweep --duration 10 --amp-deg 5 --freq-hz 0.3

  # 自定义目标（度）
  PYTHONPATH=src python tools/move_to_joints.py \\
    --left-joints 0.287,0,0,-0.634,30.124,3.356 \\
    --right-joints 0,0,0,1.580,31.510,2.634
"""

from __future__ import annotations

import argparse
import math
import time

from piper_towel_fold.config import PiperRobotConfig
from piper_towel_fold.piper import PiperRobot

# 起始 / home 位姿（度 / 米）——--preset home 使用
HOME_LEFT_DEG = [0.287, 0.000, 0.000, -0.634, 25.124, 3.356]
HOME_RIGHT_DEG = [0.000, 0.000, 0.000, 1.580, 26.510, 2.634]
HOME_LEFT_GRIPPER_M = 0.000280
HOME_RIGHT_GRIPPER_M = 0.000140

DEMO_J5_OFFSET_DEG = 5.0
# sweep 默认：在 j5 上做正弦往返，其它关节保持当前角
SWEEP_JOINT_INDEX = 5  # 1-based
SWEEP_DEFAULT_DURATION_S = 10.0
SWEEP_DEFAULT_AMP_DEG = 5.0
SWEEP_DEFAULT_FREQ_HZ = 0.3


def parse_joints_deg(text: str) -> list[float]:
    parts = [p.strip() for p in text.split(",") if p.strip()]
    if len(parts) != 6:
        raise argparse.ArgumentTypeError("需要恰好 6 个关节角（度），用逗号分隔")
    return [float(p) for p in parts]


def joints_to_action(side: str, joints_deg: list[float], gripper_m: float) -> dict[str, float]:
    action = {
        f"{side}_joint_{index}.pos": math.radians(value)
        for index, value in enumerate(joints_deg, start=1)
    }
    action[f"{side}_gripper.pos"] = float(gripper_m)
    return action


def arm_joints_deg(obs: dict, side: str) -> list[float]:
    return [math.degrees(obs[f"{side}_joint_{index}.pos"]) for index in range(1, 7)]


def max_joint_error_rad(obs: dict, side: str, joints_deg: list[float]) -> float:
    return max(
        abs(obs[f"{side}_joint_{index}.pos"] - math.radians(value))
        for index, value in enumerate(joints_deg, start=1)
    )


def joint_errors_deg(obs: dict, side: str, joints_deg: list[float]) -> list[float]:
    return [
        abs(math.degrees(obs[f"{side}_joint_{index}.pos"]) - value)
        for index, value in enumerate(joints_deg, start=1)
    ]


def format_arm(obs: dict, side: str) -> str:
    joints = ", ".join(
        f"j{index}={math.degrees(obs[f'{side}_joint_{index}.pos']):8.3f} deg"
        for index in range(1, 7)
    )
    gripper = obs[f"{side}_gripper.pos"]
    return f"{side}: {joints}, gripper={gripper:.6f}"


def is_all_joints_near_zero(obs: dict, *, eps_deg: float = 0.05) -> bool:
    for side in ("left", "right"):
        for index in range(1, 7):
            if abs(math.degrees(obs[f"{side}_joint_{index}.pos"])) > eps_deg:
                return False
    return True


def wait_for_valid_state(
    robot: PiperRobot,
    *,
    timeout_s: float = 5.0,
    period_s: float = 0.05,
) -> dict:
    """ConnectPort 后首帧常为全 0，需等到 CAN 关节反馈到达再控。"""
    deadline = time.monotonic() + timeout_s
    last = robot.get_observation()
    while time.monotonic() < deadline:
        if not is_all_joints_near_zero(last):
            return last
        time.sleep(period_s)
        last = robot.get_observation()

    raise RuntimeError(
        "连接后关节反馈仍为全 0。请确认已 bringup_can、从臂已使能，"
        "并先用 scripts/run_read_state.sh 能读到非零角度。"
    )


def move_to(
    robot: PiperRobot,
    left_deg: list[float],
    right_deg: list[float],
    left_gripper_m: float,
    right_gripper_m: float,
    *,
    tol_deg: float,
    period_s: float,
    timeout_s: float,
) -> None:
    tol_rad = math.radians(tol_deg)
    deadline = time.monotonic() + timeout_s
    step = 0

    while True:
        action = {}
        action.update(joints_to_action("left", left_deg, left_gripper_m))
        action.update(joints_to_action("right", right_deg, right_gripper_m))
        robot.send_action(action)

        obs = robot.get_observation()
        err_left = max_joint_error_rad(obs, "left", left_deg)
        err_right = max_joint_error_rad(obs, "right", right_deg)
        err = max(err_left, err_right)
        step += 1

        if step == 1 or step % 25 == 0:
            left_errs = joint_errors_deg(obs, "left", left_deg)
            right_errs = joint_errors_deg(obs, "right", right_deg)
            print(
                f"step={step}  max_err={math.degrees(err):.3f} deg  "
                f"(left={math.degrees(err_left):.3f}, right={math.degrees(err_right):.3f})"
            )
            print(
                "  left errs : "
                + ", ".join(f"j{i}={e:.3f}" for i, e in enumerate(left_errs, start=1))
            )
            print(
                "  right errs: "
                + ", ".join(f"j{i}={e:.3f}" for i, e in enumerate(right_errs, start=1))
            )

        if err < tol_rad:
            print("到位。")
            print(format_arm(obs, "left"))
            print(format_arm(obs, "right"))
            return

        if time.monotonic() > deadline:
            print("超时，未完全到位。当前状态：")
            print(format_arm(obs, "left"))
            print(format_arm(obs, "right"))
            raise TimeoutError(f"未能在 {timeout_s:.1f}s 内到达目标（容差 {tol_deg}°）")

        time.sleep(period_s)


def sweep_joints(
    robot: PiperRobot,
    base_left_deg: list[float],
    base_right_deg: list[float],
    left_gripper_m: float,
    right_gripper_m: float,
    *,
    duration_s: float,
    amp_deg: float,
    freq_hz: float,
    joint_index: int,
    period_s: float,
) -> None:
    """以当前位姿为中心，对指定关节做正弦连续运动，便于目视/统计抖动。"""
    if joint_index < 1 or joint_index > 6:
        raise ValueError("joint_index 必须在 1..6")
    if duration_s <= 0:
        raise ValueError("duration 必须 > 0")
    if amp_deg < 0:
        raise ValueError("amp-deg 必须 >= 0")
    if freq_hz < 0:
        raise ValueError("freq-hz 必须 >= 0")

    ji = joint_index - 1
    print(
        f"sweep：关节 j{joint_index}，幅值 ±{amp_deg:.2f}°，频率 {freq_hz:.3f} Hz，"
        f"时长 {duration_s:.1f}s，周期 {period_s:.3f}s"
    )
    print("其它关节保持起始角。Ctrl+C 可中断。")

    t0 = time.monotonic()
    next_tick = t0
    step = 0
    max_track_err = {"left": 0.0, "right": 0.0}
    max_jerk_proxy = {"left": 0.0, "right": 0.0}
    prev_fb = {"left": None, "right": None}
    prev_dfb = {"left": None, "right": None}

    while True:
        now = time.monotonic()
        t = now - t0
        if t >= duration_s:
            break

        # 正弦从 0 出发，结束时尽量回到中心（duration 不必整周期）
        offset = amp_deg * math.sin(2.0 * math.pi * freq_hz * t)
        left_deg = list(base_left_deg)
        right_deg = list(base_right_deg)
        left_deg[ji] = base_left_deg[ji] + offset
        right_deg[ji] = base_right_deg[ji] + offset

        action = {}
        action.update(joints_to_action("left", left_deg, left_gripper_m))
        action.update(joints_to_action("right", right_deg, right_gripper_m))
        robot.send_action(action)
        obs = robot.get_observation()

        for side, cmd_deg in (("left", left_deg), ("right", right_deg)):
            fb = math.degrees(obs[f"{side}_joint_{joint_index}.pos"])
            track_err = abs(fb - cmd_deg[ji])
            max_track_err[side] = max(max_track_err[side], track_err)

            prev = prev_fb[side]
            if prev is not None:
                dfb = fb - prev
                prev_d = prev_dfb[side]
                if prev_d is not None:
                    # 相邻采样一阶差分的变化，粗略反映不平滑/抖动
                    max_jerk_proxy[side] = max(max_jerk_proxy[side], abs(dfb - prev_d))
                prev_dfb[side] = dfb
            prev_fb[side] = fb

        step += 1
        if step == 1 or step % 25 == 0:
            print(
                f"t={t:5.2f}s  offset={offset:+6.2f}°  "
                f"cmd L/R j{joint_index}={left_deg[ji]:7.2f}/{right_deg[ji]:7.2f}  "
                f"fb={math.degrees(obs[f'left_joint_{joint_index}.pos']):7.2f}/"
                f"{math.degrees(obs[f'right_joint_{joint_index}.pos']):7.2f}"
            )

        next_tick += period_s
        sleep_s = next_tick - time.monotonic()
        if sleep_s > 0:
            time.sleep(sleep_s)
        else:
            next_tick = time.monotonic()

    # 收尾：回到中心位姿
    print("sweep 结束，回到起始中心角…")
    move_to(
        robot,
        list(base_left_deg),
        list(base_right_deg),
        left_gripper_m,
        right_gripper_m,
        tol_deg=1.0,
        period_s=period_s,
        timeout_s=max(10.0, duration_s),
    )
    print(
        "抖动粗测（越大越不平）：\n"
        f"  max |cmd-fb|  left={max_track_err['left']:.3f}°  right={max_track_err['right']:.3f}°\n"
        f"  max |ΔΔfb|   left={max_jerk_proxy['left']:.3f}°  right={max_jerk_proxy['right']:.3f}°"
    )
    print("目视：若运动过程有明显顿挫/颤动，即存在较大抖动。")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="将 Piper 双臂从臂移动到给定关节角目标。")
    parser.add_argument("--left-can", default="can2")
    parser.add_argument("--right-can", default="can0")
    parser.add_argument(
        "--preset",
        choices=("home", "demo", "sweep"),
        default=None,
        help="home=回到记录位姿；demo=当前位姿 j5 +5°；sweep=连续正弦扫动测抖动",
    )
    parser.add_argument(
        "--left-joints",
        type=parse_joints_deg,
        default=None,
        help="左臂 6 关节目标角（度）",
    )
    parser.add_argument(
        "--right-joints",
        type=parse_joints_deg,
        default=None,
        help="右臂 6 关节目标角（度）",
    )
    parser.add_argument("--left-gripper", type=float, default=None)
    parser.add_argument("--right-gripper", type=float, default=None)
    parser.add_argument("--control-speed", type=int, default=15)
    parser.add_argument("--max-joint-step-rad", type=float, default=0.04)
    parser.add_argument("--tol-deg", type=float, default=1.0, help="到位容差（度）")
    parser.add_argument("--period", type=float, default=0.02, help="控制周期（秒）")
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument(
        "--state-wait",
        type=float,
        default=5.0,
        help="连接后等待非零关节反馈的最长时间（秒）",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=SWEEP_DEFAULT_DURATION_S,
        help="sweep 时长（秒），默认 10",
    )
    parser.add_argument(
        "--amp-deg",
        type=float,
        default=SWEEP_DEFAULT_AMP_DEG,
        help="sweep 正弦幅值（度），默认 ±5",
    )
    parser.add_argument(
        "--freq-hz",
        type=float,
        default=SWEEP_DEFAULT_FREQ_HZ,
        help="sweep 正弦频率（Hz），默认 0.3",
    )
    parser.add_argument(
        "--sweep-joint",
        type=int,
        default=SWEEP_JOINT_INDEX,
        help="sweep 的关节序号 1..6，默认 5",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只打印目标，不连接/不下发",
    )
    return parser.parse_args()


def resolve_static_targets(
    args: argparse.Namespace,
) -> tuple[list[float] | None, list[float] | None, float | None, float | None]:
    """demo/sweep 需要读真实状态后再算，这里先返回 None。"""
    if args.preset in ("demo", "sweep"):
        return None, None, None, None
    if args.preset == "home":
        return (
            list(HOME_LEFT_DEG),
            list(HOME_RIGHT_DEG),
            HOME_LEFT_GRIPPER_M,
            HOME_RIGHT_GRIPPER_M,
        )
    if args.left_joints is not None and args.right_joints is not None:
        lg = HOME_LEFT_GRIPPER_M if args.left_gripper is None else args.left_gripper
        rg = HOME_RIGHT_GRIPPER_M if args.right_gripper is None else args.right_gripper
        return args.left_joints, args.right_joints, lg, rg
    raise SystemExit(
        "请指定 --preset home|demo|sweep，或同时提供 --left-joints 与 --right-joints"
    )


def targets_from_demo(obs: dict) -> tuple[list[float], list[float], float, float]:
    left = arm_joints_deg(obs, "left")
    right = arm_joints_deg(obs, "right")
    left[4] += DEMO_J5_OFFSET_DEG
    right[4] += DEMO_J5_OFFSET_DEG
    return left, right, obs["left_gripper.pos"], obs["right_gripper.pos"]


def print_targets(
    left_deg: list[float],
    right_deg: list[float],
    left_g: float,
    right_g: float,
) -> None:
    print("目标位姿（度）：")
    print(
        "left:  "
        + ", ".join(f"j{i}={v:8.3f}" for i, v in enumerate(left_deg, start=1))
        + f", gripper={left_g:.6f}"
    )
    print(
        "right: "
        + ", ".join(f"j{i}={v:8.3f}" for i, v in enumerate(right_deg, start=1))
        + f", gripper={right_g:.6f}"
    )


def main() -> None:
    args = parse_args()
    left_deg, right_deg, left_g, right_g = resolve_static_targets(args)

    if args.dry_run and args.preset in ("demo", "sweep"):
        print(f"{args.preset} 需连接后读取真实位姿；dry-run 跳过。")
        return

    if args.dry_run:
        assert left_deg is not None and right_deg is not None
        assert left_g is not None and right_g is not None
        if args.left_gripper is not None:
            left_g = args.left_gripper
        if args.right_gripper is not None:
            right_g = args.right_gripper
        print_targets(left_deg, right_deg, left_g, right_g)
        print("dry-run：未连接机械臂。")
        return

    # sweep 需要限速能跟上正弦；默认 0.04rad/step 对慢扫足够，仍允许用户加大
    max_step = args.max_joint_step_rad
    if args.preset == "sweep" and args.max_joint_step_rad <= 0.04:
        # 略放宽，避免限速把正弦切成折线（看起来像抖）
        max_step = 0.08

    config = PiperRobotConfig(
        follower_left_port=args.left_can,
        follower_right_port=args.right_can,
        enable_control=True,
        control_speed=args.control_speed,
        max_joint_step_rad=max_step,
        cameras={},
    )
    robot = PiperRobot(config)

    try:
        robot.connect()
        print(f"等待关节反馈（最多 {args.state_wait:.1f}s）…")
        obs = wait_for_valid_state(robot, timeout_s=args.state_wait)
        print("当前位姿：")
        print(format_arm(obs, "left"))
        print(format_arm(obs, "right"))

        if args.preset == "sweep":
            base_left = arm_joints_deg(obs, "left")
            base_right = arm_joints_deg(obs, "right")
            left_g = obs["left_gripper.pos"]
            right_g = obs["right_gripper.pos"]
            if args.left_gripper is not None:
                left_g = args.left_gripper
            if args.right_gripper is not None:
                right_g = args.right_gripper
            sweep_joints(
                robot,
                base_left,
                base_right,
                left_g,
                right_g,
                duration_s=args.duration,
                amp_deg=args.amp_deg,
                freq_hz=args.freq_hz,
                joint_index=args.sweep_joint,
                period_s=args.period,
            )
            return

        if args.preset == "demo":
            left_deg, right_deg, left_g, right_g = targets_from_demo(obs)
            print(f"demo：在当前位姿上左右 j5 +{DEMO_J5_OFFSET_DEG}°")
        else:
            assert left_deg is not None and right_deg is not None
            assert left_g is not None and right_g is not None
            if args.left_gripper is not None:
                left_g = args.left_gripper
            if args.right_gripper is not None:
                right_g = args.right_gripper

        print_targets(left_deg, right_deg, left_g, right_g)
        print("开始移动…")
        move_to(
            robot,
            left_deg,
            right_deg,
            left_g,
            right_g,
            tol_deg=args.tol_deg,
            period_s=args.period,
            timeout_s=args.timeout,
        )
    except KeyboardInterrupt:
        print("\n已中断。")
    finally:
        if robot.is_connected:
            robot.disconnect()


if __name__ == "__main__":
    main()
