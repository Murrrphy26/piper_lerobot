import argparse
import json
import signal
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from lerobot.cameras.opencv import OpenCVCameraConfig

from .config import PiperRobotConfig
from .offline_infer import action_tensor_to_dict, load_dataset, load_policy
from .piper import PiperRobot
from .preprocessing import FramePreprocessor, load_preprocessing_config
from .recorder import ARM_STATE_KEYS


def parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_camera_ref(value: str) -> int | str:
    return int(value) if value.isdigit() else value


def make_camera_configs(args: argparse.Namespace) -> dict[str, OpenCVCameraConfig]:
    camera_refs = parse_csv(args.camera_indices)
    camera_names = parse_csv(args.camera_names)
    if len(camera_refs) != len(camera_names):
        raise ValueError("--camera-indices and --camera-names must have the same number of items.")

    return {
        camera_name: OpenCVCameraConfig(
            index_or_path=parse_camera_ref(camera_ref),
            fps=args.camera_fps,
            width=args.camera_width,
            height=args.camera_height,
        )
        for camera_name, camera_ref in zip(camera_names, camera_refs, strict=True)
    }


def install_stop_handler() -> dict[str, bool]:
    stop_requested = {"value": False}

    def request_stop(signum: int, frame: object) -> None:
        del signum, frame
        stop_requested["value"] = True
        print("\nStop requested. Finishing current loop.")

    signal.signal(signal.SIGINT, request_stop)
    return stop_requested


def image_to_tensor(image: np.ndarray) -> torch.Tensor:
    if image.ndim != 3:
        raise ValueError(f"Expected HWC image, got shape {image.shape}.")
    tensor = torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1)
    return tensor.float() / 255.0


def observation_to_policy_input(
    observation: dict[str, Any],
    camera_names: list[str],
    task: str,
) -> dict[str, Any]:
    policy_input: dict[str, Any] = {
        "observation.state": torch.tensor(
            [float(observation.get(key, 0.0)) for key in ARM_STATE_KEYS],
            dtype=torch.float32,
        ),
        "task": task,
    }

    for camera_name in camera_names:
        image = observation[camera_name]
        policy_input[f"observation.images.{camera_name}"] = image_to_tensor(image)

    return policy_input


def smooth_action(
    action: dict[str, float],
    previous_action: dict[str, float] | None,
    alpha: float,
) -> dict[str, float]:
    if previous_action is None:
        return dict(action)

    return {
        key: previous_action.get(key, value) * (1.0 - alpha) + value * alpha
        for key, value in action.items()
    }


def current_action_from_observation(observation: dict[str, Any]) -> dict[str, float]:
    return {key: float(observation.get(key, 0.0)) for key in ARM_STATE_KEYS}


def max_abs_delta(left: dict[str, float], right: dict[str, float]) -> float:
    return max(abs(left[key] - right[key]) for key in left.keys() & right.keys())


def right_arm_delta(left: dict[str, float], right: dict[str, float]) -> float:
    keys = {
        *(f"right_joint_{index}.pos" for index in range(1, 7)),
        "right_gripper.pos",
    }
    return max(abs(left[key] - right[key]) for key in keys & left.keys() & right.keys())


def json_ready(action: dict[str, float]) -> dict[str, float]:
    return {key: float(value) for key, value in action.items()}


def get_action_names(policy_config: Any) -> list[str]:
    output_features = getattr(policy_config, "output_features", None)
    if isinstance(output_features, dict):
        action_feature = output_features.get("action")
        if isinstance(action_feature, dict):
            names = action_feature.get("names")
        else:
            names = getattr(action_feature, "names", None)
        if names is not None:
            return list(names)

    return list(ARM_STATE_KEYS)


def load_dataset_stats(args: argparse.Namespace) -> Any | None:
    if not args.dataset_root:
        return None

    try:
        dataset = load_dataset(
            repo_id=args.repo_id,
            dataset_root=args.dataset_root,
            video_backend=args.video_backend,
        )
    except Exception as exc:
        print(f"Warning: could not load dataset stats from {args.dataset_root}: {exc}")
        return None

    return getattr(dataset.meta, "stats", None)


def print_action_summary(prefix: str, action: dict[str, float]) -> None:
    right = ", ".join(
        f"rj{index}={action[f'right_joint_{index}.pos']:.3f}" for index in range(1, 7)
    )
    left = ", ".join(
        f"lj{index}={action[f'left_joint_{index}.pos']:.3f}" for index in range(1, 7)
    )
    print(
        f"{prefix} {left}, lg={action['left_gripper.pos']:.4f} | "
        f"{right}, rg={action['right_gripper.pos']:.4f}"
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a trained LeRobot policy on the real Piper robot.")
    parser.add_argument(
        "--policy-path",
        default="outputs/train/act_piper_pick_cube/checkpoints/last/pretrained_model",
        help="Path to the local pretrained_model checkpoint directory.",
    )
    parser.add_argument("--repo-id", default="local/piper_pick_cube")
    parser.add_argument("--dataset-root", default="data/lerobot/local/piper_pick_cube")
    parser.add_argument("--video-backend", default="pyav", choices=("pyav", "torchcodec"))
    parser.add_argument("--task", default="pick_cube", help="Task string passed to the policy.")
    parser.add_argument("--execute", action="store_true", help="Actually send actions to the robot.")
    parser.add_argument("--duration", type=float, default=10.0, help="Run duration in seconds.")
    parser.add_argument("--fps", type=float, default=10.0, help="Policy control frequency.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--follower-left-can", default="can2")
    parser.add_argument("--follower-right-can", default="can0")
    parser.add_argument("--camera-indices", default="2,4,0")
    parser.add_argument("--camera-names", default="cam_top,cam_left,cam_right")
    parser.add_argument("--camera-width", type=int, default=640)
    parser.add_argument("--camera-height", type=int, default=480)
    parser.add_argument("--camera-fps", type=int, default=30)
    parser.add_argument("--control-speed", type=int, default=10)
    parser.add_argument("--max-joint-step-rad", type=float, default=0.025)
    parser.add_argument("--max-gripper-step-m", type=float, default=0.001)
    parser.add_argument("--gripper-effort", type=int, default=1000)
    parser.add_argument("--smoothing-alpha", type=float, default=0.25)
    parser.add_argument(
        "--pace-by-reach",
        action="store_true",
        help="Hold the current policy target until the right arm gets close, instead of advancing every tick.",
    )
    parser.add_argument(
        "--advance-threshold-rad",
        type=float,
        default=0.08,
        help="Right-arm max error threshold for advancing to the next policy action.",
    )
    parser.add_argument(
        "--max-hold-steps",
        type=int,
        default=30,
        help="Maximum control ticks to hold one policy action when --pace-by-reach is enabled.",
    )
    parser.add_argument("--print-every", type=int, default=1, help="Print every N policy steps.")
    parser.add_argument("--log-jsonl", default="", help="Optional JSONL path for live rollout diagnostics.")
    return parser


def parse_args() -> argparse.Namespace:
    return build_arg_parser().parse_args()


def run_live_policy(args: argparse.Namespace) -> None:
    if args.fps <= 0:
        raise ValueError("--fps must be greater than 0.")
    if not 0.0 < args.smoothing_alpha <= 1.0:
        raise ValueError("--smoothing-alpha must be in (0, 1].")

    camera_configs = make_camera_configs(args)
    config = PiperRobotConfig(
        follower_left_port=args.follower_left_can,
        follower_right_port=args.follower_right_can,
        cameras=camera_configs,
        enable_control=args.execute,
        control_speed=args.control_speed,
        max_joint_step_rad=args.max_joint_step_rad,
        max_gripper_step_m=args.max_gripper_step_m,
        gripper_effort=args.gripper_effort,
    )

    policy_config, policy, make_pre_post_processors = load_policy(args.policy_path, args.device)
    dataset_stats = load_dataset_stats(args)
    preprocessing = getattr(args, "preprocessing", None)
    frame_preprocessor = (
        FramePreprocessor(load_preprocessing_config(preprocessing), mode="online")
        if preprocessing is not None
        else None
    )
    if frame_preprocessor is not None and frame_preprocessor.enabled:
        print("Frame preprocessing enabled for live inference.")
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_config,
        pretrained_path=args.policy_path,
        dataset_stats=dataset_stats,
        preprocessor_overrides={"device_processor": {"device": args.device}},
    )

    robot = PiperRobot(config)
    stop_requested = install_stop_handler()
    period = 1.0 / args.fps
    started_at = time.monotonic()
    previous_action: dict[str, float] | None = None
    held_predicted_action: dict[str, float] | None = None
    hold_steps = 0
    camera_names = list(camera_configs.keys())
    action_feature_names = get_action_names(policy_config)
    log_file = None
    if args.log_jsonl:
        log_path = Path(args.log_jsonl)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_file = log_path.open("w", encoding="utf-8")

    reset = getattr(policy, "reset", None)
    if callable(reset):
        reset()

    print("Running live policy.")
    print(f"  mode: {'EXECUTE' if args.execute else 'DRY RUN'}")
    print(f"  policy: {args.policy_path}")
    print(f"  fps: {args.fps}")
    print(f"  cameras: {', '.join(camera_names)}")
    if not args.execute:
        print("  no actions will be sent; add --execute only after dry-run output looks sane")
    print("Press Ctrl+C to stop.")
    print()

    try:
        robot.connect()
        if frame_preprocessor is not None:
            frame_preprocessor.reset()
        step = 0
        while not stop_requested["value"]:
            loop_started_at = time.monotonic()
            if time.monotonic() - started_at >= args.duration:
                break

            observation = robot.get_observation()
            if frame_preprocessor is not None and frame_preprocessor.enabled:
                observation = frame_preprocessor.process_observation(observation, camera_names)
            current_action = current_action_from_observation(observation)
            if previous_action is None:
                previous_action = current_action

            policy_input = observation_to_policy_input(observation, camera_names, args.task)
            should_advance = True
            if args.pace_by_reach and held_predicted_action is not None:
                target_error = right_arm_delta(held_predicted_action, current_action)
                should_advance = (
                    target_error <= args.advance_threshold_rad
                    or hold_steps >= args.max_hold_steps
                )

            if should_advance:
                processed = preprocessor(policy_input)
                with torch.inference_mode():
                    action_tensor = policy.select_action(processed)
                    action_tensor = postprocessor(action_tensor)
                predicted_action = action_tensor_to_dict(action_tensor, action_feature_names)
                held_predicted_action = predicted_action
                hold_steps = 0
            elif held_predicted_action is not None:
                predicted_action = held_predicted_action
                hold_steps += 1
            else:
                raise RuntimeError("No held action is available.")
            smoothed_action = smooth_action(predicted_action, previous_action, args.smoothing_alpha)

            if args.execute:
                sent_action = robot.send_action(smoothed_action)
                previous_action = dict(sent_action)
            else:
                sent_action = smoothed_action
                previous_action = smoothed_action

            if step % args.print_every == 0:
                print_action_summary(f"step {step:04d} pred:", predicted_action)
                print_action_summary(f"step {step:04d} send:", sent_action)
                print(
                    f"step {step:04d} delta:"
                    f" pred-current={max_abs_delta(predicted_action, current_action):.4f}"
                    f" send-current={max_abs_delta(sent_action, current_action):.4f}"
                    f" right-target={right_arm_delta(predicted_action, current_action):.4f}"
                    f" hold={hold_steps}"
                )

            if log_file is not None:
                log_record = {
                    "timestamp": time.time(),
                    "step": step,
                    "loop_s": time.monotonic() - loop_started_at,
                    "current_action": json_ready(current_action),
                    "predicted_action": json_ready(predicted_action),
                    "sent_action": json_ready(sent_action),
                    "pred_current_max_abs_delta": max_abs_delta(predicted_action, current_action),
                    "sent_current_max_abs_delta": max_abs_delta(sent_action, current_action),
                    "right_target_delta": right_arm_delta(predicted_action, current_action),
                    "hold_steps": hold_steps,
                }
                log_file.write(json.dumps(log_record, ensure_ascii=False) + "\n")
                log_file.flush()

            step += 1
            elapsed = time.monotonic() - loop_started_at
            time.sleep(max(0.0, period - elapsed))
    finally:
        if log_file is not None:
            log_file.close()
        if robot.is_connected:
            robot.disconnect()


def main() -> None:
    run_live_policy(parse_args())


if __name__ == "__main__":
    main()
