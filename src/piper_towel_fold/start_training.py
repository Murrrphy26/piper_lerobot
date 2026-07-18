import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from .train_loss import report_from_log


DEFAULTS: dict[str, Any] = {
    "policy_type": "act",
    "device": "cuda",
    "steps": 5000,
    "batch_size": 4,
    "log_freq": 100,
    "save_freq": 1000,
    "wandb_enable": False,
    "video_backend": "pyav",
    "pytorch_alloc_conf": "expandable_segments:True",
    "tolerance_s": 5.0,
    "outcome_filter": "exclude-failures",
    "include_unknown": False,
    "exclude_unlabeled": True,
    "push_to_hub": False,
    # 训完后自动根据 train.log 打印 loss 收敛判断（ACT / PI05 通用）
    "report_loss_on_finish": True,
}


def load_config(config_path: Path) -> dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as config_file:
        data = json.load(config_file)
    if not isinstance(data, dict):
        raise ValueError("Config file must contain a JSON object.")
    return data


def training_config(config: dict[str, Any]) -> dict[str, Any]:
    training = config.get("training", {})
    if training is None:
        training = {}
    if not isinstance(training, dict):
        raise ValueError("'training' must be an object when present.")

    result = dict(DEFAULTS)
    result.update(training)
    return result


def dataset_root(config: dict[str, Any]) -> Path:
    repo_id = config.get("repo_id")
    root = config.get("root", "data/lerobot")
    if not repo_id:
        raise ValueError("Config must contain 'repo_id'.")
    return Path(str(root)) / str(repo_id)


def patch_image_feature_names(info_path: Path) -> None:
    info = json.loads(info_path.read_text(encoding="utf-8"))
    features = info.get("features", {})
    if not isinstance(features, dict):
        return

    updated = False
    for feature_name, feature in features.items():
        if not isinstance(feature, dict):
            continue
        if not feature_name.startswith("observation.images."):
            continue
        if feature.get("dtype") not in {"image", "video"}:
            continue
        if "names" in feature:
            continue

        shape = feature.get("shape")
        if isinstance(shape, list) and len(shape) == 3:
            feature["names"] = ["height", "width", "channels"]
            updated = True

    if updated:
        info_path.write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
        print(f"Patched dataset metadata: {info_path}")


def validate_dataset(path: Path) -> None:
    info_path = path / "meta" / "info.json"
    if not info_path.exists():
        raise FileNotFoundError(
            f"Dataset metadata not found: {info_path}. Record data first or check repo_id/root."
        )

    patch_image_feature_names(info_path)
    if not any((path / "data").rglob("*.parquet")):
        raise FileNotFoundError(f"No parquet files found under {path / 'data'}.")


def append_training_episodes(
    cmd: list[str],
    dataset_root: Path,
    training: dict[str, Any],
) -> list[int] | None:
    from .episode_outcomes import resolve_training_episodes, summarize_outcomes

    episodes = resolve_training_episodes(dataset_root, training)
    if episodes is not None:
        cmd.append(f"--dataset.episodes={json.dumps(episodes)}")

    summary = summarize_outcomes(dataset_root)
    counts = summary["counts"]
    print("Episode outcome filter")
    print(f"  mode: {training.get('outcome_filter', 'exclude-failures')}")
    print(
        "  dataset labels: "
        f"success={counts['success']} "
        f"failure={counts['failure']} "
        f"unknown={counts['unknown']} "
        f"unlabeled={counts['unlabeled']}"
    )
    if episodes is None:
        print("  training episodes: all")
    else:
        print(f"  training episodes: {len(episodes)} of {summary['num_episodes']} ({episodes})")
    print()
    return episodes


def command_from_config(config: dict[str, Any], training: dict[str, Any], path: Path) -> list[str]:
    repo_id = str(config["repo_id"])
    policy_type = str(training["policy_type"])
    job_name = str(training.get("job_name") or f"{policy_type}_{repo_id.split('/')[-1]}")
    output_dir = str(training.get("output_dir") or Path("outputs") / "train" / job_name)
    policy_repo_id = str(training.get("policy_repo_id") or f"local/{job_name}")

    cmd = [
        "lerobot-train",
        f"--dataset.repo_id={repo_id}",
        f"--dataset.root={path}",
        f"--policy.type={policy_type}",
        f"--output_dir={output_dir}",
        f"--job_name={job_name}",
        f"--policy.device={training['device']}",
        f"--dataset.video_backend={training['video_backend']}",
        f"--steps={training['steps']}",
        f"--batch_size={training['batch_size']}",
        f"--log_freq={training['log_freq']}",
        f"--save_freq={training['save_freq']}",
        f"--wandb.enable={str(training['wandb_enable']).lower()}",
        f"--policy.repo_id={policy_repo_id}",
        f"--policy.push_to_hub={str(training.get('push_to_hub', False)).lower()}",
    ]
    
    # 添加容差参数（如果配置中有的话）
    if "tolerance_s" in training:
        cmd.append(f"--tolerance_s={training['tolerance_s']}")

    append_act_piper_policy_options(cmd, training)
    append_pi05_policy_options(cmd, training)
    append_training_episodes(cmd, path, training)
    return cmd


def append_pi05_policy_options(cmd: list[str], training: dict[str, Any]) -> None:
    if training.get("policy_type") != "pi05":
        return

    if training.get("pretrained_path"):
        cmd.append(f"--policy.pretrained_path={training['pretrained_path']}")

    if "dtype" in training:
        cmd.append(f"--policy.dtype={training['dtype']}")

    if "compile_model" in training:
        cmd.append(f"--policy.compile_model={str(training['compile_model']).lower()}")

    if "gradient_checkpointing" in training:
        cmd.append(
            f"--policy.gradient_checkpointing={str(training['gradient_checkpointing']).lower()}"
        )

    if "freeze_vision_encoder" in training:
        cmd.append(
            f"--policy.freeze_vision_encoder={str(training['freeze_vision_encoder']).lower()}"
        )

    if "train_expert_only" in training:
        cmd.append(f"--policy.train_expert_only={str(training['train_expert_only']).lower()}")

    normalization_mapping = training.get("normalization_mapping")
    if normalization_mapping:
        cmd.append(f"--policy.normalization_mapping={json.dumps(normalization_mapping)}")

    if "use_relative_actions" in training:
        cmd.append(f"--policy.use_relative_actions={str(training['use_relative_actions']).lower()}")


def append_act_piper_policy_options(cmd: list[str], training: dict[str, Any]) -> None:
    if training.get("policy_type") != "act_piper":
        return

    camera_scales = training.get("camera_scales")
    if camera_scales:
        cmd.append(f"--policy.camera_scales={json.dumps(camera_scales)}")

    if "learnable_camera_scales" in training:
        cmd.append(
            f"--policy.learnable_camera_scales={str(training['learnable_camera_scales']).lower()}"
        )

    if "use_camera_id_embed" in training:
        cmd.append(f"--policy.use_camera_id_embed={str(training['use_camera_id_embed']).lower()}")

    if "chunk_size" in training:
        cmd.append(f"--policy.chunk_size={int(training['chunk_size'])}")

    if "n_action_steps" in training:
        cmd.append(f"--policy.n_action_steps={int(training['n_action_steps'])}")


def resolve_output_dir(config: dict[str, Any], training: dict[str, Any]) -> Path:
    repo_id = str(config["repo_id"])
    policy_type = str(training["policy_type"])
    job_name = str(training.get("job_name") or f"{policy_type}_{repo_id.split('/')[-1]}")
    return Path(str(training.get("output_dir") or Path("outputs") / "train" / job_name))


def run_training_with_log(command: list[str], env: dict[str, str], log_path: Path) -> None:
    """Run lerobot-train, stream stdout live, and tee into log_path."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"  train log: {log_path}")
    print()

    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            command,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        try:
            for line in process.stdout:
                sys.stdout.write(line)
                sys.stdout.flush()
                log_file.write(line)
                log_file.flush()
        finally:
            process.stdout.close()
            returncode = process.wait()

    if returncode != 0:
        raise subprocess.CalledProcessError(returncode, command)


def main() -> None:
    parser = argparse.ArgumentParser(description="Start LeRobot training from a JSON recording config.")
    parser.add_argument(
        "--config",
        default="configs/record_pick_cube.json",
        help="Path to the JSON config file.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print the command without running it.")
    parser.add_argument(
        "--no-loss-report",
        action="store_true",
        help="Skip post-training loss convergence report.",
    )
    args = parser.parse_args()

    config = load_config(Path(args.config))
    training = training_config(config)
    path = dataset_root(config)
    validate_dataset(path)

    command = command_from_config(config, training, path)
    output_dir = resolve_output_dir(config, training)
    # Keep the train log beside output_dir so we do not create output_dir before
    # lerobot-train (it refuses to start if the directory already exists).
    log_path = output_dir.parent / f"{output_dir.name}.train.log"

    print("Training policy")
    print(f"  dataset: {config['repo_id']}")
    print(f"  dataset root: {path}")
    print(f"  policy: {training['policy_type']}")
    print(f"  steps: {training['steps']}")
    print(f"  batch size: {training['batch_size']}")
    print(f"  video backend: {training['video_backend']}")
    print(f"  output dir: {output_dir}")
    print()
    print(" ".join(command))

    if args.dry_run:
        return

    if shutil.which("lerobot-train") is None:
        raise FileNotFoundError("lerobot-train was not found in PATH. Activate your LeRobot env first.")

    env = os.environ.copy()
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", str(training["pytorch_alloc_conf"]))
    run_training_with_log(command, env, log_path)

    should_report = bool(training.get("report_loss_on_finish", True)) and not args.no_loss_report
    if should_report:
        log_freq = training.get("log_freq")
        report_from_log(
            log_path,
            output_dir=output_dir,
            log_freq=int(log_freq) if log_freq is not None else None,
        )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Training failed: {exc}", file=sys.stderr)
        raise
