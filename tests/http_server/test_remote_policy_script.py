import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


class RemotePolicyScriptTest(unittest.TestCase):
    def test_fold_cloth_uses_actual_remote_repository(self):
        config = json.loads(
            (REPO_ROOT / "configs/fold_cloth.json").read_text(encoding="utf-8")
        )

        self.assertEqual(
            config["remote_gpu"]["gpu_repo_root"],
            "/mnt/disk/fyx/piper_lerobot",
        )
        self.assertEqual(
            config["deployment"]["gpu_server"]["repo_root"],
            "/mnt/disk/fyx/piper_lerobot",
        )

    def test_remote_config_exports_conda_environment(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config = root / "config.json"
            config.write_text(
                json.dumps({"remote_gpu": {"conda_env": "piper-test"}}),
                encoding="utf-8",
            )
            fake_bin = root / "bin"
            fake_bin.mkdir()
            (fake_bin / "python").symlink_to("/usr/bin/python3")
            env = os.environ.copy()
            env["PATH"] = f"{fake_bin}:{env['PATH']}"

            result = subprocess.run(
                [
                    "bash",
                    "-c",
                    'source scripts/lib/remote_gpu_config.sh; '
                    'remote_gpu_load_config "$1" "$2"; '
                    'printf "%s" "$REMOTE_CONDA_ENV"',
                    "bash",
                    str(config),
                    str(root),
                ],
                cwd=REPO_ROOT,
                env=env,
                text=True,
                capture_output=True,
                check=True,
            )

            self.assertEqual(result.stdout, "piper-test")

    def test_remote_script_activates_configured_conda_without_tty(self):
        script = (REPO_ROOT / "scripts/start_policy_server_pi05_remote.sh").read_text(
            encoding="utf-8"
        )

        self.assertNotIn('ssh -t "${REMOTE_SSH_HOST}"', script)
        self.assertIn('REMOTE_CONDA_ENV="$3"', script)
        self.assertIn('conda activate "${REMOTE_CONDA_ENV}"', script)
        self.assertIn('[[ ! -d "src/piper_train" ]]', script)
        self.assertLess(
            script.index('conda activate "${REMOTE_CONDA_ENV}"'),
            script.index("python -m piper_train.start_async_policy_server"),
        )


if __name__ == "__main__":
    unittest.main()
