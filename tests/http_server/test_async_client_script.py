import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


class AsyncClientScriptTest(unittest.TestCase):
    def test_home_move_opens_both_grippers_without_dataset_reset(self):
        script = (REPO_ROOT / "scripts/run_async_policy_client_pi05.sh").read_text(
            encoding="utf-8"
        )

        self.assertNotIn("reset_pose.sh", script)
        self.assertIn('OPEN_GRIPPER_M="${OPEN_GRIPPER_M:-0.07}"', script)
        self.assertIn('"${SCRIPT_DIR}/run_move_to_joints.sh"', script)
        self.assertIn("--preset home", script)
        self.assertIn('--left-gripper "${OPEN_GRIPPER_M}"', script)
        self.assertIn('--right-gripper "${OPEN_GRIPPER_M}"', script)


if __name__ == "__main__":
    unittest.main()
