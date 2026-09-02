import tempfile
import unittest
from pathlib import Path

from http_server.log_reader import LogNotFoundError, LogReader


class LogReaderTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.log_dir = Path(self.temp_dir.name)
        self.reader = LogReader(self.log_dir)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_list_logs_only_returns_files_that_exist(self):
        (self.log_dir / "http.log").write_text("started\n", encoding="utf-8")

        result = self.reader.list_logs()

        self.assertEqual([item["name"] for item in result], ["http"])
        self.assertEqual(result[0]["size_bytes"], 8)
        self.assertIn("modified_at", result[0])

    def test_read_tail_returns_only_requested_final_lines(self):
        (self.log_dir / "async_policy_client.log").write_text(
            "one\ntwo\nthree\nfour\n", encoding="utf-8"
        )

        result = self.reader.read_tail("async-policy-client", lines=2)

        self.assertEqual(result["name"], "async-policy-client")
        self.assertEqual(result["requested_lines"], 2)
        self.assertEqual(result["returned_lines"], 2)
        self.assertEqual(result["content"], "three\nfour\n")

    def test_unknown_or_not_yet_created_log_is_not_found(self):
        with self.assertRaises(LogNotFoundError):
            self.reader.read_tail("unknown", lines=200)
        with self.assertRaises(LogNotFoundError):
            self.reader.read_tail("ssh-tunnel", lines=200)


if __name__ == "__main__":
    unittest.main()
