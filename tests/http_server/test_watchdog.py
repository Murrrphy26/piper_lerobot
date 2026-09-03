import subprocess
import tempfile
import threading
import unittest
from pathlib import Path

from http_server.process_manager import ProcessManager
from http_server.watchdog import HeartbeatWatchdog


class FakeProcess:
    def __init__(self, pid):
        self.pid = pid
        self.returncode = None

    def poll(self):
        return self.returncode


class FakeProcessFactory:
    def __init__(self):
        self.processes = []

    def __call__(self, command, **kwargs):
        process = FakeProcess(2000 + len(self.processes))
        self.processes.append(process)
        return process


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


class FakeManager:
    def __init__(self):
        self.server_running = False
        self.client_running = False
        self.status_calls = 0
        self.stop_client_calls = 0

    def status(self):
        self.status_calls += 1
        return {
            "policy_server_running": self.server_running,
            "policy_client_running": self.client_running,
        }

    def stop_policy_client(self):
        self.stop_client_calls += 1
        self.client_running = False
        return {"status": "stopped"}


class FakeLogger:
    def __init__(self):
        self.messages = []

    def warning(self, message, *args):
        self.messages.append(message % args if args else message)


class ProcessStatusTest(unittest.TestCase):
    def test_status_uses_process_handles_and_clears_exited_processes(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            factory = FakeProcessFactory()
            manager = ProcessManager(
                Path("/repo"),
                Path(temp_dir),
                client_delay_seconds=60,
                process_factory=factory,
            )
            manager.start_policy_server()
            manager.start_policy_client()

            self.assertEqual(
                manager.status(),
                {
                    "policy_server_running": True,
                    "policy_client_running": False,
                },
            )

            factory.processes[0].returncode = 1
            factory.processes[1].returncode = 255
            self.assertEqual(
                manager.status(),
                {
                    "policy_server_running": False,
                    "policy_client_running": False,
                },
            )


class HeartbeatWatchdogTest(unittest.TestCase):
    def setUp(self):
        self.manager = FakeManager()
        self.clock = FakeClock()
        self.logger = FakeLogger()
        self.watchdog = HeartbeatWatchdog(
            self.manager,
            timeout_seconds=12.0,
            check_interval_seconds=1.0,
            monotonic=self.clock,
            logger=self.logger,
        )

    def test_heartbeat_updates_time_and_returns_real_status(self):
        self.manager.server_running = True

        result = self.watchdog.heartbeat()

        self.assertEqual(
            result,
            {
                "alive": True,
                "policy_server_running": True,
                "policy_client_running": False,
            },
        )
        self.clock.now = 11.9
        self.watchdog.check_once()
        self.assertEqual(self.manager.stop_client_calls, 0)

    def test_never_received_heartbeat_does_not_stop_client(self):
        self.manager.client_running = True
        self.clock.now = 100.0

        self.watchdog.check_once()

        self.assertEqual(self.manager.stop_client_calls, 0)
        self.assertEqual(self.manager.status_calls, 1)

    def test_timeout_stops_only_running_client_and_logs_reason(self):
        self.manager.server_running = True
        self.manager.client_running = True
        self.watchdog.heartbeat()
        self.clock.now = 12.1

        self.watchdog.check_once()

        self.assertEqual(self.manager.stop_client_calls, 1)
        self.assertTrue(self.manager.server_running)
        self.assertTrue(any("heartbeat timeout" in item for item in self.logger.messages))

    def test_timeout_without_running_client_does_not_call_stop(self):
        self.watchdog.heartbeat()
        self.clock.now = 20.0

        self.watchdog.check_once()

        self.assertEqual(self.manager.stop_client_calls, 0)


if __name__ == "__main__":
    unittest.main()
