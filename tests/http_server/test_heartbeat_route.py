import asyncio
import tempfile
import unittest
from pathlib import Path

from http_server.app import create_app


class FakeManager:
    def __init__(self, log_dir, calls):
        self.log_dir = Path(log_dir)
        self.calls = calls

    def stop_all(self):
        self.calls.append("manager.stop_all")


class FakeWatchdog:
    def __init__(self, calls):
        self.calls = calls

    def heartbeat(self):
        self.calls.append("watchdog.heartbeat")
        return {
            "alive": True,
            "policy_server_running": True,
            "policy_client_running": False,
        }

    def start(self):
        self.calls.append("watchdog.start")

    def stop(self):
        self.calls.append("watchdog.stop")


class HeartbeatRouteTest(unittest.TestCase):
    def test_post_heartbeat_delegates_to_watchdog(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            calls = []
            app = create_app(FakeManager(temp_dir, calls), FakeWatchdog(calls))
            route = next(route for route in app.routes if route.path == "/heartbeat")

            result = route.endpoint()

            self.assertEqual(
                result,
                {
                    "alive": True,
                    "policy_server_running": True,
                    "policy_client_running": False,
                },
            )
            self.assertEqual(calls, ["watchdog.heartbeat"])
            self.assertIn("POST", route.methods)

    def test_lifespan_starts_watchdog_then_stops_it_before_processes(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            calls = []
            app = create_app(FakeManager(temp_dir, calls), FakeWatchdog(calls))

            async def run_lifespan():
                async with app.router.lifespan_context(app):
                    self.assertEqual(calls, ["watchdog.start"])

            asyncio.run(run_lifespan())

            self.assertEqual(
                calls,
                ["watchdog.start", "watchdog.stop", "manager.stop_all"],
            )


if __name__ == "__main__":
    unittest.main()
