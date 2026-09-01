import signal
import subprocess
import threading
import time
from pathlib import Path

import pytest

from http_server.process_manager import (
    AlreadyRunningError,
    ProcessManager,
    ProcessStopError,
)


class FakeProcess:
    def __init__(self, pid: int):
        self.pid = pid
        self.returncode = None
        self._exited = threading.Event()

    def poll(self):
        return self.returncode

    def exit(self, returncode: int = 0):
        self.returncode = returncode
        self._exited.set()

    def wait(self, timeout=None):
        if not self._exited.wait(timeout):
            raise subprocess.TimeoutExpired("fake", timeout)
        return self.returncode


class FakeProcessFactory:
    def __init__(self):
        self.calls = []
        self.processes = []

    def __call__(self, command, **kwargs):
        process = FakeProcess(1000 + len(self.processes))
        self.processes.append(process)
        self.calls.append((command, kwargs))
        return process


@pytest.fixture
def manager(tmp_path):
    factory = FakeProcessFactory()
    instance = ProcessManager(
        repo_root=Path("/repo"),
        log_dir=tmp_path,
        client_delay_seconds=0.02,
        stop_timeout_seconds=0.05,
        process_factory=factory,
        getpgid=lambda pid: pid + 10,
        killpg=lambda pgid, sig: None,
    )
    return instance, factory


def wait_for_calls(factory, count):
    deadline = time.monotonic() + 1
    while len(factory.calls) < count and time.monotonic() < deadline:
        time.sleep(0.005)
    assert len(factory.calls) >= count


def test_start_policy_server_uses_exact_command_and_log(manager):
    instance, factory = manager

    result = instance.start_policy_server()

    command, options = factory.calls[0]
    assert command == [
        "/repo/scripts/start_policy_server_pi05_remote.sh",
        "configs/fold_cloth.json",
    ]
    assert options["cwd"] == Path("/repo")
    assert options["start_new_session"] is True
    assert options["stderr"] is subprocess.STDOUT
    assert options["stdout"].name.endswith("remote_policy_server.log")
    assert result["pid"] == 1000
    assert result["status"] == "started"


def test_start_policy_server_rejects_duplicate(manager):
    instance, _ = manager
    instance.start_policy_server()

    with pytest.raises(AlreadyRunningError, match="Policy Server"):
        instance.start_policy_server()


def test_start_policy_client_starts_tunnel_then_client_without_confirmation(manager):
    instance, factory = manager

    result = instance.start_policy_client()
    wait_for_calls(factory, 2)

    tunnel_command, tunnel_options = factory.calls[0]
    client_command, client_options = factory.calls[1]
    assert tunnel_command == [
        "/repo/scripts/ssh_tunnel_policy_server.sh",
        "configs/fold_cloth.json",
    ]
    assert tunnel_options["stdout"].name.endswith("ssh_tunnel.log")
    assert client_command == [
        "/repo/scripts/run_async_policy_client_pi05.sh",
        "configs/fold_cloth.json",
    ]
    assert client_options["env"]["SKIP_CONFIRM"] == "true"
    assert client_options["stdout"].name.endswith("async_policy_client.log")
    assert result["pid"] == 1000
    assert result["client_starts_in_seconds"] == 0.02


def test_client_workflow_rejects_duplicate_during_delay(manager):
    instance, _ = manager
    instance.start_policy_client()

    with pytest.raises(AlreadyRunningError, match="Policy Client"):
        instance.start_policy_client()


def test_dead_tunnel_prevents_delayed_client_and_unlocks(manager):
    instance, factory = manager
    instance.start_policy_client()
    factory.processes[0].exit(255)
    time.sleep(0.05)

    assert len(factory.calls) == 1
    instance.start_policy_client()
    assert len(factory.calls) == 2


def test_stop_policy_server_sends_sigint_to_process_group(tmp_path):
    factory = FakeProcessFactory()
    signals = []
    instance = ProcessManager(
        Path("/repo"),
        tmp_path,
        process_factory=factory,
        getpgid=lambda pid: pid + 10,
        killpg=lambda pgid, sig: (signals.append((pgid, sig)), factory.processes[0].exit()),
    )
    instance.start_policy_server()

    result = instance.stop_policy_server()

    assert signals == [(1010, signal.SIGINT)]
    assert result["status"] == "stopped"


def test_stop_policy_client_cancels_delay_and_stops_client_before_tunnel(tmp_path):
    factory = FakeProcessFactory()
    signals = []

    def killpg(pgid, sig):
        signals.append((pgid, sig))
        next(process for process in factory.processes if process.pid == pgid).exit()

    instance = ProcessManager(
        Path("/repo"),
        tmp_path,
        client_delay_seconds=0.01,
        process_factory=factory,
        getpgid=lambda pid: pid,
        killpg=killpg,
    )
    instance.start_policy_client()
    wait_for_calls(factory, 2)

    instance.stop_policy_client()

    assert signals == [(1001, signal.SIGINT), (1000, signal.SIGINT)]


def test_stop_during_delay_never_starts_client(tmp_path):
    factory = FakeProcessFactory()

    def killpg(pgid, sig):
        factory.processes[0].exit()

    instance = ProcessManager(
        Path("/repo"),
        tmp_path,
        client_delay_seconds=0.2,
        process_factory=factory,
        getpgid=lambda pid: pid,
        killpg=killpg,
    )
    instance.start_policy_client()
    instance.stop_policy_client()
    time.sleep(0.25)

    assert len(factory.calls) == 1


def test_stop_is_idempotent(manager):
    instance, _ = manager

    assert instance.stop_policy_server()["status"] == "not_running"
    assert instance.stop_policy_client()["status"] == "not_running"


def test_natural_policy_server_exit_releases_lock(manager):
    instance, factory = manager
    instance.start_policy_server()
    factory.processes[0].exit()

    instance.start_policy_server()

    assert len(factory.calls) == 2


def test_stop_timeout_becomes_process_stop_error(tmp_path):
    factory = FakeProcessFactory()
    instance = ProcessManager(
        Path("/repo"),
        tmp_path,
        stop_timeout_seconds=0.01,
        process_factory=factory,
        getpgid=lambda pid: pid,
        killpg=lambda pgid, sig: None,
    )
    instance.start_policy_server()

    with pytest.raises(ProcessStopError, match="Policy Server"):
        instance.stop_policy_server()

