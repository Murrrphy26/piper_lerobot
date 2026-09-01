"""Thread-safe lifecycle management for the processes exposed over HTTP."""

from __future__ import annotations

import os
import signal
import subprocess
import threading
from pathlib import Path
from typing import Callable


class AlreadyRunningError(RuntimeError):
    def __init__(self, service: str):
        self.service = service
        super().__init__(f"{service} 服务正在运行")


class ProcessStopError(RuntimeError):
    def __init__(self, service: str, pid: int):
        self.service = service
        self.pid = pid
        super().__init__(f"停止 {service} 超时（PID {pid}）")


class ProcessManager:
    """Own the child processes started by one HTTP server instance."""

    CONFIG_PATH = "configs/fold_cloth.json"

    def __init__(
        self,
        repo_root: Path,
        log_dir: Path,
        client_delay_seconds: float = 5.0,
        stop_timeout_seconds: float = 10.0,
        *,
        process_factory: Callable[..., subprocess.Popen] = subprocess.Popen,
        getpgid: Callable[[int], int] = os.getpgid,
        killpg: Callable[[int, int], None] = os.killpg,
    ) -> None:
        self.repo_root = Path(repo_root).resolve()
        self.log_dir = Path(log_dir).resolve()
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.client_delay_seconds = client_delay_seconds
        self.stop_timeout_seconds = stop_timeout_seconds
        self._process_factory = process_factory
        self._getpgid = getpgid
        self._killpg = killpg
        self._lock = threading.RLock()

        self._policy_server = None
        self._policy_server_stopping = False
        self._ssh_tunnel = None
        self._async_client = None
        self._client_pending = False
        self._client_stopping = False
        self._client_cancel = threading.Event()

    @staticmethod
    def _alive(process) -> bool:
        return process is not None and process.poll() is None

    def _spawn(self, script: str, log_name: str, *, env=None):
        command = [str(self.repo_root / "scripts" / script), self.CONFIG_PATH]
        log_path = self.log_dir / log_name
        log_stream = log_path.open("a", encoding="utf-8", buffering=1)
        try:
            return self._process_factory(
                command,
                cwd=self.repo_root,
                env=env,
                stdout=log_stream,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        finally:
            log_stream.close()

    def start_policy_server(self) -> dict[str, object]:
        with self._lock:
            if self._policy_server_stopping or self._alive(self._policy_server):
                raise AlreadyRunningError("Policy Server")
            self._policy_server = self._spawn(
                "start_policy_server_pi05_remote.sh",
                "remote_policy_server.log",
            )
            return {
                "status": "started",
                "pid": self._policy_server.pid,
                "log": str(self.log_dir / "remote_policy_server.log"),
            }

    def _client_workflow_active(self) -> bool:
        return (
            self._client_stopping
            or self._client_pending
            or self._alive(self._ssh_tunnel)
            or self._alive(self._async_client)
        )

    def start_policy_client(self) -> dict[str, object]:
        with self._lock:
            if self._client_workflow_active():
                raise AlreadyRunningError("Policy Client")

            self._client_cancel = threading.Event()
            cancel_event = self._client_cancel
            self._ssh_tunnel = self._spawn(
                "ssh_tunnel_policy_server.sh",
                "ssh_tunnel.log",
            )
            self._async_client = None
            self._client_pending = True
            thread = threading.Thread(
                target=self._start_client_after_delay,
                args=(cancel_event,),
                name="policy-client-delay",
                daemon=True,
            )
            thread.start()
            return {
                "status": "started",
                "pid": self._ssh_tunnel.pid,
                "client_starts_in_seconds": self.client_delay_seconds,
                "ssh_log": str(self.log_dir / "ssh_tunnel.log"),
                "client_log": str(self.log_dir / "async_policy_client.log"),
            }

    def _start_client_after_delay(self, cancel_event: threading.Event) -> None:
        if cancel_event.wait(self.client_delay_seconds):
            return

        with self._lock:
            if cancel_event is not self._client_cancel or cancel_event.is_set():
                return
            if not self._alive(self._ssh_tunnel):
                self._ssh_tunnel = None
                self._client_pending = False
                return

            env = os.environ.copy()
            env["SKIP_CONFIRM"] = "true"
            try:
                self._async_client = self._spawn(
                    "run_async_policy_client_pi05.sh",
                    "async_policy_client.log",
                    env=env,
                )
            finally:
                self._client_pending = False

    def _stop_process(self, process, service: str) -> None:
        if not self._alive(process):
            return
        try:
            process_group = self._getpgid(process.pid)
            self._killpg(process_group, signal.SIGINT)
        except ProcessLookupError:
            return
        try:
            process.wait(timeout=self.stop_timeout_seconds)
        except subprocess.TimeoutExpired as error:
            raise ProcessStopError(service, process.pid) from error

    def stop_policy_server(self) -> dict[str, object]:
        with self._lock:
            process = self._policy_server
            if not self._alive(process):
                self._policy_server = None
                return {"status": "not_running"}
            self._policy_server_stopping = True

        try:
            self._stop_process(process, "Policy Server")
        finally:
            with self._lock:
                if not self._alive(process):
                    self._policy_server = None
                self._policy_server_stopping = False
        return {"status": "stopped"}

    def stop_policy_client(self) -> dict[str, object]:
        with self._lock:
            if not self._client_workflow_active():
                self._ssh_tunnel = None
                self._async_client = None
                return {"status": "not_running"}
            self._client_stopping = True
            self._client_cancel.set()
            self._client_pending = False
            client = self._async_client
            tunnel = self._ssh_tunnel

        try:
            self._stop_process(client, "Policy Client")
            self._stop_process(tunnel, "SSH Tunnel")
        finally:
            with self._lock:
                if not self._alive(client):
                    self._async_client = None
                if not self._alive(tunnel):
                    self._ssh_tunnel = None
                self._client_stopping = False
        return {"status": "stopped"}

    def stop_all(self) -> None:
        self.stop_policy_client()
        self.stop_policy_server()
