"""Heartbeat tracking and automatic Policy Client shutdown."""

from __future__ import annotations

import logging
import threading
import time
from typing import Callable

from .process_manager import ProcessManager

LOGGER = logging.getLogger(__name__)


class HeartbeatWatchdog:
    def __init__(
        self,
        manager: ProcessManager,
        timeout_seconds: float = 12.0,
        check_interval_seconds: float = 1.0,
        *,
        monotonic: Callable[[], float] = time.monotonic,
        logger: logging.Logger = LOGGER,
    ) -> None:
        self.manager = manager
        self.timeout_seconds = timeout_seconds
        self.check_interval_seconds = check_interval_seconds
        self._monotonic = monotonic
        self._logger = logger
        self._lock = threading.Lock()
        self._last_heartbeat: float | None = None
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def heartbeat(self) -> dict[str, bool]:
        with self._lock:
            self._last_heartbeat = self._monotonic()
        return {"alive": True, **self.manager.status()}

    def check_once(self) -> None:
        status = self.manager.status()
        with self._lock:
            if self._last_heartbeat is None:
                return
            elapsed = self._monotonic() - self._last_heartbeat
            if elapsed <= self.timeout_seconds or not status["policy_client_running"]:
                return
            self._logger.warning(
                "Stopping Policy Client: heartbeat timeout (%.1fs without heartbeat)",
                elapsed,
            )
            self.manager.stop_policy_client()

    def _run(self) -> None:
        while not self._stop_event.wait(self.check_interval_seconds):
            try:
                self.check_once()
            except Exception:
                self._logger.exception("Heartbeat watchdog check failed")

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run,
            name="heartbeat-watchdog",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=self.check_interval_seconds + 1.0)
        self._thread = None
