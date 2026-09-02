"""Safe listing and tail-reading for the current HTTP server run logs."""

from __future__ import annotations

from collections import deque
from datetime import datetime
from pathlib import Path


class LogNotFoundError(LookupError):
    def __init__(self, name: str):
        self.name = name
        super().__init__(f"日志不存在：{name}")


class LogReader:
    LOG_FILES = {
        "http": "http.log",
        "remote-policy-server": "remote_policy_server.log",
        "ssh-tunnel": "ssh_tunnel.log",
        "async-policy-client": "async_policy_client.log",
    }

    def __init__(self, log_dir: Path) -> None:
        self.log_dir = Path(log_dir).resolve()

    def _path(self, name: str) -> Path:
        filename = self.LOG_FILES.get(name)
        if filename is None:
            raise LogNotFoundError(name)
        path = self.log_dir / filename
        if not path.is_file():
            raise LogNotFoundError(name)
        return path

    def list_logs(self) -> list[dict[str, object]]:
        logs = []
        for name, filename in self.LOG_FILES.items():
            path = self.log_dir / filename
            if not path.is_file():
                continue
            stat = path.stat()
            logs.append(
                {
                    "name": name,
                    "size_bytes": stat.st_size,
                    "modified_at": datetime.fromtimestamp(stat.st_mtime)
                    .astimezone()
                    .isoformat(),
                }
            )
        return logs

    def read_tail(self, name: str, lines: int) -> dict[str, object]:
        path = self._path(name)
        tail = deque(maxlen=lines)
        with path.open("r", encoding="utf-8", errors="replace") as stream:
            tail.extend(stream)
        content = "".join(tail)
        return {
            "name": name,
            "requested_lines": lines,
            "returned_lines": len(tail),
            "content": content,
        }
