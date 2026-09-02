"""FastAPI routes for controlling the Piper policy processes."""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Query, Request
from fastapi.responses import JSONResponse

from .log_reader import LogNotFoundError, LogReader
from .process_manager import AlreadyRunningError, ProcessManager, ProcessStopError

LOGGER = logging.getLogger(__name__)
REPO_ROOT = Path(__file__).resolve().parents[1]


def _default_manager() -> ProcessManager:
    configured_log_dir = os.environ.get("HTTP_SERVER_LOG_DIR")
    log_dir = (
        Path(configured_log_dir)
        if configured_log_dir
        else REPO_ROOT / "logs" / "http_server" / "manual"
    )
    return ProcessManager(repo_root=REPO_ROOT, log_dir=log_dir)


def create_app(manager: ProcessManager | None = None) -> FastAPI:
    process_manager = manager or _default_manager()
    log_reader = LogReader(process_manager.log_dir)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        yield
        process_manager.stop_all()

    app = FastAPI(title="Piper HTTP Control Server", lifespan=lifespan)
    app.state.manager = process_manager

    @app.exception_handler(AlreadyRunningError)
    async def already_running_handler(_: Request, error: AlreadyRunningError):
        return JSONResponse(status_code=409, content={"detail": str(error)})

    @app.exception_handler(ProcessStopError)
    async def process_stop_handler(_: Request, error: ProcessStopError):
        LOGGER.error("Managed process did not stop: %s", error)
        return JSONResponse(status_code=500, content={"detail": str(error)})

    @app.exception_handler(LogNotFoundError)
    async def log_not_found_handler(_: Request, error: LogNotFoundError):
        return JSONResponse(status_code=404, content={"detail": str(error)})

    @app.post("/policy-server/start")
    def start_policy_server():
        return process_manager.start_policy_server()

    @app.post("/policy-server/stop")
    def stop_policy_server():
        return process_manager.stop_policy_server()

    @app.post("/policy-client/start")
    def start_policy_client():
        return process_manager.start_policy_client()

    @app.post("/policy-client/stop")
    def stop_policy_client():
        return process_manager.stop_policy_client()

    @app.get("/logs")
    def list_logs():
        return {"logs": log_reader.list_logs()}

    @app.get("/logs/{name}")
    def read_log(name: str, lines: int = Query(default=200, ge=1, le=2000)):
        return log_reader.read_tail(name, lines)

    return app
