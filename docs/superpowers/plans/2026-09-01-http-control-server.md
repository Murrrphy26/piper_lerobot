# HTTP Control Server Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a headless FastAPI service that starts and stops the remote Policy Server and local Policy Client workflow with duplicate-start locking and per-run logs.

**Architecture:** A standalone `http_server` package owns a thread-safe process manager and a thin FastAPI routing layer. A shell entrypoint creates the second-precision run directory, configures logging, and launches Uvicorn on port 12123.

**Tech Stack:** Python 3.10+, FastAPI, Uvicorn, pytest, standard-library subprocess/threading/signal APIs

**Spec:** `docs/superpowers/specs/2026-09-01-http-control-server-design.md`

## Global Constraints

- Do not put HTTP implementation code under `src/piper_train`.
- Use `configs/fold_cloth.json` for every managed command.
- Set `SKIP_CONFIRM=true` only for the async Policy Client.
- Start the client five seconds after the SSH tunnel.
- Stop the client before the SSH tunnel.
- Use SIGINT for graceful process-group shutdown; never escalate to SIGKILL.
- Default to `0.0.0.0:12123`.
- Create `logs/http_server/YYYYMMDD_HHMMSS/` for each HTTP Server launch.
- Do not create any Git commits.

---

### Task 1: Process manager start and locking

**Files:**
- Create: `http_server/__init__.py`
- Create: `http_server/process_manager.py`
- Create: `tests/http_server/test_process_manager.py`

**Interfaces:**
- Produces: `ProcessManager(repo_root: Path, log_dir: Path, client_delay_seconds: float = 5.0, stop_timeout_seconds: float = 10.0)`
- Produces: `start_policy_server() -> dict[str, object]`
- Produces: `start_policy_client() -> dict[str, object]`
- Produces: `AlreadyRunningError(service: str)`

- [ ] Write tests using a fake process factory to assert the exact Policy Server and SSH commands, repository working directory, process-group isolation, and separate log handles.
- [ ] Run `pytest -q tests/http_server/test_process_manager.py` and verify failure because `http_server.process_manager` does not exist.
- [ ] Implement constructor injection for process creation and the two atomic start methods. Store process handles under a `threading.RLock`; raise `AlreadyRunningError` while a workflow owns a live process or pending delayed start.
- [ ] Add a test proving two Policy Server start calls raise `AlreadyRunningError("Policy Server")`; run it red, then implement the minimum state check and run green.
- [ ] Add a test proving Policy Client delay spawns `scripts/run_async_policy_client_pi05.sh configs/fold_cloth.json` with `SKIP_CONFIRM=true`; run it red, implement the cancellable delayed worker, then run green.
- [ ] Add a test proving a tunnel that exits during the delay prevents client spawn and releases the workflow; run red, implement, and run green.

### Task 2: Graceful stop and lifecycle cleanup

**Files:**
- Modify: `http_server/process_manager.py`
- Modify: `tests/http_server/test_process_manager.py`

**Interfaces:**
- Produces: `stop_policy_server() -> dict[str, object]`
- Produces: `stop_policy_client() -> dict[str, object]`
- Produces: `stop_all() -> None`
- Produces: `ProcessStopError(service: str, pid: int)`

- [ ] Add a test that `stop_policy_server` sends `signal.SIGINT` to the server process group and waits with the configured timeout; run red.
- [ ] Implement a shared `_stop_process` helper using injected `getpgid`, `killpg`, and process `wait`; run green.
- [ ] Add a test that `stop_policy_client` cancels a pending delayed start and never spawns the client; run red, implement cancellation synchronization, and run green.
- [ ] Add a test recording signal order and assert client is stopped before SSH; run red, implement ordered shutdown, and run green.
- [ ] Add tests for idempotent stops, natural child exit unlocking, and timeout conversion to `ProcessStopError`; implement each behavior after observing its test fail.
- [ ] Add a test that `stop_all` calls client shutdown before server shutdown; implement and run the full process-manager test file.

### Task 3: FastAPI routes and lifespan

**Files:**
- Create: `http_server/app.py`
- Create: `tests/http_server/test_app.py`

**Interfaces:**
- Consumes: `ProcessManager`, `AlreadyRunningError`, and `ProcessStopError`
- Produces: `create_app(manager: ProcessManager | None = None) -> FastAPI`
- Produces: module-level `app` for Uvicorn

- [ ] Write a failing test with FastAPI `TestClient` asserting `POST /policy-server/start` delegates and returns PID/log information.
- [ ] Implement the app factory and Policy Server start route; run green.
- [ ] Add a failing test that maps `AlreadyRunningError("Policy Server")` to HTTP 409 and `{"detail": "Policy Server 服务正在运行"}`; implement the exception handler and run green.
- [ ] Add failing tests for `/policy-server/stop`, `/policy-client/start`, and `/policy-client/stop`; implement thin delegating routes and run green.
- [ ] Add a failing test mapping `ProcessStopError` to HTTP 500; implement and run green.
- [ ] Add a lifespan test proving normal app shutdown invokes `manager.stop_all()`; implement and run the full app tests.

### Task 4: Timestamped launch and logging

**Files:**
- Create: `scripts/start_http_server.sh`
- Create: `tests/http_server/test_start_http_server.py`
- Modify: `.gitignore`

**Interfaces:**
- Consumes: `http_server.app:app`
- Produces: shell entrypoint controlled by `HTTP_SERVER_HOST` and `HTTP_SERVER_PORT`, defaulting to `0.0.0.0` and `12123`
- Produces: `HTTP_SERVER_LOG_DIR` environment variable consumed by the default app factory

- [ ] Write a failing test that runs the entrypoint with a fake `uvicorn` executable and fixed `date`, asserting creation of `logs/http_server/YYYYMMDD_HHMMSS` plus all four log files.
- [ ] Implement the shell entrypoint with `set -euo pipefail`, repository-root resolution, timestamp creation, log touching, exported absolute `HTTP_SERVER_LOG_DIR`, and Uvicorn invocation whose stdout/stderr append to `http.log`; run green.
- [ ] Add a failing test for default host/port and environment overrides; implement argument construction and run green.
- [ ] Add `logs/http_server/` to `.gitignore` without changing unrelated ignore rules.
- [ ] Run `bash -n scripts/start_http_server.sh` and the launch-script tests.

### Task 5: Dependencies and end-to-end verification

**Files:**
- Modify: `pyproject.toml`
- Modify: `tests/http_server/test_app.py` if dependency-version compatibility requires test setup changes

**Interfaces:**
- Makes `fastapi`, `uvicorn`, and the HTTP test client dependencies available from the project metadata.

- [ ] Add a failing metadata test or assertion that the required runtime dependencies are declared.
- [ ] Add compatible FastAPI and Uvicorn dependency declarations to `pyproject.toml`; keep robot/training dependencies unchanged.
- [ ] Run `pytest -q tests/http_server`.
- [ ] Run `python -m compileall -q http_server`.
- [ ] Run `bash -n scripts/start_http_server.sh scripts/start_policy_server_pi05_remote.sh scripts/ssh_tunnel_policy_server.sh scripts/run_async_policy_client_pi05.sh`.
- [ ] Run `git diff --check` and inspect `git status --short` to verify only intended working-tree changes exist and no new commit was created.
