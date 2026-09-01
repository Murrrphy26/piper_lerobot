from pathlib import Path

import httpx
import pytest

from http_server.app import create_app
from http_server.process_manager import AlreadyRunningError, ProcessStopError


class FakeManager:
    def __init__(self):
        self.calls = []
        self.failure = None

    def _call(self, name, result):
        self.calls.append(name)
        if self.failure:
            raise self.failure
        return result

    def start_policy_server(self):
        return self._call("start_policy_server", {"status": "started", "pid": 10})

    def stop_policy_server(self):
        return self._call("stop_policy_server", {"status": "stopped"})

    def start_policy_client(self):
        return self._call("start_policy_client", {"status": "started", "pid": 11})

    def stop_policy_client(self):
        return self._call("stop_policy_client", {"status": "stopped"})

    def stop_all(self):
        self.calls.append("stop_all")


@pytest.fixture
def anyio_backend():
    return "asyncio"


async def request(app, method, path):
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        return await client.request(method, path)


@pytest.mark.anyio
async def test_policy_server_start_delegates_and_returns_result():
    manager = FakeManager()
    response = await request(create_app(manager), "POST", "/policy-server/start")

    assert response.status_code == 200
    assert response.json() == {"status": "started", "pid": 10}
    assert manager.calls[:1] == ["start_policy_server"]


@pytest.mark.anyio
async def test_duplicate_start_returns_409_in_chinese():
    manager = FakeManager()
    manager.failure = AlreadyRunningError("Policy Server")
    response = await request(create_app(manager), "POST", "/policy-server/start")

    assert response.status_code == 409
    assert response.json() == {"detail": "Policy Server 服务正在运行"}


@pytest.mark.anyio
async def test_all_control_routes_delegate():
    manager = FakeManager()
    app = create_app(manager)
    assert (await request(app, "POST", "/policy-server/stop")).json()["status"] == "stopped"
    assert (await request(app, "POST", "/policy-client/start")).json()["pid"] == 11
    assert (await request(app, "POST", "/policy-client/stop")).json()["status"] == "stopped"

    assert manager.calls[:3] == [
        "stop_policy_server",
        "start_policy_client",
        "stop_policy_client",
    ]


@pytest.mark.anyio
async def test_stop_timeout_returns_500():
    manager = FakeManager()
    manager.failure = ProcessStopError("Policy Server", 99)
    response = await request(create_app(manager), "POST", "/policy-server/stop")

    assert response.status_code == 500
    assert response.json() == {"detail": "停止 Policy Server 超时（PID 99）"}


@pytest.mark.anyio
async def test_lifespan_stops_all_managed_processes():
    manager = FakeManager()
    app = create_app(manager)
    async with app.router.lifespan_context(app):
        pass

    assert manager.calls == ["stop_all"]


@pytest.mark.anyio
async def test_unknown_route_and_wrong_method_use_fastapi_defaults():
    manager = FakeManager()
    app = create_app(manager)
    assert (await request(app, "POST", "/missing")).status_code == 404
    assert (await request(app, "GET", "/policy-server/start")).status_code == 405


def test_default_manager_uses_environment_log_directory(tmp_path, monkeypatch):
    monkeypatch.setenv("HTTP_SERVER_LOG_DIR", str(tmp_path))
    app = create_app()

    assert app.state.manager.log_dir == Path(tmp_path).resolve()
