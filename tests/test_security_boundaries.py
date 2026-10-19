# Copyright (c) Microsoft. All rights reserved.

"""Exercise HTTP boundaries without importing optional GPU training packages."""

import ast
import gzip
import json
import random
import re
import socket
import subprocess
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable
from unittest.mock import AsyncMock, Mock
from urllib.parse import quote, urlencode, urlsplit

import pytest
import urllib3
from flask import Flask, Response, abort, request
from werkzeug.serving import make_server

ROOT = Path(__file__).resolve().parents[1]
DAEMONS = [
    "agentlightning/verl/daemon.py",
    "contrib/agentlightning/contrib/algorithm/env_verl/daemon.py",
]


def load_method(path: Path, method_name: str, namespace: dict[str, Any]) -> Any:
    tree = ast.parse(path.read_text())
    method = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == method_name)
    module = ast.Module(body=[method], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return namespace[method_name]


@pytest.mark.parametrize("source", DAEMONS)
def test_proxy_http_boundaries(source: str, monkeypatch: pytest.MonkeyPatch) -> None:
    app = Flask(__name__)
    backend = Mock(return_value=SimpleNamespace(status=200, data=b'{"ok":true}', headers={}))
    pool = Mock()
    pool.return_value.__enter__ = Mock(return_value=SimpleNamespace(urlopen=backend))
    pool.return_value.__exit__ = Mock(return_value=False)
    monkeypatch.setattr(urllib3, "HTTPConnectionPool", pool)
    namespace: dict[str, Any] = {
        "Flask": Mock(return_value=app),
        "Response": Response,
        "abort": abort,
        "request": request,
        "urllib3": urllib3,
        "random": random,
        "re": re,
        "time": time,
        "json": json,
        "quote": quote,
        "urlencode": urlencode,
        "threading": SimpleNamespace(Thread=Mock(return_value=SimpleNamespace(start=Mock()))),
        "__name__": __name__,
    }
    owner = SimpleNamespace(backend_llm_server_addresses=["127.0.0.1:8001"], llm_timeout_seconds=5, proxy_port=80)
    load_method(ROOT / source, "_start_proxy_server_v0", namespace)(owner)
    client = app.test_client()
    for headers in [
        {"Host": "attacker.example"},
        {"Host": "localhost.attacker.example"},
        {"Host": "localhost:9000"},
        {"Origin": "https://attacker.example"},
        {"Origin": "null"},
        {"Origin": "http://localhost:9000"},
        {"Sec-Fetch-Site": "cross-site"},
    ]:
        backend.reset_mock()
        assert client.post("/v1/chat/completions", headers=headers, json={"model": "example"}).status_code == 403
        backend.assert_not_called()
    for headers in [{"Origin": "http://localhost"}, {"Host": "127.0.0.1:80"}, {}]:
        assert client.get("/v1/models", headers=headers).status_code == 200
    owner.proxy_port = 8123
    assert (
        client.get("/v1/models", headers={"Host": "127.0.0.1:8123", "Origin": "http://127.0.0.1:8123"}).status_code
        == 200
    )
    backend.reset_mock()
    assert client.get("/v1/models").status_code == 403
    backend.assert_not_called()
    owner.proxy_port = 80
    for path in [
        "../admin",
        "%2e%2e/admin",
        "models/../admin",
        "models%5c..%5cadmin",
        "models/a%00b",
        "models/a%0db",
        "models/a%7fb",
    ]:
        backend.reset_mock()
        assert client.get("/v1/" + path).status_code == 400
        backend.assert_not_called()
    backend.reset_mock()
    assert client.get("/v1/models/a%0ab").status_code == 404
    backend.assert_not_called()
    response = client.post("/v1/chat/completions?version=1", json={"model": "example"})
    assert response.status_code == 200
    assert response.json == {"ok": True}
    assert backend.call_args.kwargs["url"] == "/v1/chat/completions?version=1"
    assert backend.call_args.kwargs["redirect"] is False
    namespace["urlencode"] = Mock(return_value="version=1\r\nHost: untrusted.invalid")
    backend.reset_mock()
    assert client.get("/v1/models?version=1").status_code == 400
    backend.assert_not_called()
    namespace["urlencode"] = urlencode
    for path in ["models/a%3fb%23c", "models/http:%2f%2fevil.invalid"]:
        assert client.get("/v1/" + path).status_code == 200
        url = urlsplit(str(backend.call_args.kwargs["url"]))
        assert not url.netloc
        assert pool.call_args.kwargs["host"] == "127.0.0.1"
        assert pool.call_args.kwargs["port"] == 8001
        assert url.path.startswith("/v1/")
        assert not url.query and not url.fragment
    for path, encoded in [
        ("models/provider/model%20v2", "models/provider/model%20v2"),
        ("models/模型", "models/%E6%A8%A1%E5%9E%8B"),
        ("models/a%25b", "models/a%25b"),
        ("models/model.v2_~", "models/model.v2_~"),
    ]:
        assert client.get("/v1/" + path).status_code == 200
        assert backend.call_args.kwargs["url"] == "/v1/" + encoded
    backend.return_value = SimpleNamespace(
        status=400,
        data=b'<script>alert("unsafe")</script>',
        headers={"Content-Type": "text/html"},
    )
    response = client.get("/v1/chat/completions")
    assert response.status_code == 400
    assert response.content_type == "application/json"
    assert response.headers["X-Content-Type-Options"] == "nosniff"
    backend.return_value = SimpleNamespace(
        status=200,
        data=b"data: streamed\n\n",
        headers={"Content-Type": "text/event-stream; charset=utf-8"},
    )
    response = client.get("/v1/chat/completions")
    assert response.mimetype == "text/event-stream"
    assert response.data == b"data: streamed\n\n"
    assert response.headers["X-Content-Type-Options"] == "nosniff"
    for address in [
        "user@evil.invalid:80",
        "127.0.0.1:80/admin",
        "127.0.0.1:80?x",
        "127.0.0.1:80#x",
        "127.0.0.1:0",
        "127.0.0.1:65536",
        "127.0.0.1:80\n",
    ]:
        owner.backend_llm_server_addresses = [address]
        backend.reset_mock()
        assert client.get("/v1/models").status_code == 503
        backend.assert_not_called()
    for address in ["gpu-worker.cluster.local:8001", "10.0.0.12:8001", "[::1]:8001"]:
        owner.backend_llm_server_addresses = [address]
        assert client.get("/v1/models").status_code == 200
        assert backend.call_args.kwargs["url"] == "/v1/models"
        assert pool.call_args.kwargs["host"] == address.rsplit(":", 1)[0].strip("[]")
        assert pool.call_args.kwargs["port"] == 8001
    owner.backend_llm_server_addresses = ["127.0.0.1:8001"]
    backend.side_effect = urllib3.exceptions.HTTPError("private backend details")
    response = client.get("/v1/chat/completions")
    assert response.status_code == 502
    assert b"private backend details" not in response.data


@pytest.mark.parametrize("source", DAEMONS)
def test_legacy_proxy_only_listens_on_advertised_loopback(source: str, monkeypatch: pytest.MonkeyPatch) -> None:
    app = Flask(__name__)
    listeners: list[Any] = []

    def run_app(*, host: str, port: int, **kwargs: Any) -> None:
        listeners.append(make_server(host, port, app))

    def immediate_thread(*, target: Callable[[], None], daemon: bool) -> SimpleNamespace:
        return SimpleNamespace(start=target)

    monkeypatch.setattr(app, "run", run_app)
    namespace: dict[str, Any] = {
        "Flask": Mock(return_value=app),
        "threading": SimpleNamespace(Thread=immediate_thread),
        "__name__": __name__,
    }
    owner = SimpleNamespace(proxy_port=0)
    try:
        load_method(ROOT / source, "_start_proxy_server_v0", namespace)(owner)
        assert len(listeners) == 1
        listener = listeners[0]
        assert listener.socket.getsockname()[0] == "127.0.0.1"
        with socket.create_connection(listener.socket.getsockname(), timeout=1):
            pass
    finally:
        for listener in listeners:
            listener.server_close()


@pytest.mark.parametrize("source", DAEMONS)
def test_proxy_transport_stays_on_configured_backend(source: str, monkeypatch: pytest.MonkeyPatch) -> None:
    upstream = Flask("upstream")
    received: list[dict[str, Any]] = []

    def capture(path: str) -> Response:
        received.append(
            {
                "path": path,
                "query": list(request.args.items(multi=True)),
                "body": request.get_data(),
                "cookie": request.headers.get("Cookie"),
                "authorization": request.headers.get("Authorization"),
                "host": request.host,
            }
        )
        if path == "redirect":
            return Response(status=302, headers={"Location": "http://untrusted.invalid/secret"})
        response = Response(
            gzip.compress(b'{"ok":true}'), headers={"Content-Encoding": "gzip"}, content_type="application/json"
        )
        response.headers.add("Set-Cookie", "first=1; HttpOnly")
        response.headers.add("Set-Cookie", "second=2; HttpOnly")
        return response

    upstream.add_url_rule("/v1/<path:path>", view_func=capture, methods=["GET", "POST"])
    listener = make_server("127.0.0.1", 0, upstream)
    server_thread = threading.Thread(target=listener.serve_forever, daemon=True)
    server_thread.start()
    app = Flask("proxy")
    namespace: dict[str, Any] = {
        "Flask": Mock(return_value=app),
        "Response": Response,
        "abort": abort,
        "request": request,
        "urllib3": urllib3,
        "random": random,
        "re": re,
        "time": time,
        "json": json,
        "quote": quote,
        "urlencode": urlencode,
        "threading": SimpleNamespace(Thread=Mock(return_value=SimpleNamespace(start=Mock()))),
        "__name__": __name__,
    }
    owner = SimpleNamespace(
        backend_llm_server_addresses=[f"127.0.0.1:{listener.server_port}"], llm_timeout_seconds=2, proxy_port=80
    )
    load_method(ROOT / source, "_start_proxy_server_v0", namespace)(owner)
    client = app.test_client()
    client.set_cookie("session", "retained")
    monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:1")
    monkeypatch.setenv("ALL_PROXY", "http://127.0.0.1:1")
    monkeypatch.setenv("NO_PROXY", "")
    try:
        for path in ["chat/completions", "models/http:%2F%2Funtrusted.invalid", "models/a%3Fb%23c"]:
            response = client.post(
                "/v1/" + path + "?key=one&key=two&query=a%26b", data=b"body", headers={"Authorization": "Bearer test"}
            )
            assert response.status_code == 200
            assert response.json == {"ok": True}
            assert "Content-Encoding" not in response.headers
            assert response.headers.getlist("Set-Cookie") == ["first=1; HttpOnly", "second=2; HttpOnly"]
            assert received[-1]["query"] == [("key", "one"), ("key", "two"), ("query", "a&b")]
            assert received[-1]["body"] == b"body"
            assert "session=retained" in received[-1]["cookie"]
            assert received[-1]["authorization"] == "Bearer test"
            assert received[-1]["host"] == f"127.0.0.1:{listener.server_port}"
        assert client.get("/v1/redirect").status_code == 302
        assert len(received) == 4
    finally:
        listener.shutdown()
        server_thread.join(timeout=2)
        listener.server_close()


def test_command_output_does_not_enter_logs() -> None:
    logs = Mock()
    owner = SimpleNamespace(logger=logs)
    result = SimpleNamespace(output="password=hunter2\nANTHROPIC_AUTH_TOKEN=secret", metadata={"secret": "hunter2"})
    method = load_method(
        ROOT / "examples/claude_code/swebench_utils/docker_runtime.py",
        "_log_command_result",
        {"CommandResult": Any, "claude_code_logger": logs},
    )
    method(owner, result)
    assert "hunter2" not in str(logs.mock_calls)
    assert "ANTHROPIC_AUTH_TOKEN" not in str(logs.mock_calls)
    assert result.output.endswith("TOKEN=secret")


def test_browser_security_boundaries() -> None:
    subprocess.run(["node", str(ROOT / "scripts/test_browser_security.cjs")], check=True, cwd=ROOT)


def test_store_error_keeps_traceback_server_side(monkeypatch: pytest.MonkeyPatch) -> None:
    from fastapi.testclient import TestClient

    from agentlightning.store.client_server import LightningStoreServer
    from agentlightning.store.memory import InMemoryLightningStore

    server = LightningStoreServer(InMemoryLightningStore(), "127.0.0.1", 4747)
    assert server.app is not None

    monkeypatch.setattr(server, "statistics", AsyncMock(side_effect=RuntimeError("private path and secret")))
    with TestClient(server.app) as client:
        response = client.get("/v1/agl/statistics")
    assert response.status_code == 500
    assert response.json() == {"detail": "Internal server error", "error_type": "RuntimeError"}
    assert "private path" not in response.text


def test_command_input_does_not_enter_logs() -> None:
    logs = Mock()
    owner = SimpleNamespace(
        output_queue=SimpleNamespace(empty=lambda: True),
        _send_bytes=Mock(),
        _read_raw_output=Mock(return_value=("result", SimpleNamespace(exit_code=0))),
        _log_command_result=Mock(),
    )
    method = load_method(
        ROOT / "examples/claude_code/swebench_utils/docker_runtime.py",
        "send_command",
        {"CommandResult": SimpleNamespace, "claude_code_logger": logs},
    )
    command = "export PASSWORD='private value'; echo ready"
    result = method(owner, command)
    assert "private value" not in str(logs.mock_calls)
    owner._send_bytes.assert_called_once_with((command + "\n").encode())
    assert result.output == "result"
