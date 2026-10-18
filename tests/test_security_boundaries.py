# Copyright (c) Microsoft. All rights reserved.

"""Exercise HTTP boundaries without importing optional GPU training packages."""

import ast
import json
import random
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock
from urllib.parse import quote, urlsplit

import pytest
import requests
from flask import Flask, Response, abort, request

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
    backend = Mock(
        return_value=SimpleNamespace(status_code=200, content=b'{"ok":true}', raw=SimpleNamespace(headers={}))
    )
    monkeypatch.setattr(requests, "request", backend)
    namespace: dict[str, Any] = {
        "Flask": Mock(return_value=app),
        "Response": Response,
        "abort": abort,
        "request": request,
        "requests": requests,
        "random": random,
        "time": time,
        "json": json,
        "quote": quote,
        "threading": SimpleNamespace(Thread=Mock(return_value=SimpleNamespace(start=Mock()))),
        "__name__": __name__,
    }
    owner = SimpleNamespace(backend_llm_server_addresses=["127.0.0.1:8001"], llm_timeout_seconds=5, proxy_port=0)
    load_method(ROOT / source, "_start_proxy_server_v0", namespace)(owner)
    client = app.test_client()
    for path in ["../admin", "%2e%2e/admin", "models/../admin", "models%5c..%5cadmin"]:
        backend.reset_mock()
        assert client.get("/v1/" + path).status_code == 400
        backend.assert_not_called()
    response = client.post("/v1/chat/completions?version=1", json={"model": "example"})
    assert response.status_code == 200
    assert response.json == {"ok": True}
    assert backend.call_args.kwargs["url"] == "http://127.0.0.1:8001/v1/chat/completions"
    assert backend.call_args.kwargs["allow_redirects"] is False
    for path in ["models/a%3fb%23c", "models/http:%2f%2fevil.invalid"]:
        assert client.get("/v1/" + path).status_code == 200
        url = urlsplit(str(backend.call_args.kwargs["url"]))
        assert url.netloc == "127.0.0.1:8001"
        assert url.path.startswith("/v1/")
        assert not url.query and not url.fragment
    backend.return_value = SimpleNamespace(
        status_code=400,
        content=b'<script>alert("unsafe")</script>',
        raw=SimpleNamespace(headers={"Content-Type": "text/html"}),
    )
    response = client.get("/v1/chat/completions")
    assert response.status_code == 400
    assert response.content_type == "application/json"
    assert response.headers["X-Content-Type-Options"] == "nosniff"
    backend.return_value = SimpleNamespace(
        status_code=200,
        content=b"data: streamed\n\n",
        raw=SimpleNamespace(headers={"Content-Type": "text/event-stream; charset=utf-8"}),
    )
    response = client.get("/v1/chat/completions")
    assert response.mimetype == "text/event-stream"
    assert response.data == b"data: streamed\n\n"
    assert response.headers["X-Content-Type-Options"] == "nosniff"
    backend.side_effect = requests.exceptions.ConnectionError("private backend details")
    response = client.get("/v1/chat/completions")
    assert response.status_code == 502
    assert b"private backend details" not in response.data


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


def test_store_error_keeps_traceback_server_side() -> None:
    from fastapi.testclient import TestClient

    from agentlightning.store.client_server import LightningStoreServer
    from agentlightning.store.memory import InMemoryLightningStore

    server = LightningStoreServer(InMemoryLightningStore(), "127.0.0.1", 4747)
    assert server.app is not None

    async def error_route() -> None:
        raise RuntimeError("private path and secret")

    server.app.add_api_route("/v1/agl/security-error", error_route)
    with TestClient(server.app) as client:
        response = client.get("/v1/agl/security-error")
    assert response.status_code == 500
    assert response.json() == {"detail": "Internal server error", "error_type": "RuntimeError"}
    assert "private path" not in response.text
