# Copyright (c) Microsoft. All rights reserved.

import warnings
from typing import Any, Dict

import pytest
from fastapi.testclient import TestClient

from agentlightning import AgentLightningClient, AgentLightningServer

TOKEN = "dummy-shared-token"


@pytest.fixture(autouse=True)
def _clear_token_env(monkeypatch: pytest.MonkeyPatch) -> None:  # pyright: ignore[reportUnusedFunction]
    monkeypatch.delenv("AGL_SERVER_TOKEN", raising=False)


def make_server(**kwargs: Any) -> AgentLightningServer:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        return AgentLightningServer(port=8011, **kwargs)


@pytest.mark.parametrize("headers", [{}, {"Authorization": "Bearer wrong"}])
def test_legacy_routes_reject_without_token(headers: Dict[str, str]) -> None:
    server = make_server(auth_token=TOKEN)
    with TestClient(server._app) as client:  # pyright: ignore[reportPrivateUsage]
        assert client.get("/task", headers=headers).status_code == 401
        assert client.get("/resources/latest", headers=headers).status_code == 401
        assert client.get("/resources/res-1", headers=headers).status_code == 401
        rollout = client.post("/rollout", json={"rollout_id": "rollout-1", "final_reward": 0.25}, headers=headers)
        assert rollout.status_code == 401
        # Nothing was stored by the rejected completion.
        assert server._store is not None  # pyright: ignore[reportPrivateUsage]
        assert not server._store._completed_rollouts  # pyright: ignore[reportPrivateUsage]


def test_legacy_routes_accept_matching_token() -> None:
    server = make_server(auth_token=TOKEN)
    headers = {"Authorization": f"Bearer {TOKEN}"}
    with TestClient(server._app) as client:  # pyright: ignore[reportPrivateUsage]
        assert client.get("/task", headers=headers).json() == {"is_available": False, "task": None}
        rollout = client.post("/rollout", json={"rollout_id": "rollout-1", "final_reward": 0.25}, headers=headers)
        assert rollout.status_code == 200


def test_legacy_server_without_token_is_unchanged() -> None:
    server = make_server()
    with TestClient(server._app) as client:  # pyright: ignore[reportPrivateUsage]
        assert client.get("/task").status_code == 200


def test_legacy_server_token_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AGL_SERVER_TOKEN", TOKEN)
    server = make_server()
    with TestClient(server._app) as client:  # pyright: ignore[reportPrivateUsage]
        assert client.get("/task").status_code == 401


def test_legacy_client_sends_token(monkeypatch: pytest.MonkeyPatch) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        explicit = AgentLightningClient("http://127.0.0.1:1", auth_token=TOKEN)
        plain = AgentLightningClient("http://127.0.0.1:1")
        monkeypatch.setenv("AGL_SERVER_TOKEN", TOKEN)
        from_env = AgentLightningClient("http://127.0.0.1:1")
    assert explicit._default_headers["Authorization"] == f"Bearer {TOKEN}"  # pyright: ignore[reportPrivateUsage]
    assert "Authorization" not in plain._default_headers  # pyright: ignore[reportPrivateUsage]
    assert from_env._default_headers["Authorization"] == f"Bearer {TOKEN}"  # pyright: ignore[reportPrivateUsage]
