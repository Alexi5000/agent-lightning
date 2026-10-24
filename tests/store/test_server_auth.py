# Copyright (c) Microsoft. All rights reserved.

from typing import Any, Dict

import aiohttp
import pytest
from fastapi.testclient import TestClient
from portpicker import pick_unused_port

from agentlightning.store.client_server import LightningStoreClient, LightningStoreServer
from agentlightning.store.memory import InMemoryLightningStore

TOKEN = "dummy-shared-token"


def make_server(**kwargs: Any) -> LightningStoreServer:
    return LightningStoreServer(InMemoryLightningStore(thread_safe=True), port=pick_unused_port(), **kwargs)


@pytest.fixture(autouse=True)
def _clear_token_env(monkeypatch: pytest.MonkeyPatch) -> None:  # pyright: ignore[reportUnusedFunction]
    monkeypatch.delenv("AGL_SERVER_TOKEN", raising=False)


@pytest.mark.parametrize(
    "headers",
    [{}, {"Authorization": "Bearer wrong"}, {"Authorization": f"Basic {TOKEN}"}, {"Authorization": TOKEN}],
)
def test_token_rejects_missing_or_wrong_credentials(headers: Dict[str, str]) -> None:
    server = make_server(auth_token=TOKEN)
    assert server.app is not None
    with TestClient(server.app) as client:
        update = client.post("/v1/agl/resources/dummy-resource", json={}, headers=headers)
        assert update.status_code == 401
        assert client.get("/v1/agl/resources/latest", headers=headers).status_code == 401
        assert client.post("/v1/traces", content=b"", headers=headers).status_code == 401
        # Health stays open so launchers and clients can probe readiness.
        assert client.get("/v1/agl/health", headers=headers).status_code == 200


@pytest.mark.asyncio
async def test_token_rejection_leaves_store_untouched() -> None:
    server = make_server(auth_token=TOKEN)
    assert server.app is not None
    with TestClient(server.app) as client:
        assert client.post("/v1/agl/resources/dummy-resource", json={}).status_code == 401
    assert await server.store.get_latest_resources() is None  # type: ignore[union-attr]


def test_token_accepts_matching_bearer() -> None:
    server = make_server(auth_token=TOKEN)
    assert server.app is not None
    with TestClient(server.app) as client:
        response = client.post(
            "/v1/agl/resources/dummy-resource", json={}, headers={"Authorization": f"Bearer {TOKEN}"}
        )
        assert response.status_code == 200
        assert response.json()["resources_id"] == "dummy-resource"


def test_token_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AGL_SERVER_TOKEN", TOKEN)
    server = make_server()
    assert server.app is not None
    with TestClient(server.app) as client:
        assert client.get("/v1/agl/resources/latest").status_code == 401
        assert client.get("/v1/agl/statistics", headers={"Authorization": f"Bearer {TOKEN}"}).status_code == 200


def test_no_token_keeps_open_behavior() -> None:
    server = make_server()
    assert server.app is not None
    with TestClient(server.app) as client:
        assert client.post("/v1/agl/resources/dummy-resource", json={}).status_code == 200


def test_otlp_headers_follow_token() -> None:
    assert make_server().otlp_traces_headers() == {}
    assert make_server(auth_token=TOKEN).otlp_traces_headers() == {"Authorization": f"Bearer {TOKEN}"}
    assert LightningStoreClient("http://127.0.0.1:1", auth_token=TOKEN).otlp_traces_headers() == {
        "Authorization": f"Bearer {TOKEN}"
    }


@pytest.mark.asyncio
async def test_client_sends_shared_token_end_to_end() -> None:
    server = make_server(auth_token=TOKEN)
    await server.start()
    authed = LightningStoreClient(server.endpoint, auth_token=TOKEN, retry_delays=(), health_retry_delays=())
    anonymous = LightningStoreClient(server.endpoint, retry_delays=(), health_retry_delays=())
    try:
        update = await authed.update_resources("dummy-resource", {})
        assert update.resources_id == "dummy-resource"
        latest = await authed.get_latest_resources()
        assert latest is not None and latest.resources_id == "dummy-resource"
        # The client swallows read errors and returns None; writes surface the 401.
        assert await anonymous.get_latest_resources() is None
        with pytest.raises(aiohttp.ClientResponseError) as exc:
            await anonymous.update_resources("other-resource", {})
        assert exc.value.status == 401
        assert await server.store.get_resources_by_id("other-resource") is None  # type: ignore[union-attr]
    finally:
        await authed.close()
        await anonymous.close()
        await server.stop()


def test_token_survives_pickling() -> None:
    import pickle

    client = pickle.loads(pickle.dumps(LightningStoreClient("http://127.0.0.1:1", auth_token=TOKEN)))
    assert client.otlp_traces_headers() == {"Authorization": f"Bearer {TOKEN}"}
    state = make_server(auth_token=TOKEN).__getstate__()
    assert state["_auth_token"] == TOKEN
