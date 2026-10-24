# Copyright (c) Microsoft. All rights reserved.

import pytest
from fastapi.testclient import TestClient

from agentlightning.server.app import create_app


def config(key="", host="127.0.0.1"):
    return {
        "key": key,
        "host": host,
        "default_proxy": {"model_name": "test", "train": {"temperature": 1}, "val": {"temperature": 0}},
    }


def test_public_bind_requires_key():
    with pytest.raises(ValueError, match="server key"):
        create_app(config(host="0.0.0.0"))
    create_app(config(key="regression-key", host="0.0.0.0"))


def test_loopback_without_key_and_browser_origin_boundary():
    with TestClient(create_app(config()), base_url="http://127.0.0.1:8080", client=("127.0.0.1", 12345)) as client:
        assert client.get("/api/rollouts?state_in=queuing").status_code == 200
        assert (
            client.get("/api/rollouts?state_in=queuing", headers={"origin": "https://attacker.example"}).status_code
            == 403
        )
        assert client.get("/api/rollouts?state_in=queuing", headers={"host": "attacker.example"}).status_code == 403


def test_remote_peer_cannot_use_keyless_api():
    with TestClient(create_app(config()), base_url="http://127.0.0.1:8080", client=("203.0.113.2", 12345)) as client:
        assert client.get("/api/rollouts?state_in=queuing").status_code == 403


def test_keyed_remote_api_rejects_wrong_key():
    with TestClient(create_app(config(key="regression-key", host="0.0.0.0"))) as client:
        assert (
            client.get("/api/rollouts?state_in=queuing", headers={"authorization": "Bearer wrong"}).status_code == 401
        )
        assert (
            client.get("/api/rollouts?state_in=queuing", headers={"authorization": "Bearer regression-key"}).status_code
            == 200
        )
