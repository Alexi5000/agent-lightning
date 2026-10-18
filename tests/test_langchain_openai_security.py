# Copyright (c) Microsoft. All rights reserved.

import base64
import socket
from io import BytesIO
from typing import Any

import httpx
import pytest


@pytest.mark.parametrize(
    "url", ["http://127.0.0.1/image.png", "http://169.254.169.254/image.png", "http://[::1]/image.png"]
)
def test_image_token_count_blocks_private_connections(monkeypatch: pytest.MonkeyPatch, url: str) -> None:
    chat = pytest.importorskip("langchain_openai.chat_models.base")
    pytest.importorskip("PIL.Image")
    connections: list[object] = []

    def connect(*args: object, **kwargs: object) -> None:
        connections.append(args)
        raise AssertionError("Private image URL attempted a network connection")

    monkeypatch.setattr(socket.socket, "connect", connect)
    assert chat._url_to_size(url) is None
    assert not connections


def test_image_token_count_preserves_embedded_images() -> None:
    chat = pytest.importorskip("langchain_openai.chat_models.base")
    image = pytest.importorskip("PIL.Image")
    buffer = BytesIO()
    image.new("RGB", (24, 16)).save(buffer, format="PNG")
    url = "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()
    assert chat._url_to_size(url) == (24, 16)


def test_chat_completion_with_current_openai_client() -> None:
    chat = pytest.importorskip("langchain_openai")

    def response(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/chat/completions"
        payload: dict[str, Any] = {
            "id": "local-completion",
            "object": "chat.completion",
            "created": 1,
            "model": "gpt-4o-mini",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "verified"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3},
        }
        return httpx.Response(200, json=payload)

    with httpx.Client(transport=httpx.MockTransport(response)) as client:
        model = chat.ChatOpenAI(model="gpt-4o-mini", api_key="local-test-key", http_client=client)
        result = model.invoke("Check client compatibility")
    assert result.content == "verified"
