# Copyright (c) Microsoft. All rights reserved.

from http.cookies import SimpleCookie
from urllib.parse import parse_qs, urlencode, urlparse

import httpx
import pytest
from fastapi_sso.sso.base import SSOLoginError
from fastapi_sso.sso.google import GoogleSSO
from litellm.proxy.management_endpoints.ui_sso import SSOAuthenticationHandler
from starlette.requests import Request


@pytest.mark.asyncio
@pytest.mark.parametrize("callback_state", ["valid", "missing", "wrong", "missing-cookie"])
async def test_litellm_sso_state_and_offline_token_exchange(
    monkeypatch: pytest.MonkeyPatch, callback_state: str
) -> None:
    requests: list[str] = []

    def provider(request: httpx.Request) -> httpx.Response:
        requests.append(request.url.path)
        if request.url.path == "/.well-known/openid-configuration":
            return httpx.Response(
                200,
                json={
                    "authorization_endpoint": "https://provider.test/authorize",
                    "token_endpoint": "https://provider.test/token",
                    "userinfo_endpoint": "https://provider.test/userinfo",
                },
            )
        if request.url.path == "/token":
            assert request.method == "POST"
            assert parse_qs(request.content.decode())["code"] == ["offline-code"]
            return httpx.Response(200, json={"access_token": "offline-token", "token_type": "Bearer"})
        if request.url.path == "/userinfo":
            assert request.headers["authorization"] == "Bearer offline-token"
            return httpx.Response(
                200, json={"sub": "offline-user", "email": "offline@example.com", "email_verified": True}
            )
        raise AssertionError(f"Unexpected provider request: {request.url}")

    async_client = httpx.AsyncClient
    monkeypatch.setattr(httpx, "AsyncClient", lambda: async_client(transport=httpx.MockTransport(provider)))
    monkeypatch.setenv("GOOGLE_CLIENT_SECRET", "offline-secret")
    redirect = await SSOAuthenticationHandler.get_sso_login_redirect(
        redirect_url="https://proxy.test/sso/callback", google_client_id="offline-client"
    )
    assert redirect is not None
    state = parse_qs(urlparse(redirect.headers["location"]).query)["state"][0]
    cookies = SimpleCookie()
    cookies.load(redirect.headers["set-cookie"])
    assert cookies["sso_state"].value == state
    assert cookies["sso_state"]["httponly"]
    assert cookies["sso_state"]["secure"]

    params = {"code": "offline-code"}
    if callback_state != "missing":
        params["state"] = "attacker-state" if callback_state == "wrong" else state
    headers = [] if callback_state == "missing-cookie" else [(b"cookie", f"sso_state={state}".encode())]
    request = Request(
        {
            "type": "http",
            "method": "GET",
            "scheme": "https",
            "server": ("proxy.test", 443),
            "path": "/sso/callback",
            "query_string": urlencode(params).encode(),
            "headers": headers,
        }
    )
    async with GoogleSSO("offline-client", "offline-secret", redirect_uri="https://proxy.test/sso/callback") as sso:
        if callback_state == "valid":
            user = await sso.verify_and_process(request)
            assert user is not None
            assert user.email == "offline@example.com"
            assert user.id == "offline-user"
            assert "/token" in requests
            assert "/userinfo" in requests
        else:
            with pytest.raises(SSOLoginError):
                await sso.verify_and_process(request)
            assert "/token" not in requests
