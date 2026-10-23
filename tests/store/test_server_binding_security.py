from dataclasses import asdict
from typing import Any, Dict
from unittest.mock import AsyncMock, patch

import httpx
import pytest
from portpicker import pick_unused_port

from agentlightning.store.client_server import LightningStoreServer
from agentlightning.store.memory import InMemoryLightningStore
from agentlightning.utils.server_launcher import PythonServerLauncherArgs


def make_server(**kwargs: Any) -> LightningStoreServer:
    return LightningStoreServer(InMemoryLightningStore(thread_safe=True), **kwargs)


@pytest.mark.parametrize("kwargs", [{}, {"launcher_args": PythonServerLauncherArgs(port=4747)}])
def test_omitted_store_host_is_loopback(kwargs: Dict[str, Any]) -> None:
    server = make_server(**kwargs)
    assert server.launcher_args.host == "127.0.0.1"
    assert server.server_launcher.endpoint.startswith("http://127.0.0.1:")


@pytest.mark.parametrize("host", ["0.0.0.0", "::", "192.0.2.1"])
def test_explicit_distributed_host_is_preserved(host: str) -> None:
    args = PythonServerLauncherArgs(host=host, port=4747)
    original = asdict(args)
    assert make_server(host=host).launcher_args.host == host
    assert make_server(launcher_args=args).launcher_args.host == host
    assert asdict(args) == original


@pytest.mark.asyncio
async def test_default_listener_serves_local_workers() -> None:
    server = make_server(port=pick_unused_port())
    await server.start()
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(server.endpoint + "/v1/agl/health")
        assert response.status_code == 200
        assert server.server_launcher.endpoint.startswith("http://127.0.0.1:")
    finally:
        await server.stop()


def test_cli_defaults_to_loopback() -> None:
    from agentlightning.cli.store import main

    with patch("agentlightning.cli.store.LightningStoreServer") as server:
        server.return_value.run_forever = AsyncMock()
        main([])
    assert server.call_args.kwargs["host"] == "127.0.0.1"
