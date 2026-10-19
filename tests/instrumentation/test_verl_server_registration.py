# Copyright (c) Microsoft. All rights reserved.

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any
from unittest.mock import Mock

import pytest


@pytest.fixture
def adapter(monkeypatch: pytest.MonkeyPatch) -> Any:
    events: list[Any] = []

    class Server:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            events.append(("server", args, kwargs))

        def launch_server(self) -> str:
            return "upstream routes"

    class Replica:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            events.append(("replica", args, kwargs))
            self.server_class = Server

    registry: dict[str, Any] = {}

    class Registry:
        @classmethod
        def register(cls, name: str, loader: Any) -> None:
            registry[name] = loader

    ray: Any = ModuleType("ray")

    def remote(cls: Any) -> tuple[str, Any]:
        return "ray actor", cls

    ray.remote = Mock(side_effect=remote)
    replicas: Any = ModuleType("verl.workers.rollout.replica")
    replicas.RolloutReplicaRegistry = Registry
    server: Any = ModuleType("verl.workers.rollout.vllm_rollout.vllm_async_server")
    server.vLLMHttpServer = Server
    server.vLLMReplica = Replica
    instrumentation: Any = ModuleType("agentlightning.instrumentation.vllm")
    instrumentation.instrument_vllm = lambda: events.append(("instrument",))
    for name, module in {
        "ray": ray,
        replicas.__name__: replicas,
        server.__name__: server,
        instrumentation.__name__: instrumentation,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)
    path = Path(__file__).parents[2] / "agentlightning/verl/modern_server.py"
    spec = importlib.util.spec_from_file_location("modern_server_contract", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, events, registry, Server, Replica


def test_instruments_actor_before_engine_initialization_preserving_routes(adapter: Any) -> None:
    module, events, _, Server, _ = adapter
    server = module.InstrumentedVLLMHttpServer("config", model_config="model")
    assert events == [("instrument",), ("server", ("config",), {"model_config": "model"})]
    assert server.launch_server() == "upstream routes"
    assert module.InstrumentedVLLMHttpServer.launch_server is Server.launch_server


def test_replica_uses_instrumented_actor_preserving_constructor(adapter: Any) -> None:
    module, events, _, Server, Replica = adapter
    replica = module.InstrumentedVLLMReplica(3, "config", "model", gpus_per_node=4)
    assert events == [("replica", (3, "config", "model"), {"gpus_per_node": 4})]
    assert replica.server_class == ("ray actor", module.InstrumentedVLLMHttpServer)
    assert Replica().server_class is Server


def test_registration_is_explicit_and_keeps_other_backends(adapter: Any) -> None:
    module, _, registry, _, _ = adapter
    other = lambda: "sglang"
    registry["sglang"] = other
    assert "vllm" not in registry
    module.register_instrumented_vllm()
    module.register_instrumented_vllm()
    assert registry["vllm"]() is module.InstrumentedVLLMReplica
    assert registry["sglang"] is other
