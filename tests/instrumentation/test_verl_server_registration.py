# Copyright (c) Microsoft. All rights reserved.

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import Mock

import pytest


@pytest.fixture
def adapter(monkeypatch):
    events = []

    class Server:
        def __init__(self, *args, **kwargs):
            events.append(("server", args, kwargs))

        def launch_server(self):
            return "upstream routes"

    class Replica:
        def __init__(self, *args, **kwargs):
            events.append(("replica", args, kwargs))
            self.server_class = Server

    registry = {}

    class Registry:
        @classmethod
        def register(cls, name, loader):
            registry[name] = loader

    ray = ModuleType("ray")
    ray.remote = Mock(side_effect=lambda cls: ("ray actor", cls))
    replicas = ModuleType("verl.workers.rollout.replica")
    replicas.RolloutReplicaRegistry = Registry
    server = ModuleType("verl.workers.rollout.vllm_rollout.vllm_async_server")
    server.vLLMHttpServer = Server
    server.vLLMReplica = Replica
    instrumentation = ModuleType("agentlightning.instrumentation.vllm")
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
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, events, registry, Server, Replica


def test_instruments_actor_before_engine_initialization_preserving_routes(adapter):
    module, events, _, Server, _ = adapter
    server = module.InstrumentedVLLMHttpServer("config", model_config="model")
    assert events == [("instrument",), ("server", ("config",), {"model_config": "model"})]
    assert server.launch_server() == "upstream routes"
    assert module.InstrumentedVLLMHttpServer.launch_server is Server.launch_server


def test_replica_uses_instrumented_actor_preserving_constructor(adapter):
    module, events, _, Server, Replica = adapter
    replica = module.InstrumentedVLLMReplica(3, "config", "model", gpus_per_node=4)
    assert events == [("replica", (3, "config", "model"), {"gpus_per_node": 4})]
    assert replica.server_class == ("ray actor", module.InstrumentedVLLMHttpServer)
    assert Replica().server_class is Server


def test_registration_is_explicit_and_keeps_other_backends(adapter):
    module, _, registry, _, _ = adapter
    other = lambda: "sglang"
    registry["sglang"] = other
    assert "vllm" not in registry
    module.register_instrumented_vllm()
    module.register_instrumented_vllm()
    assert registry["vllm"]() is module.InstrumentedVLLMReplica
    assert registry["sglang"] is other
