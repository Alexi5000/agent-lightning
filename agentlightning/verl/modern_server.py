# Copyright (c) Microsoft. All rights reserved.

from typing import Any, cast

import ray
from verl.workers.rollout.replica import RolloutReplicaRegistry
from verl.workers.rollout.vllm_rollout.vllm_async_server import vLLMHttpServer, vLLMReplica

from agentlightning.instrumentation.vllm import instrument_vllm


class InstrumentedVLLMHttpServer(vLLMHttpServer):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        instrument_vllm()
        cast(Any, super()).__init__(*args, **kwargs)


class InstrumentedVLLMReplica(vLLMReplica):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        cast(Any, super()).__init__(*args, **kwargs)
        self.server_class = ray.remote(InstrumentedVLLMHttpServer)


def register_instrumented_vllm() -> None:
    RolloutReplicaRegistry.register("vllm", lambda: InstrumentedVLLMReplica)
