# Copyright (c) Microsoft. All rights reserved.

"""Compatibility import for the instrumented modern VERL server actor."""

import ray

from .modern_server import InstrumentedVLLMHttpServer

PatchedvLLMServer = ray.remote(num_cpus=1)(InstrumentedVLLMHttpServer)
