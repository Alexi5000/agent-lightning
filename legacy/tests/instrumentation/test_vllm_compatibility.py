# Copyright (c) Microsoft. All rights reserved.

"""CPU unit contracts for the legacy and split vLLM serving interfaces."""

import importlib
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import BaseModel


class Completion(BaseModel):
    id: str = "completion"


class LegacyError(BaseModel):
    code: int = 400


class ErrorInfo(BaseModel):
    code: int = 400


class ModernError(BaseModel):
    error: ErrorInfo


@pytest.fixture(params=[False, True], ids=["legacy", "split"])
def serving_module(monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest) -> Any:
    modern = request.param
    seen: list[Any] = []
    error_type = ModernError if modern else LegacyError

    class Serving:
        async def chat_completion_full_generator(
            self: Any,
            request: Any,
            results: Any,
            request_id: str,
            model: str,
            conversation: Any,
            tokenizer: Any,
            metadata: Any,
            *args: Any,
            **kwargs: Any,
        ) -> Any:
            seen.append((args, kwargs))
            async for result in results:
                seen.append(result)
            if request == "error":
                return ModernError(error=ErrorInfo()) if modern else LegacyError()
            return Completion()

    protocol = SimpleNamespace(
        ChatCompletionResponse=Completion, ChatCompletionRequest=Completion, ErrorResponse=error_type
    )
    original_import = importlib.import_module

    def import_module(name: str, package: Any = None) -> Any:
        if name == "vllm.entrypoints.openai.chat_completion.protocol":
            if modern:
                return protocol
            raise ModuleNotFoundError(name=name)
        if name == "vllm.entrypoints.openai.protocol":
            return protocol
        if name in {"vllm.entrypoints.openai.serving_chat", "vllm.entrypoints.openai.chat_completion.serving"}:
            return SimpleNamespace(OpenAIServingChat=Serving)
        if name == "vllm.entrypoints.openai.engine.protocol":
            return SimpleNamespace(ErrorResponse=ModernError)
        return original_import(name, package)

    monkeypatch.setattr(importlib, "import_module", import_module)
    path = Path(__file__).parents[2] / "agentlightning/instrumentation/vllm.py"
    spec = importlib.util.spec_from_file_location("vllm_compatibility_under_test", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return SimpleNamespace(module=module, protocol=protocol, serving=Serving, seen=seen, error_type=error_type)


@pytest.mark.asyncio
async def test_tokens_and_added_serving_arguments(serving_module: Any) -> None:
    async def results() -> Any:
        yield SimpleNamespace(prompt_token_ids=[1, 2], outputs=[SimpleNamespace(token_ids=[3, 4])])
        yield SimpleNamespace(
            prompt_token_ids=[1, 2], outputs=[SimpleNamespace(token_ids=[3, 4, 5]), SimpleNamespace(token_ids=[6])]
        )

    module = serving_module.module
    module.instrument_vllm()
    parser = object()
    response = await serving_module.serving().chat_completion_full_generator(
        "ok", results(), "request", "model", [], None, None, parser, mm_token_counts={"image": 10}
    )
    assert response.prompt_token_ids == [1, 2]
    assert response.response_token_ids == [[3, 4, 5], [6]]
    assert serving_module.seen[0] == ((parser,), {"mm_token_counts": {"image": 10}})


@pytest.mark.asyncio
async def test_errors_are_returned_unchanged(serving_module: Any) -> None:
    async def results() -> Any:
        outputs: list[Any] = []
        for result in outputs:
            yield result

    serving_module.module.instrument_vllm()
    response = await serving_module.serving().chat_completion_full_generator(
        "error", results(), "request", "model", [], None, None
    )
    assert isinstance(response, serving_module.error_type)
    assert "prompt_token_ids" not in response.model_dump()
    assert "response_token_ids" not in response.model_dump()


def test_uninstrument_restores_protocol_and_allows_reinstrumentation(serving_module: Any) -> None:
    module = serving_module.module
    original = serving_module.serving.chat_completion_full_generator
    module.instrument_vllm()
    with pytest.warns(UserWarning, match="already instrumented"):
        module.instrument_vllm()
    module.uninstrument_vllm()
    assert serving_module.protocol.ChatCompletionResponse is Completion
    assert serving_module.serving.chat_completion_full_generator is original
    module.instrument_vllm()
    assert serving_module.protocol.ChatCompletionResponse is module.ChatCompletionResponsePatched
    parsed = module.ChatCompletionResponsePatched(response_token_ids=[[1, 2], [3]])
    assert parsed.response_token_ids == [[1, 2], [3]]


@pytest.mark.asyncio
async def test_server_preserves_legacy_and_nested_error_status(
    serving_module: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    def remote(**kwargs: Any) -> Any:
        def decorate(cls: Any) -> Any:
            return cls

        return decorate

    monkeypatch.setitem(sys.modules, "ray", SimpleNamespace(remote=remote))
    monkeypatch.setitem(
        sys.modules,
        "verl.workers.rollout.vllm_rollout.vllm_async_server",
        SimpleNamespace(AsyncvLLMServer=object),
    )
    monkeypatch.setitem(sys.modules, "agentlightning.instrumentation.vllm", serving_module.module)
    path = Path(__file__).parents[2] / "agentlightning/verl/async_server.py"
    spec = importlib.util.spec_from_file_location("vllm_server_under_test", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    server = module.PatchedvLLMServer.__new__(module.PatchedvLLMServer)
    error = (
        ModernError(error=ErrorInfo(code=429)) if serving_module.error_type is ModernError else LegacyError(code=429)
    )

    async def create_completion(*args: Any) -> Any:
        return error

    async def body() -> dict[str, Any]:
        return {}

    server.openai_serving_chat = SimpleNamespace(create_chat_completion=create_completion)
    response = await server.chat_completion(SimpleNamespace(json=body))
    assert response.status_code == 429
    assert response.body == error.model_dump_json().encode()
