# Copyright (c) Microsoft. All rights reserved.

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, create_autospec

import pytest
from agl_tinker import train
from tinker_cookbook.checkpoint_utils import CheckpointManager


@pytest.mark.asyncio
@pytest.mark.parametrize("kl_penalty_coef", [0.0, 0.1])
async def test_modern_training_preserves_checkpoint_and_reference_clients(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, kl_penalty_coef: float
) -> None:
    initial_sampler = object()
    updated_sampler = object()
    training_client = SimpleNamespace(
        save_weights_and_get_sampling_client_async=AsyncMock(return_value=initial_sampler)
    )
    service_client = SimpleNamespace(create_sampling_client=Mock(return_value=object()))
    provider = Mock()
    provider.rewrite_litellm_custom_providers.return_value = provider
    monkeypatch.setattr(train, "TinkerLLM", Mock(return_value=provider))
    monkeypatch.setattr(train, "get_renderer", Mock(return_value=object()))
    trajectories = [object()]
    monkeypatch.setattr(train, "do_group_of_group_rollouts", AsyncMock(return_value=trajectories))
    train_step = create_autospec(train.do_train_step_and_get_sampling_client)
    train_step.return_value = (updated_sampler, {"loss": 0.25})
    monkeypatch.setattr(train, "do_train_step_and_get_sampling_client", train_step)
    cfg = SimpleNamespace(
        model_name="example-model",
        renderer_name="example",
        log_path=str(tmp_path),
        save_every=20,
        kl_penalty_coef=kl_penalty_coef,
        max_tokens=16,
        train_temperature=1.0,
        top_k=-1,
        top_p=1.0,
        learning_rate=1e-5,
        eval_every=0,
        remove_constant_reward_groups=False,
        concurrency=1,
    )
    builders = [object()]
    dataset = SimpleNamespace(get_batch=Mock(return_value=builders))
    store = SimpleNamespace(add_resources=AsyncMock(return_value=SimpleNamespace(resources_id="resources-1")))
    proxy = SimpleNamespace(
        update_model_list=Mock(), restart=AsyncMock(), as_resource=Mock(), model_list=[], stop=AsyncMock()
    )
    logger = SimpleNamespace(log_metrics=Mock())
    tokenizer = object()
    await train.do_sync_training(
        start_batch=3,
        end_batch=4,
        num_batches=4,
        cfg=cfg,
        training_client=training_client,
        service_client=service_client,
        evaluators=[],
        dataset=dataset,
        ml_logger=logger,
        tokenizer=tokenizer,
        store=store,
        adapter=object(),
        llm_proxy=proxy,
    )
    training_client.save_weights_and_get_sampling_client_async.assert_awaited_once_with()
    args = train_step.await_args.args
    assert args[:3] == (cfg, 3, training_client)
    assert isinstance(args[3], CheckpointManager)
    assert args[3]._ttl_seconds is None
    assert args[5:] == (tokenizer, builders, trajectories)
    if kl_penalty_coef:
        service_client.create_sampling_client.assert_called_once_with(base_model="example-model")
        assert args[4] is service_client.create_sampling_client.return_value
    else:
        service_client.create_sampling_client.assert_not_called()
        assert args[4] is None
    provider.update_sampling_client.assert_called_once_with(updated_sampler)
    proxy.stop.assert_awaited_once()
    assert logger.log_metrics.call_args.kwargs["step"] == 3
