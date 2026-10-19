# Copyright (c) Microsoft. All rights reserved.

from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import Mock

import numpy as np
import pytest

pytest.importorskip("verl", minversion="0.9.1")

import torch
from omegaconf import OmegaConf
from tensordict import TensorDict
from verl import DataProto
from verl.utils import tensordict_utils

tu: Any = tensordict_utils

from agentlightning.verl.trainer import AgentLightningTrainer


@pytest.fixture
def trainer() -> Any:
    trainer: Any = object.__new__(AgentLightningTrainer)
    trainer.checkpoint_manager = Mock()
    trainer.agent_mode_daemon = Mock()
    trainer.config = OmegaConf.create(
        {
            "actor_rollout_ref": {
                "rollout": {"n": 2, "temperature": 0.7, "multi_turn": {"enable": True}},
                "actor": {
                    "calculate_entropy": True,
                    "entropy_coeff": 0.0,
                    "ppo_mini_batch_size": 3,
                    "ppo_epochs": 2,
                    "data_loader_seed": 7,
                    "shuffle": True,
                },
            },
            "critic": {"ppo_mini_batch_size": 4, "ppo_epochs": 3, "data_loader_seed": 8, "shuffle": False},
        }
    )
    return trainer


@pytest.fixture
def batch() -> Any:
    return cast(Any, DataProto).from_single_dict(
        {
            "input_ids": torch.tensor([[0, 4, 5, 6], [7, 8, 9, 0]]),
            "attention_mask": torch.tensor([[0, 1, 1, 1], [1, 1, 1, 0]]),
            "position_ids": torch.tensor([[0, 0, 1, 2], [0, 1, 2, 0]]),
            "prompts": torch.tensor([[0, 4], [7, 8]]),
            "responses": torch.tensor([[5, 6], [9, 0]]),
            "response_mask": torch.tensor([[1, 1], [1, 0]]),
        }
    )


@pytest.mark.parametrize("lora", [False, True])
def test_reference_rpc_converts_padding_and_preserves_lora_routing(trainer: Any, batch: Any, lora: bool) -> None:
    trainer.ref_in_actor = lora

    def infer(data: Any) -> Any:
        assert isinstance(data, TensorDict)
        input_ids = cast(torch.Tensor, data["input_ids"])
        assert input_ids.is_nested
        assert torch.equal(input_ids.values(), torch.tensor([4, 5, 6, 7, 8, 9]))
        assert tu.get(data, "compute_loss") is False
        assert tu.get(data, "calculate_entropy") is False
        assert tu.get(data, "no_lora_adapter") is (True if lora else None)
        return tu.get_tensordict({"log_probs": torch.arange(6, dtype=torch.float32)})

    trainer.actor_rollout_wg = SimpleNamespace(compute_log_prob=Mock(side_effect=infer))
    trainer.ref_policy_wg = SimpleNamespace(compute_ref_log_prob=Mock(side_effect=infer))
    output = trainer._compute_reference_log_prob(batch)
    assert output.batch["ref_log_prob"].tolist() == [[0, 1], [4, 0]]
    assert trainer.actor_rollout_wg.compute_log_prob.call_count == int(lora)
    assert trainer.ref_policy_wg.compute_ref_log_prob.call_count == int(not lora)
    assert not batch.batch["input_ids"].is_nested


def test_actor_update_keeps_training_metadata_and_metric_names(trainer: Any, batch: Any) -> None:
    def update(data: Any) -> Any:
        assert data["input_ids"].is_nested
        assert tu.get(data, "global_batch_size") == 6
        assert tu.get(data, "epochs") == 2
        assert tu.get(data, "temperature") == 0.7
        assert tu.get(data, "multi_turn") is True
        return tu.get_tensordict({}, {"metrics": {"mfu": 0.25, "loss": 1.5}})

    trainer.actor_rollout_wg = SimpleNamespace(update_actor=update)
    output = trainer._update_actor(batch)
    assert output.meta_info["metrics"] == {"perf/mfu/actor": 0.25, "actor/loss": 1.5}


def test_critic_update_awaits_rpc_and_keeps_metadata(trainer: Any, batch: Any) -> None:
    def update(data: Any) -> Any:
        assert data["input_ids"].is_nested
        assert tu.get(data, "global_batch_size") == 8
        assert tu.get(data, "epochs") == 3
        return SimpleNamespace(get=lambda: tu.get_tensordict({}, {"metrics": {"mfu": 0.4}}))

    trainer.critic_wg = SimpleNamespace(train_mini_batch=update)
    assert trainer._update_critic(batch).meta_info["metrics"] == {"perf/mfu/critic": 0.4}


def test_old_policy_returns_entropy_padding_and_utilization(trainer: Any, batch: Any) -> None:
    def infer(data: Any) -> Any:
        assert data["input_ids"].is_nested
        assert tu.get(data, "calculate_entropy") is True
        return tu.get_tensordict(
            {"log_probs": torch.arange(6, dtype=torch.float32), "entropy": torch.ones(6)},
            {"metrics": {"mfu": 0.3}},
        )

    trainer.actor_rollout_wg = SimpleNamespace(compute_log_prob=infer)
    output, utilization = trainer._compute_old_log_prob(batch)
    assert utilization == 0.3
    assert output.batch["old_log_probs"].tolist() == [[0, 1], [4, 0]]
    assert output.batch["entropys"].tolist() == [[1, 1], [1, 0]]


def test_critic_values_restore_response_padding(trainer: Any, batch: Any) -> None:
    def infer(data: Any) -> Any:
        assert data["input_ids"].is_nested
        assert tu.get(data, "compute_loss") is False
        return SimpleNamespace(get=lambda: tu.get_tensordict({"values": torch.arange(6, dtype=torch.float32)}))

    trainer.critic_wg = SimpleNamespace(infer_batch=infer)
    assert trainer._compute_values(batch).batch["values"].tolist() == [[0, 1], [4, 0]]


@pytest.mark.parametrize("cleanup_fails", [False, True])
def test_rollout_failure_releases_inference_memory(trainer: Any, cleanup_fails: bool) -> None:
    if cleanup_fails:
        trainer.agent_mode_daemon.clear_data_and_server.side_effect = RuntimeError("cleanup")
    with pytest.raises(RuntimeError):
        with trainer._rollout_context():
            raise RuntimeError("rollout")
    trainer.checkpoint_manager.wake_up_replicas.assert_called_once_with()
    trainer.agent_mode_daemon.clear_data_and_server.assert_called_once_with()
    trainer.checkpoint_manager.sleep_replicas.assert_called_once_with()


@pytest.mark.parametrize("adv_estimator", ["grpo", "gae"])
def test_training_step_runs_real_advantages_padding_and_updates(trainer: Any, batch: Any, adv_estimator: str) -> None:
    trainer.config = OmegaConf.merge(
        trainer.config,
        {
            "agentlightning": {"trace_aggregator": {"level": "transition"}},
            "data": {"max_prompt_length": 2, "max_response_length": 2},
            "algorithm": {"adv_estimator": adv_estimator, "use_kl_in_reward": False, "gamma": 1.0, "lam": 1.0},
            "trainer": {"balance_batch": False, "critic_warmup": 0},
            "actor_rollout_ref": {"actor": {"ppo_mini_batch_size": 1, "loss_agg_mode": "token-mean"}},
        },
    )
    batch.batch["token_level_scores"] = torch.tensor([[0.0, 1.0], [2.0, 0.0]])
    batch.batch["is_drop_mask"] = torch.zeros(2, dtype=torch.bool)
    batch.non_tensor_batch["data_id_list"] = np.array(["same-task", "same-task"])
    trainer.agent_mode_daemon.get_train_data_batch.return_value = (batch, {})
    trainer.llm_server_manager = SimpleNamespace(server_addresses=["127.0.0.1:9000"])
    trainer.use_rm = False
    trainer.use_reference_policy = False
    trainer.use_critic = adv_estimator == "gae"
    trainer.global_steps = 1
    trainer.resource_pool_manager = SimpleNamespace(get_n_gpus=lambda: 1)

    def infer(data: Any) -> Any:
        assert data["input_ids"].is_nested
        assert tu.get(data, "temperature") == 0.7
        return tu.get_tensordict({"log_probs": torch.zeros(6), "entropy": torch.ones(6)}, {"metrics": {"mfu": 0.2}})

    def update(data: Any) -> Any:
        assert data["input_ids"].is_nested
        assert torch.isfinite(data["advantages"]).all()
        assert data["advantages"].abs().sum() > 0
        return tu.get_tensordict({}, {"metrics": {"mfu": 0.4, "loss": 0.5}})

    trainer.actor_rollout_wg = SimpleNamespace(world_size=1, compute_log_prob=infer, update_actor=update)

    def infer_values(data: Any) -> Any:
        assert data["input_ids"].is_nested
        return SimpleNamespace(get=lambda: tu.get_tensordict({"values": torch.zeros(6)}))

    def train_critic(data: Any) -> Any:
        assert data["input_ids"].is_nested
        return SimpleNamespace(get=lambda: tu.get_tensordict({}, {"metrics": {"mfu": 0.3}}))

    trainer.critic_wg = SimpleNamespace(infer_batch=infer_values, train_mini_batch=train_critic)
    metrics = trainer._train_step({"fake_ids": torch.tensor([[1], [2]])})
    assert metrics["actor/loss"] == 0.5
    assert metrics["training/n_triplets_dropped_remainder"] == 0
    assert metrics["critic/score/mean_before_processing"] == 1.5
    trainer.checkpoint_manager.sleep_replicas.assert_called_once_with()
