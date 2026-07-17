from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

NUM_GPUS = 0


@pytest.mark.unit
def test_sao_critic_trains_twice_then_returns_post_update_values(monkeypatch) -> None:
    from slime.backends.megatron_utils import actor as actor_module
    from slime.backends.megatron_utils import data as data_module

    critic = object.__new__(actor_module.MegatronTrainRayActor)
    critic.args = SimpleNamespace(
        critic_train_epochs=2,
        loss_type="policy_loss",
        policy_loss_type="sao_dis",
    )
    critic.model = object()
    critic.optimizer = object()
    critic.opt_param_scheduler = object()
    rollout_data = {
        "num_microbatches": [1],
        "global_batch_sizes": [2],
    }
    data_iterator = object()
    calls: list[tuple[str, int | None]] = []
    forward_values = iter(
        [
            {"values": [torch.tensor([1.0])]},
            {"values": [torch.tensor([2.0])]},
        ]
    )

    monkeypatch.setattr(actor_module, "get_data_iterator", lambda data: data_iterator)

    def fake_forward_only(*args, **kwargs):
        calls.append(("forward", None))
        return next(forward_values)

    def fake_compute_advantages(args, data):
        calls.append(("advantages", None))
        torch.testing.assert_close(data["values"][0], torch.tensor([1.0]))
        data["returns"] = [torch.tensor([3.0])]

    def fake_train(*args, train_epoch_id=0, **kwargs):
        calls.append(("train", train_epoch_id))
        # K updates must keep the original old-value clipping anchor.
        torch.testing.assert_close(rollout_data["values"][0], torch.tensor([1.0]))

    monkeypatch.setattr(actor_module, "forward_only", fake_forward_only)
    monkeypatch.setattr(actor_module, "compute_advantages_and_returns", fake_compute_advantages)
    monkeypatch.setattr(actor_module, "train", fake_train)
    monkeypatch.setattr(actor_module.mpu, "is_pipeline_last_stage", lambda: True)
    monkeypatch.setattr(data_module, "tensors_to_cpu", lambda tensors: tensors)

    result = critic.train_critic(rollout_id=9, rollout_data=rollout_data)

    assert calls == [
        ("forward", None),
        ("advantages", None),
        ("train", 0),
        ("train", 1),
        ("forward", None),
    ]
    torch.testing.assert_close(result["values"][0], torch.tensor([2.0]))
    torch.testing.assert_close(rollout_data["values"][0], torch.tensor([1.0]))


@pytest.mark.unit
def test_vanilla_ppo_critic_keeps_single_update_and_pre_update_values(monkeypatch) -> None:
    from slime.backends.megatron_utils import actor as actor_module
    from slime.backends.megatron_utils import data as data_module

    critic = object.__new__(actor_module.MegatronTrainRayActor)
    critic.args = SimpleNamespace(
        critic_train_epochs=1,
        loss_type="policy_loss",
        policy_loss_type="ppo",
    )
    critic.model = object()
    critic.optimizer = object()
    critic.opt_param_scheduler = object()
    rollout_data = {
        "num_microbatches": [1],
        "global_batch_sizes": [2],
    }
    calls: list[str] = []

    monkeypatch.setattr(actor_module, "get_data_iterator", lambda data: object())

    def fake_forward_only(*args, **kwargs):
        calls.append("forward")
        return {"values": [torch.tensor([1.0])]}

    def fake_compute_advantages(args, data):
        calls.append("advantages")

    def fake_train(*args, **kwargs):
        calls.append("train")
        assert "train_epoch_id" not in kwargs

    monkeypatch.setattr(actor_module, "forward_only", fake_forward_only)
    monkeypatch.setattr(actor_module, "compute_advantages_and_returns", fake_compute_advantages)
    monkeypatch.setattr(actor_module, "train", fake_train)
    monkeypatch.setattr(actor_module.mpu, "is_pipeline_last_stage", lambda: True)
    monkeypatch.setattr(data_module, "tensors_to_cpu", lambda tensors: tensors)

    result = critic.train_critic(rollout_id=9, rollout_data=rollout_data)

    assert calls == ["forward", "advantages", "train"]
    torch.testing.assert_close(result["values"][0], torch.tensor([1.0]))

