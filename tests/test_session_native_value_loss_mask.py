from types import SimpleNamespace

import pytest
import torch

from slime.backends.megatron_utils import loss as loss_module


def test_session_native_value_loss_uses_only_pre_action_boundaries(monkeypatch):
    monkeypatch.setattr(
        loss_module.mpu,
        "get_context_parallel_world_size",
        lambda: 1,
    )
    monkeypatch.setattr(
        loss_module.mpu,
        "get_data_parallel_world_size",
        lambda **_kwargs: 1,
    )

    def fake_get_values(_logits, **_kwargs):
        return torch.empty(0), {
            "values": [torch.zeros(3), torch.zeros(2)],
        }

    monkeypatch.setattr(loss_module, "get_values", fake_get_values)
    batch = {
        "values": [torch.zeros(3), torch.zeros(2)],
        # Large non-boundary targets make accidental actor-mask reuse obvious.
        "returns": [torch.tensor([2.0, 100.0, 100.0]), torch.tensor([4.0, 100.0])],
        "loss_masks": [torch.ones(3), torch.ones(2)],
        "critic_action_boundary_masks": [
            torch.tensor([1, 0, 0]),
            torch.tensor([1, 0]),
        ],
        "rollout_mask_sums": torch.tensor([5.0, 5.0]),
        "critic_rollout_mask_sums": torch.tensor([2.0, 2.0]),
        "total_lengths": [5, 4],
        "response_lengths": [3, 2],
        "unconcat_tokens": [torch.zeros(5), torch.zeros(4)],
    }
    args = SimpleNamespace(
        session_native_gae=True,
        loss_type="value_loss",
        calculate_per_token_loss=False,
        recompute_loss_function=False,
        allgather_cp=False,
        value_clip=1_000_000.0,
    )

    loss, normalizer, _ = loss_module.loss_function(
        args,
        batch,
        num_microbatches=1,
        step_global_batch_size=1,
        logits=torch.zeros(1),
    )

    # One rollout with two action boundaries: (2^2 + 4^2) / 2 = 10.
    assert loss.item() == pytest.approx(10.0)
    assert normalizer.item() == 1
