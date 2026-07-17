from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
import torch

NUM_GPUS = 0


def _args() -> SimpleNamespace:
    return SimpleNamespace(
        use_rollout_logprobs=True,
        rollout_top_p=1.0,
        rollout_temperature=1.0,
        entropy_coef=0.0,
        use_opsm=False,
        advantage_estimator="ppo",
        policy_loss_type="sao_dis",
        sao_dis_eps_low=0.3,
        sao_dis_eps_high=5.0,
        get_mismatch_metrics=False,
        use_tis=False,
        custom_pg_loss_reducer_function_path=None,
        calculate_per_token_loss=False,
        use_kl_loss=False,
    )


@pytest.mark.unit
def test_sao_policy_loss_uses_rollout_behavior_not_trainer_old_policy(monkeypatch) -> None:
    from slime.backends.megatron_utils import loss as loss_module

    current_log_probs = torch.tensor([math.log(0.4), math.log(0.2)], requires_grad=True)
    rollout_behavior = torch.tensor([math.log(0.2), math.log(0.2)])
    unrelated_old_policy = torch.tensor([math.log(0.8), math.log(0.8)])

    monkeypatch.setattr(loss_module.mpu, "get_context_parallel_world_size", lambda: 1)
    monkeypatch.setattr(
        loss_module,
        "get_log_probs_and_entropy",
        lambda *args, **kwargs: (
            None,
            {
                "log_probs": [current_log_probs],
                "entropy": [torch.zeros_like(current_log_probs)],
            },
        ),
    )
    batch = {
        "advantages": [torch.tensor([1.0, 1.0])],
        "rollout_log_probs": [rollout_behavior],
        "log_probs": [unrelated_old_policy],
        "loss_masks": [torch.ones(2)],
        "response_lengths": [2],
        "total_lengths": [4],
        "unconcat_tokens": [torch.zeros(4, dtype=torch.long)],
    }

    loss, metrics = loss_module.policy_loss_function(
        _args(),
        batch,
        logits=torch.zeros(1, 4, 8),
        sum_of_sample_mean=lambda tensor: tensor.sum() / 2,
    )

    # Direct ratios are [2, 1], hence mean loss = -(2 + 1) / 2.
    torch.testing.assert_close(loss.detach(), torch.tensor(-1.5))
    torch.testing.assert_close(metrics["train_rollout_logprob_abs_diff"], torch.tensor(math.log(2.0) / 2))
    assert metrics["sao_dis_masked_frac"] == 0
    loss.backward()
    assert current_log_probs.grad is not None
    torch.testing.assert_close(current_log_probs.grad, torch.tensor([-1.0, -0.5]))


@pytest.mark.unit
def test_sao_policy_loss_slices_full_action_mask_for_context_parallelism(monkeypatch) -> None:
    from slime.backends.megatron_utils import loss as loss_module

    current_log_probs = torch.tensor([math.log(0.4)], requires_grad=True)
    slice_calls = []
    monkeypatch.setattr(loss_module.mpu, "get_context_parallel_world_size", lambda: 2)

    def fake_slice(mask, total_length, response_length):
        slice_calls.append((mask.clone(), total_length, response_length))
        return mask[:1]

    monkeypatch.setattr(loss_module, "slice_log_prob_with_cp", fake_slice)
    monkeypatch.setattr(
        loss_module,
        "get_log_probs_and_entropy",
        lambda *args, **kwargs: (
            None,
            {
                "log_probs": [current_log_probs],
                "entropy": [torch.zeros_like(current_log_probs)],
            },
        ),
    )
    batch = {
        "advantages": [torch.tensor([1.0])],
        "rollout_log_probs": [torch.tensor([math.log(0.2)])],
        "log_probs": [torch.tensor([math.log(0.8)])],
        "loss_masks": [torch.tensor([1.0, 0.0])],
        "response_lengths": [2],
        "total_lengths": [4],
        "unconcat_tokens": [torch.zeros(4, dtype=torch.long)],
    }

    loss, _ = loss_module.policy_loss_function(
        _args(),
        batch,
        logits=torch.zeros(1, 4, 8),
        sum_of_sample_mean=lambda tensor: tensor.sum(),
    )

    torch.testing.assert_close(loss.detach(), torch.tensor(-2.0))
    assert len(slice_calls) == 1
    assert slice_calls[0][1:] == (4, 2)

