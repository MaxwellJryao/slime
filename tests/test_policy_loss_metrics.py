from __future__ import annotations

import sys
from types import SimpleNamespace

import _cp_dist_helpers
import pytest
import torch

from slime.backends.megatron_utils import loss as loss_module  # noqa: E402

for _module_name, _fake_module in (
    ("megatron.core.mpu", _cp_dist_helpers._fake_mpu),
    ("megatron.core", _cp_dist_helpers._fake_core),
    ("megatron", _cp_dist_helpers._fake_megatron),
):
    if sys.modules.get(_module_name) is _fake_module:
        sys.modules.pop(_module_name)

NUM_GPUS = 0


def _args(*, use_rollout_logprobs: bool) -> SimpleNamespace:
    return SimpleNamespace(
        advantage_estimator="ppo",
        calculate_per_token_loss=False,
        custom_pg_loss_reducer_function_path=None,
        entropy_coef=0.0,
        eps_clip=0.2,
        eps_clip_high=0.2,
        get_mismatch_metrics=False,
        policy_loss_type="ppo",
        rollout_temperature=1.0,
        rollout_top_p=1.0,
        use_kl_loss=False,
        use_opsm=False,
        use_rollout_logprobs=use_rollout_logprobs,
        use_tis=False,
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    ("use_rollout_logprobs", "expected_abs_diff"),
    [(True, 2.5), (False, 2.0)],
)
def test_policy_metric_compares_rollout_behavior_with_the_training_policy(
    monkeypatch,
    use_rollout_logprobs,
    expected_abs_diff,
):
    current_log_probs = torch.tensor([1.0, 4.0], requires_grad=True)
    rollout_log_probs = torch.tensor([0.0, 0.0])
    cached_train_log_probs = torch.tensor([2.0, 2.0])
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
    monkeypatch.setattr(
        loss_module,
        "compute_policy_loss",
        lambda ppo_kl, advantages, *_args: (
            torch.zeros_like(ppo_kl),
            torch.zeros_like(ppo_kl),
        ),
    )
    batch = {
        "advantages": [torch.ones(2)],
        "log_probs": [cached_train_log_probs],
        "loss_masks": [torch.ones(2)],
        "response_lengths": [2],
        "rollout_log_probs": [rollout_log_probs],
        "total_lengths": [4],
        "unconcat_tokens": [torch.zeros(4, dtype=torch.long)],
    }

    _, metrics = loss_module.policy_loss_function(
        _args(use_rollout_logprobs=use_rollout_logprobs),
        batch,
        logits=torch.zeros(1, 4, 8),
        sum_of_sample_mean=torch.mean,
    )

    torch.testing.assert_close(
        metrics["train_rollout_logprob_abs_diff"],
        torch.tensor(expected_abs_diff),
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
