from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
import torch

NUM_GPUS = 0


@pytest.mark.unit
def test_value_loss_emits_exact_masked_global_error_sufficient_stats(monkeypatch) -> None:
    from slime.backends.megatron_utils import cp_utils
    from slime.backends.megatron_utils import loss as loss_module

    predictions = torch.tensor([0.0, 2.0, 10.0, 10.0, 999.0], requires_grad=True)
    targets = torch.tensor([0.0, 1.0, 10.0, 20.0, -999.0])
    loss_mask = torch.tensor([1.0, 1.0, 1.0, 1.0, 0.0])
    old_values = predictions.detach().clone()

    monkeypatch.setattr(loss_module.mpu, "get_context_parallel_world_size", lambda: 1)
    monkeypatch.setattr(
        loss_module.mpu,
        "get_data_parallel_world_size",
        lambda with_context_parallel=True: 1,
    )
    monkeypatch.setattr(
        loss_module,
        "get_values",
        lambda *args, **kwargs: (
            torch.empty(0),
            {"values": [predictions]},
        ),
    )
    batch = {
        "values": [old_values],
        "returns": [targets],
        "loss_masks": [loss_mask],
        "rollout_mask_sums": torch.tensor([4.0]),
        "response_lengths": [5],
        "total_lengths": [7],
        "unconcat_tokens": [torch.zeros(7, dtype=torch.long)],
    }
    _, _, loss_log = loss_module.loss_function(
        SimpleNamespace(
            value_clip=1.0,
            loss_type="value_loss",
            calculate_per_token_loss=False,
            recompute_loss_function=False,
            allgather_cp=False,
        ),
        batch,
        num_microbatches=1,
        step_global_batch_size=1,
        logits=torch.zeros(1, 7, 1),
    )

    monkeypatch.setattr(cp_utils.dist, "all_reduce", lambda tensor, group=None: None)
    reduced = cp_utils.reduce_train_step_metrics(
        [loss_log],
        calculate_per_token_loss=False,
        step_global_batch_size=1,
        cp_size=1,
        dp_with_cp_group=object(),
    )

    target_centered_ss = 501.0 - 31.0**2 / 4.0
    residual_centered_ss = 101.0 - (-9.0) ** 2 / 4.0
    assert reduced["value_mae"] == pytest.approx(11.0 / 4.0)
    assert reduced["value_rmse"] == pytest.approx(math.sqrt(101.0 / 4.0))
    assert reduced["value_residual_bias"] == pytest.approx(-9.0 / 4.0)
    assert reduced["value_explained_variance"] == pytest.approx(1.0 - residual_centered_ss / target_centered_ss)

    # These base names become train/critic-<name> in model.train's existing
    # role namespace. Hidden sufficient-stat keys must never leak to W&B.
    assert set(loss_module.CRITIC_OBSERVABILITY_METRICS) <= set(reduced)
    assert not any(name.startswith("__train_metric_stat__:") for name in reduced)
