"""Reduction contracts for non-linear train observability metrics."""

from __future__ import annotations

import math
import sys

import _cp_dist_helpers
import pytest
import torch

from slime.backends.megatron_utils.cp_utils import (  # noqa: E402
    masked_explained_variance_train_metric_stats,
    masked_mean_train_metric_stats,
    masked_root_mean_square_train_metric_stats,
    reduce_train_step_metrics,
)

# ``cp_utils`` keeps the fake MPU it imported.  Do not leak the helper-owned
# modules to tests that need the real Megatron package.
for _module_name, _fake_module in (
    ("megatron.core.mpu", _cp_dist_helpers._fake_mpu),
    ("megatron.core", _cp_dist_helpers._fake_core),
    ("megatron", _cp_dist_helpers._fake_megatron),
):
    if sys.modules.get(_module_name) is _fake_module:
        sys.modules.pop(_module_name)


NUM_GPUS = 0


@pytest.fixture
def no_op_all_reduce(monkeypatch):
    import torch.distributed as dist

    monkeypatch.setattr(dist, "all_reduce", lambda tensor, group=None: None)
    return object()


def _loss_log(stats: dict[str, torch.Tensor]) -> dict[str, object]:
    return {
        "keys": list(stats),
        "values": torch.stack(
            [torch.zeros((), dtype=torch.float64)]
            + [value.to(dtype=torch.float64).reshape(()) for value in stats.values()]
        ),
    }


def _critic_stats(predictions: torch.Tensor, targets: torch.Tensor, loss_mask: torch.Tensor):
    residual = predictions - targets
    stats: dict[str, torch.Tensor] = {}
    stats.update(masked_mean_train_metric_stats("value_mae", residual.abs(), loss_mask))
    stats.update(masked_root_mean_square_train_metric_stats("value_rmse", residual, loss_mask))
    stats.update(masked_mean_train_metric_stats("value_residual_bias", residual, loss_mask))
    stats.update(
        masked_explained_variance_train_metric_stats(
            "value_explained_variance",
            predictions=predictions,
            targets=targets,
            loss_mask=loss_mask,
        )
    )
    return stats


def _reduce_chunks(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    loss_mask: torch.Tensor,
    chunks: list[slice],
    group,
) -> dict[str, float]:
    return reduce_train_step_metrics(
        [_loss_log(_critic_stats(predictions[chunk], targets[chunk], loss_mask[chunk])) for chunk in chunks],
        calculate_per_token_loss=False,
        step_global_batch_size=1,
        cp_size=1,
        dp_with_cp_group=group,
    )


@pytest.mark.unit
def test_masked_mean_excludes_masked_nonfinite_values_before_arithmetic(no_op_all_reduce) -> None:
    values = torch.tensor([1.0, float("inf"), 3.0, float("nan")])
    loss_mask = torch.tensor([1.0, 0.0, 0.5, 0.0])
    reduced = reduce_train_step_metrics(
        [_loss_log(masked_mean_train_metric_stats("masked_mean", values, loss_mask))],
        calculate_per_token_loss=False,
        step_global_batch_size=1,
        cp_size=1,
        dp_with_cp_group=no_op_all_reduce,
    )

    assert reduced == {"masked_mean": pytest.approx(5.0 / 3.0)}


@pytest.mark.unit
@pytest.mark.parametrize(
    "chunks",
    [
        [slice(0, 4)],
        [slice(0, 2), slice(2, 4)],
        [slice(0, 1), slice(1, 2), slice(2, 3), slice(3, 4)],
    ],
)
def test_critic_metrics_are_microbatch_partition_invariant(chunks, no_op_all_reduce) -> None:
    predictions = torch.tensor([0.0, 2.0, 10.0, 10.0])
    targets = torch.tensor([0.0, 1.0, 10.0, 20.0])
    loss_mask = torch.ones(4)

    reduced = _reduce_chunks(predictions, targets, loss_mask, chunks, no_op_all_reduce)
    target_centered_ss = 501.0 - 31.0**2 / 4.0
    residual_centered_ss = 101.0 - (-9.0) ** 2 / 4.0
    expected_ev = 1.0 - residual_centered_ss / target_centered_ss

    assert reduced["value_mae"] == pytest.approx(11.0 / 4.0)
    assert reduced["value_rmse"] == pytest.approx(math.sqrt(101.0 / 4.0))
    assert reduced["value_residual_bias"] == pytest.approx(-9.0 / 4.0)
    assert reduced["value_explained_variance"] == pytest.approx(expected_ev)


@pytest.mark.unit
def test_explained_variance_uses_global_moments_not_mean_of_local_ev(no_op_all_reduce) -> None:
    predictions = torch.tensor([0.0, 2.0, 10.0, 10.0])
    targets = torch.tensor([0.0, 1.0, 10.0, 20.0])
    loss_mask = torch.ones(4)

    reduced = _reduce_chunks(
        predictions,
        targets,
        loss_mask,
        [slice(0, 2), slice(2, 4)],
        no_op_all_reduce,
    )

    # Each local partition has EV=0, while the correct global moments include
    # between-partition target variance and yield a materially different value.
    assert reduced["value_explained_variance"] == pytest.approx(0.6903163950143816)
    assert reduced["value_explained_variance"] != pytest.approx(0.0)


@pytest.mark.unit
def test_empty_mask_emits_finite_neutral_metrics(no_op_all_reduce) -> None:
    reduced = _reduce_chunks(
        torch.tensor([float("inf")]),
        torch.tensor([float("-inf")]),
        torch.zeros(1),
        [slice(0, 1)],
        no_op_all_reduce,
    )

    assert reduced == {
        "value_mae": 0.0,
        "value_rmse": 0.0,
        "value_residual_bias": 0.0,
        "value_explained_variance": 0.0,
    }
