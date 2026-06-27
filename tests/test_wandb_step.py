from argparse import Namespace

import pytest

from slime.utils.metric_utils import set_wandb_step


def _args(*, always_use_train_step: bool) -> Namespace:
    return Namespace(
        wandb_always_use_train_step=always_use_train_step,
        rollout_batch_size=5,
        n_samples_per_prompt=8,
        global_batch_size=20,
    )


@pytest.mark.unit
def test_set_wandb_step_uses_rollout_axis_by_default():
    metrics = {"polar/reward_mean": 0.5}

    step_key = set_wandb_step(
        _args(always_use_train_step=False),
        metrics,
        rollout_id=3,
        default_step_key="rollout/step",
    )

    assert step_key == "rollout/step"
    assert metrics["rollout/step"] == 3
    assert "train/step" not in metrics


@pytest.mark.unit
def test_set_wandb_step_attaches_scaled_train_axis():
    metrics = {"polar/reward_mean": 0.5}

    step_key = set_wandb_step(
        _args(always_use_train_step=True),
        metrics,
        rollout_id=3,
        default_step_key="rollout/step",
    )

    assert step_key == "train/step"
    assert metrics["rollout/step"] == 6
    assert metrics["train/step"] == 6


@pytest.mark.unit
def test_set_wandb_step_aligns_eval_metrics_to_train_axis():
    metrics = {"eval/reward": 0.5}

    step_key = set_wandb_step(
        _args(always_use_train_step=True),
        metrics,
        rollout_id=4,
        default_step_key="eval/step",
    )

    assert step_key == "train/step"
    assert metrics["eval/step"] == 8
    assert metrics["train/step"] == 8
