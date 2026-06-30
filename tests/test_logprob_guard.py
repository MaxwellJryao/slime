from types import SimpleNamespace

import _cp_dist_helpers  # noqa: F401
import pytest
import torch
import torch.distributed as dist
from megatron.core import mpu

from slime.backends.megatron_utils.logprob_guard import enforce_train_rollout_logprob_abs_diff


NUM_GPUS = 0


def _args(threshold):
    return SimpleNamespace(max_train_rollout_logprob_abs_diff=threshold)


def _rollout_data(train, rollout, mask=None):
    response_length = len(train)
    if mask is None:
        mask = [1] * response_length
    return {
        "log_probs": [torch.tensor(train, dtype=torch.float32)],
        "rollout_log_probs": [torch.tensor(rollout, dtype=torch.float32)],
        "loss_masks": [torch.tensor(mask, dtype=torch.int32)],
        "total_lengths": [response_length + 2],
        "response_lengths": [response_length],
    }


@pytest.fixture(autouse=True)
def _single_pipeline_stage(monkeypatch):
    monkeypatch.setattr(mpu, "is_pipeline_last_stage", lambda **_kwargs: True, raising=False)
    monkeypatch.setattr(mpu, "get_context_parallel_world_size", lambda: 1, raising=False)


@pytest.mark.unit
def test_logprob_guard_is_disabled_by_default():
    assert enforce_train_rollout_logprob_abs_diff(_args(None), {}, rollout_id=0) is None


@pytest.mark.unit
def test_logprob_guard_uses_true_masked_token_mean():
    value = enforce_train_rollout_logprob_abs_diff(
        _args(0.2),
        _rollout_data([-1.0, -2.0, -3.0], [-1.1, -2.2, -30.0], mask=[1, 1, 0]),
        rollout_id=4,
    )

    assert value == pytest.approx(0.15, abs=1e-6)


@pytest.mark.unit
def test_logprob_guard_fails_before_training_on_large_mismatch():
    with pytest.raises(RuntimeError, match=r"rollout_id=7.*masked_mean_abs_diff=9"):
        enforce_train_rollout_logprob_abs_diff(
            _args(1.0),
            _rollout_data([-10.0, -10.0], [-1.0, -1.0]),
            rollout_id=7,
        )


@pytest.mark.unit
def test_logprob_guard_uses_one_global_decision_on_all_ranks(monkeypatch):
    calls = []

    def fake_all_reduce(stats, op=None):
        calls.append(op)
        # Local mean is 0.05 while a remote rank has a severe mismatch.
        stats += torch.tensor([4.0, 2.0, 0.0, 0.0], dtype=stats.dtype)

    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "all_reduce", fake_all_reduce)

    with pytest.raises(RuntimeError, match="exceeded the fail-fast threshold"):
        enforce_train_rollout_logprob_abs_diff(
            _args(1.0),
            _rollout_data([-1.0, -1.0], [-1.05, -1.05]),
            rollout_id=1,
        )

    assert calls == [dist.ReduceOp.SUM]


@pytest.mark.unit
def test_logprob_guard_rejects_nonfinite_values():
    with pytest.raises(RuntimeError, match="nonfinite_tokens=1"):
        enforce_train_rollout_logprob_abs_diff(
            _args(1.0),
            _rollout_data([-1.0, float("nan")], [-1.0, -1.0]),
            rollout_id=2,
        )
