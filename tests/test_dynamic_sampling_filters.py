from types import SimpleNamespace

import pytest

from slime.rollout.filter_hub.dynamic_sampling_filters import check_reward_nonzero_std
from slime.utils.types import Sample

NUM_GPUS = 0
ARGS = SimpleNamespace(reward_key=None)


def _sample(
    reward: float,
    *,
    index: int,
    rollout_id: int | None,
    group_index: int = 0,
    loss_mask: list[int] | None = None,
) -> Sample:
    if loss_mask is None:
        loss_mask = [1]
    return Sample(
        group_index=group_index,
        index=index,
        rollout_id=rollout_id,
        response_length=len(loss_mask),
        reward=reward,
        loss_mask=loss_mask,
        status=Sample.Status.COMPLETED,
    )


def test_multiple_traces_from_one_trajectory_do_not_pass() -> None:
    samples = [
        _sample(0.0, index=0, rollout_id=10),
        _sample(1.0, index=1, rollout_id=10),
    ]

    output = check_reward_nonzero_std(ARGS, samples)

    assert output.keep is False
    assert output.reason == "insufficient_trajectories_1"


def test_two_mixed_trajectory_means_pass_with_index_fallback() -> None:
    samples = [
        _sample(0.0, index=10, rollout_id=None),
        _sample(0.0, index=10, rollout_id=None),
        _sample(1.0, index=11, rollout_id=None),
        _sample(1.0, index=11, rollout_id=None),
    ]

    output = check_reward_nonzero_std(ARGS, samples)

    assert output.keep is True
    assert output.reason is None


def test_uniform_trajectory_means_do_not_pass_despite_trace_variance() -> None:
    samples = [
        _sample(0.0, index=0, rollout_id=10),
        _sample(1.0, index=1, rollout_id=10),
        _sample(0.0, index=2, rollout_id=11),
        _sample(1.0, index=3, rollout_id=11),
    ]

    output = check_reward_nonzero_std(ARGS, samples)

    assert output.keep is False
    assert output.reason == "zero_std_0.5"


def test_fully_masked_trajectory_does_not_count_as_preference_signal() -> None:
    samples = [
        _sample(0.0, index=0, rollout_id=10, loss_mask=[1]),
        _sample(1.0, index=1, rollout_id=11, loss_mask=[0]),
    ]

    output = check_reward_nonzero_std(ARGS, samples)

    assert output.keep is False
    assert output.reason == "insufficient_trajectories_1"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
