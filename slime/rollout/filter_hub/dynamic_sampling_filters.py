import torch

from slime.rollout.filter_hub.base_types import DynamicFilterOutput
from slime.utils.types import Sample

__all__ = ["check_reward_nonzero_std"]

_REWARD_STD_EPSILON = 1e-6


def _is_trainable_sample(sample: Sample) -> bool:
    if bool(getattr(sample, "remove_sample", False)):
        return False

    status = getattr(sample, "status", None)
    status_name = getattr(status, "name", None) or str(status).rsplit(".", 1)[-1]
    if status_name.upper() in ("FAILED", "ABORTED"):
        return False

    loss_mask = getattr(sample, "loss_mask", None)
    if loss_mask is None:
        return int(getattr(sample, "response_length", 0) or 0) > 0
    return any(int(value) != 0 for value in loss_mask)


def _key_value(value, default):
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return str(value)


def _trajectory_key(sample: Sample, sample_position: int) -> tuple[object, object]:
    group_index = _key_value(getattr(sample, "group_index", None), -1)
    trajectory_index = getattr(sample, "rollout_id", None)
    if trajectory_index is None:
        trajectory_index = getattr(sample, "index", None)
    return group_index, _key_value(trajectory_index, sample_position)


def check_reward_nonzero_std(args, samples: list[Sample], **kwargs):
    """Keep groups with non-degenerate reward variation across trajectories.

    One rollout trajectory may fan out into multiple trace samples. Those
    traces are one exchangeable training unit, so compute a reward mean per
    ``(group_index, rollout_id/index)`` before checking standard deviation.
    Fully masked, removed, failed, and aborted samples cannot contribute
    gradients and therefore cannot manufacture preference signal.
    """
    rewards_by_trajectory: dict[tuple[object, object], list[float]] = {}
    for sample_position, sample in enumerate(samples):
        if not _is_trainable_sample(sample):
            continue
        key = _trajectory_key(sample, sample_position)
        rewards_by_trajectory.setdefault(key, []).append(float(sample.get_reward_value(args)))

    trajectory_means = [sum(rewards) / len(rewards) for rewards in rewards_by_trajectory.values()]
    if len(trajectory_means) < 2:
        return DynamicFilterOutput(
            keep=False,
            reason=f"insufficient_trajectories_{len(trajectory_means)}",
        )

    keep = bool(torch.tensor(trajectory_means, dtype=torch.float64).std() > _REWARD_STD_EPSILON)
    return DynamicFilterOutput(
        keep=keep,
        reason=None if keep else f"zero_std_{round(trajectory_means[0], 1)}",
    )
