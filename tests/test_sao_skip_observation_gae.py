from __future__ import annotations

import pytest
import torch

from slime.utils.ppo_utils import (
    get_segmented_skip_observation_advantages_and_returns_batch,
    get_skip_observation_advantages_and_returns,
    get_skip_observation_advantages_and_returns_batch,
)

NUM_GPUS = 0


@pytest.mark.unit
def test_skip_observation_gae_bridges_only_action_indices() -> None:
    values = torch.tensor([1.0, 2.0, 100.0, -100.0, 4.0, 5.0], dtype=torch.float64)
    rewards = torch.tensor([0.0, 0.0, 999.0, -999.0, 0.0, 1.0], dtype=torch.float64)
    action_mask = torch.tensor([True, True, False, False, True, True])

    advantages, returns = get_skip_observation_advantages_and_returns(
        values,
        rewards,
        action_mask,
        gamma=1.0,
        policy_lambd=0.5,
        critic_lambd=1.0,
    )

    torch.testing.assert_close(
        advantages,
        torch.tensor([1.75, 1.5, 0.0, 0.0, -1.0, -4.0], dtype=torch.float64),
    )
    torch.testing.assert_close(
        returns,
        torch.tensor([1.0, 1.0, 0.0, 0.0, 1.0, 1.0], dtype=torch.float64),
    )


@pytest.mark.unit
def test_skip_observation_gae_is_invariant_to_observation_values() -> None:
    action_mask = torch.tensor([True, False, False, True])
    rewards = torch.tensor([0.0, 13.0, -7.0, 1.0])
    base_values = torch.tensor([0.25, 0.0, 0.0, 0.75])
    perturbed_values = torch.tensor([0.25, 1.0e9, -1.0e9, 0.75])

    base = get_skip_observation_advantages_and_returns(base_values, rewards, action_mask, gamma=0.9, policy_lambd=0.8)
    perturbed = get_skip_observation_advantages_and_returns(
        perturbed_values, rewards, action_mask, gamma=0.9, policy_lambd=0.8
    )

    torch.testing.assert_close(base[0][action_mask], perturbed[0][action_mask])
    torch.testing.assert_close(base[1][action_mask], perturbed[1][action_mask])


@pytest.mark.unit
def test_skip_observation_batch_places_terminal_reward_on_last_action(monkeypatch) -> None:
    from megatron.core import mpu

    monkeypatch.setattr(mpu, "get_context_parallel_world_size", lambda: 1)
    advantages, returns = get_skip_observation_advantages_and_returns_batch(
        total_lengths=[6],
        response_lengths=[4],
        values_list=[torch.zeros(4)],
        token_rewards_list=[torch.zeros(4)],
        sequence_rewards=[10.0],
        action_masks=[torch.tensor([True, False, True, False])],
        gamma=1.0,
        length_adaptive_alpha=1.5,
        critic_lambd=1.0,
    )

    expected_lambda = 1.0 - 1.0 / (1.5 * 4)
    torch.testing.assert_close(
        advantages[0],
        torch.tensor([10.0 * expected_lambda, 0.0, 10.0, 0.0]),
    )
    torch.testing.assert_close(returns[0], torch.tensor([10.0, 0.0, 10.0, 0.0]))


@pytest.mark.unit
def test_length_adaptive_lambda_uses_raw_response_length(monkeypatch) -> None:
    from megatron.core import mpu

    monkeypatch.setattr(mpu, "get_context_parallel_world_size", lambda: 1)
    advantages, _ = get_skip_observation_advantages_and_returns_batch(
        total_lengths=[4, 6],
        response_lengths=[2, 4],
        values_list=[torch.zeros(2), torch.zeros(4)],
        token_rewards_list=[torch.zeros(2), torch.zeros(4)],
        sequence_rewards=[1.0, 1.0],
        action_masks=[
            torch.tensor([True, True]),
            torch.tensor([True, False, True, False]),
        ],
        gamma=1.0,
        length_adaptive_alpha=1.5,
    )

    short_lambda = 1.0 - 1.0 / (1.5 * 2)
    long_lambda = 1.0 - 1.0 / (1.5 * 4)
    torch.testing.assert_close(advantages[0], torch.tensor([short_lambda, 1.0]))
    torch.testing.assert_close(advantages[1], torch.tensor([long_lambda, 0.0, 1.0, 0.0]))
    assert long_lambda > short_lambda


@pytest.mark.unit
def test_skip_observation_gae_rejects_trajectory_without_actions() -> None:
    with pytest.raises(ValueError, match="at least one generated action token"):
        get_skip_observation_advantages_and_returns(
            torch.zeros(3),
            torch.zeros(3),
            torch.zeros(3, dtype=torch.bool),
            gamma=1.0,
            policy_lambd=0.5,
        )


@pytest.mark.unit
@pytest.mark.parametrize(
    ("gamma", "alpha", "critic_lambd"),
    [(1.0, 1.5, 1.0), (0.83, 0.75, 0.4)],
)
def test_segmented_targets_are_representation_invariant(
    monkeypatch,
    gamma: float,
    alpha: float,
    critic_lambd: float,
) -> None:
    """Merged CTG targets must match two semantically identical traces."""

    from megatron.core import mpu

    monkeypatch.setattr(mpu, "get_context_parallel_world_size", lambda: 1)
    merged_values = torch.tensor([0.1, -0.2, 0.3, 99.0, -99.0, 0.4, 0.2, -0.1])
    merged_token_rewards = torch.tensor(
        [-0.01, -0.02, -0.03, 100.0, -100.0, -0.04, -0.05, -0.06]
    )
    merged_mask = torch.tensor([True, True, True, False, False, True, True, True])

    merged_advantages, merged_returns = (
        get_segmented_skip_observation_advantages_and_returns_batch(
            total_lengths=[10],
            response_lengths=[8],
            values_list=[merged_values],
            token_rewards_list=[merged_token_rewards],
            segment_ranges_list=[[(0, 3), (5, 8)]],
            segment_terminal_targets_list=[[0.9, 1.0]],
            action_masks=[merged_mask],
            gamma=gamma,
            length_adaptive_alpha=alpha,
            critic_lambd=critic_lambd,
        )
    )
    legacy_advantages, legacy_returns = (
        get_skip_observation_advantages_and_returns_batch(
            total_lengths=[5, 9],
            response_lengths=[3, 3],
            values_list=[merged_values[:3], merged_values[5:]],
            token_rewards_list=[merged_token_rewards[:3], merged_token_rewards[5:]],
            sequence_rewards=[0.9, 1.0],
            action_masks=[torch.ones(3, dtype=torch.bool)] * 2,
            gamma=gamma,
            length_adaptive_alpha=alpha,
            critic_lambd=critic_lambd,
        )
    )

    torch.testing.assert_close(merged_advantages[0][:3], legacy_advantages[0])
    torch.testing.assert_close(merged_advantages[0][5:], legacy_advantages[1])
    torch.testing.assert_close(merged_returns[0][:3], legacy_returns[0])
    torch.testing.assert_close(merged_returns[0][5:], legacy_returns[1])
    torch.testing.assert_close(merged_advantages[0][3:5], torch.zeros(2))
    torch.testing.assert_close(merged_returns[0][3:5], torch.zeros(2))


@pytest.mark.unit
def test_segmented_terminal_targets_do_not_accumulate_across_actions(
    monkeypatch,
) -> None:
    from megatron.core import mpu

    monkeypatch.setattr(mpu, "get_context_parallel_world_size", lambda: 1)
    advantages, returns = (
        get_segmented_skip_observation_advantages_and_returns_batch(
            total_lengths=[10],
            response_lengths=[8],
            values_list=[torch.zeros(8)],
            token_rewards_list=[torch.zeros(8)],
            segment_ranges_list=[[(0, 3), (5, 8)]],
            segment_terminal_targets_list=[[0.9, 1.0]],
            action_masks=[
                torch.tensor([True, True, True, False, False, True, True, True])
            ],
            gamma=1.0,
            length_adaptive_alpha=1.5,
            critic_lambd=1.0,
        )
    )

    assert returns[0][2].item() == pytest.approx(0.9)
    assert returns[0][7].item() == pytest.approx(1.0)
    assert returns[0][0].item() == pytest.approx(0.9)
    assert advantages[0][3:5].tolist() == [0.0, 0.0]
