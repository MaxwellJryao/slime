from __future__ import annotations

import pytest
import torch

from slime.utils.ppo_utils import compute_dppo_loss

NUM_GPUS = 0


def _log_probs(values: list[float], *, requires_grad: bool = False) -> torch.Tensor:
    return torch.log(torch.tensor(values, dtype=torch.float64)).requires_grad_(requires_grad)


@pytest.mark.unit
def test_dppo_binary_tv_matches_sampled_token_definition() -> None:
    behavior = _log_probs([0.1, 0.5, 0.9])
    policy = _log_probs([0.2, 0.5, 0.3])

    _, _, divergence = compute_dppo_loss(
        behavior,
        policy,
        advantages=torch.ones_like(policy),
        response_mask=torch.ones_like(policy, dtype=torch.bool),
        divergence_threshold=1.0,
    )

    torch.testing.assert_close(divergence, torch.tensor([0.1, 0.0, 0.6], dtype=torch.float64))


@pytest.mark.unit
def test_dppo_masks_only_updates_moving_farther_from_behavior() -> None:
    # Tokens 0/1 move above behavior; token 2/3 move below behavior. Each move
    # exceeds the binary-TV threshold. Positive advantages are blocked only on
    # the upward move; negative advantages are blocked only on the downward one.
    behavior = _log_probs([0.2, 0.2, 0.4, 0.4])
    policy = _log_probs([0.4, 0.4, 0.1, 0.1], requires_grad=True)
    advantages = torch.tensor([1.0, -1.0, 1.0, -1.0], dtype=torch.float64)

    loss, blocked, _ = compute_dppo_loss(
        behavior,
        policy,
        advantages=advantages,
        response_mask=torch.ones_like(policy, dtype=torch.bool),
        divergence_threshold=0.1,
    )

    assert blocked.tolist() == [1.0, 0.0, 0.0, 1.0]
    ratio = torch.exp(policy.detach() - behavior)
    torch.testing.assert_close(loss.detach(), -ratio * advantages * (1.0 - blocked))

    loss.sum().backward()
    assert policy.grad is not None
    assert policy.grad[0] == 0
    assert policy.grad[1] != 0
    assert policy.grad[2] != 0
    assert policy.grad[3] == 0


@pytest.mark.unit
def test_dppo_keeps_out_of_region_corrective_update_and_masks_padding() -> None:
    behavior = _log_probs([0.2, 0.4, 0.2])
    policy = _log_probs([0.5, 0.1, 0.9])
    response_mask = torch.tensor([True, True, False])

    loss, blocked, divergence = compute_dppo_loss(
        behavior,
        policy,
        # Both valid updates point back toward behavior and must remain active.
        advantages=torch.tensor([-1.0, 1.0, 1.0], dtype=torch.float64),
        response_mask=response_mask,
        divergence_threshold=0.05,
    )

    assert blocked.tolist() == [0.0, 0.0, 0.0]
    assert loss[0] != 0
    assert loss[1] != 0
    assert loss[2] == 0
    assert divergence[2] == 0


@pytest.mark.unit
def test_dppo_rejects_unsupported_divergence() -> None:
    values = _log_probs([0.5])
    with pytest.raises(ValueError, match="Unsupported DPPO divergence"):
        compute_dppo_loss(
            values,
            values,
            advantages=torch.ones_like(values),
            response_mask=torch.ones_like(values, dtype=torch.bool),
            divergence_threshold=0.1,
            divergence_type="kl",
        )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
