from __future__ import annotations

import math

import pytest
import torch

from slime.utils.ppo_utils import compute_sao_dis_loss

NUM_GPUS = 0


@pytest.mark.unit
def test_sao_dis_strictly_masks_both_tails_for_both_advantage_signs() -> None:
    eps_low = 0.3
    eps_high = 5.0
    lower = torch.log1p(torch.tensor(-eps_low, dtype=torch.float64))
    upper = torch.log1p(torch.tensor(eps_high, dtype=torch.float64))
    log_ratios = torch.stack(
        [
            lower - 0.01,
            lower,
            torch.log(torch.tensor(0.8, dtype=torch.float64)),
            torch.tensor(0.0, dtype=torch.float64),
            torch.log(torch.tensor(5.9, dtype=torch.float64)),
            upper,
            upper + 0.01,
        ]
    ).requires_grad_()
    behavior = torch.zeros_like(log_ratios)
    advantages = torch.tensor([1.0, -1.0, 1.0, -1.0, -1.0, 1.0, -1.0], dtype=torch.float64)

    loss, below, above, ratio = compute_sao_dis_loss(
        behavior,
        log_ratios,
        advantages,
        torch.ones_like(log_ratios, dtype=torch.bool),
        eps_low,
        eps_high,
    )

    assert below.tolist() == [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    assert above.tolist() == [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0]
    kept = 1.0 - below - above
    torch.testing.assert_close(loss.detach(), -ratio.detach() * advantages * kept)

    loss.sum().backward()
    assert log_ratios.grad is not None
    assert log_ratios.grad[[0, 1, 5, 6]].tolist() == [0.0, 0.0, 0.0, 0.0]
    assert torch.all(log_ratios.grad[[2, 3, 4]] != 0)


@pytest.mark.unit
def test_sao_dis_masks_non_response_tokens_without_rescaling_kept_tokens() -> None:
    behavior = torch.zeros(3, dtype=torch.float64)
    policy = torch.tensor([math.log(2.0), math.log(2.0), math.log(2.0)], dtype=torch.float64, requires_grad=True)
    advantages = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)

    loss, below, above, ratio = compute_sao_dis_loss(
        behavior,
        policy,
        advantages,
        torch.tensor([True, False, True]),
        eps_low=0.3,
        eps_high=5.0,
    )

    torch.testing.assert_close(loss.detach(), torch.tensor([-2.0, 0.0, -6.0], dtype=torch.float64))
    assert below.sum() == 0
    assert above.sum() == 0
    torch.testing.assert_close(ratio, torch.tensor([2.0, 0.0, 2.0], dtype=torch.float64))
    loss.sum().backward()
    assert policy.grad is not None
    torch.testing.assert_close(policy.grad, torch.tensor([-2.0, 0.0, -6.0], dtype=torch.float64))


@pytest.mark.unit
def test_sao_dis_rejects_nonfinite_behavior_ratio() -> None:
    with pytest.raises(FloatingPointError, match="non-finite"):
        compute_sao_dis_loss(
            torch.tensor([0.0]),
            torch.tensor([float("nan")]),
            torch.tensor([1.0]),
            torch.tensor([True]),
            eps_low=0.3,
            eps_high=5.0,
        )


@pytest.mark.unit
def test_sao_dis_extreme_finite_staleness_stays_finite() -> None:
    policy = torch.tensor([-1000.0, 0.0, 1000.0], requires_grad=True)
    loss, below, above, retained_ratio = compute_sao_dis_loss(
        torch.zeros_like(policy),
        policy,
        torch.ones_like(policy),
        torch.ones_like(policy, dtype=torch.bool),
        eps_low=0.3,
        eps_high=5.0,
    )

    assert below.tolist() == [1.0, 0.0, 0.0]
    assert above.tolist() == [0.0, 0.0, 1.0]
    assert torch.all(torch.isfinite(loss))
    assert torch.all(torch.isfinite(retained_ratio))
    loss.sum().backward()
    assert policy.grad is not None
    assert torch.all(torch.isfinite(policy.grad))
    torch.testing.assert_close(policy.grad, torch.tensor([0.0, -1.0, 0.0]))
