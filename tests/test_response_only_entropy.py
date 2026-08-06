"""Regression tests for response-row-only policy entropy."""

import sys

import _cp_dist_helpers
import pytest
import torch

from slime.backends.megatron_utils import loss as loss_module  # noqa: E402
from slime.utils import ppo_utils  # noqa: E402

mpu = loss_module.mpu

for _module_name, _fake_module in (
    ("megatron.core.mpu", _cp_dist_helpers._fake_mpu),
    ("megatron.core", _cp_dist_helpers._fake_core),
    ("megatron", _cp_dist_helpers._fake_megatron),
):
    if sys.modules.get(_module_name) is _fake_module:
        sys.modules.pop(_module_name)

NUM_GPUS = 0


def _set_parallel_layout(monkeypatch, *, cp_size: int, cp_rank: int) -> None:
    monkeypatch.setattr(mpu, "get_context_parallel_world_size", lambda: cp_size)
    monkeypatch.setattr(mpu, "get_context_parallel_rank", lambda: cp_rank)
    monkeypatch.setattr(
        mpu,
        "get_tensor_model_parallel_group",
        lambda: None,
        raising=False,
    )


def _dense_entropy(logits: torch.Tensor, _tp_group) -> torch.Tensor:
    probabilities = logits.softmax(dim=-1)
    return -(probabilities * logits.log_softmax(dim=-1)).sum(dim=-1)


def _dense_log_probs(
    logits: torch.Tensor,
    target_tokens: torch.Tensor,
    _tp_group,
    *,
    keep_mask=None,
) -> torch.Tensor:
    assert keep_mask is None
    return logits.log_softmax(dim=-1).gather(
        dim=-1,
        index=target_tokens.unsqueeze(-1),
    )


@pytest.mark.unit
def test_response_only_entropy_matches_full_then_slice_values_and_gradients(
    monkeypatch,
):
    """Selecting response rows first is identical to the legacy objective."""
    _set_parallel_layout(monkeypatch, cp_size=1, cp_rank=0)
    entropy_inputs: list[torch.Tensor] = []

    def recording_entropy(logits: torch.Tensor, tp_group) -> torch.Tensor:
        entropy_inputs.append(logits.detach().clone())
        return _dense_entropy(logits, tp_group)

    monkeypatch.setattr(ppo_utils, "compute_entropy_from_logits", recording_entropy)
    monkeypatch.setattr(ppo_utils, "compute_log_probs", _dense_log_probs)

    temperature = 0.5
    args = type(
        "Args",
        (),
        {
            "rollout_temperature": temperature,
            "allgather_cp": False,
            "entropy_coef": 1.0,
            "log_probs_chunk_size": 2,
        },
    )()
    total_lengths = [5, 4]
    response_lengths = [2, 3]
    tokens = [
        torch.tensor([0, 1, 2, 3, 4]),
        torch.tensor([4, 3, 2, 1]),
    ]
    logits = torch.linspace(-2.0, 2.0, steps=45).reshape(1, 9, 5).requires_grad_()

    _, result = loss_module.get_log_probs_and_entropy(
        logits,
        args=args,
        unconcat_tokens=tokens,
        total_lengths=total_lengths,
        response_lengths=response_lengths,
        with_entropy=True,
    )
    response_entropy = torch.cat(result["entropy"])

    reference_logits = logits.detach().clone().requires_grad_()
    scaled_reference = reference_logits.squeeze(0) / temperature
    full_entropy = _dense_entropy(scaled_reference, None)
    legacy_response_entropy = torch.cat([full_entropy[2:4], full_entropy[5:8]])

    torch.testing.assert_close(response_entropy, legacy_response_entropy)
    response_entropy.sum().backward()
    legacy_response_entropy.sum().backward()
    torch.testing.assert_close(logits.grad, reference_logits.grad)

    recorded_rows = torch.cat(entropy_inputs)
    expected_rows = torch.cat(
        [
            scaled_reference.detach()[2:4],
            scaled_reference.detach()[5:8],
        ]
    )
    assert recorded_rows.size(0) == sum(response_lengths)
    torch.testing.assert_close(recorded_rows, expected_rows)


@pytest.mark.unit
@pytest.mark.parametrize(
    (
        "cp_size",
        "cp_rank",
        "allgather_cp",
        "total_length",
        "response_length",
        "local_logit_positions",
        "expected_logits",
        "expected_tokens",
        "expected_local_grad_positions",
    ),
    [
        (
            1,
            0,
            False,
            8,
            4,
            list(range(8)),
            [3, 4, 5, 6],
            [4, 5, 6, 7],
            [3, 4, 5, 6],
        ),
        (2, 0, False, 8, 4, [0, 1, 6, 7], [6], [7], [2]),
        (2, 1, False, 8, 4, [2, 3, 4, 5], [3, 4, 5], [4, 5, 6], [1, 2, 3]),
        (2, 0, True, 6, 4, [0, 1, 2], [1, 2], [2, 3], [1, 2]),
        (2, 1, True, 6, 4, [3, 4, 5], [3, 4], [4, 5], [0, 1]),
    ],
    ids=[
        "cp1",
        "zigzag-rank0",
        "zigzag-rank1",
        "allgather-rank0",
        "allgather-rank1",
    ],
)
def test_get_responses_selects_exact_cp_entropy_rows(
    monkeypatch,
    cp_size,
    cp_rank,
    allgather_cp,
    total_length,
    response_length,
    local_logit_positions,
    expected_logits,
    expected_tokens,
    expected_local_grad_positions,
):
    """The row selector matches the response layout consumed by the loss."""
    _set_parallel_layout(monkeypatch, cp_size=cp_size, cp_rank=cp_rank)
    args = type(
        "Args",
        (),
        {
            "rollout_temperature": 1.0,
            "allgather_cp": allgather_cp,
        },
    )()
    logits = torch.tensor(local_logit_positions, dtype=torch.float32).reshape(1, -1, 1).requires_grad_()
    tokens = torch.arange(total_length)

    [(selected_logits, selected_tokens)] = list(
        loss_module.get_responses(
            logits,
            args=args,
            unconcat_tokens=[tokens],
            total_lengths=[total_length],
            response_lengths=[response_length],
        )
    )

    assert selected_logits.flatten().tolist() == expected_logits
    assert selected_tokens.tolist() == expected_tokens
    selected_logits.sum().backward()
    expected_grad = torch.zeros_like(logits)
    expected_grad[0, expected_local_grad_positions, 0] = 1
    torch.testing.assert_close(logits.grad, expected_grad)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
