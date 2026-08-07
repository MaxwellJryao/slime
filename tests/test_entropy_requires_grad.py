"""Focused CPU tests for optional entropy autograd graph construction."""

import pytest
import torch

from slime.utils import ppo_utils


NUM_GPUS = 0


@pytest.mark.parametrize("chunk_size", [-1, 2], ids=["unchunked", "chunked"])
@pytest.mark.parametrize(
    "entropy_kwargs, expected_requires_grad",
    [
        pytest.param({}, True, id="default_requires_grad"),
        pytest.param({"entropy_requires_grad": False}, False, id="detached"),
    ],
)
def test_calculate_log_probs_and_entropy_controls_entropy_graph(monkeypatch, chunk_size, entropy_kwargs, expected_requires_grad):
    logits = torch.randn(5, 7, requires_grad=True)
    tokens = torch.tensor([0, 2, 4, 6, 1])
    source_storage = logits.untyped_storage().data_ptr()
    entropy_inputs = []

    def fake_entropy(entropy_logits, _tp_group):
        entropy_inputs.append(
            (
                torch.is_grad_enabled(),
                entropy_logits.requires_grad,
                entropy_logits.untyped_storage().data_ptr(),
            )
        )
        return entropy_logits.square().sum(dim=-1)

    def fake_log_probs(log_prob_logits, target_tokens, _tp_group, keep_mask=None):
        assert keep_mask is None
        return log_prob_logits.gather(dim=-1, index=target_tokens.unsqueeze(-1))

    monkeypatch.setattr(ppo_utils, "compute_entropy_from_logits", fake_entropy)
    monkeypatch.setattr(ppo_utils, "compute_log_probs", fake_log_probs)

    log_probs, entropy = ppo_utils.calculate_log_probs_and_entropy(
        logits,
        tokens,
        tp_group=None,
        with_entropy=True,
        chunk_size=chunk_size,
        **entropy_kwargs,
    )

    torch.testing.assert_close(entropy, logits.square().sum(dim=-1))
    assert entropy.requires_grad is expected_requires_grad
    assert log_probs.requires_grad

    for grad_enabled, input_requires_grad, input_storage in entropy_inputs:
        assert grad_enabled is expected_requires_grad
        assert input_requires_grad is expected_requires_grad
        if expected_requires_grad:
            assert input_storage != source_storage
        else:
            assert input_storage == source_storage


def test_without_entropy_reuses_logits_for_log_probs(monkeypatch):
    logits = torch.randn(5, 7, requires_grad=True)
    tokens = torch.tensor([0, 2, 4, 6, 1])
    source_storage = logits.untyped_storage().data_ptr()
    log_prob_storages = []

    def fake_log_probs(log_prob_logits, target_tokens, _tp_group, keep_mask=None):
        assert keep_mask is None
        log_prob_storages.append(log_prob_logits.untyped_storage().data_ptr())
        return log_prob_logits.gather(dim=-1, index=target_tokens.unsqueeze(-1))

    monkeypatch.setattr(ppo_utils, "compute_log_probs", fake_log_probs)

    log_probs, entropy = ppo_utils.calculate_log_probs_and_entropy(
        logits,
        tokens,
        tp_group=None,
        with_entropy=False,
    )

    assert entropy is None
    assert log_probs.requires_grad
    assert log_prob_storages == [source_storage]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
