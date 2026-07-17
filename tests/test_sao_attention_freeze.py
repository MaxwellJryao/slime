from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from slime.backends.megatron_utils.model_provider import freeze_model_params

NUM_GPUS = 0


class _TinyCritic(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.self_attention = torch.nn.Linear(2, 2)
        self.mlp = torch.nn.Linear(2, 2)
        self.output_layer = torch.nn.Linear(2, 1)


def _args(**overrides):
    values = dict(
        policy_loss_type="sao_dis",
        sao_attention_param_pattern=r"(?:^|\.)self_attention(?:\.|$)",
        only_train_params_name_list=None,
        freeze_params_name_list=["self_attention"],
    )
    values.update(overrides)
    return SimpleNamespace(**values)


@pytest.mark.unit
def test_sao_critic_attention_validation_keeps_mlp_and_value_head_trainable() -> None:
    model = _TinyCritic()

    freeze_model_params(model, _args(), role="critic")

    assert all(not parameter.requires_grad for parameter in model.self_attention.parameters())
    assert all(parameter.requires_grad for parameter in model.mlp.parameters())
    assert all(parameter.requires_grad for parameter in model.output_layer.parameters())

    output = model.output_layer(model.mlp(model.self_attention(torch.ones(1, 2)))).sum()
    output.backward()
    assert all(parameter.grad is None for parameter in model.self_attention.parameters())
    assert all(parameter.grad is not None for parameter in model.mlp.parameters())
    assert all(parameter.grad is not None for parameter in model.output_layer.parameters())


@pytest.mark.unit
def test_sao_critic_attention_validation_rejects_trainable_attention() -> None:
    with pytest.raises(RuntimeError, match="remain trainable"):
        freeze_model_params(
            _TinyCritic(),
            _args(freeze_params_name_list=["does_not_match"]),
            role="critic",
        )


@pytest.mark.unit
def test_sao_critic_attention_validation_rejects_zero_matches() -> None:
    with pytest.raises(RuntimeError, match="matched zero parameters"):
        freeze_model_params(
            _TinyCritic(),
            _args(sao_attention_param_pattern=r"missing_attention", freeze_params_name_list=["self_attention"]),
            role="critic",
        )

