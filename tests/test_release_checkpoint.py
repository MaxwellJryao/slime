from types import SimpleNamespace

import torch

from slime.backends.megatron_utils import checkpoint
from slime.backends.megatron_utils.checkpoint import is_release_checkpoint


def test_release_checkpoint_is_model_seed(tmp_path):
    (tmp_path / "latest_checkpointed_iteration.txt").write_text("release\n")

    assert is_release_checkpoint(tmp_path)


def test_numeric_checkpoint_is_training_resume(tmp_path):
    (tmp_path / "latest_checkpointed_iteration.txt").write_text("17\n")

    assert not is_release_checkpoint(tmp_path)


def test_missing_tracker_is_not_release(tmp_path):
    assert not is_release_checkpoint(tmp_path)


def test_load_checkpoint_disables_autograd_for_state_restore(monkeypatch, tmp_path):
    (tmp_path / "latest_checkpointed_iteration.txt").write_text("4\n")
    model_parameter = torch.nn.Parameter(torch.zeros(4))
    parameter_shard = model_parameter.view(-1)[1:3]
    grad_enabled_during_load = None

    def fake_load_checkpoint(**_kwargs):
        nonlocal grad_enabled_during_load
        grad_enabled_during_load = torch.is_grad_enabled()
        parameter_shard.copy_(torch.tensor([1.0, 2.0]))
        return 4, 0

    monkeypatch.setattr(checkpoint, "get_args", lambda: SimpleNamespace(load=str(tmp_path)))
    monkeypatch.setattr(checkpoint, "_load_checkpoint_megatron", fake_load_checkpoint)

    with torch.enable_grad():
        result = checkpoint.load_checkpoint(
            ddp_model=None,
            optimizer=None,
            opt_param_scheduler=None,
            checkpointing_context=None,
            skip_load_to_model_and_opt=False,
        )
        assert torch.is_grad_enabled()

    assert result == (4, 0)
    assert grad_enabled_during_load is False
    torch.testing.assert_close(model_parameter, torch.tensor([0.0, 1.0, 2.0, 0.0]))
    assert parameter_shard.requires_grad

    model_parameter.sum().backward()
    torch.testing.assert_close(model_parameter.grad, torch.ones_like(model_parameter))
