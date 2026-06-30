from __future__ import annotations

import importlib.util
import tempfile
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F

_MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "slime"
    / "backends"
    / "megatron_utils"
    / "fp32_lm_head.py"
)
_SPEC = importlib.util.spec_from_file_location("slime_fp32_lm_head_test", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
assert_fp32_lm_head = _MODULE.assert_fp32_lm_head
enable_fp32_lm_head = _MODULE.enable_fp32_lm_head
fp32_lm_head_weight_dtypes = _MODULE.fp32_lm_head_weight_dtypes


class _FakeOutputLayer(torch.nn.Module):
    def __init__(self, hidden: int = 4, vocab: int = 7) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(vocab, hidden, dtype=torch.bfloat16))
        self.last_input_dtype = None

    def forward(self, input_, weight=None, runtime_gather_output=None):
        del runtime_gather_output
        self.last_input_dtype = input_.dtype
        return F.linear(input_, self.weight if weight is None else weight), None


class _FakeGPT(torch.nn.Module):
    def __init__(self, *, tied: bool = False) -> None:
        super().__init__()
        self.body = torch.nn.Linear(4, 4, bias=False, dtype=torch.bfloat16)
        self.output_layer = _FakeOutputLayer()
        self.share_embeddings_and_output_weights = tied

    def forward(self, hidden):
        return self.output_layer(hidden)[0]


def test_fp32_head_survives_mixed_precision_apply_and_projects_fp32_hidden() -> None:
    model = _FakeGPT()
    original_key_set = set(model.state_dict())

    assert enable_fp32_lm_head(model)
    model.bfloat16()

    assert model.body.weight.dtype == torch.bfloat16
    assert model.output_layer.weight.dtype == torch.float32
    assert set(model.state_dict()) == original_key_set == {
        "body.weight",
        "output_layer.weight",
    }

    hidden = torch.randn(3, 4, dtype=torch.bfloat16, requires_grad=True)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        logits = model(hidden)
    assert model.output_layer.last_input_dtype == torch.float32
    assert logits.dtype == torch.float32

    logits.square().mean().backward()
    assert model.output_layer.weight.grad is not None
    assert model.output_layer.weight.grad.dtype == torch.float32
    assert hidden.grad is not None and hidden.grad.dtype == torch.bfloat16


def test_fp32_head_is_ddp_optimizer_checkpoint_and_sync_surface_safe() -> None:
    model = _FakeGPT()
    enable_fp32_lm_head(model)
    model.bfloat16()

    with tempfile.TemporaryDirectory() as temp_dir:
        rendezvous = Path(temp_dir) / "gloo-rendezvous"
        dist.init_process_group(
            backend="gloo",
            init_method=f"file://{rendezvous}",
            rank=0,
            world_size=1,
        )
        try:
            ddp = torch.nn.parallel.DistributedDataParallel(model)
            optimizer = torch.optim.Adam(ddp.parameters(), lr=1e-3)
            loss = ddp(torch.randn(2, 4, dtype=torch.bfloat16)).float().square().mean()
            loss.backward()
            optimizer.step()

            assert_fp32_lm_head(ddp)
            assert fp32_lm_head_weight_dtypes(ddp) == (torch.float32,)
            # This is the same named-parameter surface consumed by Slime's
            # weight-sync iterators; the protocol carries each tensor's dtype.
            assert dict(ddp.named_parameters())["module.output_layer.weight"].dtype == torch.float32
            state = ddp.module.state_dict()
        finally:
            dist.destroy_process_group()

    restored = _FakeGPT()
    enable_fp32_lm_head(restored)
    restored.bfloat16()
    # Simulate an older BF16 release checkpoint. load_state_dict copies into,
    # rather than replacing, the FP32 destination Parameter.
    bf16_state = {name: tensor.bfloat16() for name, tensor in state.items()}
    restored.load_state_dict(bf16_state)
    assert restored.output_layer.weight.dtype == torch.float32
    assert_fp32_lm_head(restored)


def test_fp32_head_rejects_tied_output_weight() -> None:
    with pytest.raises(ValueError, match="requires untied"):
        enable_fp32_lm_head(_FakeGPT(tied=True))
