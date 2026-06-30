"""Keep Megatron's actor LM projection in true FP32 mixed precision."""

from __future__ import annotations

from collections.abc import Iterable

import torch


_FP32_LAYER_TYPES: dict[type[torch.nn.Module], type[torch.nn.Module]] = {}


class _FP32LMHeadMixin:
    """Mixin installed on an existing output-layer instance without renaming it.

    Megatron's ``Float16Module`` recursively calls ``module.bfloat16()`` or
    ``module.half()`` before DDP and optimizer construction.  Overriding
    ``_apply`` is therefore necessary: casting logits after a low-precision
    GEMM is not an FP32 LM head.
    """

    _slime_fp32_lm_head = True

    def _slime_restore_fp32_parameters(self) -> None:
        for parameter in self.parameters(recurse=False):
            if parameter.is_floating_point() and parameter.dtype != torch.float32:
                parameter.data = parameter.data.float()
                if parameter.grad is not None:
                    parameter.grad.data = parameter.grad.data.float()

    def _apply(self, fn, *args, **kwargs):
        result = super()._apply(fn, *args, **kwargs)
        self._slime_restore_fp32_parameters()
        return result

    def forward(self, input_: torch.Tensor, *args, **kwargs):
        if not torch.is_floating_point(input_):
            raise TypeError(
                "FP32 LM head expected floating hidden states, "
                f"got dtype={input_.dtype}"
            )
        supplied_weight = kwargs.get("weight")
        if supplied_weight is not None and supplied_weight.dtype != torch.float32:
            raise RuntimeError(
                "FP32 LM head received a non-FP32 external/tied output weight; "
                "use untied embeddings and output weights"
            )
        hidden_fp32 = input_.float()
        # Slime normally uses explicit BF16/FP16 modules rather than autocast,
        # but disabling an ambient autocast makes the projection contract local
        # and prevents a caller from silently demoting this GEMM.
        with torch.autocast(device_type=hidden_fp32.device.type, enabled=False):
            return super().forward(hidden_fp32, *args, **kwargs)


def _fp32_layer_type(base: type[torch.nn.Module]) -> type[torch.nn.Module]:
    layer_type = _FP32_LAYER_TYPES.get(base)
    if layer_type is None:
        layer_type = type(
            f"SlimeFP32{base.__name__}",
            (_FP32LMHeadMixin, base),
            {"__module__": __name__},
        )
        _FP32_LAYER_TYPES[base] = layer_type
    return layer_type


def enable_fp32_lm_head(model: torch.nn.Module) -> bool:
    """Enable a real FP32 output projection on one post-process actor chunk.

    The existing layer instance and parameter names are retained, so Megatron
    distributed checkpoints and HF weight conversion keep their normal keys.
    The function must run inside the model provider, before Float16Module/DDP
    and optimizer construction.
    """

    output_layer = getattr(model, "output_layer", None)
    if output_layer is None:
        return False
    if bool(getattr(model, "share_embeddings_and_output_weights", False)):
        raise ValueError(
            "--enable-fp32-lm-head requires untied embeddings/output weights; "
            "a tied weight cannot be FP32 only in the LM projection"
        )
    weight = getattr(output_layer, "weight", None)
    if not isinstance(weight, torch.nn.Parameter):
        raise TypeError(
            "--enable-fp32-lm-head requires output_layer.weight to be a Parameter"
        )
    if not getattr(output_layer, "_slime_fp32_lm_head", False):
        output_layer.__class__ = _fp32_layer_type(type(output_layer))
    output_layer._slime_restore_fp32_parameters()
    assert_fp32_lm_head(output_layer)
    return True


def iter_fp32_lm_heads(models: torch.nn.Module | Iterable[torch.nn.Module]):
    """Yield enabled heads through Float16Module/DDP wrapper hierarchies."""

    roots = [models] if isinstance(models, torch.nn.Module) else models
    seen: set[int] = set()
    for root in roots:
        for module in root.modules():
            if id(module) in seen:
                continue
            seen.add(id(module))
            if getattr(module, "_slime_fp32_lm_head", False):
                yield module


def assert_fp32_lm_head(
    models: torch.nn.Module | Iterable[torch.nn.Module],
) -> None:
    """Fail if an enabled head was demoted by load/offload/wrapping code."""

    for output_layer in iter_fp32_lm_heads(models):
        parameters = list(output_layer.parameters(recurse=False))
        if not parameters:
            raise RuntimeError("enabled FP32 LM head has no direct parameters")
        bad = [parameter.dtype for parameter in parameters if parameter.dtype != torch.float32]
        if bad:
            raise RuntimeError(f"FP32 LM head parameter dtype invariant violated: {bad}")


def fp32_lm_head_weight_dtypes(
    models: torch.nn.Module | Iterable[torch.nn.Module],
) -> tuple[torch.dtype, ...]:
    """Expose the sync/checkpoint-visible head dtypes for diagnostics/tests."""

    return tuple(
        parameter.dtype
        for output_layer in iter_fp32_lm_heads(models)
        for parameter in output_layer.parameters(recurse=False)
    )
