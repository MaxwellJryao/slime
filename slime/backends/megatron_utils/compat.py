"""Small compatibility shims for version-skewed Megatron environments."""

from __future__ import annotations

import enum
import logging
from types import ModuleType


def ensure_checkpoint_enum_compat(transformer_enums: ModuleType | None = None) -> bool:
    """Register a checkpoint enum absent from older Megatron releases.

    Returns ``True`` only when the compatibility enum was installed.  Passing a
    module explicitly keeps the capability check unit-testable without loading
    Megatron or initializing CUDA.
    """

    if transformer_enums is None:
        from megatron.core.transformer import enums as transformer_enums

    if hasattr(transformer_enums, "InferenceCudaGraphScope"):
        return False

    inference_scope = enum.Enum(
        "InferenceCudaGraphScope",
        {"none": 1, "layer": 2, "block": 3},
        module=transformer_enums.__name__,
        qualname="InferenceCudaGraphScope",
    )
    transformer_enums.InferenceCudaGraphScope = inference_scope
    return True


def warn_if_numpy_compatibility_is_unverified(
    version: str,
    logger: logging.Logger,
) -> bool:
    """Warn when running beyond Megatron's historically asserted NumPy 1.x."""

    if version.startswith("1."):
        return False
    logger.warning(
        "running Megatron initialization with NumPy %s; this compatibility path "
        "must be revalidated when the pinned Megatron version changes",
        version,
    )
    return True


def repair_distributed_optimizer_param_index_maps(optimizer: object) -> int:
    """Align Megatron's checkpoint lookup map with mixed-dtype param order.

    The pinned Megatron builds ``model_param_group_index_map`` in gradient-
    buffer order, then reorders each inner optimizer group to FP32 parameters
    followed by BF16/FP16 parameters. With a true FP32 LM head, checkpoint
    save/load can consequently look up a state tensor belonging to another
    parameter. Rebuild the private lookup from the model lists that define the
    final optimizer order. Returns the number of repaired optimizers.
    """

    repaired = 0
    for distributed_optimizer in getattr(optimizer, "chained_optimizers", [optimizer]):
        if distributed_optimizer.__class__.__name__ != "DistributedOptimizer":
            continue
        if getattr(
            getattr(distributed_optimizer, "ddp_config", None),
            "use_megatron_fsdp",
            False,
        ):
            continue
        model_fp32_groups = getattr(distributed_optimizer, "model_fp32_groups", None)
        model_float16_groups = getattr(distributed_optimizer, "model_float16_groups", None)
        shard_fp32_groups = getattr(distributed_optimizer, "shard_fp32_groups", None)
        precision_aware = getattr(
            getattr(distributed_optimizer, "config", None),
            "use_precision_aware_optimizer_no_fp8_or_ds_fp8",
            False,
        )
        float16_shard_attr = (
            "shard_float16_groups"
            if precision_aware
            else "shard_fp32_from_float16_groups"
        )
        shard_float16_groups = getattr(
            distributed_optimizer,
            float16_shard_attr,
            None,
        )
        inner_param_groups = getattr(
            getattr(distributed_optimizer, "optimizer", None),
            "param_groups",
            None,
        )
        current = getattr(distributed_optimizer, "model_param_group_index_map", None)
        groups = (
            model_fp32_groups,
            model_float16_groups,
            shard_fp32_groups,
            shard_float16_groups,
            inner_param_groups,
        )
        if any(group is None for group in groups) or current is None:
            continue
        if len({len(group) for group in groups}) != 1:
            raise RuntimeError(
                "Megatron distributed optimizer has inconsistent mixed-dtype group counts"
            )

        expected = {}
        for group_index, group_values in enumerate(
            zip(*groups, strict=True)
        ):
            model_fp32, model_float16, shard_fp32, shard_float16, inner_group = group_values
            ordered_model_params = [*model_fp32, *model_float16]
            expected_shards = [*shard_fp32, *shard_float16]
            actual_shards = inner_group["params"]
            if len(actual_shards) != len(expected_shards) or any(
                actual is not expected
                for actual, expected in zip(actual_shards, expected_shards, strict=True)
            ):
                raise RuntimeError(
                    "Megatron distributed optimizer shard order does not match mixed-dtype groups"
                )
            range_lookup = getattr(
                distributed_optimizer,
                "_get_model_param_range_map",
                None,
            )
            if callable(range_lookup):
                for model_param, shard_param in zip(
                    ordered_model_params, expected_shards, strict=True
                ):
                    param_range = range_lookup(model_param)["param"]
                    expected_numel = param_range.end - param_range.start
                    if shard_param.numel() != expected_numel:
                        raise RuntimeError(
                            "Megatron distributed optimizer shard size does not match model range"
                        )
            for group_order, model_param in enumerate(ordered_model_params):
                if model_param in expected:
                    raise RuntimeError(
                        "Megatron distributed optimizer contains a duplicate model parameter"
                    )
                expected[model_param] = (group_index, group_order)

        if set(current) != set(expected):
            raise RuntimeError(
                "Megatron distributed optimizer checkpoint map has inconsistent parameters"
            )
        if current != expected:
            distributed_optimizer.model_param_group_index_map = expected
            repaired += 1
    return repaired
