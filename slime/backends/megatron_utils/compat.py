"""Small compatibility shims for version-skewed Megatron environments."""

from __future__ import annotations

import enum
import logging
from collections.abc import Mapping
from types import ModuleType
from types import MethodType


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


_HYBRID_FP32_OFFLOAD_PATCH_MARKER = "_slime_mixed_native_fp32_offload_sync_patch"


def _copy_hybrid_inner_params_from_model(hybrid_optimizer: object) -> None:
    """Synchronize detached HybridDeviceOptimizer params after a model-only load."""

    param_to_inner = getattr(hybrid_optimizer, "param_to_inner_param", None)
    if not isinstance(param_to_inner, Mapping):
        raise RuntimeError("HybridDeviceOptimizer is missing its model-to-inner parameter map")

    copies = []
    for model_param, inner_param in param_to_inner.items():
        if inner_param is model_param:
            continue
        if getattr(model_param, "shape", None) != getattr(inner_param, "shape", None):
            raise RuntimeError("HybridDeviceOptimizer model/inner parameter shape mismatch")
        copies.append((inner_param, model_param))

    # Validate the complete map before mutating any optimizer parameter.
    for inner_param, model_param in copies:
        inner_param.data.copy_(model_param.data, non_blocking=False)


def _copy_hybrid_inner_params_from_loaded_state(hybrid_optimizer: object) -> None:
    """Synchronize rebuilt HybridDeviceOptimizer params from loaded master state."""

    param_to_inner = getattr(hybrid_optimizer, "param_to_inner_param", None)
    state = getattr(hybrid_optimizer, "state", None)
    if not isinstance(param_to_inner, Mapping) or not isinstance(state, Mapping):
        raise RuntimeError("HybridDeviceOptimizer is missing parameter maps required for checkpoint load")

    copies = []
    for model_param, param_state in state.items():
        if model_param not in param_to_inner:
            raise RuntimeError("HybridDeviceOptimizer checkpoint state contains an unknown parameter")
        if not isinstance(param_state, Mapping) or "master_param" not in param_state:
            raise RuntimeError("HybridDeviceOptimizer checkpoint state is missing master_param")
        inner_param = param_to_inner[model_param]
        master_param = param_state["master_param"]
        if getattr(inner_param, "shape", None) != getattr(master_param, "shape", None):
            raise RuntimeError("HybridDeviceOptimizer inner/master parameter shape mismatch")
        copies.append((inner_param, master_param))

    # Validate the complete state before mutating any optimizer parameter.
    for inner_param, master_param in copies:
        inner_param.data.copy_(master_param.data, non_blocking=False)


def repair_hybrid_optimizer_native_fp32_offload_sync(optimizer: object) -> int:
    """Repair mixed-dtype CPU-offload synchronization in pinned Megatron.

    ``HybridDeviceOptimizer`` creates detached CPU parameters for every offloaded
    shard, but its precision-aware synchronization maps include only parameters
    that were cast *to* FP32.  A model parameter that is already FP32 (for
    example Slime's true-FP32 LM head) is therefore omitted.  A model-only load
    leaves that CPU master at its random initialization, while a numbered
    checkpoint load indexes the incomplete map and fails.

    Patch only affected optimizer instances: precision-aware hybrid optimizers
    that contain both cast-to-FP32 parameters and an offloaded native-FP32
    parameter.  Returns the number of newly patched instances.
    """

    repaired = 0
    for distributed_optimizer in getattr(optimizer, "chained_optimizers", [optimizer]):
        if distributed_optimizer.__class__.__name__ != "DistributedOptimizer":
            continue
        hybrid_optimizer = getattr(distributed_optimizer, "optimizer", None)
        if hybrid_optimizer is None or hybrid_optimizer.__class__.__name__ != "HybridDeviceOptimizer":
            continue
        if not getattr(hybrid_optimizer, "param_update_in_fp32", False):
            continue

        param_to_inner = getattr(hybrid_optimizer, "param_to_inner_param", None)
        param_to_fp32 = getattr(hybrid_optimizer, "param_to_fp32_param", None)
        if not isinstance(param_to_inner, Mapping) or not isinstance(param_to_fp32, Mapping):
            raise RuntimeError("HybridDeviceOptimizer precision-aware parameter maps are unavailable")

        # A cast-to-FP32 map proves this is a mixed-precision optimizer.  A
        # parameter absent from that map but backed by a distinct inner tensor
        # is a native-FP32 parameter that was offloaded to CPU.
        native_fp32_offloaded = [model_param for model_param, inner_param in param_to_inner.items() if model_param not in param_to_fp32 and inner_param is not model_param]
        if not param_to_fp32 or not native_fp32_offloaded:
            continue

        if getattr(hybrid_optimizer, _HYBRID_FP32_OFFLOAD_PATCH_MARKER, False):
            if (
                getattr(
                    getattr(hybrid_optimizer, "update_fp32_param_by_new_param", None),
                    "__func__",
                    None,
                )
                is not _copy_hybrid_inner_params_from_model
                or getattr(
                    getattr(hybrid_optimizer, "_update_fp32_params_by_new_state", None),
                    "__func__",
                    None,
                )
                is not _copy_hybrid_inner_params_from_loaded_state
            ):
                raise RuntimeError("HybridDeviceOptimizer mixed-FP32 compatibility patch was replaced")
            continue

        if not callable(getattr(hybrid_optimizer, "update_fp32_param_by_new_param", None)) or not callable(getattr(hybrid_optimizer, "_update_fp32_params_by_new_state", None)):
            raise RuntimeError("HybridDeviceOptimizer synchronization API is incompatible with Slime")

        hybrid_optimizer.update_fp32_param_by_new_param = MethodType(_copy_hybrid_inner_params_from_model, hybrid_optimizer)
        hybrid_optimizer._update_fp32_params_by_new_state = MethodType(_copy_hybrid_inner_params_from_loaded_state, hybrid_optimizer)
        setattr(hybrid_optimizer, _HYBRID_FP32_OFFLOAD_PATCH_MARKER, True)
        repaired += 1

    return repaired
