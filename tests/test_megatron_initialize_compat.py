import logging
import pickle
import sys
import types
from unittest.mock import Mock

import pytest
import torch

from slime.backends.megatron_utils.compat import (
    ensure_checkpoint_enum_compat,
    repair_distributed_optimizer_param_index_maps,
    repair_hybrid_optimizer_native_fp32_offload_sync,
    warn_if_numpy_compatibility_is_unverified,
)


NUM_GPUS = 0


def test_checkpoint_enum_compat_installs_missing_capability() -> None:
    transformer_enums = types.ModuleType("fake_megatron.transformer.enums")

    assert ensure_checkpoint_enum_compat(transformer_enums) is True
    scope = transformer_enums.InferenceCudaGraphScope
    assert [member.name for member in scope] == ["none", "layer", "block"]
    assert [member.value for member in scope] == [1, 2, 3]
    assert scope.__module__ == transformer_enums.__name__


def test_checkpoint_enum_compat_supports_pickle_round_trip(monkeypatch) -> None:
    transformer_enums = sys.modules[__name__]
    monkeypatch.delattr(
        transformer_enums,
        "InferenceCudaGraphScope",
        raising=False,
    )

    assert ensure_checkpoint_enum_compat(transformer_enums) is True
    scope = transformer_enums.InferenceCudaGraphScope
    assert pickle.loads(pickle.dumps(scope.block)) is scope.block


def test_checkpoint_enum_compat_preserves_native_megatron_enum() -> None:
    native_scope = object()
    transformer_enums = types.ModuleType("fake_megatron.transformer.enums")
    transformer_enums.InferenceCudaGraphScope = native_scope

    assert ensure_checkpoint_enum_compat(transformer_enums) is False
    assert transformer_enums.InferenceCudaGraphScope is native_scope


def test_numpy_compatibility_warning_is_version_gated() -> None:
    logger = Mock(spec=logging.Logger)

    assert warn_if_numpy_compatibility_is_unverified("1.26.4", logger) is False
    logger.warning.assert_not_called()

    assert warn_if_numpy_compatibility_is_unverified("2.2.1", logger) is True
    logger.warning.assert_called_once()
    assert logger.warning.call_args.args[-1] == "2.2.1"


def test_mixed_dtype_distributed_optimizer_checkpoint_map_is_repaired() -> None:
    distributed_optimizer_type = type("DistributedOptimizer", (), {})
    distributed_optimizer = distributed_optimizer_type()
    fp32_head = object()
    bf16_mlp = object()
    bf16_norm = object()
    fp32_head_shard = types.SimpleNamespace(numel=lambda: 8)
    bf16_mlp_shard = types.SimpleNamespace(numel=lambda: 4)
    bf16_norm_shard = types.SimpleNamespace(numel=lambda: 2)
    local_numel = {fp32_head: 8, bf16_mlp: 4, bf16_norm: 2}
    distributed_optimizer._get_model_param_range_map = lambda model_param: {
        "param": types.SimpleNamespace(start=0, end=local_numel[model_param])
    }
    distributed_optimizer.model_fp32_groups = [[fp32_head]]
    distributed_optimizer.model_float16_groups = [[bf16_mlp, bf16_norm]]
    distributed_optimizer.shard_fp32_groups = [[fp32_head_shard]]
    distributed_optimizer.shard_fp32_from_float16_groups = [
        [bf16_mlp_shard, bf16_norm_shard]
    ]
    distributed_optimizer.config = types.SimpleNamespace(
        use_precision_aware_optimizer_no_fp8_or_ds_fp8=False
    )
    distributed_optimizer.ddp_config = types.SimpleNamespace(use_megatron_fsdp=False)
    distributed_optimizer.optimizer = types.SimpleNamespace(
        param_groups=[{"params": [fp32_head_shard, bf16_mlp_shard, bf16_norm_shard]}]
    )
    # Megatron originally records gradient-buffer order, but the inner
    # optimizer is subsequently reordered to FP32 followed by BF16/FP16.
    distributed_optimizer.model_param_group_index_map = {
        bf16_mlp: (0, 0),
        bf16_norm: (0, 1),
        fp32_head: (0, 2),
    }
    optimizer = types.SimpleNamespace(chained_optimizers=[distributed_optimizer])

    assert repair_distributed_optimizer_param_index_maps(optimizer) == 1
    assert distributed_optimizer.model_param_group_index_map == {
        fp32_head: (0, 0),
        bf16_mlp: (0, 1),
        bf16_norm: (0, 2),
    }
    assert repair_distributed_optimizer_param_index_maps(optimizer) == 0


def test_mixed_dtype_checkpoint_map_rejects_unexpected_shard_order() -> None:
    distributed_optimizer_type = type("DistributedOptimizer", (), {})
    distributed_optimizer = distributed_optimizer_type()
    fp32_head = object()
    bf16_mlp = object()
    fp32_head_shard = object()
    bf16_mlp_shard = object()
    distributed_optimizer.model_fp32_groups = [[fp32_head]]
    distributed_optimizer.model_float16_groups = [[bf16_mlp]]
    distributed_optimizer.shard_fp32_groups = [[fp32_head_shard]]
    distributed_optimizer.shard_fp32_from_float16_groups = [[bf16_mlp_shard]]
    distributed_optimizer.config = types.SimpleNamespace(
        use_precision_aware_optimizer_no_fp8_or_ds_fp8=False
    )
    distributed_optimizer.ddp_config = types.SimpleNamespace(use_megatron_fsdp=False)
    distributed_optimizer.optimizer = types.SimpleNamespace(
        param_groups=[{"params": [bf16_mlp_shard, fp32_head_shard]}]
    )
    distributed_optimizer.model_param_group_index_map = {
        bf16_mlp: (0, 0),
        fp32_head: (0, 1),
    }

    with pytest.raises(RuntimeError, match="shard order"):
        repair_distributed_optimizer_param_index_maps(distributed_optimizer)


def _mixed_native_fp32_offload_optimizer():
    hybrid_optimizer_type = type("HybridDeviceOptimizer", (), {})
    hybrid_optimizer = hybrid_optimizer_type()
    hybrid_optimizer.param_update_in_fp32 = True

    model_fp32 = torch.tensor([3.0, 5.0], dtype=torch.float32)
    model_bf16 = torch.tensor([7.0, 11.0], dtype=torch.bfloat16)
    inner_fp32 = torch.zeros_like(model_fp32)
    inner_bf16 = torch.zeros_like(model_bf16, dtype=torch.float32)
    hybrid_optimizer.param_to_inner_param = {
        model_fp32: inner_fp32,
        model_bf16: inner_bf16,
    }
    hybrid_optimizer.param_to_fp32_param = {model_bf16: inner_bf16}
    hybrid_optimizer.state = {}
    hybrid_optimizer.update_fp32_param_by_new_param = lambda: None
    hybrid_optimizer._update_fp32_params_by_new_state = lambda: None

    distributed_optimizer_type = type("DistributedOptimizer", (), {})
    distributed_optimizer = distributed_optimizer_type()
    distributed_optimizer.optimizer = hybrid_optimizer
    optimizer = types.SimpleNamespace(chained_optimizers=[distributed_optimizer])
    return optimizer, hybrid_optimizer, model_fp32, model_bf16, inner_fp32, inner_bf16


def test_mixed_native_fp32_cpu_offload_sync_is_repaired_and_idempotent() -> None:
    (
        optimizer,
        hybrid_optimizer,
        model_fp32,
        model_bf16,
        inner_fp32,
        inner_bf16,
    ) = _mixed_native_fp32_offload_optimizer()

    assert repair_hybrid_optimizer_native_fp32_offload_sync(optimizer) == 1
    assert repair_hybrid_optimizer_native_fp32_offload_sync(optimizer) == 0

    # Release/model-only load: both the cast BF16 shard and the already-FP32
    # offloaded shard must refresh their detached CPU masters.
    hybrid_optimizer.update_fp32_param_by_new_param()
    torch.testing.assert_close(inner_fp32, model_fp32)
    torch.testing.assert_close(inner_bf16, model_bf16.float())

    # Numbered checkpoint load: HDO rebuilds its inner parameters before this
    # hook, so both masters must be refreshed from loaded optimizer state.
    loaded_fp32 = torch.tensor([13.0, 17.0])
    loaded_bf16_master = torch.tensor([19.0, 23.0])
    reloaded_inner_fp32 = torch.zeros_like(inner_fp32)
    reloaded_inner_bf16 = torch.zeros_like(inner_bf16)
    hybrid_optimizer.param_to_inner_param = {
        model_fp32: reloaded_inner_fp32,
        model_bf16: reloaded_inner_bf16,
    }
    hybrid_optimizer.param_to_fp32_param = {
        model_bf16: reloaded_inner_bf16,
    }
    hybrid_optimizer.state = {
        model_fp32: {"master_param": loaded_fp32},
        model_bf16: {"master_param": loaded_bf16_master},
    }
    hybrid_optimizer._update_fp32_params_by_new_state()
    torch.testing.assert_close(reloaded_inner_fp32, loaded_fp32)
    torch.testing.assert_close(reloaded_inner_bf16, loaded_bf16_master)


def test_native_fp32_offload_sync_patch_is_narrowly_gated() -> None:
    optimizer, hybrid_optimizer, model_fp32, _, _, _ = _mixed_native_fp32_offload_optimizer()

    # A native FP32 parameter that remains on device does not need the repair.
    hybrid_optimizer.param_to_inner_param[model_fp32] = model_fp32
    assert repair_hybrid_optimizer_native_fp32_offload_sync(optimizer) == 0

    # Likewise, an all-lower-precision optimizer has complete upstream maps.
    hybrid_optimizer.param_to_inner_param.pop(model_fp32)
    assert repair_hybrid_optimizer_native_fp32_offload_sync(optimizer) == 0


def test_mixed_native_fp32_cpu_offload_sync_fails_closed_on_bad_state() -> None:
    optimizer, hybrid_optimizer, model_fp32, _, _, _ = _mixed_native_fp32_offload_optimizer()
    assert repair_hybrid_optimizer_native_fp32_offload_sync(optimizer) == 1

    hybrid_optimizer.state = {model_fp32: {}}
    with pytest.raises(RuntimeError, match="missing master_param"):
        hybrid_optimizer._update_fp32_params_by_new_state()

    unknown_param = torch.tensor([29.0, 31.0])
    hybrid_optimizer.state = {unknown_param: {"master_param": unknown_param.clone()}}
    with pytest.raises(RuntimeError, match="unknown parameter"):
        hybrid_optimizer._update_fp32_params_by_new_state()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
