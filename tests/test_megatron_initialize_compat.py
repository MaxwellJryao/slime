import logging
import pickle
import sys
import types
from unittest.mock import Mock

import pytest

from slime.backends.megatron_utils.compat import (
    ensure_checkpoint_enum_compat,
    repair_distributed_optimizer_param_index_maps,
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


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
