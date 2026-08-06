from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import _cp_dist_helpers
import pytest
import torch


def _stub_module(name, **attributes):
    module = ModuleType(name)
    module.__dict__.update(attributes)
    sys.modules[name] = module
    return module


class _StubConfig:
    pass


_import_stubs = {
    "megatron.core.packed_seq_params": _stub_module(
        "megatron.core.packed_seq_params",
        PackedSeqParams=_StubConfig,
    ),
    "sglang.srt.constants": _stub_module(
        "sglang.srt.constants",
        GPU_MEMORY_TYPE_CUDA_GRAPH="cuda_graph",
        GPU_MEMORY_TYPE_KV_CACHE="kv_cache",
        GPU_MEMORY_TYPE_WEIGHTS="weights",
    ),
    "sglang.srt": _stub_module("sglang.srt"),
    "sglang": _stub_module("sglang"),
    "slime.backends.sglang_utils.external": _stub_module(
        "slime.backends.sglang_utils.external",
        start_external_rollout_servers=lambda *_args, **_kwargs: None,
    ),
    "slime.backends.sglang_utils.sglang_config": _stub_module(
        "slime.backends.sglang_utils.sglang_config",
        ModelConfig=_StubConfig,
        ServerGroupConfig=_StubConfig,
        SglangConfig=_StubConfig,
    ),
    "slime.backends.sglang_utils.sglang_engine": _stub_module(
        "slime.backends.sglang_utils.sglang_engine",
        SGLangEngine=_StubConfig,
    ),
}

from slime.backends.megatron_utils import data as data_module  # noqa: E402
from slime.ray import rollout as rollout_module  # noqa: E402
from slime.utils.data import process_rollout_data  # noqa: E402
from slime.utils.types import Sample  # noqa: E402

for _module_name, _stub in _import_stubs.items():
    if sys.modules.get(_module_name) is _stub:
        sys.modules.pop(_module_name)
for _module_name, _fake_module in (
    ("megatron.core.mpu", _cp_dist_helpers._fake_mpu),
    ("megatron.core", _cp_dist_helpers._fake_core),
    ("megatron", _cp_dist_helpers._fake_megatron),
):
    if sys.modules.get(_module_name) is _fake_module:
        sys.modules.pop(_module_name)

NUM_GPUS = 0


def _sample(index, rollout_id, train_metadata):
    return Sample(
        index=index,
        rollout_id=rollout_id,
        tokens=[10, 11, 12],
        response_length=1,
        reward=1.0,
        loss_mask=[1],
        status=Sample.Status.COMPLETED,
        train_metadata=train_metadata,
    )


def _manager(transport):
    manager_cls = rollout_module.RolloutManager.__ray_metadata__.modified_class
    manager = object.__new__(manager_cls)
    manager.args = SimpleNamespace(
        advantage_estimator="ppo",
        balance_by_flops=False,
        balance_data=False,
        global_batch_size=2,
        grpo_std_normalization=False,
        micro_batch_size=1,
        n_samples_per_prompt=1,
        reward_key=None,
        rewards_normalization=False,
        rollout_data_transport=transport,
        use_dynamic_batch_size=False,
    )
    manager.custom_convert_samples_to_train_data_func = None
    manager.custom_reward_post_process_func = None
    manager.train_parallel_config = {
        "cp_size": 1,
        "dp_size": 2,
        "microbatch_group_size_per_vp_stage": 1,
        "vpp_size": 1,
    }
    return manager


@pytest.mark.unit
@pytest.mark.parametrize("transport", ["object-store", "nixl"])
def test_mixed_train_metadata_reaches_each_dp_trainer(monkeypatch, transport):
    expected_metadata = {
        0: None,
        1: {"loss_type": "policy_loss"},
        2: {"source": "tool"},
        3: {"source": "environment", "weight": 0.5},
    }
    samples = [
        _sample(0, 7, expected_metadata[0]),
        _sample(1, 7, expected_metadata[1]),
        _sample(2, 8, expected_metadata[2]),
        _sample(3, 8, expected_metadata[3]),
    ]
    manager = _manager(transport)
    monkeypatch.setattr(rollout_module.ray, "put", lambda value, **_kwargs: value)
    monkeypatch.setattr(rollout_module.ray, "get", lambda value, **_kwargs: value)

    converted = manager._convert_samples_to_train_data(samples)

    assert converted["metadata"] == [expected_metadata[index] for index in range(4)]
    refs = manager._split_train_data_by_dp(converted)
    observed_metadata = {}
    for dp_rank in range(2):
        trainer_payload = process_rollout_data(manager.args, refs, dp_rank, 2)
        assert len(trainer_payload["metadata"]) == len(trainer_payload["sample_indices"])
        observed_metadata.update(
            zip(
                trainer_payload["sample_indices"],
                trainer_payload["metadata"],
                strict=True,
            )
        )

    assert observed_metadata == expected_metadata


@pytest.mark.unit
def test_rollout_logging_ignores_non_numeric_train_metadata(monkeypatch):
    captured = {}
    monkeypatch.setattr(data_module.mpu, "get_tensor_model_parallel_rank", lambda: 0, raising=False)
    monkeypatch.setattr(data_module.mpu, "is_pipeline_last_stage", lambda: True, raising=False)
    monkeypatch.setattr(data_module.mpu, "get_context_parallel_world_size", lambda: 1, raising=False)
    monkeypatch.setattr(
        data_module.mpu,
        "get_data_parallel_world_size",
        lambda **_kwargs: 1,
        raising=False,
    )
    monkeypatch.setattr(
        data_module,
        "gather_log_data",
        lambda _name, _args, _rollout_id, log_dict: captured.update(log_dict) or {},
    )
    rollout_data = {
        "global_batch_sizes": [2],
        "loss_masks": [torch.ones(1), torch.ones(1)],
        "metadata": [None, {"loss_type": "policy_loss"}],
        "response_lengths": [1, 1],
        "rewards": [1.0, 2.0],
        "total_lengths": [3, 3],
    }
    args = SimpleNamespace(
        ci_test=False,
        log_correct_samples=False,
        log_multi_turn=False,
        log_passrate=False,
    )

    data_module.log_rollout_data(3, args, rollout_data)

    assert "metadata" not in captured
    assert captured["rewards"] == (3.0, 2)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
