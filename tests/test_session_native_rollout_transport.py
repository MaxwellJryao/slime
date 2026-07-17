from types import ModuleType, SimpleNamespace
import sys

import torch


def _stub_module(name, **attributes):
    module = ModuleType(name)
    module.__dict__.update(attributes)
    sys.modules[name] = module
    return module


class _StubConfig:
    pass


# RolloutManager's conversion/split methods do not instantiate SGLang. Stub
# only those heavyweight imports so this CPU test exercises the real methods.
_stub_module(
    "sglang.srt.constants",
    GPU_MEMORY_TYPE_CUDA_GRAPH="cuda_graph",
    GPU_MEMORY_TYPE_KV_CACHE="kv_cache",
    GPU_MEMORY_TYPE_WEIGHTS="weights",
)
_stub_module("sglang.srt")
_stub_module("sglang")
_stub_module(
    "slime.backends.sglang_utils.external",
    start_external_rollout_servers=lambda *_args, **_kwargs: None,
)
_stub_module(
    "slime.backends.sglang_utils.sglang_config",
    ModelConfig=_StubConfig,
    ServerGroupConfig=_StubConfig,
    SglangConfig=_StubConfig,
)
_stub_module(
    "slime.backends.sglang_utils.sglang_engine",
    SGLangEngine=_StubConfig,
)

from slime.ray import rollout as rollout_module  # noqa: E402
from slime.utils.session_native_gae_runtime import (  # noqa: E402
    CONTRACT_KEY,
    CRITIC_DENOM_FIELD,
    CRITIC_MASK_FIELD,
    RUNTIME_CAPABILITIES,
    RUNTIME_CAPABILITIES_FIELD,
    RUNTIME_VERSION,
    RUNTIME_VERSION_FIELD,
)
from slime.utils.types import Sample  # noqa: E402


def _contract(rollout_id, action_index, action_type, response_length):
    return {
        "schema": "spilot.session_native_action_gae/v1",
        "available": True,
        "session_id": f"session-{rollout_id}",
        "rollout_id": rollout_id,
        "objective": "terminal_broadcast",
        "boundary_value_index": 0,
        "action_index": action_index,
        "action_count": 2,
        "action_type": action_type,
        "action_reward": 1.0 if action_index == 1 else 0.0,
        "terminal_reward": 1.0,
        "response_token_count": response_length,
        "trainable_token_count": response_length,
    }


def _sample(rollout_id, action_index, action_type, response_length, index):
    return Sample(
        index=index,
        rollout_id=rollout_id,
        tokens=list(range(response_length + 2)),
        response_length=response_length,
        reward=1.0,
        loss_mask=[1] * response_length,
        status=Sample.Status.COMPLETED,
        train_metadata={
            CONTRACT_KEY: _contract(
                rollout_id,
                action_index,
                action_type,
                response_length,
            )
        },
    )


def _manager():
    manager_cls = rollout_module.RolloutManager.__ray_metadata__.modified_class
    manager = object.__new__(manager_cls)
    manager.args = SimpleNamespace(
        session_native_gae=True,
        reward_key=None,
        advantage_estimator="ppo",
        rewards_normalization=False,
        n_samples_per_prompt=1,
        grpo_std_normalization=False,
        global_batch_size=2,
        use_dynamic_batch_size=True,
        max_tokens_per_gpu=1_000,
        micro_batch_size=1,
        balance_data=False,
        balance_by_flops=False,
        rollout_data_transport="object-store",
    )
    manager.custom_reward_post_process_func = None
    manager.custom_convert_samples_to_train_data_func = None
    manager.train_parallel_config = {
        "dp_size": 2,
        "cp_size": 1,
        "vpp_size": 1,
        "microbatch_group_size_per_vp_stage": 1,
    }
    return manager


def test_metadata_and_separate_critic_masks_survive_conversion_and_dp_split(monkeypatch):
    samples = [
        _sample(7, 0, "ROUTE", 3, 7),
        _sample(7, 1, "SUBMIT", 2, 7),
        _sample(8, 0, "ROUTE", 4, 8),
        _sample(8, 1, "VERIFY", 1, 8),
    ]
    manager = _manager()

    converted = manager._convert_samples_to_train_data(samples)

    assert converted["loss_masks"] == [[1, 1, 1], [1, 1], [1, 1, 1, 1], [1]]
    assert converted[CRITIC_MASK_FIELD] == [
        [1, 0, 0],
        [1, 0],
        [1, 0, 0, 0],
        [1],
    ]
    assert [item[CONTRACT_KEY]["action_index"] for item in converted["metadata"]] == [
        0,
        1,
        0,
        1,
    ]

    monkeypatch.setattr(rollout_module.ray, "put", lambda value, **_kwargs: value)
    partitions = manager._split_train_data_by_dp(converted)

    assert len(partitions) == 2
    transported_actions = []
    for boxed in partitions:
        payload = boxed.inner
        assert payload[RUNTIME_VERSION_FIELD] == RUNTIME_VERSION
        assert payload[RUNTIME_CAPABILITIES_FIELD] == RUNTIME_CAPABILITIES
        assert len(payload["metadata"]) == len(payload[CRITIC_MASK_FIELD])
        assert torch.is_tensor(payload[CRITIC_DENOM_FIELD])
        assert all(torch.is_tensor(mask) for mask in payload[CRITIC_MASK_FIELD])
        transported_actions.extend(
            (
                item[CONTRACT_KEY]["rollout_id"],
                item[CONTRACT_KEY]["action_index"],
            )
            for item in payload["metadata"]
        )
    assert sorted(transported_actions) == [(7, 0), (7, 1), (8, 0), (8, 1)]
