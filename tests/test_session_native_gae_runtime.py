from types import SimpleNamespace

import pytest

from slime.utils.session_native_gae_runtime import (
    CONTRACT_KEY,
    CRITIC_DENOM_FIELD,
    CRITIC_MASK_FIELD,
    EXPECTED_METRICS,
    METRICS_FIELD,
    NORMALIZED_FLAG,
    RUNTIME_CAPABILITIES,
    RUNTIME_CAPABILITIES_FIELD,
    RUNTIME_VERSION,
    RUNTIME_VERSION_FIELD,
    SessionNativeGAERuntimeError,
    copy_runtime_attestation,
    prepare_train_data,
    validate_args,
    validate_optimizer_payload,
)


def _args(**overrides):
    values = {
        "session_native_gae": True,
        "use_critic": True,
        "advantage_estimator": "ppo",
        "policy_loss_type": "ppo",
        "loss_type": "policy_loss",
        "normalize_advantages": True,
        "use_rollout_logprobs": True,
        "compute_advantages_and_returns": True,
        "custom_advantage_function_path": (
            "slime_bridge.session_native_gae.compute_session_native_advantages_and_returns"
        ),
        "context_parallel_size": 1,
        "kl_coef": 0.0,
        "use_opd": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _contract(
    *,
    rollout_id: int,
    action_index: int,
    action_type: str,
    response_length: int,
    trainable_count: int,
    terminal_reward: float = 1.0,
):
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
        "action_reward": terminal_reward if action_index == 1 else 0.0,
        "terminal_reward": terminal_reward,
        "response_token_count": response_length,
        "trainable_token_count": trainable_count,
    }


def _samples_and_data():
    contracts = [
        _contract(
            rollout_id=7,
            action_index=0,
            action_type="ROUTE",
            response_length=3,
            trainable_count=2,
        ),
        _contract(
            rollout_id=7,
            action_index=1,
            action_type="SUBMIT",
            response_length=2,
            trainable_count=1,
        ),
        _contract(
            rollout_id=8,
            action_index=0,
            action_type="ROUTE",
            response_length=1,
            trainable_count=1,
            terminal_reward=0.5,
        ),
        _contract(
            rollout_id=8,
            action_index=1,
            action_type="VERIFY",
            response_length=4,
            trainable_count=3,
            terminal_reward=0.5,
        ),
    ]
    masks = [[1, 0, 1], [1, 0], [1], [0, 1, 1, 1]]
    samples = [SimpleNamespace(train_metadata={CONTRACT_KEY: contract}) for contract in contracts]
    data = {
        "metadata": [sample.train_metadata for sample in samples],
        "rollout_ids": [7, 7, 8, 8],
        "response_lengths": [3, 2, 1, 4],
        "loss_masks": [list(mask) for mask in masks],
    }
    return samples, data, masks


def _metrics():
    return {name: float(index) / 10.0 for index, name in enumerate(sorted(EXPECTED_METRICS))}


def test_prepare_train_data_preserves_actor_masks_and_builds_distinct_critic_contract():
    samples, data, original_masks = _samples_and_data()

    prepare_train_data(_args(), samples, data)

    assert data["loss_masks"] == original_masks
    assert data[CRITIC_MASK_FIELD] == [
        [1, 0, 0],
        [1, 0],
        [1],
        [1, 0, 0, 0],
    ]
    assert data[CRITIC_DENOM_FIELD] == [2, 2, 2, 2]
    assert data[RUNTIME_VERSION_FIELD] == RUNTIME_VERSION
    assert data[RUNTIME_CAPABILITIES_FIELD] == RUNTIME_CAPABILITIES


def test_prepare_train_data_rejects_unavailable_cost_to_go_contract():
    samples, data, _ = _samples_and_data()
    unavailable = dict(data["metadata"][0][CONTRACT_KEY])
    unavailable.update(
        {
            "available": False,
            "invalid_reason": ("session-native GAE requires terminal_broadcast; cost_to_go is not a critic target"),
        }
    )
    data["metadata"][0] = {CONTRACT_KEY: unavailable}
    samples[0].train_metadata = data["metadata"][0]

    with pytest.raises(SessionNativeGAERuntimeError, match="cost_to_go"):
        prepare_train_data(_args(), samples, data)


def test_prepare_train_data_rejects_available_cost_to_go_objective():
    samples, data, _ = _samples_and_data()
    contract = dict(data["metadata"][0][CONTRACT_KEY])
    contract["objective"] = "cost_to_go"
    data["metadata"][0] = {CONTRACT_KEY: contract}
    samples[0].train_metadata = data["metadata"][0]

    with pytest.raises(SessionNativeGAERuntimeError, match="cost_to_go"):
        prepare_train_data(_args(), samples, data)


def test_prepare_train_data_rejects_missing_or_split_action_trajectory():
    samples, data, _ = _samples_and_data()
    samples.pop(1)
    for field in ("metadata", "rollout_ids", "response_lengths", "loss_masks"):
        data[field].pop(1)

    with pytest.raises(SessionNativeGAERuntimeError, match="missing action samples"):
        prepare_train_data(_args(), samples, data)


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"use_critic": False}, "--use-critic"),
        ({"normalize_advantages": False}, "--normalize-advantages"),
        ({"context_parallel_size": 2}, "context parallel size 1"),
        ({"kl_coef": 0.01}, "--kl-coef=0"),
        ({"use_opd": True}, "OPD disabled"),
    ],
)
def test_session_native_cli_envelope_fails_closed(override, message):
    with pytest.raises(SessionNativeGAERuntimeError, match=message):
        validate_args(_args(**override))


def test_independent_trace_fallback_has_no_session_native_requirements():
    validate_args(SimpleNamespace(session_native_gae=False))
    samples, data, original_masks = _samples_and_data()

    prepare_train_data(SimpleNamespace(session_native_gae=False), samples, data)

    assert data["loss_masks"] == original_masks
    assert CRITIC_MASK_FIELD not in data
    assert RUNTIME_VERSION_FIELD not in data


def test_optimizer_payload_requires_boundary_mask_and_prebroadcast_normalization():
    payload = {
        "response_lengths": [3, 2],
        "loss_masks": [[1, 0, 1], [1, 0]],
        CRITIC_MASK_FIELD: [[1, 0, 0], [1, 0]],
        CRITIC_DENOM_FIELD: [2.0, 2.0],
        "advantages": [[-1.0, -1.0, -1.0], [1.0, 1.0]],
        "returns": [[0.8, 0.0, 0.0], [1.0, 0.0]],
        NORMALIZED_FLAG: True,
        METRICS_FIELD: _metrics(),
        RUNTIME_VERSION_FIELD: RUNTIME_VERSION,
        RUNTIME_CAPABILITIES_FIELD: dict(RUNTIME_CAPABILITIES),
    }

    validate_optimizer_payload(_args(), payload)

    payload[CRITIC_MASK_FIELD] = payload["loss_masks"]
    with pytest.raises(SessionNativeGAERuntimeError, match="boundary-only"):
        validate_optimizer_payload(_args(), payload)

    payload[CRITIC_MASK_FIELD] = [[1, 0, 0], [1, 0]]
    payload[NORMALIZED_FLAG] = False
    with pytest.raises(SessionNativeGAERuntimeError, match="action level"):
        validate_optimizer_payload(_args(), payload)

    payload[NORMALIZED_FLAG] = True
    payload[CRITIC_DENOM_FIELD] = [0.0, 2.0]
    with pytest.raises(SessionNativeGAERuntimeError, match="critic denominators"):
        validate_optimizer_payload(_args(), payload)


def test_dp_partition_attestation_copy_is_exact_and_versioned():
    source = {
        RUNTIME_VERSION_FIELD: RUNTIME_VERSION,
        RUNTIME_CAPABILITIES_FIELD: dict(RUNTIME_CAPABILITIES),
    }
    target = {}

    copy_runtime_attestation(source, target)

    assert target == source
    target[RUNTIME_CAPABILITIES_FIELD]["critic_loss_mask"] = "wrong"
    assert source[RUNTIME_CAPABILITIES_FIELD] == RUNTIME_CAPABILITIES

    source[RUNTIME_VERSION_FIELD] = "stale"
    with pytest.raises(SessionNativeGAERuntimeError, match="attestation mismatch"):
        copy_runtime_attestation(source, {})
