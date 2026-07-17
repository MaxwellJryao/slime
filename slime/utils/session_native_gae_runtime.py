"""Fail-closed runtime contract for SPilot session-native action GAE.

The rollout adapter emits one independent token sequence per Router action.
Those sequences must remain independent for policy training, while their
action metadata is transported together so a custom advantage function can
reconstruct ``ROUTE -> SUBMIT/VERIFY`` trajectories after data-parallel
partitioning.

This module intentionally owns only transport and optimizer-boundary
invariants.  The environment-specific GAE math lives in the configured custom
advantage function.  Session-native mode is opt-in; ordinary Slime training is
unchanged when it is disabled.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
from numbers import Integral, Real
from typing import Any


RUNTIME_VERSION = "slime.session_native_gae_runtime/v1"
CONTRACT_KEY = "spilot_session_native_gae"
CONTRACT_SCHEMA = "spilot.session_native_action_gae/v1"
OBJECTIVE = "terminal_broadcast"
BOUNDARY_VALUE_INDEX = 0

RUNTIME_CAPABILITIES = {
    "metadata_transport": "per-sample-partition-preserving",
    "critic_loss_mask": "action-boundary-separate-from-actor",
    "advantage_normalization": "action-level-before-token-broadcast",
    "metric_logging": "action-level-unwhitened",
}

RUNTIME_VERSION_FIELD = "session_native_gae_runtime_version"
RUNTIME_CAPABILITIES_FIELD = "session_native_gae_runtime_capabilities"
CRITIC_MASK_FIELD = "critic_action_boundary_masks"
CRITIC_DENOM_FIELD = "critic_rollout_mask_sums"
NORMALIZED_FLAG = "session_native_gae_action_advantages_normalized"
METRICS_FIELD = "session_native_gae_metrics"

EXPECTED_METRICS = frozenset(
    {
        "rollout/session_native_action_reward_mean",
        "rollout/session_native_action_reward_variance",
        "train/session_native_action_advantage_mean",
        "train/session_native_action_advantage_variance",
        "train/session_native_critic_mae",
        "train/session_native_critic_explained_variance",
        "rollout/session_native_trainable_token_share",
    }
)


class SessionNativeGAERuntimeError(ValueError):
    """Raised before optimization when a session-native invariant is absent."""


def enabled(args: Any) -> bool:
    return bool(getattr(args, "session_native_gae", False))


def validate_args(args: Any) -> None:
    """Validate the deliberately narrow first production runtime envelope."""

    if not enabled(args):
        return

    requirements = {
        "--use-critic": bool(getattr(args, "use_critic", False)),
        "--advantage-estimator=ppo": getattr(args, "advantage_estimator", None) == "ppo",
        "--policy-loss-type=ppo": getattr(args, "policy_loss_type", "ppo") == "ppo",
        "--loss-type=policy_loss": getattr(args, "loss_type", None) == "policy_loss",
        "--normalize-advantages": bool(getattr(args, "normalize_advantages", False)),
        "--use-rollout-logprobs": bool(getattr(args, "use_rollout_logprobs", False)),
        "advantage computation enabled": bool(getattr(args, "compute_advantages_and_returns", False)),
        "custom advantage function configured": bool(getattr(args, "custom_advantage_function_path", None)),
        "context parallel size 1": int(getattr(args, "context_parallel_size", 1) or 1) == 1,
        "--kl-coef=0": float(getattr(args, "kl_coef", 0.0)) == 0.0,
        "OPD disabled": not bool(getattr(args, "use_opd", False)),
    }
    missing = [name for name, satisfied in requirements.items() if not satisfied]
    if missing:
        raise SessionNativeGAERuntimeError("--session-native-gae requires: " + ", ".join(missing))


def prepare_train_data(
    args: Any,
    samples: Sequence[Any],
    train_data: dict[str, Any],
) -> None:
    """Validate action contracts and add optimizer-facing runtime fields.

    Actor ``loss_masks`` are never modified.  A second boundary-only mask is
    constructed for critic loss, and both masks travel through DP transport.
    """

    if not enabled(args):
        return
    if not samples:
        raise SessionNativeGAERuntimeError("session-native GAE cannot prepare an empty sample batch")

    metadata = train_data.get("metadata")
    if not _is_sequence(metadata) or len(metadata) != len(samples):
        raise SessionNativeGAERuntimeError(
            "session-native GAE requires one preserved train_metadata object per sample"
        )

    rollout_ids = train_data.get("rollout_ids")
    response_lengths = train_data.get("response_lengths")
    actor_masks = train_data.get("loss_masks")
    for name, values in (
        ("rollout_ids", rollout_ids),
        ("response_lengths", response_lengths),
        ("loss_masks", actor_masks),
    ):
        if not _is_sequence(values) or len(values) != len(samples):
            raise SessionNativeGAERuntimeError(f"session-native GAE requires aligned {name}")

    contracts: list[Mapping[str, Any]] = []
    for position, (metadata_item, rollout_id, response_length, actor_mask) in enumerate(
        zip(metadata, rollout_ids, response_lengths, actor_masks, strict=True)
    ):
        if not isinstance(metadata_item, Mapping):
            raise SessionNativeGAERuntimeError(f"session-native sample {position} train_metadata is not an object")
        contract = metadata_item.get(CONTRACT_KEY)
        if not isinstance(contract, Mapping):
            raise SessionNativeGAERuntimeError(f"session-native sample {position} has no {CONTRACT_KEY!r} contract")
        _validate_contract(
            contract,
            position=position,
            rollout_id=rollout_id,
            response_length=response_length,
            actor_mask=actor_mask,
        )
        contracts.append(contract)

    grouped: dict[int | str, list[Mapping[str, Any]]] = {}
    for contract in contracts:
        grouped.setdefault(contract["rollout_id"], []).append(contract)
    for rollout_id, trajectory in grouped.items():
        _validate_trajectory(rollout_id, trajectory)

    critic_masks: list[list[int]] = []
    critic_totals: dict[int | str, int] = {}
    for rollout_id, response_length in zip(rollout_ids, response_lengths, strict=True):
        length = _positive_int(response_length, "response_length")
        critic_masks.append([1] + [0] * (length - 1))
        critic_totals[rollout_id] = critic_totals.get(rollout_id, 0) + 1

    train_data[CRITIC_MASK_FIELD] = critic_masks
    train_data[CRITIC_DENOM_FIELD] = [critic_totals[rid] for rid in rollout_ids]
    train_data[RUNTIME_VERSION_FIELD] = RUNTIME_VERSION
    train_data[RUNTIME_CAPABILITIES_FIELD] = dict(RUNTIME_CAPABILITIES)


def copy_runtime_attestation(source: Mapping[str, Any], target: dict[str, Any]) -> None:
    """Copy batch-level attestation into one DP partition, or fail closed."""

    for field, expected in (
        (RUNTIME_VERSION_FIELD, RUNTIME_VERSION),
        (RUNTIME_CAPABILITIES_FIELD, RUNTIME_CAPABILITIES),
    ):
        value = source.get(field)
        if value != expected:
            raise SessionNativeGAERuntimeError(
                f"session-native GAE runtime attestation mismatch for {field}: {value!r}"
            )
        target[field] = dict(value) if isinstance(value, Mapping) else value


def validate_optimizer_payload(args: Any, rollout_data: Mapping[str, Any]) -> None:
    """Validate custom-function outputs before actor or critic optimization."""

    if not enabled(args):
        return
    require_attestation(rollout_data)

    response_lengths = rollout_data.get("response_lengths")
    actor_masks = rollout_data.get("loss_masks")
    critic_masks = rollout_data.get(CRITIC_MASK_FIELD)
    advantages = rollout_data.get("advantages")
    returns = rollout_data.get("returns")
    fields = {
        "response_lengths": response_lengths,
        "loss_masks": actor_masks,
        CRITIC_MASK_FIELD: critic_masks,
        "advantages": advantages,
        "returns": returns,
    }
    if not all(_is_sequence(value) for value in fields.values()):
        missing = [name for name, value in fields.items() if not _is_sequence(value)]
        raise SessionNativeGAERuntimeError("session-native optimizer payload is missing: " + ", ".join(missing))
    size = len(response_lengths)
    if size == 0 or any(len(value) != size for value in fields.values()):
        raise SessionNativeGAERuntimeError("session-native optimizer fields are empty or have different sample counts")
    critic_denoms = _flat_values(
        rollout_data.get(CRITIC_DENOM_FIELD),
        "session-native critic denominators",
    )
    if len(critic_denoms) != size or any(value <= 0 or not value.is_integer() for value in critic_denoms):
        raise SessionNativeGAERuntimeError(
            "session-native critic denominators must be one positive integer per sample"
        )

    for position, (length, actor_mask, critic_mask, advantage, return_) in enumerate(
        zip(
            response_lengths,
            actor_masks,
            critic_masks,
            advantages,
            returns,
            strict=True,
        )
    ):
        expected_length = _positive_int(length, f"sample {position} response_length")
        actor_values = _flat_values(actor_mask, f"sample {position} actor mask")
        critic_values = _flat_values(critic_mask, f"sample {position} critic mask")
        advantage_values = _flat_values(advantage, f"sample {position} advantages")
        return_values = _flat_values(return_, f"sample {position} returns")
        if any(
            len(values) != expected_length for values in (actor_values, critic_values, advantage_values, return_values)
        ):
            raise SessionNativeGAERuntimeError(
                f"session-native sample {position} token fields are not response-aligned"
            )
        if any(value not in {0.0, 1.0} for value in actor_values):
            raise SessionNativeGAERuntimeError(f"session-native sample {position} actor mask is not binary")
        if critic_values != [1.0] + [0.0] * (expected_length - 1):
            raise SessionNativeGAERuntimeError(f"session-native sample {position} critic mask is not boundary-only")
        if any(not math.isfinite(value) for value in advantage_values + return_values):
            raise SessionNativeGAERuntimeError(f"session-native sample {position} contains non-finite targets")

    if rollout_data.get(NORMALIZED_FLAG) is not True:
        raise SessionNativeGAERuntimeError(
            "session-native advantages were not normalized at action level before token broadcast"
        )
    validate_metrics(rollout_data.get(METRICS_FIELD))


def require_attestation(rollout_data: Mapping[str, Any]) -> None:
    if rollout_data.get(RUNTIME_VERSION_FIELD) != RUNTIME_VERSION:
        raise SessionNativeGAERuntimeError("session-native GAE runtime version is absent or unsupported")
    if rollout_data.get(RUNTIME_CAPABILITIES_FIELD) != RUNTIME_CAPABILITIES:
        raise SessionNativeGAERuntimeError("session-native GAE runtime capabilities are absent or incomplete")


def validate_metrics(metrics: Any) -> dict[str, float]:
    if not isinstance(metrics, Mapping) or set(metrics) != EXPECTED_METRICS:
        raise SessionNativeGAERuntimeError(
            "session-native GAE metrics are missing or do not match the seven-metric contract"
        )
    parsed: dict[str, float] = {}
    for name, value in metrics.items():
        if isinstance(value, bool) or not isinstance(value, Real):
            raise SessionNativeGAERuntimeError(f"session-native metric {name!r} is not numeric")
        parsed_value = float(value)
        if not math.isfinite(parsed_value):
            raise SessionNativeGAERuntimeError(f"session-native metric {name!r} is not finite")
        parsed[name] = parsed_value
    return parsed


def _validate_contract(
    contract: Mapping[str, Any],
    *,
    position: int,
    rollout_id: Any,
    response_length: Any,
    actor_mask: Any,
) -> None:
    if contract.get("schema") != CONTRACT_SCHEMA:
        raise SessionNativeGAERuntimeError(f"session-native sample {position} uses an unsupported contract schema")
    if contract.get("available") is not True:
        reason = contract.get("invalid_reason") or "unknown adapter failure"
        raise SessionNativeGAERuntimeError(f"session-native sample {position} contract is unavailable: {reason}")
    if contract.get("objective") != OBJECTIVE:
        raise SessionNativeGAERuntimeError("session-native GAE requires terminal_broadcast; cost_to_go is rejected")
    if contract.get("boundary_value_index") != BOUNDARY_VALUE_INDEX:
        raise SessionNativeGAERuntimeError(f"session-native sample {position} has no prompt-boundary value contract")
    if contract.get("rollout_id") != rollout_id:
        raise SessionNativeGAERuntimeError(f"session-native sample {position} rollout_id changed during conversion")
    expected_length = _positive_int(response_length, "response_length")
    if _positive_int(contract.get("response_token_count"), "response_token_count") != expected_length:
        raise SessionNativeGAERuntimeError(
            f"session-native sample {position} response length changed during conversion"
        )
    mask = _flat_values(actor_mask, f"sample {position} actor mask")
    if len(mask) != expected_length or any(value not in {0.0, 1.0} for value in mask):
        raise SessionNativeGAERuntimeError(
            f"session-native sample {position} actor mask is not response-aligned binary data"
        )
    trainable_count = sum(int(value) for value in mask)
    if trainable_count <= 0 or trainable_count != _positive_int(
        contract.get("trainable_token_count"), "trainable_token_count"
    ):
        raise SessionNativeGAERuntimeError(
            f"session-native sample {position} trainable-token count changed during conversion"
        )


def _validate_trajectory(
    rollout_id: int | str,
    trajectory: Sequence[Mapping[str, Any]],
) -> None:
    counts = {_positive_int(contract.get("action_count"), "action_count") for contract in trajectory}
    if len(counts) != 1 or next(iter(counts)) != len(trajectory):
        raise SessionNativeGAERuntimeError(f"session-native rollout {rollout_id!r} is missing action samples")
    ordered = sorted(
        trajectory,
        key=lambda contract: _nonnegative_int(contract.get("action_index"), "action_index"),
    )
    if [contract.get("action_index") for contract in ordered] != list(range(len(ordered))):
        raise SessionNativeGAERuntimeError(f"session-native rollout {rollout_id!r} action indices are not contiguous")
    if (
        len(ordered) != 2
        or ordered[0].get("action_type") != "ROUTE"
        or ordered[1].get("action_type") not in {"SUBMIT", "VERIFY"}
    ):
        raise SessionNativeGAERuntimeError(f"session-native rollout {rollout_id!r} is not ROUTE -> SUBMIT/VERIFY")
    terminal_rewards = {_finite_float(contract.get("terminal_reward"), "terminal_reward") for contract in ordered}
    if len(terminal_rewards) != 1:
        raise SessionNativeGAERuntimeError(f"session-native rollout {rollout_id!r} has inconsistent terminal rewards")
    terminal_reward = next(iter(terminal_rewards))
    action_rewards = [_finite_float(contract.get("action_reward"), "action_reward") for contract in ordered]
    if action_rewards != [0.0, terminal_reward]:
        raise SessionNativeGAERuntimeError(
            f"session-native rollout {rollout_id!r} does not contain one sparse terminal reward"
        )


def _flat_values(value: Any, label: str) -> list[float]:
    if hasattr(value, "detach") and callable(value.detach):
        value = value.detach()
    if hasattr(value, "cpu") and callable(value.cpu):
        value = value.cpu()
    if hasattr(value, "reshape") and callable(value.reshape):
        value = value.reshape(-1)
    if hasattr(value, "tolist") and callable(value.tolist):
        value = value.tolist()
    if not _is_sequence(value):
        raise SessionNativeGAERuntimeError(f"{label} is not a one-dimensional sequence")
    parsed: list[float] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, Real):
            raise SessionNativeGAERuntimeError(f"{label} contains a non-numeric value")
        parsed.append(float(item))
    return parsed


def _is_sequence(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray))


def _nonnegative_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise SessionNativeGAERuntimeError(f"{label} must be an integer")
    parsed = int(value)
    if parsed < 0:
        raise SessionNativeGAERuntimeError(f"{label} must be non-negative")
    return parsed


def _positive_int(value: Any, label: str) -> int:
    parsed = _nonnegative_int(value, label)
    if parsed == 0:
        raise SessionNativeGAERuntimeError(f"{label} must be positive")
    return parsed


def _finite_float(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise SessionNativeGAERuntimeError(f"{label} must be numeric")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise SessionNativeGAERuntimeError(f"{label} must be finite")
    return parsed
