from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from slime.ray import placement_group, rollout
from slime.ray import rollout_wal_compat as compat


def _successful_git_outputs(actual_revision: str = "a" * 40) -> dict[tuple[str, ...], bytes]:
    audited_diff = (
        b"M\tslime/backends/megatron_utils/actor.py\n"
        b"M\tslime/backends/megatron_utils/data.py\n"
        b"A\ttests/test_perf_logging_after_offload.py\n"
    )
    child_diff = (
        b"M\tslime/ray/placement_group.py\n"
        b"A\tslime/ray/rollout_wal_compat.py\n"
        b"A\ttests/test_rollout_wal_compat.py\n"
    )
    # The production constant protects the real audited patch.  Unit tests
    # substitute an equivalent synthetic digest to exercise the decision path.
    audited_patch = b"audited telemetry patch"
    return {
        ("rev-parse", "--verify", "HEAD"): f"{actual_revision}\n".encode(),
        ("status", "--porcelain", "--untracked-files=all"): b"",
        ("rev-parse", "--verify", "HEAD^"): f"{compat.AUDITED_COMPAT_GIT_COMMIT}\n".encode(),
        (
            "rev-parse",
            "--verify",
            f"{compat.AUDITED_COMPAT_GIT_COMMIT}^",
        ): f"{compat.AUDITED_RUNTIME_GIT_COMMIT}\n".encode(),
        (
            "rev-parse",
            "--verify",
            f"{compat.AUDITED_RUNTIME_GIT_COMMIT}^",
        ): f"{compat.AUDITED_LEGACY_GIT_COMMIT}\n".encode(),
        (
            "diff",
            "--name-status",
            compat.AUDITED_LEGACY_GIT_COMMIT,
            compat.AUDITED_RUNTIME_GIT_COMMIT,
        ): audited_diff,
        (
            "diff",
            "--binary",
            compat.AUDITED_LEGACY_GIT_COMMIT,
            compat.AUDITED_RUNTIME_GIT_COMMIT,
        ): audited_patch,
        (
            "diff",
            "--name-status",
            compat.AUDITED_RUNTIME_GIT_COMMIT,
            actual_revision,
        ): child_diff,
        ("test-patch-digest",): hashlib.sha256(audited_patch).hexdigest().encode(),
    }


def _install_git_stub(monkeypatch, outputs: dict[tuple[str, ...], bytes]) -> None:
    def fake_git_bytes(_repo_root: Path, *args: str) -> bytes:
        return outputs[tuple(args)]

    monkeypatch.setattr(compat, "_git_bytes", fake_git_bytes)
    monkeypatch.setattr(
        compat,
        "AUDITED_RUNTIME_PATCH_SHA256",
        outputs[("test-patch-digest",)].decode(),
    )


@pytest.mark.unit
def test_compatibility_is_disabled_by_default_without_git_access(monkeypatch) -> None:
    monkeypatch.setattr(
        compat,
        "_git_bytes",
        lambda *_args: pytest.fail("disabled compatibility must not inspect git"),
    )

    assert compat.rollout_manager_wal_compat_env(environ={}) == {}


@pytest.mark.unit
def test_compatibility_returns_actor_local_legacy_and_actual_identities(monkeypatch) -> None:
    actual_revision = "a" * 40
    outputs = _successful_git_outputs(actual_revision)
    _install_git_stub(monkeypatch, outputs)
    env = {
        compat.OPT_IN_ENV: compat.AUDITED_LEGACY_GIT_COMMIT,
        compat.SLIME_GIT_COMMIT_ENV: actual_revision,
    }

    overrides = compat.rollout_manager_wal_compat_env(environ=env, repo_root=Path("/runtime"))

    assert overrides == {
        compat.SLIME_GIT_COMMIT_ENV: compat.AUDITED_LEGACY_GIT_COMMIT,
        compat.SLIME_RUNTIME_GIT_COMMIT_ENV: actual_revision,
    }
    assert env[compat.SLIME_GIT_COMMIT_ENV] == actual_revision
    assert compat.SLIME_RUNTIME_GIT_COMMIT_ENV not in env


@pytest.mark.unit
def test_serialized_opt_in_returns_actor_local_overrides(monkeypatch) -> None:
    actual_revision = "a" * 40
    outputs = _successful_git_outputs(actual_revision)
    _install_git_stub(monkeypatch, outputs)
    env = {
        compat.SERIALIZED_OPT_IN_ENV: compat.AUDITED_LEGACY_GIT_COMMIT,
        compat.SLIME_GIT_COMMIT_ENV: actual_revision,
    }

    assert compat.rollout_manager_wal_compat_env(
        environ=env,
        repo_root=Path("/runtime"),
    ) == {
        compat.SLIME_GIT_COMMIT_ENV: compat.AUDITED_LEGACY_GIT_COMMIT,
        compat.SLIME_RUNTIME_GIT_COMMIT_ENV: actual_revision,
    }


@pytest.mark.unit
def test_conflicting_direct_and_serialized_opt_ins_fail_closed() -> None:
    env = {
        compat.OPT_IN_ENV: compat.AUDITED_LEGACY_GIT_COMMIT,
        compat.SERIALIZED_OPT_IN_ENV: "b" * 40,
    }

    with pytest.raises(compat.RolloutWalCompatibilityError, match="conflicting"):
        compat.rollout_manager_wal_compat_env(environ=env, repo_root=Path("/runtime"))


@pytest.mark.unit
@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda env, outputs: env.update({compat.SLIME_GIT_COMMIT_ENV: "b" * 40}),
            compat.SLIME_GIT_COMMIT_ENV,
        ),
        (
            lambda env, outputs: outputs.__setitem__(
                ("rev-parse", "--verify", "HEAD^"), b"c" * 40 + b"\n"
            ),
            "compatibility runtime parent",
        ),
        (
            lambda env, outputs: outputs.__setitem__(
                (
                    "diff",
                    "--name-status",
                    compat.AUDITED_RUNTIME_GIT_COMMIT,
                    "a" * 40,
                ),
                b"M\tslime/ray/placement_group.py\nM\tslime/train.py\n",
            ),
            "compatibility child diff",
        ),
    ],
)
def test_compatibility_fails_closed_on_runtime_mismatch(monkeypatch, mutate, message: str) -> None:
    actual_revision = "a" * 40
    outputs = _successful_git_outputs(actual_revision)
    env = {
        compat.OPT_IN_ENV: compat.AUDITED_LEGACY_GIT_COMMIT,
        compat.SLIME_GIT_COMMIT_ENV: actual_revision,
    }
    mutate(env, outputs)
    _install_git_stub(monkeypatch, outputs)

    with pytest.raises(compat.RolloutWalCompatibilityError, match=message):
        compat.rollout_manager_wal_compat_env(environ=env, repo_root=Path("/runtime"))


class _RemoteMethod:
    def __init__(self, func):
        self._func = func

    def remote(self, *args, **kwargs):
        return self._func(*args, **kwargs)


@pytest.mark.unit
def test_create_rollout_manager_scopes_overrides_to_actor_runtime(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class _Actor:
        ready = _RemoteMethod(lambda: {})

    class _FakeRolloutManager:
        @classmethod
        def options(cls, **kwargs):
            captured.update(kwargs)
            return cls

        @classmethod
        def remote(cls, _args, _pg):
            return _Actor()

    actual = "a" * 40
    driver_env = {
        compat.SLIME_GIT_COMMIT_ENV: actual,
        compat.OPT_IN_ENV: compat.AUDITED_LEGACY_GIT_COMMIT,
    }
    expected_overrides = {
        compat.SLIME_GIT_COMMIT_ENV: compat.AUDITED_LEGACY_GIT_COMMIT,
        compat.SLIME_RUNTIME_GIT_COMMIT_ENV: actual,
    }
    monkeypatch.setattr(rollout, "RolloutManager", _FakeRolloutManager)
    monkeypatch.setattr(placement_group.ray, "get", lambda value: value)
    monkeypatch.setattr(
        placement_group,
        "rollout_manager_wal_compat_env",
        lambda: expected_overrides,
    )
    args = SimpleNamespace(
        num_rollout=1,
        num_epoch=1,
        check_weight_update_equal=False,
        offload_rollout=False,
        rollout_data_transport="object-store",
    )

    placement_group.create_rollout_manager(args, object())

    assert captured["runtime_env"] == {
        "env_vars": {"RAY_USE_UVLOOP": "0", **expected_overrides}
    }
    assert driver_env == {
        compat.SLIME_GIT_COMMIT_ENV: actual,
        compat.OPT_IN_ENV: compat.AUDITED_LEGACY_GIT_COMMIT,
    }
