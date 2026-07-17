"""Narrow compatibility identity for replaying an audited rollout WAL.

The partial-rollout WAL fingerprints ``TMAX_SLIME_GIT_COMMIT``.  A rollout
that completed under the legacy revision below is safe to replay with the
audited telemetry-only successor, but only the RolloutManager may use that
legacy identity.  Trainer actors continue to use the real runtime revision.

This escape hatch is deliberately specific and fail closed.  It is disabled
unless the caller requests the one audited legacy revision, and it verifies
the immutable runtime lineage and both allowlisted diffs before returning any
environment overrides.
"""

from __future__ import annotations

import hashlib
import logging
import os
import subprocess
from collections.abc import Mapping
from pathlib import Path

logger = logging.getLogger(__name__)

OPT_IN_ENV = "SLIME_ROLLOUT_MANAGER_WAL_COMPAT_LEGACY_GIT_COMMIT"
SLIME_GIT_COMMIT_ENV = "TMAX_SLIME_GIT_COMMIT"
SLIME_RUNTIME_GIT_COMMIT_ENV = "TMAX_SLIME_RUNTIME_GIT_COMMIT"

AUDITED_LEGACY_GIT_COMMIT = "77c6d526d740dc31f6bdd3177ac247b43ea1ce87"
AUDITED_RUNTIME_GIT_COMMIT = "372025f1200ad83c6e5d5c5610387a17c31ea581"
AUDITED_RUNTIME_PATCH_SHA256 = "b3f9e7cf007683b8c50c01685c270c07e20d472414fbed8f7e47bceacff434c9"

_AUDITED_RUNTIME_DIFF = frozenset(
    {
        ("M", "slime/backends/megatron_utils/actor.py"),
        ("M", "slime/backends/megatron_utils/data.py"),
        ("A", "tests/test_perf_logging_after_offload.py"),
    }
)
_COMPAT_CHILD_DIFF = frozenset(
    {
        ("M", "slime/ray/placement_group.py"),
        ("A", "slime/ray/rollout_wal_compat.py"),
        ("A", "tests/test_rollout_wal_compat.py"),
    }
)


class RolloutWalCompatibilityError(RuntimeError):
    """Raised when the requested compatibility identity cannot be audited."""


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _git_bytes(repo_root: Path, *args: str) -> bytes:
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_root), *args],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RolloutWalCompatibilityError(f"git audit failed for {args!r}: {exc}") from exc
    if result.returncode != 0:
        stderr = result.stderr.decode("utf-8", errors="replace").strip()
        raise RolloutWalCompatibilityError(
            f"git audit failed for {args!r} with exit {result.returncode}: {stderr}"
        )
    return result.stdout


def _git_text(repo_root: Path, *args: str) -> str:
    return _git_bytes(repo_root, *args).decode("utf-8", errors="strict").strip()


def _parse_name_status(value: str) -> frozenset[tuple[str, str]]:
    rows: set[tuple[str, str]] = set()
    for line in value.splitlines():
        fields = line.split("\t")
        if len(fields) != 2 or fields[0] not in {"A", "M", "D"}:
            raise RolloutWalCompatibilityError(f"unexpected git diff --name-status row: {line!r}")
        rows.add((fields[0], fields[1]))
    return frozenset(rows)


def _require_equal(label: str, actual: object, expected: object) -> None:
    if actual != expected:
        raise RolloutWalCompatibilityError(f"{label} mismatch: expected {expected!r}, found {actual!r}")


def rollout_manager_wal_compat_env(
    environ: Mapping[str, str] | None = None,
    repo_root: Path | None = None,
) -> dict[str, str]:
    """Return actor-local revision overrides after a strict source audit.

    An unset opt-in is a zero-cost no-op.  The returned mapping is intended
    only for ``RolloutManager.options(runtime_env=...)`` and must never be
    merged into the driver or trainer process environment.
    """

    env = os.environ if environ is None else environ
    legacy_revision = env.get(OPT_IN_ENV)
    if legacy_revision is None:
        return {}
    _require_equal(OPT_IN_ENV, legacy_revision, AUDITED_LEGACY_GIT_COMMIT)

    root = _repository_root() if repo_root is None else Path(repo_root)
    actual_revision = _git_text(root, "rev-parse", "--verify", "HEAD")

    pinned_revision = env.get(SLIME_GIT_COMMIT_ENV)
    if not pinned_revision:
        raise RolloutWalCompatibilityError(
            f"{SLIME_GIT_COMMIT_ENV} must pin the actual immutable runtime when {OPT_IN_ENV} is set"
        )
    _require_equal(SLIME_GIT_COMMIT_ENV, pinned_revision, actual_revision)

    existing_runtime_revision = env.get(SLIME_RUNTIME_GIT_COMMIT_ENV)
    if existing_runtime_revision:
        _require_equal(SLIME_RUNTIME_GIT_COMMIT_ENV, existing_runtime_revision, actual_revision)

    dirty = _git_text(root, "status", "--porcelain", "--untracked-files=all")
    if dirty:
        raise RolloutWalCompatibilityError(
            "compatibility identity requires a clean runtime worktree; "
            f"found: {dirty.splitlines()[0]!r}"
        )

    actual_parent = _git_text(root, "rev-parse", "--verify", "HEAD^")
    _require_equal("compatibility runtime parent", actual_parent, AUDITED_RUNTIME_GIT_COMMIT)

    audited_parent = _git_text(
        root,
        "rev-parse",
        "--verify",
        f"{AUDITED_RUNTIME_GIT_COMMIT}^",
    )
    _require_equal("audited telemetry runtime parent", audited_parent, AUDITED_LEGACY_GIT_COMMIT)

    audited_diff = _parse_name_status(
        _git_text(
            root,
            "diff",
            "--name-status",
            AUDITED_LEGACY_GIT_COMMIT,
            AUDITED_RUNTIME_GIT_COMMIT,
        )
    )
    _require_equal("audited telemetry runtime diff", audited_diff, _AUDITED_RUNTIME_DIFF)

    audited_patch = _git_bytes(
        root,
        "diff",
        "--binary",
        AUDITED_LEGACY_GIT_COMMIT,
        AUDITED_RUNTIME_GIT_COMMIT,
    )
    audited_patch_digest = hashlib.sha256(audited_patch).hexdigest()
    _require_equal(
        "audited telemetry runtime patch digest",
        audited_patch_digest,
        AUDITED_RUNTIME_PATCH_SHA256,
    )

    compat_diff = _parse_name_status(
        _git_text(
            root,
            "diff",
            "--name-status",
            AUDITED_RUNTIME_GIT_COMMIT,
            actual_revision,
        )
    )
    _require_equal("compatibility child diff", compat_diff, _COMPAT_CHILD_DIFF)

    logger.warning(
        "RolloutManager WAL compatibility identity enabled: legacy=%s actual_runtime=%s",
        legacy_revision,
        actual_revision,
    )
    return {
        SLIME_GIT_COMMIT_ENV: legacy_revision,
        SLIME_RUNTIME_GIT_COMMIT_ENV: actual_revision,
    }
