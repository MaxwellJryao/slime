"""Reliably reconcile the terminal state of a shared W&B run.

The normal Slime driver is the only shared writer allowed to update a run's
terminal state.  An allocation-level failure can kill that driver before its
``run.finish()`` reaches W&B, though.  This module provides the small,
standalone reconciler used by the rank-zero allocation wrapper after every
other shared writer has stopped.

Each network attempt runs in a fresh Python process and therefore starts a
fresh ``wandb-core`` service.  This is important: retrying ``Run.finish`` on
the same object is ineffective after W&B marks that local object finished,
even when its connection to the service was reset before the server update.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import re
import subprocess
import sys
import time
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA_VERSION = 1
SOURCE = "slime-shared-wandb-terminal-reconciler-v1"
TRANSIENT_EXIT_CODE = 75
CONFLICT_EXIT_CODE = 78
_SAFE_COMPONENT_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,191}")
_DESIRED_STATES = {0: "finished"}
# W&B documents crashed runs as resumable.  Killed runs are deliberately not
# rewritten automatically because they can encode an explicit user action.
_MUTABLE_CLOUD_STATES = frozenset({"running", "crashed"})
_TERMINAL_CLOUD_STATES = frozenset({"finished", "failed", "killed"})
_STALE_OR_CONFLICTING_ENV_VARS = frozenset(
    {
        "WANDB_DISABLED",
        "WANDB_ENTITY",
        "WANDB_GROUP",
        "WANDB_JOB_TYPE",
        "WANDB_NAME",
        "WANDB_PROJECT",
        "WANDB_RESUME",
        "WANDB_RUN_ID",
        "WANDB_SERVICE",
        "WANDB_SWEEP_ID",
        "WANDB_TAGS",
    }
)


class TerminalStateError(RuntimeError):
    """Base class for expected reconciliation failures."""


class TerminalStateConflict(TerminalStateError):
    """The durable intent or cloud state conflicts with the requested state."""


class TerminalStateTransientError(TerminalStateError):
    """The attempt failed without proving an incompatible terminal state."""


@dataclass(frozen=True)
class TerminalSpec:
    entity: str
    project: str
    run_id: str
    exit_code: int
    receipt: Path
    wandb_dir: Path
    origin_job_id: str | None = None

    @property
    def desired_state(self) -> str:
        return desired_cloud_state(self.exit_code)

    @property
    def run_path(self) -> str:
        return f"{self.entity}/{self.project}/{self.run_id}"


def desired_cloud_state(exit_code: int) -> str:
    """Map a process exit status to the state documented by W&B."""

    return _DESIRED_STATES.get(exit_code, "failed")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def validate_spec(spec: TerminalSpec) -> None:
    for name, value in (
        ("entity", spec.entity),
        ("project", spec.project),
        ("run_id", spec.run_id),
    ):
        if _SAFE_COMPONENT_RE.fullmatch(value) is None:
            raise ValueError(f"{name} must be a safe W&B path component")
    if not 0 <= spec.exit_code <= 255:
        raise ValueError("exit_code must be between 0 and 255")
    if not spec.receipt.is_absolute():
        raise ValueError("receipt must be an absolute path")
    if not spec.wandb_dir.is_absolute():
        raise ValueError("wandb_dir must be an absolute path")
    if spec.origin_job_id is not None and _SAFE_COMPONENT_RE.fullmatch(spec.origin_job_id) is None:
        raise ValueError("origin_job_id must be a safe identifier")


def _intent_identity(spec: TerminalSpec) -> dict[str, Any]:
    return {
        "entity": spec.entity,
        "project": spec.project,
        "run_id": spec.run_id,
        "exit_code": spec.exit_code,
        "desired_state": spec.desired_state,
        "source": SOURCE,
    }


def _intent_sha256(identity: Mapping[str, Any]) -> str:
    encoded = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _new_receipt(spec: TerminalSpec) -> dict[str, Any]:
    identity = _intent_identity(spec)
    return {
        "schema_version": SCHEMA_VERSION,
        "intent": {
            **identity,
            "created_at": utc_now(),
            "origin_job_id": spec.origin_job_id,
        },
        "intent_sha256": _intent_sha256(identity),
        "status": "pending",
        "attempts": [],
    }


def _load_receipt(path: Path) -> dict[str, Any]:
    try:
        stat_result = path.stat()
        if stat_result.st_mode & 0o077:
            raise TerminalStateConflict(f"receipt permissions must be private: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
    except TerminalStateConflict:
        raise
    except (OSError, json.JSONDecodeError) as exc:
        raise TerminalStateConflict(f"cannot read durable terminal receipt: {path}") from exc
    if not isinstance(payload, dict):
        raise TerminalStateConflict("durable terminal receipt must contain a JSON object")
    return payload


def _validate_existing_receipt(payload: Mapping[str, Any], spec: TerminalSpec) -> None:
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise TerminalStateConflict("unsupported durable terminal receipt schema")
    identity = _intent_identity(spec)
    intent = payload.get("intent")
    if not isinstance(intent, dict):
        raise TerminalStateConflict("durable terminal receipt has no intent")
    observed_identity = {key: intent.get(key) for key in identity}
    if observed_identity != identity or payload.get("intent_sha256") != _intent_sha256(identity):
        raise TerminalStateConflict("durable terminal receipt conflicts with requested run state")


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, sort_keys=True, separators=(",", ":"))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


class _ReceiptLock:
    def __init__(self, receipt: Path, timeout_seconds: float):
        self.path = receipt.with_name(f"{receipt.name}.lock")
        self.timeout_seconds = timeout_seconds
        self._descriptor: int | None = None

    def __enter__(self) -> _ReceiptLock:
        self.path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        self._descriptor = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o600)
        deadline = time.monotonic() + self.timeout_seconds
        while True:
            try:
                fcntl.flock(self._descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                return self
            except BlockingIOError as exc:
                if time.monotonic() >= deadline:
                    raise TerminalStateTransientError("timed out acquiring terminal receipt lock") from exc
                time.sleep(min(0.1, max(0.0, deadline - time.monotonic())))

    def __exit__(self, _exc_type: Any, _exc: Any, _traceback: Any) -> None:
        if self._descriptor is None:
            return
        fcntl.flock(self._descriptor, fcntl.LOCK_UN)
        os.close(self._descriptor)
        self._descriptor = None


def _normalize_cloud_state(value: Any) -> str:
    state = str(value).strip().lower()
    if not state:
        raise TerminalStateTransientError("W&B returned an empty run state")
    return state


def _query_cloud_state(wandb_module: Any, spec: TerminalSpec, *, api_timeout_seconds: int) -> str:
    api = wandb_module.Api(timeout=api_timeout_seconds)
    return _normalize_cloud_state(api.run(spec.run_path).state)


def finish_and_verify_once(
    spec: TerminalSpec,
    *,
    wandb_module: Any,
    api_timeout_seconds: int,
    finish_timeout_seconds: float,
    verify_timeout_seconds: float,
    poll_interval_seconds: float,
    monotonic: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    """Run one W&B reconciliation attempt using an already-fresh SDK service."""

    pre_state = _query_cloud_state(wandb_module, spec, api_timeout_seconds=api_timeout_seconds)
    if pre_state == spec.desired_state:
        return {
            "outcome": "verified",
            "pre_state": pre_state,
            "observed_state": pre_state,
            "finish_called": False,
        }
    if pre_state in _TERMINAL_CLOUD_STATES:
        raise TerminalStateConflict(
            f"W&B run is already terminal as {pre_state}, expected {spec.desired_state}"
        )
    if pre_state not in _MUTABLE_CLOUD_STATES:
        raise TerminalStateTransientError(f"W&B run state {pre_state!r} is not safe to reconcile")

    settings = wandb_module.Settings(
        mode="shared",
        x_primary=True,
        x_update_finish_state=True,
        x_label="terminal-reconciler",
        x_disable_stats=True,
        x_disable_meta=True,
        x_server_side_derived_summary=True,
        console="off",
        quiet=True,
        disable_git=True,
        disable_code=True,
        disable_job_creation=True,
        save_code=False,
        finish_timeout=finish_timeout_seconds,
        finish_timeout_raises=True,
    )
    run = None
    finish_error: BaseException | None = None
    try:
        run = wandb_module.init(
            entity=spec.entity,
            project=spec.project,
            id=spec.run_id,
            resume="must",
            force=True,
            dir=str(spec.wandb_dir),
            settings=settings,
        )
        if str(run.id) != spec.run_id:
            raise TerminalStateConflict("W&B resumed an unexpected run ID")
        run.finish(exit_code=spec.exit_code)
    except TerminalStateConflict:
        raise
    except BaseException as exc:  # W&B can surface transport resets outside Exception.
        finish_error = exc

    deadline = monotonic() + verify_timeout_seconds
    observed_state = pre_state
    while True:
        try:
            observed_state = _query_cloud_state(
                wandb_module,
                spec,
                api_timeout_seconds=api_timeout_seconds,
            )
        except Exception:
            observed_state = "unavailable"
        if observed_state == spec.desired_state:
            return {
                "outcome": "verified",
                "pre_state": pre_state,
                "observed_state": observed_state,
                "finish_called": run is not None,
                "finish_error_type": type(finish_error).__name__ if finish_error else None,
            }
        if observed_state in _TERMINAL_CLOUD_STATES:
            raise TerminalStateConflict(
                f"W&B run became terminal as {observed_state}, expected {spec.desired_state}"
            )
        remaining = deadline - monotonic()
        if remaining <= 0:
            error_type = type(finish_error).__name__ if finish_error else "VerificationTimeout"
            raise TerminalStateTransientError(
                f"W&B terminal state was not verified after finish ({error_type})"
            )
        sleep(min(poll_interval_seconds, remaining))


def sanitized_attempt_environment(environment: Mapping[str, str]) -> dict[str, str]:
    """Return an online W&B environment detached from any dead shared service."""

    sanitized = dict(environment)
    for name in _STALE_OR_CONFLICTING_ENV_VARS:
        sanitized.pop(name, None)
    sanitized["WANDB_MODE"] = "shared"
    sanitized["WANDB_SILENT"] = "true"
    sanitized["WANDB_CONSOLE"] = "off"
    return sanitized


def _child_result(
    *,
    outcome: str,
    error_type: str | None = None,
    **values: Any,
) -> dict[str, Any]:
    return {
        "outcome": outcome,
        "error_type": error_type,
        **values,
    }


def run_internal_attempt(args: argparse.Namespace) -> int:
    spec = _spec_from_args(args)
    result_path = Path(args.internal_result)
    started_at = utc_now()
    try:
        import wandb

        result = finish_and_verify_once(
            spec,
            wandb_module=wandb,
            api_timeout_seconds=args.api_timeout_seconds,
            finish_timeout_seconds=args.finish_timeout_seconds,
            verify_timeout_seconds=args.verify_timeout_seconds,
            poll_interval_seconds=args.poll_interval_seconds,
        )
        result = _child_result(started_at=started_at, completed_at=utc_now(), **result)
        return_code = 0
    except TerminalStateConflict as exc:
        result = _child_result(
            outcome="conflict",
            error_type=type(exc).__name__,
            started_at=started_at,
            completed_at=utc_now(),
        )
        return_code = CONFLICT_EXIT_CODE
    except BaseException as exc:
        result = _child_result(
            outcome="transient_failure",
            error_type=type(exc).__name__,
            started_at=started_at,
            completed_at=utc_now(),
        )
        return_code = TRANSIENT_EXIT_CODE
    _atomic_write_json(result_path, result)
    return return_code


def _attempt_command(args: argparse.Namespace, result_path: Path) -> list[str]:
    command = [
        sys.executable,
        str(Path(__file__).resolve().parents[2] / "scripts" / "reconcile_wandb_terminal_state.py"),
        "--entity",
        args.entity,
        "--project",
        args.project,
        "--run-id",
        args.run_id,
        "--exit-code",
        str(args.exit_code),
        "--receipt",
        args.receipt,
        "--wandb-dir",
        args.wandb_dir,
        "--api-timeout-seconds",
        str(args.api_timeout_seconds),
        "--finish-timeout-seconds",
        str(args.finish_timeout_seconds),
        "--verify-timeout-seconds",
        str(args.verify_timeout_seconds),
        "--poll-interval-seconds",
        str(args.poll_interval_seconds),
        "--internal-result",
        str(result_path),
    ]
    if args.origin_job_id:
        command.extend(("--origin-job-id", args.origin_job_id))
    return command


def _run_fresh_attempt(
    args: argparse.Namespace,
    result_path: Path,
    *,
    run_subprocess: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> dict[str, Any]:
    started_at = utc_now()
    try:
        completed = run_subprocess(
            _attempt_command(args, result_path),
            env=sanitized_attempt_environment(os.environ),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=args.attempt_timeout_seconds,
            check=False,
            start_new_session=True,
        )
    except subprocess.TimeoutExpired:
        return _child_result(
            outcome="transient_failure",
            error_type="AttemptTimeout",
            started_at=started_at,
            completed_at=utc_now(),
            timed_out=True,
        )
    except OSError as exc:
        return _child_result(
            outcome="transient_failure",
            error_type=type(exc).__name__,
            started_at=started_at,
            completed_at=utc_now(),
        )
    try:
        result = _load_receipt(result_path)
    except TerminalStateConflict:
        result = _child_result(
            outcome="transient_failure",
            error_type="MissingAttemptReceipt",
            started_at=started_at,
            completed_at=utc_now(),
        )
    finally:
        try:
            result_path.unlink()
        except FileNotFoundError:
            pass
    result["child_returncode"] = completed.returncode
    if completed.returncode == CONFLICT_EXIT_CODE:
        result["outcome"] = "conflict"
    elif completed.returncode != 0 and result.get("outcome") == "verified":
        result["outcome"] = "transient_failure"
        result["error_type"] = "ChildExitMismatch"
    return result


def reconcile(
    args: argparse.Namespace,
    *,
    attempt_runner: Callable[[argparse.Namespace, Path], dict[str, Any]] = _run_fresh_attempt,
    sleep: Callable[[float], None] = time.sleep,
) -> int:
    spec = _spec_from_args(args)
    validate_spec(spec)
    with _ReceiptLock(spec.receipt, args.lock_timeout_seconds):
        if spec.receipt.exists():
            receipt = _load_receipt(spec.receipt)
            _validate_existing_receipt(receipt, spec)
            if receipt.get("status") == "verified":
                if receipt.get("observed_state") != spec.desired_state:
                    raise TerminalStateConflict("verified receipt has an inconsistent cloud state")
                return 0
            if receipt.get("status") == "conflict":
                return CONFLICT_EXIT_CODE
        else:
            receipt = _new_receipt(spec)
            _atomic_write_json(spec.receipt, receipt)

        attempt_root = spec.wandb_dir / "terminal-reconciler"
        attempt_root.mkdir(mode=0o700, parents=True, exist_ok=True)
        for local_attempt in range(1, args.attempts + 1):
            result_path = attempt_root / f"attempt-{os.getpid()}-{uuid.uuid4().hex}.json"
            result = attempt_runner(args, result_path)
            attempt_number = len(receipt["attempts"]) + 1
            result["attempt"] = attempt_number
            receipt["attempts"].append(result)
            receipt["last_attempt_at"] = result.get("completed_at", utc_now())
            if result.get("outcome") == "verified":
                observed_state = _normalize_cloud_state(result.get("observed_state"))
                if observed_state != spec.desired_state:
                    raise TerminalStateConflict("attempt claimed verification for the wrong cloud state")
                receipt.update(
                    status="verified",
                    observed_state=observed_state,
                    verified_at=utc_now(),
                )
                _atomic_write_json(spec.receipt, receipt)
                return 0
            if result.get("outcome") == "conflict":
                receipt.update(status="conflict", conflict_at=utc_now())
                _atomic_write_json(spec.receipt, receipt)
                return CONFLICT_EXIT_CODE
            receipt["status"] = "retryable"
            _atomic_write_json(spec.receipt, receipt)
            if local_attempt < args.attempts:
                sleep(args.retry_delay_seconds * local_attempt)
        return TRANSIENT_EXIT_CODE


def _spec_from_args(args: argparse.Namespace) -> TerminalSpec:
    return TerminalSpec(
        entity=args.entity,
        project=args.project,
        run_id=args.run_id,
        exit_code=args.exit_code,
        receipt=Path(args.receipt),
        wandb_dir=Path(args.wandb_dir),
        origin_job_id=args.origin_job_id or None,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Idempotently finish and verify a shared W&B run using a fresh primary writer.",
    )
    parser.add_argument("--entity", required=True)
    parser.add_argument("--project", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--exit-code", required=True, type=int)
    parser.add_argument("--receipt", required=True)
    parser.add_argument("--wandb-dir", required=True)
    parser.add_argument("--origin-job-id", default=os.environ.get("SLURM_JOB_ID", ""))
    parser.add_argument("--attempts", type=int, default=3)
    parser.add_argument("--attempt-timeout-seconds", type=float, default=45.0)
    parser.add_argument("--api-timeout-seconds", type=int, default=10)
    parser.add_argument("--finish-timeout-seconds", type=float, default=20.0)
    parser.add_argument("--verify-timeout-seconds", type=float, default=15.0)
    parser.add_argument("--poll-interval-seconds", type=float, default=1.0)
    parser.add_argument("--retry-delay-seconds", type=float, default=2.0)
    parser.add_argument("--lock-timeout-seconds", type=float, default=5.0)
    parser.add_argument("--internal-result", default="", help=argparse.SUPPRESS)
    return parser


def _validate_runtime_args(args: argparse.Namespace) -> None:
    for name in (
        "attempt_timeout_seconds",
        "finish_timeout_seconds",
        "verify_timeout_seconds",
        "poll_interval_seconds",
        "lock_timeout_seconds",
    ):
        if getattr(args, name) <= 0:
            raise ValueError(f"{name} must be greater than zero")
    if args.retry_delay_seconds < 0:
        raise ValueError("retry_delay_seconds must not be negative")
    if args.api_timeout_seconds <= 0:
        raise ValueError("api_timeout_seconds must be greater than zero")
    if args.attempts <= 0:
        raise ValueError("attempts must be greater than zero")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        _validate_runtime_args(args)
        validate_spec(_spec_from_args(args))
        if args.internal_result:
            result_path = Path(args.internal_result)
            if not result_path.is_absolute():
                raise ValueError("internal_result must be an absolute path")
            return run_internal_attempt(args)
        return reconcile(args)
    except TerminalStateConflict as exc:
        print(f"W&B terminal-state conflict: {exc}", file=sys.stderr)
        return CONFLICT_EXIT_CODE
    except (OSError, TerminalStateTransientError, ValueError) as exc:
        print(f"W&B terminal-state reconciliation failed: {exc}", file=sys.stderr)
        return TRANSIENT_EXIT_CODE


if __name__ == "__main__":
    raise SystemExit(main())
