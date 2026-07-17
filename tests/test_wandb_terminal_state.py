from __future__ import annotations

import argparse
import json
import stat
from pathlib import Path

import pytest

from slime.utils import wandb_terminal_state as terminal


def _args(tmp_path: Path, *, exit_code: int = 1, attempts: int = 3) -> argparse.Namespace:
    return argparse.Namespace(
        entity="hwinf_dcm",
        project="SPilot",
        run_id="stable-run-id",
        exit_code=exit_code,
        receipt=str(tmp_path / "terminal.json"),
        wandb_dir=str(tmp_path / "wandb"),
        origin_job_id="12345",
        attempts=attempts,
        attempt_timeout_seconds=5.0,
        api_timeout_seconds=2,
        finish_timeout_seconds=3.0,
        verify_timeout_seconds=4.0,
        poll_interval_seconds=0.1,
        retry_delay_seconds=0.0,
        lock_timeout_seconds=1.0,
        internal_result="",
    )


def _attempt(
    *,
    outcome: str,
    observed_state: str | None = None,
    error_type: str | None = None,
) -> dict:
    result = {
        "outcome": outcome,
        "error_type": error_type,
        "started_at": "2026-07-17T00:00:00Z",
        "completed_at": "2026-07-17T00:00:01Z",
    }
    if observed_state is not None:
        result["observed_state"] = observed_state
    return result


@pytest.mark.unit
def test_reconcile_retries_transient_failure_and_writes_private_verified_receipt(tmp_path):
    args = _args(tmp_path)
    results = iter(
        [
            _attempt(outcome="transient_failure", error_type="ConnectionResetError"),
            _attempt(outcome="verified", observed_state="failed"),
        ]
    )

    assert terminal.reconcile(args, attempt_runner=lambda _args, _path: next(results)) == 0

    receipt_path = Path(args.receipt)
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert stat.S_IMODE(receipt_path.stat().st_mode) == 0o600
    assert receipt["status"] == "verified"
    assert receipt["observed_state"] == "failed"
    assert [attempt["outcome"] for attempt in receipt["attempts"]] == [
        "transient_failure",
        "verified",
    ]
    assert receipt["intent"]["exit_code"] == 1
    assert len(receipt["intent_sha256"]) == 64


@pytest.mark.unit
def test_verified_receipt_is_idempotent_without_another_cloud_attempt(tmp_path):
    args = _args(tmp_path)
    calls = []

    def first(_args, _path):
        return _attempt(outcome="verified", observed_state="failed")

    assert terminal.reconcile(args, attempt_runner=first) == 0

    assert terminal.reconcile(
        args,
        attempt_runner=lambda *_values: calls.append(_values),
    ) == 0
    assert calls == []


@pytest.mark.unit
def test_existing_receipt_rejects_a_different_exit_code_without_cloud_mutation(tmp_path):
    failed_args = _args(tmp_path, exit_code=1)
    assert terminal.reconcile(
        failed_args,
        attempt_runner=lambda _args, _path: _attempt(
            outcome="verified", observed_state="failed"
        ),
    ) == 0

    success_args = _args(tmp_path, exit_code=0)
    with pytest.raises(terminal.TerminalStateConflict, match="conflicts"):
        terminal.reconcile(
            success_args,
            attempt_runner=lambda *_values: pytest.fail("must not contact W&B"),
        )


@pytest.mark.unit
def test_cloud_terminal_conflict_is_durable_and_fail_closed(tmp_path):
    args = _args(tmp_path)

    assert terminal.reconcile(
        args,
        attempt_runner=lambda _args, _path: _attempt(outcome="conflict"),
    ) == terminal.CONFLICT_EXIT_CODE
    receipt = json.loads(Path(args.receipt).read_text(encoding="utf-8"))
    assert receipt["status"] == "conflict"

    assert terminal.reconcile(
        args,
        attempt_runner=lambda *_values: pytest.fail("conflicts are not auto-overwritten"),
    ) == terminal.CONFLICT_EXIT_CODE


class _FakePublicRun:
    def __init__(self, state: str):
        self.state = state


class _FakeApi:
    def __init__(self, module):
        self.module = module

    def run(self, path):
        self.module.paths.append(path)
        return _FakePublicRun(next(self.module.states))


class _FakeRun:
    def __init__(self, module, run_id):
        self.module = module
        self.id = run_id

    def finish(self, *, exit_code):
        self.module.finished.append(exit_code)
        if self.module.finish_error is not None:
            raise self.module.finish_error


class _FakeWandb:
    def __init__(self, states, *, finish_error=None):
        self.states = iter(states)
        self.finish_error = finish_error
        self.paths = []
        self.settings = []
        self.initialized = []
        self.finished = []

    def Api(self, *, timeout):
        assert timeout == 2
        return _FakeApi(self)

    def Settings(self, **kwargs):
        self.settings.append(kwargs)
        return kwargs

    def init(self, **kwargs):
        self.initialized.append(kwargs)
        return _FakeRun(self, kwargs["id"])


@pytest.mark.unit
def test_fresh_primary_resumes_must_finishes_and_verifies_exact_state(tmp_path):
    spec = terminal._spec_from_args(_args(tmp_path))
    wandb = _FakeWandb(["running", "running", "failed"])
    monotonic_values = iter([0.0, 0.0, 0.1])

    result = terminal.finish_and_verify_once(
        spec,
        wandb_module=wandb,
        api_timeout_seconds=2,
        finish_timeout_seconds=3.0,
        verify_timeout_seconds=4.0,
        poll_interval_seconds=0.1,
        monotonic=lambda: next(monotonic_values),
        sleep=lambda _seconds: None,
    )

    assert result["observed_state"] == "failed"
    assert wandb.finished == [1]
    assert wandb.initialized[0]["resume"] == "must"
    assert wandb.initialized[0]["force"] is True
    assert wandb.settings[0]["mode"] == "shared"
    assert wandb.settings[0]["x_primary"] is True
    assert wandb.settings[0]["x_update_finish_state"] is True
    assert wandb.settings[0]["x_label"] == "terminal-reconciler"
    assert wandb.settings[0]["x_server_side_derived_summary"] is True
    assert wandb.settings[0]["disable_job_creation"] is True
    assert wandb.settings[0]["save_code"] is False
    assert wandb.settings[0]["finish_timeout_raises"] is True
    assert "config" not in wandb.initialized[0]
    assert "name" not in wandb.initialized[0]
    assert "group" not in wandb.initialized[0]
    assert wandb.paths == [spec.run_path, spec.run_path, spec.run_path]


@pytest.mark.unit
def test_already_desired_cloud_state_skips_resume(tmp_path):
    spec = terminal._spec_from_args(_args(tmp_path))
    wandb = _FakeWandb(["failed"])

    result = terminal.finish_and_verify_once(
        spec,
        wandb_module=wandb,
        api_timeout_seconds=2,
        finish_timeout_seconds=3.0,
        verify_timeout_seconds=4.0,
        poll_interval_seconds=0.1,
    )

    assert result["finish_called"] is False
    assert wandb.initialized == []


@pytest.mark.unit
def test_finish_connection_reset_is_accepted_only_after_cloud_verification(tmp_path):
    spec = terminal._spec_from_args(_args(tmp_path))
    wandb = _FakeWandb(
        ["running", "failed"],
        finish_error=ConnectionResetError("connection lost"),
    )

    result = terminal.finish_and_verify_once(
        spec,
        wandb_module=wandb,
        api_timeout_seconds=2,
        finish_timeout_seconds=3.0,
        verify_timeout_seconds=4.0,
        poll_interval_seconds=0.1,
    )

    assert result["observed_state"] == "failed"
    assert result["finish_error_type"] == "ConnectionResetError"


@pytest.mark.unit
def test_crashed_successful_allocation_is_resumed_and_finished(tmp_path):
    spec = terminal._spec_from_args(_args(tmp_path, exit_code=0))
    wandb = _FakeWandb(["crashed", "finished"])

    result = terminal.finish_and_verify_once(
        spec,
        wandb_module=wandb,
        api_timeout_seconds=2,
        finish_timeout_seconds=3.0,
        verify_timeout_seconds=4.0,
        poll_interval_seconds=0.1,
    )

    assert result["pre_state"] == "crashed"
    assert result["observed_state"] == "finished"
    assert wandb.finished == [0]


@pytest.mark.unit
def test_opposite_terminal_cloud_state_is_never_overwritten(tmp_path):
    spec = terminal._spec_from_args(_args(tmp_path))
    wandb = _FakeWandb(["finished"])

    with pytest.raises(terminal.TerminalStateConflict, match="already terminal"):
        terminal.finish_and_verify_once(
            spec,
            wandb_module=wandb,
            api_timeout_seconds=2,
            finish_timeout_seconds=3.0,
            verify_timeout_seconds=4.0,
            poll_interval_seconds=0.1,
        )
    assert wandb.initialized == []


@pytest.mark.unit
def test_broken_shared_service_is_removed_from_fresh_attempt_environment(monkeypatch):
    monkeypatch.setenv("WANDB_SERVICE", "dead-service-token")
    monkeypatch.setenv("WANDB_RUN_ID", "wrong-run")
    monkeypatch.setenv("WANDB_RESUME", "never")
    monkeypatch.setenv("WANDB_MODE", "offline")
    monkeypatch.setenv("WANDB_API_KEY", "preserved-credential")

    environment = terminal.sanitized_attempt_environment(dict(terminal.os.environ))

    assert "WANDB_SERVICE" not in environment
    assert "WANDB_RUN_ID" not in environment
    assert "WANDB_RESUME" not in environment
    assert environment["WANDB_MODE"] == "shared"
    assert environment["WANDB_API_KEY"] == "preserved-credential"


@pytest.mark.unit
@pytest.mark.parametrize(
    "field,value",
    [
        ("run_id", "bad/run"),
        ("entity", ""),
        ("exit_code", 256),
        ("receipt", "relative.json"),
    ],
)
def test_invalid_terminal_identity_fails_before_network(tmp_path, field, value):
    args = _args(tmp_path)
    setattr(args, field, value)

    with pytest.raises(ValueError):
        terminal.validate_spec(terminal._spec_from_args(args))


@pytest.mark.unit
def test_attempt_command_uses_the_same_pinned_repository_cli(tmp_path):
    args = _args(tmp_path)
    command = terminal._attempt_command(args, tmp_path / "attempt.json")

    assert Path(command[1]).resolve() == (
        Path(terminal.__file__).resolve().parents[2]
        / "scripts"
        / "reconcile_wandb_terminal_state.py"
    )
    assert "--internal-result" in command


@pytest.mark.unit
def test_success_exit_code_requires_finished_cloud_state(tmp_path):
    args = _args(tmp_path, exit_code=0)

    assert terminal.reconcile(
        args,
        attempt_runner=lambda _args, _path: _attempt(
            outcome="verified", observed_state="finished"
        ),
    ) == 0
    receipt = json.loads(Path(args.receipt).read_text(encoding="utf-8"))
    assert receipt["intent"]["desired_state"] == "finished"
