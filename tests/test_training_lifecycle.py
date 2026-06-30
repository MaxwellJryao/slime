import json

from slime.utils.training_lifecycle import (
    final_eval_complete_marker_matches,
    graceful_exit_due,
    write_final_eval_complete_marker,
    write_training_complete_marker,
)


def test_graceful_exit_due():
    assert not graceful_exit_due(None, now=100)
    assert not graceful_exit_due(101, now=100)
    assert graceful_exit_due(100, now=100)
    assert graceful_exit_due(99, now=100)


def test_write_training_complete_marker_is_atomic_and_descriptive(tmp_path):
    marker = tmp_path / "checkpoint" / "TRAINING_COMPLETE"

    write_training_complete_marker(str(marker), num_rollout=16)

    payload = json.loads(marker.read_text())
    assert payload["num_rollout"] == 16
    assert payload["last_rollout_id"] == 15
    assert isinstance(payload["completed_at_unix_time"], float)
    assert list(marker.parent.glob(".TRAINING_COMPLETE.tmp-*")) == []


def test_final_eval_complete_marker_matches_exact_run_end(tmp_path):
    marker = tmp_path / "checkpoint" / "FINAL_EVAL_COMPLETE"
    eval_hash = "a" * 64

    write_final_eval_complete_marker(
        str(marker),
        final_rollout_id=40,
        model_iteration=40,
        num_rollout=41,
        eval_data_sha256=eval_hash,
    )

    payload = json.loads(marker.read_text())
    assert payload["final_rollout_id"] == 40
    assert payload["model_iteration"] == 40
    assert payload["num_rollout"] == 41
    assert payload["eval_data_sha256"] == eval_hash
    assert final_eval_complete_marker_matches(
        str(marker),
        final_rollout_id=40,
        model_iteration=40,
        num_rollout=41,
        eval_data_sha256=eval_hash,
    )
    assert not final_eval_complete_marker_matches(
        str(marker),
        final_rollout_id=41,
        model_iteration=41,
        num_rollout=42,
        eval_data_sha256=eval_hash,
    )
    assert not final_eval_complete_marker_matches(
        str(marker),
        final_rollout_id=40,
        model_iteration=40,
        num_rollout=41,
        eval_data_sha256="b" * 64,
    )
    assert list(marker.parent.glob(".FINAL_EVAL_COMPLETE.tmp-*")) == []


def test_final_eval_marker_persists_recoverable_metric_payload(tmp_path):
    marker = tmp_path / "FINAL_EVAL_COMPLETE"
    metrics = {
        "eval/train_step": 123,
        "eval/tmax_holdout/reward_mean": 0.4375,
    }

    write_final_eval_complete_marker(
        str(marker),
        final_rollout_id=40,
        model_iteration=40,
        num_rollout=41,
        eval_data_sha256="c" * 64,
        metrics=metrics,
        primary_tracking_flush_attempted=True,
    )

    payload = json.loads(marker.read_text())
    assert payload["metrics"] == metrics
    assert payload["primary_tracking_flush_attempted"] is True


def test_final_eval_complete_marker_rejects_malformed_payload(tmp_path):
    marker = tmp_path / "FINAL_EVAL_COMPLETE"
    marker.write_text("[]\n")

    assert not final_eval_complete_marker_matches(
        str(marker),
        final_rollout_id=1,
        model_iteration=1,
        num_rollout=2,
        eval_data_sha256="b" * 64,
    )
