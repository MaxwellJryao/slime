import json

from slime.utils.training_lifecycle import graceful_exit_due, write_training_complete_marker


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
