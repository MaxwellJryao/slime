from slime.backends.megatron_utils.checkpoint import is_release_checkpoint


def test_release_checkpoint_is_model_seed(tmp_path):
    (tmp_path / "latest_checkpointed_iteration.txt").write_text("release\n")

    assert is_release_checkpoint(tmp_path)


def test_numeric_checkpoint_is_training_resume(tmp_path):
    (tmp_path / "latest_checkpointed_iteration.txt").write_text("17\n")

    assert not is_release_checkpoint(tmp_path)


def test_missing_tracker_is_not_release(tmp_path):
    assert not is_release_checkpoint(tmp_path)
