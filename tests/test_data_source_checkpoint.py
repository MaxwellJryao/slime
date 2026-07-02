from types import SimpleNamespace

import pytest

from slime.rollout import data_source as data_source_module
from slime.rollout.data_source import RolloutDataSource

NUM_GPUS = 0


def _args(tmp_path):
    return SimpleNamespace(
        rollout_global_dataset=True,
        prompt_data=None,
        n_samples_per_prompt=2,
        rollout_shuffle=False,
        save=str(tmp_path),
        load=str(tmp_path),
    )


def test_data_source_checkpoint_is_atomic_and_round_trips(tmp_path):
    source = RolloutDataSource(_args(tmp_path))
    source.sample_offset = 17
    source.epoch_id = 2
    source.sample_group_index = 31
    source.sample_index = 62
    source.metadata = {"committed": True}

    source.save(7)

    path = tmp_path / "rollout" / "global_dataset_state_dict_7.pt"
    assert path.is_file()
    assert list(path.parent.glob("*.tmp.*")) == []

    restored = RolloutDataSource(_args(tmp_path))
    restored.load(7)
    assert (
        restored.sample_offset,
        restored.epoch_id,
        restored.sample_group_index,
        restored.sample_index,
        restored.metadata,
    ) == (17, 2, 31, 62, {"committed": True})


def test_failed_data_source_save_keeps_previous_checkpoint(monkeypatch, tmp_path):
    source = RolloutDataSource(_args(tmp_path))
    source.sample_offset = 5
    source.save(3)
    path = tmp_path / "rollout" / "global_dataset_state_dict_3.pt"
    committed = path.read_bytes()

    def fail_after_partial_write(_state, file_obj):
        file_obj.write(b"partial")
        raise OSError("injected save failure")

    monkeypatch.setattr(data_source_module.torch, "save", fail_after_partial_write)
    source.sample_offset = 99
    with pytest.raises(OSError, match="injected save failure"):
        source.save(3)

    assert path.read_bytes() == committed
    assert list(path.parent.glob("*.tmp.*")) == []


def test_resume_requires_exact_rollout_state_but_initial_seed_does_not(tmp_path):
    source = RolloutDataSource(_args(tmp_path))

    source.load(-1)
    with pytest.raises(FileNotFoundError, match="iteration 4.*exact matching rollout data-source state"):
        source.load(4)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
