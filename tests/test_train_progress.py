from pathlib import Path

import pytest

from slime.utils.train_progress import atomic_write_train_step

NUM_GPUS = 0


def test_atomic_write_train_step_replaces_complete_integer(tmp_path: Path):
    path = tmp_path / "progress" / "train.step"

    assert atomic_write_train_step(7, path)
    assert path.read_text() == "7\n"
    assert atomic_write_train_step(123, path)
    assert path.read_text() == "123\n"
    assert not list(path.parent.glob(f".{path.name}.*.tmp"))


def test_atomic_write_train_step_is_optional(monkeypatch):
    monkeypatch.delenv("SLIME_TRAIN_PROGRESS_FILE", raising=False)

    assert not atomic_write_train_step(4)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
