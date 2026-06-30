"""Publish optimizer progress for allocation-level monitoring sidecars."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path


TRAIN_PROGRESS_ENV = "SLIME_TRAIN_PROGRESS_FILE"


def atomic_write_train_step(step: int, path: str | os.PathLike[str] | None = None) -> bool:
    """Atomically publish ``step`` and return whether a target was configured."""

    raw_path = path if path is not None else os.environ.get(TRAIN_PROGRESS_ENV, "")
    if not raw_path:
        return False

    target = Path(raw_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=target.parent,
            prefix=f".{target.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
            stream.write(f"{int(step)}\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, target)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return True
