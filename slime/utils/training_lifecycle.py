from __future__ import annotations

import json
import os
import time
from pathlib import Path


def graceful_exit_due(deadline: float | None, *, now: float | None = None) -> bool:
    if deadline is None:
        return False
    return (time.time() if now is None else now) >= deadline


def write_training_complete_marker(path: str | None, *, num_rollout: int) -> None:
    if path is None:
        return

    marker = Path(path)
    marker.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "completed_at_unix_time": time.time(),
        "last_rollout_id": num_rollout - 1,
        "num_rollout": num_rollout,
    }
    temporary = marker.with_name(f".{marker.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, marker)
