from __future__ import annotations

import json
import math
import os
import tempfile
import time
from pathlib import Path
from typing import Any


def _json_safe(value: Any) -> Any:
    """Convert scalar telemetry containers to a durable JSON representation."""

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]

    # NumPy/PyTorch scalar values occur in metric payloads but importing those
    # heavy packages in this lifecycle helper would make simple launcher tools
    # unnecessarily expensive.
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return _json_safe(item())
        except (TypeError, ValueError, RuntimeError):
            pass
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        try:
            return _json_safe(tolist())
        except (TypeError, ValueError, RuntimeError):
            pass
    return repr(value)


def graceful_exit_due(deadline: float | None, *, now: float | None = None) -> bool:
    if deadline is None:
        return False
    return (time.time() if now is None else now) >= deadline


def _write_json_marker(path: str | None, payload: dict[str, Any]) -> None:
    if path is None:
        return

    marker = Path(path)
    marker.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=marker.parent,
            prefix=f".{marker.name}.tmp-",
            delete=False,
        ) as output:
            temporary = Path(output.name)
            output.write(json.dumps(payload, sort_keys=True) + "\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, marker)
        temporary = None

        # Persist the rename as well as the file contents. Some network file
        # systems do not support fsync on directories, so retain the atomic
        # marker when that final durability hint is unavailable.
        directory_fd = os.open(
            marker.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        )
        try:
            try:
                os.fsync(directory_fd)
            except OSError:
                pass
        finally:
            os.close(directory_fd)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def write_training_complete_marker(path: str | None, *, num_rollout: int) -> None:
    _write_json_marker(
        path,
        {
            "completed_at_unix_time": time.time(),
            "last_rollout_id": num_rollout - 1,
            "num_rollout": num_rollout,
        },
    )


def write_final_eval_complete_marker(
    path: str | None,
    *,
    final_rollout_id: int,
    model_iteration: int,
    num_rollout: int,
    eval_data_sha256: str,
    metrics: dict[str, Any] | None = None,
    primary_tracking_flush_attempted: bool = False,
) -> None:
    """Commit a successful fixed-set final evaluation durably and atomically.

    The optional metric snapshot makes a missing W&B row recoverable when its
    client reports an asynchronous upload timeout only as a warning.  The
    ``primary_tracking_flush_attempted`` flag deliberately records an attempt,
    not a claim that the remote service acknowledged the row.
    """

    payload = {
        "completed_at_unix_time": time.time(),
        "final_rollout_id": final_rollout_id,
        "model_iteration": model_iteration,
        "num_rollout": num_rollout,
        "eval_data_sha256": eval_data_sha256,
    }
    if metrics is not None:
        payload["metrics"] = _json_safe(metrics)
    if primary_tracking_flush_attempted:
        payload["primary_tracking_flush_attempted"] = True
    _write_json_marker(path, payload)


def final_eval_complete_marker_matches(
    path: str | None,
    *,
    final_rollout_id: int,
    model_iteration: int,
    num_rollout: int,
    eval_data_sha256: str,
) -> bool:
    """Return whether ``path`` proves eval completion for this exact run end."""

    if path is None:
        return False
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return False
    if not isinstance(payload, dict):
        return False
    return (
        type(payload.get("final_rollout_id")) is int
        and payload["final_rollout_id"] == final_rollout_id
        and type(payload.get("model_iteration")) is int
        and payload["model_iteration"] == model_iteration
        and type(payload.get("num_rollout")) is int
        and payload["num_rollout"] == num_rollout
        and type(payload.get("eval_data_sha256")) is str
        and payload["eval_data_sha256"] == eval_data_sha256
    )
