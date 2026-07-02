"""Small, side-effect-free helpers for cross-process startup timing.

Launcher phases happen in bash while model phases happen in Ray workers, so a
monotonic clock cannot span the complete startup path.  The launcher exports
Unix nanosecond markers and Python converts only ordered, finite pairs into
durations.  Local phase timers should still use ``perf_counter`` directly.
"""

from __future__ import annotations

import math
import os
import time
from collections.abc import Mapping

from slime.utils.timer import Timer


_UNIX_NS_PER_SECOND = 1_000_000_000


def elapsed_seconds_between_unix_ns(start_ns: object, end_ns: object) -> float | None:
    """Return an elapsed duration only for a valid ordered timestamp pair."""

    try:
        start = int(start_ns)
        end = int(end_ns)
    except (TypeError, ValueError):
        return None
    if start <= 0 or end < start:
        return None
    elapsed = (end - start) / _UNIX_NS_PER_SECOND
    if not math.isfinite(elapsed):
        return None
    return elapsed


def elapsed_seconds_from_env(
    start_name: str,
    end_name: str | None = None,
    *,
    environ: Mapping[str, str] | None = None,
    now_ns: int | None = None,
) -> float | None:
    """Measure an exported launcher/worker span without inventing a marker."""

    values = os.environ if environ is None else environ
    start = values.get(start_name)
    end: object
    if end_name is None:
        end = time.time_ns() if now_ns is None else now_ns
    else:
        end = values.get(end_name)
    return elapsed_seconds_between_unix_ns(start, end)


def launcher_startup_metrics(
    *,
    environ: Mapping[str, str] | None = None,
) -> dict[str, float]:
    """Build real launcher envelopes from markers exported by ``run.sh``."""

    marker_pairs = {
        "timing/startup_submit_to_container_entry_time": (
            "SLIME_SUBMIT_UNIX_NS",
            "SLIME_CONTAINER_ENTRY_UNIX_NS",
        ),
        "timing/startup_outer_container_entry_time": (
            "SLIME_SLURM_BATCH_START_UNIX_NS",
            "SLIME_CONTAINER_ENTRY_UNIX_NS",
        ),
        "timing/startup_container_entry_to_job_script_time": (
            "SLIME_CONTAINER_ENTRY_UNIX_NS",
            "SLIME_JOB_SCRIPT_START_UNIX_NS",
        ),
        "timing/startup_job_script_to_ray_ready_time": (
            "SLIME_JOB_SCRIPT_START_UNIX_NS",
            "SLIME_RAY_READY_UNIX_NS",
        ),
        "timing/startup_rollout_service_time": (
            "SLIME_ROLLOUT_SERVICE_START_UNIX_NS",
            "SLIME_ROLLOUT_SERVICE_READY_UNIX_NS",
        ),
        "timing/startup_gateway_time": (
            "SLIME_GATEWAY_START_UNIX_NS",
            "SLIME_GATEWAY_READY_UNIX_NS",
        ),
        "timing/startup_uds_tunnel_time": (
            "SLIME_UDS_TUNNEL_START_UNIX_NS",
            "SLIME_UDS_TUNNEL_READY_UNIX_NS",
        ),
        "timing/startup_services_time": (
            "SLIME_RAY_READY_UNIX_NS",
            "SLIME_SERVICES_READY_UNIX_NS",
        ),
        "timing/startup_job_script_to_services_ready_time": (
            "SLIME_JOB_SCRIPT_START_UNIX_NS",
            "SLIME_SERVICES_READY_UNIX_NS",
        ),
    }
    metrics: dict[str, float] = {}
    for metric_name, (start_name, end_name) in marker_pairs.items():
        elapsed = elapsed_seconds_from_env(
            start_name,
            end_name,
            environ=environ,
        )
        if elapsed is not None:
            metrics[metric_name] = elapsed
    return metrics


def record_train_phase_duration(
    name: str,
    started_at: float,
    *,
    finished_at: float | None = None,
) -> float:
    """Record one local trainer phase in seconds and return the duration."""

    end = time.perf_counter() if finished_at is None else float(finished_at)
    duration = max(0.0, end - float(started_at))
    if not math.isfinite(duration):
        raise ValueError(f"non-finite trainer phase duration for {name!r}")
    Timer().add(name, duration)
    return duration
