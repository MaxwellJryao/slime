from __future__ import annotations

from types import SimpleNamespace

import pytest

from train_async import _attach_startup_step
from slime.utils.startup_timing import (
    elapsed_seconds_between_unix_ns,
    launcher_startup_metrics,
    record_train_phase_duration,
)
from slime.utils.timer import Timer


@pytest.mark.unit
def test_launcher_startup_metrics_require_real_ordered_marker_pairs() -> None:
    second = 1_000_000_000
    metrics = launcher_startup_metrics(
        environ={
            "SLIME_SUBMIT_UNIX_NS": str(1 * second),
            "SLIME_SLURM_BATCH_START_UNIX_NS": str(8 * second),
            "SLIME_CONTAINER_ENTRY_UNIX_NS": str(9 * second),
            "SLIME_JOB_SCRIPT_START_UNIX_NS": str(10 * second),
            "SLIME_RAY_READY_UNIX_NS": str(12 * second),
            "SLIME_POLAR_ROLLOUT_START_UNIX_NS": str(12 * second),
            "SLIME_POLAR_ROLLOUT_READY_UNIX_NS": str(13 * second),
            "SLIME_POLAR_GATEWAY_START_UNIX_NS": str(13 * second),
            "SLIME_POLAR_GATEWAY_READY_UNIX_NS": str(15 * second),
            "SLIME_POLAR_UDS_START_UNIX_NS": str(15 * second),
            "SLIME_POLAR_UDS_READY_UNIX_NS": str(16 * second),
            "SLIME_POLAR_READY_UNIX_NS": str(16 * second),
        }
    )

    assert metrics == {
        "timing/startup_submit_to_container_entry_time": 8.0,
        "timing/startup_outer_container_entry_time": 1.0,
        "timing/startup_container_entry_to_job_script_time": 1.0,
        "timing/startup_job_script_to_ray_ready_time": 2.0,
        "timing/startup_polar_rollout_server_time": 1.0,
        "timing/startup_polar_gateway_time": 2.0,
        "timing/startup_polar_uds_tunnel_time": 1.0,
        "timing/startup_polar_services_time": 4.0,
        "timing/startup_job_script_to_polar_ready_time": 6.0,
    }
    assert elapsed_seconds_between_unix_ns("bad", 20 * second) is None
    assert elapsed_seconds_between_unix_ns(20 * second, 19 * second) is None


@pytest.mark.unit
def test_forward_backward_and_optimizer_phase_durations_accumulate_in_seconds() -> None:
    timer = Timer()
    timer.reset()

    assert record_train_phase_duration(
        "train_forward_backward_dispatch",
        10.0,
        finished_at=11.25,
    ) == pytest.approx(1.25)
    assert record_train_phase_duration(
        "optimizer_dispatch",
        20.0,
        finished_at=20.5,
    ) == pytest.approx(0.5)
    assert record_train_phase_duration(
        "train_forward_backward_dispatch",
        30.0,
        finished_at=30.75,
    ) == pytest.approx(0.75)

    assert timer.log_dict()["train_forward_backward_dispatch"] == pytest.approx(2.0)
    assert timer.log_dict()["optimizer_dispatch"] == pytest.approx(0.5)
    timer.reset()


@pytest.mark.unit
@pytest.mark.parametrize(
    ("always_train_step", "expected_step_key", "expected_rollout_step"),
    [
        (True, "train/step", 12),
        (False, "rollout/step", 3),
    ],
)
def test_startup_metrics_carry_the_axis_used_by_timing_wandb_metrics(
    always_train_step: bool,
    expected_step_key: str,
    expected_rollout_step: int,
) -> None:
    args = SimpleNamespace(
        rollout_batch_size=5,
        n_samples_per_prompt=8,
        global_batch_size=10,
        start_rollout_id=3,
        wandb_always_use_train_step=always_train_step,
    )
    metrics = {"timing/startup_trainer_model_init_time": 10.0}

    step_key = _attach_startup_step(args, metrics)

    assert step_key == expected_step_key
    assert metrics["train/step"] == 12
    assert metrics["rollout/step"] == expected_rollout_step
