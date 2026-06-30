from types import SimpleNamespace

import pytest

from slime.ray import rollout as rollout_module


def _args() -> SimpleNamespace:
    return SimpleNamespace(
        custom_rollout_log_function_path=None,
        load_debug_rollout_data=False,
    )


@pytest.mark.parametrize(
    ("extra_metrics", "handoff_time", "expected_perf_time"),
    [
        (
            {"timing/service_window": 8.0, "timing/service_time_max": 5.0},
            0.001,
            8.0,
        ),
        ({"timing/service_time_max": 4.5}, 0.001, 4.5),
        ({}, 3.0, 3.0),
    ],
    ids=(
        "buffered-async-service-window",
        "legacy-buffered-max-service-time",
        "ordinary-rollout-handoff-time",
    ),
)
def test_rollout_perf_uses_service_time_without_losing_handoff_time(
    monkeypatch,
    extra_metrics,
    handoff_time,
    expected_perf_time,
) -> None:
    logged = []
    perf_durations = []
    monkeypatch.setattr(rollout_module, "compute_metrics_from_samples", lambda *_args: {})

    def compute_perf(_args, _samples, duration):
        perf_durations.append(duration)
        return {"rollout_time": duration}

    monkeypatch.setattr(rollout_module, "compute_perf_metrics_from_samples", compute_perf)
    monkeypatch.setattr(rollout_module, "set_wandb_step", lambda *_args, **_kwargs: "rollout/step")
    monkeypatch.setattr(
        rollout_module.logging_utils,
        "log",
        lambda _args, metrics, **_kwargs: logged.append(dict(metrics)),
    )

    rollout_module._log_rollout_data(
        rollout_id=8,
        args=_args(),
        samples=[],
        rollout_extra_metrics=extra_metrics,
        rollout_time=handoff_time,
    )

    assert perf_durations == [expected_perf_time]
    assert logged[0]["timing/rollout_time"] == expected_perf_time
    assert logged[0]["timing/handoff_time"] == handoff_time


@pytest.mark.parametrize(
    "invalid_service_time",
    [0, -1, float("nan"), float("inf"), "not-a-number"],
)
def test_invalid_service_window_falls_back_to_valid_max_latency(
    invalid_service_time,
) -> None:
    assert (
        rollout_module._resolve_rollout_perf_time(
            {
                "timing/service_window": invalid_service_time,
                "timing/service_time_max": 1.5,
            },
            2.0,
        )
        == 1.5
    )


@pytest.mark.parametrize("invalid_service_time", [0, -1, float("nan"), float("inf")])
def test_invalid_max_latency_falls_back_to_handoff(invalid_service_time) -> None:
    assert (
        rollout_module._resolve_rollout_perf_time(
            {"timing/service_time_max": invalid_service_time},
            2.0,
        )
        == 2.0
    )


def test_speculative_rollout_metrics_wait_for_actor_commit(monkeypatch) -> None:
    manager_cls = rollout_module.RolloutManager.__ray_metadata__.modified_class
    manager = manager_cls.__new__(manager_cls)
    manager.args = SimpleNamespace(
        ci_test=False,
        use_fault_tolerance=False,
        debug_rollout_only=False,
    )
    manager._pending_rollout_logs = {}
    manager.health_monitoring_resume = lambda: None
    manager._get_rollout_data = lambda *, rollout_id: (
        [f"sample-{rollout_id}"],
        {"polar/reward_mean": 0.5},
    )
    manager._save_debug_rollout_data = lambda *_args, **_kwargs: None
    manager._convert_samples_to_train_data = lambda data: data
    manager._split_train_data_by_dp = lambda data: data
    times = iter((10.0, 12.0, 12.0, 12.5, 12.5, 12.75, 12.75))
    monkeypatch.setattr(rollout_module.time, "perf_counter", lambda: next(times))
    logged = []

    monkeypatch.setattr(
        rollout_module,
        "_log_rollout_data",
        lambda *args, **kwargs: logged.append((args, kwargs)),
    )

    assert manager.generate(4) == ["sample-4"]
    assert logged == []
    assert list(manager._pending_rollout_logs) == [4]

    manager.commit_rollout_metrics(4)

    assert len(logged) == 1
    assert logged[0][0][0] == 4
    assert logged[0][1]["completed_train_batch"] is True
    extra_metrics = logged[0][0][3]
    assert extra_metrics["timing/rollout_to_train_data_time"] == 0.5
    assert extra_metrics["timing/dp_split_time"] == 0.25
    assert extra_metrics["timing/generate_e2e_time"] == 2.75
    assert manager._pending_rollout_logs == {}


def test_rollout_performance_metrics_separate_durations_from_throughput() -> None:
    assert rollout_module._prefix_rollout_performance_metrics(
        {
            "rollout_time": 8.0,
            "non_generation_time/mean": 1.0,
            "request/e2e_latency/p95": 7.0,
            "prefill/forward_duration/max": 2.0,
            "tokens_per_gpu_per_sec": 100.0,
            "decode/throughput/mean": 25.0,
            "prefill/transfer_speed_gb_s/mean": 12.0,
        }
    ) == {
        "timing/rollout_time": 8.0,
        "timing/non_generation_time/mean": 1.0,
        "timing/request/e2e_latency/p95": 7.0,
        "timing/prefill/forward_duration/max": 2.0,
        "perf/tokens_per_gpu_per_sec": 100.0,
        "perf/decode/throughput/mean": 25.0,
        "perf/prefill/transfer_speed_gb_s/mean": 12.0,
    }


def test_failed_rollout_metric_emit_remains_pending_for_fail_fast_retry(monkeypatch) -> None:
    manager_cls = rollout_module.RolloutManager.__ray_metadata__.modified_class
    manager = manager_cls.__new__(manager_cls)
    manager.args = SimpleNamespace()
    manager._pending_rollout_logs = {
        5: rollout_module._PendingRolloutLog([], {}, 1.0),
    }
    monkeypatch.setattr(
        rollout_module,
        "_log_rollout_data",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("wandb failed")),
    )

    with pytest.raises(RuntimeError, match="wandb failed"):
        manager.commit_rollout_metrics(5)

    assert 5 in manager._pending_rollout_logs
