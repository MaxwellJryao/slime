from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from _actor_import_helpers import import_actor_module

actor_module = import_actor_module()

NUM_GPUS = 0


@pytest.mark.unit
def test_actor_captures_perf_context_before_offload_and_flushes_after(monkeypatch) -> None:
    from slime.backends.megatron_utils import data as data_module

    events: list[str] = []
    reloadable_group = SimpleNamespace(group=SimpleNamespace(rank=lambda: 0))

    @contextmanager
    def phase_timer(name: str):
        events.append(f"{name}:start")
        yield
        events.append(f"{name}:end")

    actor = object.__new__(actor_module.MegatronTrainRayActor)
    actor.args = SimpleNamespace(debug_rollout_only=False, offload_train=True)
    actor.role = "actor"
    actor.wake_up = lambda: events.append("wake_up")
    actor._get_rollout_data = lambda _ref: events.append("get_rollout_data") or {"sample": object()}
    actor.train_actor = lambda *_args, **_kwargs: events.append("train_actor")

    def capture_primary_rank() -> bool:
        assert reloadable_group.group is not None
        events.append("capture_primary_rank")
        return True

    def capture_world_size() -> int:
        assert reloadable_group.group is not None
        events.append("capture_world_size")
        return 16

    def sleep() -> None:
        events.append("sleep")
        # Match ReloadableProcessGroup.destroy_process_groups(): the wrapper
        # remains installed in Megatron but its inner group is torn down.
        reloadable_group.group = None

    def destroyed_group_rank(*_args, **_kwargs) -> int:
        # This reproduces the production failure if post-sleep telemetry ever
        # falls back to Megatron's live process-group queries.
        return reloadable_group.group.rank()

    def log_perf_data_raw(**kwargs) -> None:
        assert reloadable_group.group is None
        assert kwargs["rollout_id"] == 84
        assert kwargs["args"] is actor.args
        assert kwargs["is_primary_rank"] is True
        assert kwargs["compute_total_fwd_flops"]([8]) == pytest.approx(320.0 / 16 / 1e12)
        events.append("log_perf_data")

    actor.sleep = sleep
    monkeypatch.setattr(actor_module, "timer", phase_timer)
    monkeypatch.setattr(actor_module, "is_megatron_main_rank", capture_primary_rank)
    monkeypatch.setattr(actor_module.dist, "get_world_size", capture_world_size)
    monkeypatch.setattr(data_module.mpu, "get_tensor_model_parallel_rank", destroyed_group_rank, raising=False)
    monkeypatch.setattr(data_module.mpu, "is_pipeline_last_stage", destroyed_group_rank, raising=False)
    monkeypatch.setattr(data_module.mpu, "get_data_parallel_rank", destroyed_group_rank, raising=False)
    monkeypatch.setattr(data_module, "calculate_fwd_flops", lambda *, seqlens, args: 320.0)
    monkeypatch.setattr(data_module.train_metric_utils, "log_perf_data_raw", log_perf_data_raw)
    monkeypatch.setattr(actor_module, "log_perf_data", data_module.log_perf_data)

    result = actor.train(84, object())

    assert result is None
    assert events == [
        "wake_up",
        "data_preprocess:start",
        "get_rollout_data",
        "data_preprocess:end",
        "train_actor",
        "capture_primary_rank",
        "capture_world_size",
        "sleep",
        "log_perf_data",
    ]


@pytest.mark.unit
def test_explicit_perf_context_never_queries_destroyed_process_groups(monkeypatch) -> None:
    from slime.backends.megatron_utils import data as data_module

    def destroyed_process_group(*_args, **_kwargs):
        raise AssertionError("post-offload telemetry queried a destroyed process group")

    monkeypatch.setattr(data_module.mpu, "get_tensor_model_parallel_rank", destroyed_process_group, raising=False)
    monkeypatch.setattr(data_module.mpu, "is_pipeline_last_stage", destroyed_process_group, raising=False)
    monkeypatch.setattr(data_module.mpu, "get_data_parallel_rank", destroyed_process_group, raising=False)
    monkeypatch.setattr(data_module.dist, "get_world_size", destroyed_process_group)
    monkeypatch.setattr(data_module, "calculate_fwd_flops", lambda *, seqlens, args: 320.0)

    captured: dict[str, object] = {}

    def log_perf_data_raw(**kwargs) -> None:
        captured.update(kwargs)
        captured["total_fwd_tflops"] = kwargs["compute_total_fwd_flops"]([8, 4])

    monkeypatch.setattr(data_module.train_metric_utils, "log_perf_data_raw", log_perf_data_raw)

    args = SimpleNamespace()
    data_module.log_perf_data(
        84,
        args,
        is_primary_rank=True,
        world_size=16,
    )

    assert captured["rollout_id"] == 84
    assert captured["args"] is args
    assert captured["is_primary_rank"] is True
    assert captured["extra_metrics"] is None
    assert captured["total_fwd_tflops"] == pytest.approx(320.0 / 16 / 1e12)


@pytest.mark.unit
def test_perf_context_fallback_preserves_live_process_group_behavior(monkeypatch) -> None:
    from slime.backends.megatron_utils import data as data_module

    monkeypatch.setattr(data_module.mpu, "get_tensor_model_parallel_rank", lambda: 0, raising=False)
    monkeypatch.setattr(data_module.mpu, "is_pipeline_last_stage", lambda: True, raising=False)
    monkeypatch.setattr(
        data_module.mpu,
        "get_data_parallel_rank",
        lambda *, with_context_parallel: 0 if with_context_parallel else 1,
        raising=False,
    )
    monkeypatch.setattr(data_module.dist, "get_world_size", lambda: 8)
    monkeypatch.setattr(data_module, "calculate_fwd_flops", lambda *, seqlens, args: 80.0)

    captured: dict[str, object] = {}

    def log_perf_data_raw(**kwargs) -> None:
        captured.update(kwargs)
        captured["total_fwd_tflops"] = kwargs["compute_total_fwd_flops"]([1])

    monkeypatch.setattr(data_module.train_metric_utils, "log_perf_data_raw", log_perf_data_raw)

    data_module.log_perf_data(9, SimpleNamespace())

    assert captured["is_primary_rank"] is True
    assert captured["total_fwd_tflops"] == pytest.approx(80.0 / 8 / 1e12)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
