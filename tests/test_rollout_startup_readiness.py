from __future__ import annotations

from types import SimpleNamespace

import pytest

from slime.ray import placement_group, rollout

NUM_GPUS = 0


class _RemoteMethod:
    def __init__(self, func):
        self._func = func

    def remote(self, *args, **kwargs):
        return self._func(*args, **kwargs)


class _FakeRolloutManagerActor:
    def __init__(self, events):
        self.ready = _RemoteMethod(lambda: events.append("ready") or {"timing/startup_sglang_router_engines_ready_time": 3.0})


@pytest.mark.unit
def test_create_rollout_manager_preserves_default_ready_barrier(monkeypatch) -> None:
    events = []

    class _FakeRolloutManager:
        @classmethod
        def options(cls, **_kwargs):
            return cls

        @classmethod
        def remote(cls, _args, _pg):
            return _FakeRolloutManagerActor(events)

    monkeypatch.setattr(rollout, "RolloutManager", _FakeRolloutManager)
    monkeypatch.setattr(placement_group.ray, "get", lambda value: value)
    args = SimpleNamespace(
        num_rollout=2,
        num_epoch=1,
        check_weight_update_equal=False,
        offload_rollout=False,
        rollout_data_transport="object-store",
    )

    manager, num_rollout_per_epoch = placement_group.create_rollout_manager(
        args,
        object(),
    )

    assert isinstance(manager, _FakeRolloutManagerActor)
    assert num_rollout_per_epoch is None
    assert events == ["ready"]
    assert args.rollout_startup_metrics == {"timing/startup_sglang_router_engines_ready_time": 3.0}


@pytest.mark.unit
def test_create_rollout_manager_can_defer_ready_for_parallel_model_init(
    monkeypatch,
) -> None:
    events = []

    class _FakeRolloutManager:
        @classmethod
        def options(cls, **_kwargs):
            return cls

        @classmethod
        def remote(cls, _args, _pg):
            return _FakeRolloutManagerActor(events)

    monkeypatch.setattr(rollout, "RolloutManager", _FakeRolloutManager)
    args = SimpleNamespace(
        num_rollout=2,
        num_epoch=1,
        check_weight_update_equal=False,
        offload_rollout=False,
        rollout_data_transport="object-store",
    )

    manager, _ = placement_group.create_rollout_manager(
        args,
        object(),
        wait_ready=False,
    )

    assert isinstance(manager, _FakeRolloutManagerActor)
    assert events == []
    assert not hasattr(args, "rollout_startup_metrics")


@pytest.mark.unit
@pytest.mark.parametrize(
    ("check_weight_update_equal", "offload_rollout"),
    [(True, False), (False, True)],
)
def test_deferred_ready_rejects_startup_operations_that_need_live_engines(
    check_weight_update_equal: bool,
    offload_rollout: bool,
) -> None:
    args = SimpleNamespace(
        check_weight_update_equal=check_weight_update_equal,
        offload_rollout=offload_rollout,
    )

    with pytest.raises(ValueError, match="wait_ready=False"):
        placement_group.create_rollout_manager(
            args,
            object(),
            wait_ready=False,
        )


@pytest.mark.unit
def test_rollout_manager_ready_waits_and_starts_health_monitors_once(
    monkeypatch,
) -> None:
    events = []

    class _Monitor:
        def __init__(self, group, args):
            events.append(("monitor-created", group, args))

        def start(self):
            events.append("monitor-started")

    manager_cls = rollout.RolloutManager.__ray_metadata__.modified_class
    manager = object.__new__(manager_cls)
    manager.args = SimpleNamespace(
        debug_train_only=False,
        use_fault_tolerance=True,
        ci_test=True,
    )
    group = object()
    manager.servers = {
        "default": SimpleNamespace(server_groups=[group]),
    }
    manager._rollout_init_handles = ["engine-init-ref"]
    manager._engines_started_at = 10.0
    manager._engines_ready = False
    manager._health_monitors = []
    manager._health_monitors_started = False
    manager._ci_fault_injection_pending = False
    manager._startup_timing = {
        "timing/startup_rollout_manager_remote_init_time": 0.5,
    }

    monkeypatch.setattr(rollout.ray, "get", lambda refs: events.append(("wait", refs)))
    monkeypatch.setattr(rollout.time, "perf_counter", lambda: 13.0)
    monkeypatch.setattr(rollout, "RolloutHealthMonitor", _Monitor)

    first = manager_cls.ready(manager)
    second = manager_cls.ready(manager)

    assert (
        first
        == second
        == {
            "timing/startup_rollout_manager_remote_init_time": 0.5,
            "timing/startup_sglang_router_engines_ready_time": 3.0,
        }
    )
    assert events == [
        ("wait", ["engine-init-ref"]),
        ("monitor-created", group, manager.args),
        "monitor-started",
    ]
    assert manager._rollout_init_handles == []
    assert manager._engines_ready is True
    assert manager._health_monitors_started is True
    assert manager._ci_fault_injection_pending is True


@pytest.mark.unit
def test_start_rollout_id_sync_is_idempotent_and_pre_generation_only() -> None:
    manager_cls = rollout.RolloutManager.__ray_metadata__.modified_class
    manager = object.__new__(manager_cls)
    manager.args = SimpleNamespace(start_rollout_id=None)
    manager.rollout_id = -1

    assert manager_cls.set_start_rollout_id(manager, 7) == 7
    assert manager.args.start_rollout_id == 7
    assert manager_cls.set_start_rollout_id(manager, 7) == 7

    with pytest.raises(RuntimeError, match="conflicts with the trainer checkpoint"):
        manager_cls.set_start_rollout_id(manager, 8)
    with pytest.raises(ValueError, match="must be non-negative"):
        manager_cls.set_start_rollout_id(manager, -1)

    manager.rollout_id = 7
    with pytest.raises(RuntimeError, match="after rollout generation starts"):
        manager_cls.set_start_rollout_id(manager, 7)


@pytest.mark.unit
def test_connect_syncs_checkpoint_cursor_before_dataset_load(monkeypatch) -> None:
    events: list[tuple[str, object]] = []
    manager_cls = rollout.RolloutManager.__ray_metadata__.modified_class
    actor_manager = object.__new__(manager_cls)
    actor_manager.args = SimpleNamespace(start_rollout_id=None)
    actor_manager.rollout_id = -1

    class _Manager:
        def __init__(self) -> None:
            self.set_start_rollout_id = _RemoteMethod(self._sync)
            self.load = _RemoteMethod(self._load)

        def _sync(self, rollout_id: int) -> int:
            events.append(("sync", rollout_id))
            return manager_cls.set_start_rollout_id(actor_manager, rollout_id)

        def _load(self, checkpoint_id: int) -> None:
            assert actor_manager.args.start_rollout_id == checkpoint_id + 1
            events.append(("load", checkpoint_id))

    class _Actor:
        def set_rollout_manager(self, manager) -> None:
            events.append(("wire", manager))

    manager = _Manager()
    args = SimpleNamespace(
        rollout_global_dataset=True,
        start_rollout_id=7,
        use_critic=False,
    )
    monkeypatch.setattr(placement_group.ray, "get", lambda value: value)

    placement_group.connect_training_models_to_rollout(
        args,
        _Actor(),
        None,
        manager,
    )

    assert events == [
        ("sync", 7),
        ("wire", manager),
        ("load", 6),
    ]
    assert actor_manager.args.start_rollout_id == 7


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
