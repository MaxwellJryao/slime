from types import SimpleNamespace

import pytest

import train_async
from slime.ray import rollout as rollout_module


class _RemoteMethod:
    def __init__(self, func):
        self._func = func

    def remote(self):
        return self._func()


class _DriverRolloutManager:
    def __init__(self, events, *, pause_result=True, pause_error=None):
        def pause():
            events.append("pause")
            if pause_error is not None:
                raise pause_error
            return pause_result

        self.pause_generation_for_weight_update = _RemoteMethod(pause)
        self.resume_generation_after_weight_update = _RemoteMethod(
            lambda: events.append("resume") or True
        )


class _ActorModel:
    def __init__(self, events, *, error=None):
        self._events = events
        self._error = error

    def update_weights(self, rollout_id=None):
        self._events.append(("update", rollout_id))
        if self._error is not None:
            raise self._error
        return "updated"


@pytest.fixture(autouse=True)
def _identity_ray_get(monkeypatch):
    monkeypatch.setattr(train_async.ray, "get", lambda value: value)


def test_weight_update_barrier_orders_pause_update_resume():
    events = []
    result = train_async._update_actor_weights_with_generation_barrier(
        _DriverRolloutManager(events),
        _ActorModel(events),
        rollout_id=7,
    )

    assert result == "updated"
    assert events == ["pause", ("update", 7), "resume"]


def test_weight_update_barrier_pause_failure_prevents_weight_mutation():
    events = []
    with pytest.raises(RuntimeError, match="drain failed"):
        train_async._update_actor_weights_with_generation_barrier(
            _DriverRolloutManager(
                events,
                pause_error=RuntimeError("drain failed"),
            ),
            _ActorModel(events),
            rollout_id=7,
        )

    assert events == ["pause"]


def test_weight_update_barrier_update_failure_is_fail_closed():
    events = []
    with pytest.raises(RuntimeError, match="sync failed"):
        train_async._update_actor_weights_with_generation_barrier(
            _DriverRolloutManager(events),
            _ActorModel(events, error=RuntimeError("sync failed")),
            rollout_id=7,
        )

    # Resuming here could admit traffic while SGLang is still paused or only
    # partially updated, so the lifecycle intentionally remains fail-closed.
    assert events == ["pause", ("update", 7)]


def test_weight_update_barrier_without_custom_hook_preserves_legacy_update():
    events = []
    result = train_async._update_actor_weights_with_generation_barrier(
        _DriverRolloutManager(events, pause_result=False),
        _ActorModel(events),
        rollout_id=None,
    )

    assert result == "updated"
    assert events == ["pause", ("update", None)]


def _bare_rollout_manager(generate_rollout):
    manager_cls = rollout_module.RolloutManager.__ray_metadata__.modified_class
    manager = object.__new__(manager_cls)
    manager.args = SimpleNamespace(marker="args")
    manager.generate_rollout = generate_rollout
    manager._weight_update_generation_paused = False
    return manager


def test_rollout_manager_without_lifecycle_hooks_is_backward_compatible():
    def generate_rollout(*_args, **_kwargs):
        raise AssertionError("generation is not part of this test")

    manager = _bare_rollout_manager(generate_rollout)

    assert manager.pause_generation_for_weight_update() is False
    assert manager.resume_generation_after_weight_update() is False
    assert manager._weight_update_generation_paused is False


def test_rollout_manager_requires_paired_lifecycle_hooks():
    def generate_rollout(*_args, **_kwargs):
        raise AssertionError("generation is not part of this test")

    generate_rollout.pause_for_weight_update = lambda _args: None
    manager = _bare_rollout_manager(generate_rollout)

    with pytest.raises(RuntimeError, match="must both be callable or both be absent"):
        manager.pause_generation_for_weight_update()
    assert manager._weight_update_generation_paused is False


def test_rollout_manager_tracks_successful_pause_and_resume():
    events = []

    def generate_rollout(*_args, **_kwargs):
        raise AssertionError("generation is not part of this test")

    generate_rollout.pause_for_weight_update = (
        lambda args: events.append(("hook-pause", args.marker))
    )
    generate_rollout.resume_after_weight_update = (
        lambda args: events.append(("hook-resume", args.marker))
    )
    manager = _bare_rollout_manager(generate_rollout)

    assert manager.pause_generation_for_weight_update() is True
    assert manager._weight_update_generation_paused is True
    assert manager.resume_generation_after_weight_update() is True
    assert manager._weight_update_generation_paused is False
    assert events == [("hook-pause", "args"), ("hook-resume", "args")]


def test_rollout_manager_resume_failure_keeps_fail_closed_state():
    def generate_rollout(*_args, **_kwargs):
        raise AssertionError("generation is not part of this test")

    generate_rollout.pause_for_weight_update = lambda _args: None
    generate_rollout.resume_after_weight_update = lambda _args: (_ for _ in ()).throw(
        RuntimeError("resume failed")
    )
    manager = _bare_rollout_manager(generate_rollout)

    assert manager.pause_generation_for_weight_update() is True
    with pytest.raises(RuntimeError, match="resume failed"):
        manager.resume_generation_after_weight_update()
    assert manager._weight_update_generation_paused is True
