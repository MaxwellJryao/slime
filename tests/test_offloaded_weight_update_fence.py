from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace

import pytest

NUM_GPUS = 0


class _RemoteCall:
    def __init__(self, function):
        self._function = function

    def remote(self, *args, **kwargs):
        return self._function(*args, **kwargs)


class _RolloutManager:
    def __init__(self, events: list[str]):
        self.get_updatable_engines_and_lock = _RemoteCall(lambda: ([object()], object(), 0, [1], [0]))
        self.clear_updatable_num_new_engines = _RemoteCall(lambda: events.append("clear_new_engines"))


class _WeightUpdater:
    def __init__(self, events: list[str]):
        self._events = events
        self.weight_version = 0

    def connect_rollout_engines(self, *args, **kwargs) -> None:
        self._events.append("connect")

    def update_weights(self) -> None:
        self._events.append("update")

    def pop_metrics(self) -> dict[str, float]:
        return {}


class _WeightsBackuper:
    def __init__(self, events: list[str]):
        self._events = events

    def copy(self, *, src_tag: str, dst_tag: str) -> None:
        assert (src_tag, dst_tag) == ("rollout_actor", "old_actor")
        self._events.append("backup_old_actor")

    def backup(self, tag: str) -> None:
        assert tag == "rollout_actor"
        self._events.append("backup_rollout_actor")


class _Timer:
    def log_dict(self) -> dict[str, float]:
        return {}

    def reset(self, name: str) -> None:
        raise AssertionError(f"unexpected timer reset: {name}")


def _actor(
    actor_module,
    events: list[str],
    *,
    offload_train: bool,
    use_critic: bool,
    keep_old_actor: bool = False,
):
    actor = object.__new__(actor_module.MegatronTrainRayActor)
    actor.args = SimpleNamespace(
        ci_test=False,
        colocate=False,
        debug_rollout_only=False,
        debug_train_only=False,
        keep_old_actor=keep_old_actor,
        offload_train=offload_train,
        update_weights_interval=1,
        use_critic=use_critic,
        use_fault_tolerance=False,
    )
    actor.rollout_manager = _RolloutManager(events)
    actor.weight_updater = _WeightUpdater(events)
    actor.weights_backuper = _WeightsBackuper(events)
    actor.wake_up = lambda: events.append("wake_up")
    actor.sleep = lambda: events.append("sleep")
    return actor


def _patch_runtime(monkeypatch, actor_module, events: list[str]) -> None:
    @contextmanager
    def disable():
        events.append("disable_enter")
        try:
            yield
        finally:
            events.append("disable_exit")

    monkeypatch.setattr(actor_module, "torch_memory_saver", SimpleNamespace(disable=disable))
    monkeypatch.setattr(actor_module.torch.cuda, "synchronize", lambda: events.append("cuda_sync"))
    monkeypatch.setattr(actor_module.ray, "get", lambda value: value)
    monkeypatch.setattr(actor_module.dist, "barrier", lambda *args, **kwargs: events.append("barrier"))
    monkeypatch.setattr(actor_module.dist, "get_rank", lambda: 0)
    monkeypatch.setattr(actor_module, "get_gloo_group", lambda: object())
    monkeypatch.setattr(actor_module, "print_memory", lambda label: events.append(label))
    monkeypatch.setattr(actor_module, "Timer", _Timer)


@pytest.mark.unit
def test_offloaded_ppo_fences_temporary_pool_before_exit_and_sleep(monkeypatch) -> None:
    from slime.backends.megatron_utils import actor as actor_module

    events: list[str] = []
    _patch_runtime(monkeypatch, actor_module, events)
    actor = _actor(
        actor_module,
        events,
        offload_train=True,
        use_critic=True,
        keep_old_actor=True,
    )

    actor.update_weights()

    assert events == [
        "wake_up",
        "connect",
        "barrier",
        "clear_new_engines",
        "disable_enter",
        "before update_weights",
        "update",
        "after update_weights",
        "backup_old_actor",
        "backup_rollout_actor",
        "cuda_sync",
        "disable_exit",
        "sleep",
    ]


@pytest.mark.unit
def test_non_offloaded_weight_update_does_not_add_cuda_fence(monkeypatch) -> None:
    from slime.backends.megatron_utils import actor as actor_module

    events: list[str] = []
    _patch_runtime(monkeypatch, actor_module, events)
    actor = _actor(actor_module, events, offload_train=False, use_critic=False)

    actor.update_weights()

    assert events == [
        "before update_weights",
        "update",
        "after update_weights",
    ]
