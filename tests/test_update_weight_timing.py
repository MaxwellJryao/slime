from __future__ import annotations

from types import SimpleNamespace

import pytest

import slime.ray.actor_group as actor_group_module
from slime.ray.actor_group import RayTrainGroup

NUM_GPUS = 0


class _RemoteCall:
    def __init__(self, value):
        self.value = value

    def remote(self):
        return self.value


class _Actor:
    def __init__(self, value):
        self.update_weights = _RemoteCall(value)


class _SaveRemoteCall:
    def remote(self, rollout_id, *, force_sync=False):
        assert rollout_id == 3
        assert force_sync is True
        return {"timing/sleep_time": 0.5}


class _SaveActor:
    save_model = _SaveRemoteCall()


def _group() -> RayTrainGroup:
    group = object.__new__(RayTrainGroup)
    group.args = SimpleNamespace(
        wandb_always_use_train_step=True,
        rollout_batch_size=16,
        n_samples_per_prompt=8,
        global_batch_size=64,
    )
    group._actor_handlers = [
        _Actor({"timing/delta_encode_time": 1.25}),
        _Actor({}),
    ]
    return group


@pytest.mark.unit
def test_update_weights_logs_at_producing_train_step(monkeypatch):
    group = _group()
    times = iter((10.0, 12.5))
    logged = []

    monkeypatch.setattr(actor_group_module, "perf_counter", lambda: next(times))
    monkeypatch.setattr(actor_group_module.ray, "get", lambda refs: refs)
    monkeypatch.setattr(
        actor_group_module.logging_utils,
        "log",
        lambda args, metrics, *, step_key: logged.append((dict(metrics), step_key)),
    )

    group.update_weights(rollout_id=3)

    assert logged == [
        (
            {
                "timing/update_weights_time": 2.5,
                "timing/delta_encode_time": 1.25,
                # 16*8/64 = two optimizer steps per rollout; rollout 3 owns
                # train steps 6 and 7, so its completed sync belongs to 7.
                "rollout/step": 7,
                "train/step": 7,
            },
            "train/step",
        )
    ]


@pytest.mark.unit
def test_initial_weight_seed_is_not_attached_to_train_step(monkeypatch):
    group = _group()
    times = iter((1.0, 4.0))
    logged = []

    monkeypatch.setattr(actor_group_module, "perf_counter", lambda: next(times))
    monkeypatch.setattr(actor_group_module.ray, "get", lambda refs: refs)
    monkeypatch.setattr(
        actor_group_module.logging_utils,
        "log",
        lambda *args, **kwargs: logged.append((args, kwargs)),
    )

    group.update_weights()

    assert logged == []


@pytest.mark.unit
def test_save_model_time_is_logged_at_producing_train_step(monkeypatch):
    group = _group()
    group._actor_handlers = [_SaveActor()]
    times = iter((20.0, 23.0))
    logged = []

    monkeypatch.setattr(actor_group_module, "perf_counter", lambda: next(times))
    monkeypatch.setattr(actor_group_module.ray, "get", lambda refs: refs)
    monkeypatch.setattr(
        actor_group_module.logging_utils,
        "log",
        lambda args, metrics, *, step_key: logged.append((dict(metrics), step_key)),
    )

    result = group.save_model(rollout_id=3, force_sync=True)

    assert result == [{"timing/sleep_time": 0.5}]
    assert logged == [
        (
            {
                "timing/save_model_time": 3.0,
                "timing/sleep_time": 0.5,
                "rollout/step": 7,
                "train/step": 7,
            },
            "train/step",
        )
    ]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
