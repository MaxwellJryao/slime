from argparse import Namespace

import pytest

from _actor_import_helpers import import_actor_module

actor_module = import_actor_module()
from slime.ray import actor_group as actor_group_module
from slime.ray.actor_group import RayTrainGroup
from slime.utils import logging_utils

NUM_GPUS = 0


class _RemoteMethod:
    def __init__(self, ref, events, index):
        self._ref = ref
        self._events = events
        self._index = index

    def remote(self):
        self._events.append(("submit", self._index))
        return self._ref


class _TrainerHandler:
    def __init__(self, ref, events, index):
        self.finish_tracking = _RemoteMethod(ref, events, index)


class _TrackingModel:
    def __init__(self, events, role, *, fail=False):
        self._events = events
        self._role = role
        self._fail = fail

    def finish_tracking(self):
        self._events.append(self._role)
        if self._fail:
            raise RuntimeError(f"{self._role} failed")


@pytest.mark.unit
@pytest.mark.parametrize("is_main_rank", [False, True])
def test_megatron_tracking_writer_is_owned_only_by_logging_rank(monkeypatch, is_main_rank):
    trainer = object.__new__(actor_module.MegatronTrainRayActor)
    args = Namespace(use_wandb=True, wandb_run_id="shared-run")
    init_calls = []
    monkeypatch.setattr(actor_module, "is_megatron_main_rank", lambda: is_main_rank)
    monkeypatch.setattr(
        actor_module,
        "init_tracking",
        lambda observed_args, *, primary, role: init_calls.append((observed_args, primary, role)),
    )

    trainer._init_tracking_writer(args, "actor")

    assert trainer._tracking_writer_initialized is is_main_rank
    assert init_calls == ([(args, False, "actor")] if is_main_rank else [])


@pytest.mark.unit
def test_megatron_tracking_finish_is_owner_only_and_idempotent(monkeypatch):
    trainer = object.__new__(actor_module.MegatronTrainRayActor)
    trainer.args = Namespace(use_wandb=True, wandb_run_id="shared-run")
    trainer._tracking_writer_initialized = True
    finish_calls = []
    monkeypatch.setattr(
        actor_module,
        "finish_tracking_client",
        lambda observed_args, *, raise_on_error: finish_calls.append((observed_args, raise_on_error)),
    )

    assert trainer.finish_tracking() is True
    assert trainer.finish_tracking() is False
    assert finish_calls == [(trainer.args, True)]
    assert trainer._tracking_writer_initialized is False


@pytest.mark.unit
def test_primary_tracking_finish_forwards_nonzero_exit_code(monkeypatch):
    args = Namespace(use_wandb=True)
    observed = []
    monkeypatch.setattr(
        logging_utils,
        "wandb",
        Namespace(
            run=object(),
            finish=lambda *, exit_code: observed.append(exit_code),
        ),
    )

    logging_utils.finish_tracking(args, exit_code=17)

    assert observed == [17]


@pytest.mark.unit
def test_distributed_tracking_closes_all_secondaries_before_primary(
    monkeypatch,
):
    events = []
    args = Namespace()
    monkeypatch.setattr(
        logging_utils,
        "finish_tracking",
        lambda observed_args, *, raise_on_error, exit_code: events.append(
            ("primary", observed_args, raise_on_error, exit_code)
        ),
    )

    logging_utils.finish_distributed_tracking(
        args,
        _TrackingModel(events, "actor"),
        _TrackingModel(events, "critic"),
        finish_rollout_tracking=lambda: events.append("rollout"),
        raise_on_primary_error=True,
    )

    assert events == [
        "actor",
        "critic",
        "rollout",
        ("primary", args, True, 0),
    ]


@pytest.mark.unit
def test_distributed_tracking_isolates_trainer_failure(monkeypatch):
    events = []
    monkeypatch.setattr(
        logging_utils,
        "finish_tracking",
        lambda _args, *, raise_on_error, exit_code: events.append(("primary", exit_code)),
    )

    logging_utils.finish_distributed_tracking(
        Namespace(),
        _TrackingModel(events, "actor", fail=True),
        _TrackingModel(events, "critic"),
        finish_rollout_tracking=lambda: events.append("rollout"),
    )

    assert events == ["actor", "critic", "rollout", ("primary", 0)]


@pytest.mark.unit
def test_distributed_tracking_preserves_rollout_failure_after_primary(
    monkeypatch,
):
    events = []
    monkeypatch.setattr(
        logging_utils,
        "finish_tracking",
        lambda _args, *, raise_on_error, exit_code: events.append(("primary", exit_code)),
    )

    def fail_rollout():
        events.append("rollout")
        raise RuntimeError("rollout teardown failed")

    with pytest.raises(RuntimeError, match="rollout teardown failed"):
        logging_utils.finish_distributed_tracking(
            Namespace(),
            _TrackingModel(events, "actor"),
            None,
            finish_rollout_tracking=fail_rollout,
        )

    assert events == ["actor", "rollout", ("primary", 1)]


@pytest.mark.unit
def test_distributed_tracking_does_not_mask_rollout_failure_with_primary_failure(
    monkeypatch,
):
    monkeypatch.setattr(
        logging_utils,
        "finish_tracking",
        lambda _args, *, raise_on_error, exit_code: ((_ for _ in ()).throw(RuntimeError("primary teardown failed"))),
    )

    def fail_rollout():
        raise ValueError("rollout teardown failed")

    with pytest.raises(ValueError, match="rollout teardown failed"):
        logging_utils.finish_distributed_tracking(
            Namespace(),
            None,
            None,
            finish_rollout_tracking=fail_rollout,
            raise_on_primary_error=True,
        )


@pytest.mark.unit
def test_ray_train_group_waits_for_every_rank_with_bounded_timeout(
    monkeypatch,
):
    events = []
    refs = [object(), object()]
    group = object.__new__(RayTrainGroup)
    group.args = Namespace(use_wandb=True)
    group.role = "actor"
    group._actor_handlers = [_TrainerHandler(ref, events, index) for index, ref in enumerate(refs)]
    observed = []
    monkeypatch.setenv("SLIME_TRAINER_TRACKING_FINISH_TIMEOUT_SECONDS", "7.5")
    monkeypatch.setattr(
        actor_group_module.ray,
        "get",
        lambda actual_refs, *, timeout: (observed.append((actual_refs, timeout)) or [True, False]),
    )

    assert group.finish_tracking() == [True, False]
    assert events == [("submit", 0), ("submit", 1)]
    assert observed == [(refs, 7.5)]


@pytest.mark.unit
def test_ray_train_group_timeout_terminates_terminal_gpu_actors(
    monkeypatch,
):
    events = []
    refs = [object(), object()]
    group = object.__new__(RayTrainGroup)
    group.args = Namespace(use_wandb=True)
    group.role = "critic"
    group._actor_handlers = [_TrainerHandler(ref, events, index) for index, ref in enumerate(refs)]
    monkeypatch.setenv("SLIME_TRAINER_TRACKING_FINISH_TIMEOUT_SECONDS", "9")

    def timeout_get(actual_refs, *, timeout):
        assert actual_refs == refs
        assert timeout == 9.0
        raise actor_group_module.ray.exceptions.GetTimeoutError("timed out")

    killed = []
    monkeypatch.setattr(actor_group_module.ray, "get", timeout_get)
    monkeypatch.setattr(
        actor_group_module.ray,
        "kill",
        lambda actor, *, no_restart: killed.append((actor, no_restart)),
    )

    assert group.finish_tracking() is None
    assert killed == [
        (group._actor_handlers[0], True),
        (group._actor_handlers[1], True),
    ]


@pytest.mark.unit
@pytest.mark.parametrize(
    ("configured", "expected"),
    [
        ("invalid", 45.0),
        ("nan", 45.0),
        ("0", 45.0),
        ("999", 120.0),
    ],
)
def test_trainer_tracking_finish_timeout_is_finite_and_capped(monkeypatch, configured, expected):
    monkeypatch.setenv("SLIME_TRAINER_TRACKING_FINISH_TIMEOUT_SECONDS", configured)

    assert actor_group_module._trainer_tracking_finish_timeout_seconds() == expected


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
