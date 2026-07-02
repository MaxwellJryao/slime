from types import SimpleNamespace

import pytest

import train

NUM_GPUS = 0


class _RemoteMethod:
    def __init__(self, func):
        self._func = func

    def remote(self, *args, **kwargs):
        return self._func(*args, **kwargs)


class _RolloutManager:
    def __init__(self, events):
        self.generate = _RemoteMethod(
            lambda rollout_id: events.append(("generate", rollout_id)) or "data"
        )
        self.commit_rollout_metrics = _RemoteMethod(
            lambda rollout_id: events.append(("metrics-commit", rollout_id))
        )
        self.eval = _RemoteMethod(
            lambda rollout_id, completed_train_batch=True: events.append(
                ("eval", rollout_id, completed_train_batch)
            )
        )
        self.dispose = _RemoteMethod(lambda: events.append(("dispose", None)))


class _ActorModel:
    def __init__(self, events, *, fail: bool = False):
        self._events = events
        self._fail = fail

    def update_weights(self, rollout_id=None):
        self._events.append(("weights", rollout_id))

    def async_train(self, rollout_id, _data, external_data=None):
        assert external_data is None
        self._events.append(("train", rollout_id))
        if self._fail:
            raise RuntimeError("actor failed")
        return "trained"

    def clear_memory(self):
        self._events.append(("clear-memory", None))


def _args() -> SimpleNamespace:
    return SimpleNamespace(
        offload_rollout=False,
        offload_train=False,
        check_weight_update_equal=False,
        num_rollout=1,
        eval_interval=None,
        start_rollout_id=0,
        skip_eval_before_train=True,
        eval_resumed_checkpoint_before_train=False,
        use_critic=False,
        num_critic_only_steps=0,
        save_interval=None,
        rollout_global_dataset=False,
    )


def _install_stubs(monkeypatch, events, *, fail_actor: bool = False):
    rollout_manager = _RolloutManager(events)
    actor_model = _ActorModel(events, fail=fail_actor)
    monkeypatch.setattr(train, "configure_logger", lambda: None)
    monkeypatch.setattr(train, "create_placement_groups", lambda _args: {"rollout": object()})
    monkeypatch.setattr(
        train,
        "create_rollout_manager",
        lambda _args, _pg: (rollout_manager, 1),
    )
    monkeypatch.setattr(
        train,
        "create_training_models",
        lambda _args, _pgs, _manager: (actor_model, None),
    )
    monkeypatch.setattr(train, "init_tracking", lambda _args: None)
    monkeypatch.setattr(train, "finish_tracking", lambda _args: None)
    monkeypatch.setattr(train, "should_run_periodic_action", lambda *_args, **_kwargs: False)
    monkeypatch.setattr(train.ray, "get", lambda value, **_kwargs: value)
    return rollout_manager


def test_sync_trainer_commits_rollout_metrics_after_actor_success(monkeypatch):
    events = []
    _install_stubs(monkeypatch, events)

    train.train(_args())

    names = [event[0] for event in events]
    assert names.index("generate") < names.index("train")
    assert names.index("train") < names.index("metrics-commit")
    assert names.index("metrics-commit") < names.index("dispose")
    assert [event for event in events if event[0] == "weights"] == [
        ("weights", None),
        ("weights", 0),
    ]


def test_sync_trainer_does_not_commit_failed_actor_batch(monkeypatch):
    events = []
    _install_stubs(monkeypatch, events, fail_actor=True)

    with pytest.raises(RuntimeError, match="actor failed"):
        train.train(_args())

    assert ("metrics-commit", 0) not in events


def test_sync_trainer_evaluates_resumed_checkpoint_before_generation(monkeypatch):
    events = []
    _install_stubs(monkeypatch, events)
    args = _args()
    args.num_rollout = 42
    args.start_rollout_id = 40
    args.eval_interval = 10
    args.skip_eval_before_train = False
    args.eval_resumed_checkpoint_before_train = True

    train.train(args)

    initial_sync = events.index(("weights", None))
    resumed_eval = events.index(("eval", 39, True))
    first_generate = events.index(("generate", 40))
    assert initial_sync < resumed_eval < first_generate


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("graceful_exit_at_unix_time", 123),
        ("graceful_exit_at_unix_time", 0),
        ("training_complete_marker", "/tmp/complete"),
        ("final_eval_complete_marker", "/tmp/eval-complete"),
        ("final_eval_data_sha256", "a" * 64),
        ("concurrent_pretrain_eval", True),
    ],
)
def test_sync_trainer_rejects_async_only_lifecycle_arguments(name, value):
    args = _args()
    setattr(args, name, value)

    with pytest.raises(ValueError, match=name):
        train.train(args)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
