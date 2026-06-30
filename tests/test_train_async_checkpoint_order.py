from types import SimpleNamespace

import pytest

import train_async
from slime.utils.training_lifecycle import write_final_eval_complete_marker


class _RemoteMethod:
    def __init__(self, func):
        self._func = func

    def remote(self, *args, **kwargs):
        return self._func(*args, **kwargs)


class _RolloutManager:
    def __init__(self, events):
        self.ready = _RemoteMethod(
            lambda: (
                events.append(("ready-submit", None))
                or {"timing/startup_sglang_router_engines_ready_time": 1.0}
            )
        )
        self.generate = _RemoteMethod(
            lambda rollout_id: events.append(("generate", rollout_id)) or "data"
        )
        self.generate_with_pretrain_eval = _RemoteMethod(
            lambda rollout_id: (
                events.extend(
                    [
                        ("generate", rollout_id),
                        ("eval", rollout_id, False),
                    ]
                )
                or "data"
            )
        )
        self.save = _RemoteMethod(
            lambda rollout_id: events.append(("state-save", rollout_id))
        )
        self.eval = _RemoteMethod(
            lambda rollout_id, completed_train_batch=True: (
                events.append(("eval", rollout_id, completed_train_batch))
                or {
                    "eval/tmax_holdout/reward_mean": 0.5,
                    "eval/train_step": rollout_id + int(completed_train_batch),
                }
            )
        )
        self.wait_pretrain_eval = _RemoteMethod(
            lambda: events.append(("pretrain-eval-wait", None))
        )
        self.commit_rollout_metrics = _RemoteMethod(
            lambda rollout_id: events.append(("metrics-commit", rollout_id))
        )
        self.dispose = _RemoteMethod(lambda: events.append(("dispose", None)))


class _ActorModel:
    def __init__(self, events):
        self._events = events

    def update_weights(self, rollout_id=None):
        self._events.append(("weights", rollout_id))

    def async_train(self, rollout_id, _data, external_data=None):
        assert external_data is None
        self._events.append(("train", rollout_id))
        return "trained"

    def save_model(self, rollout_id, force_sync=False):
        self._events.append(("model-save", rollout_id, force_sync))


def test_rollout_state_precedes_training_and_model_commit(monkeypatch, tmp_path):
    events = []
    rollout_manager = _RolloutManager(events)
    actor_model = _ActorModel(events)
    args = SimpleNamespace(
        colocate=False,
        check_weight_update_equal=False,
        graceful_exit_at_unix_time=None,
        training_complete_marker=None,
        final_eval_complete_marker=str(tmp_path / "FINAL_EVAL_COMPLETE"),
        final_eval_data_sha256="a" * 64,
        start_rollout_id=0,
        num_rollout=1,
        use_critic=False,
        num_critic_only_steps=0,
        save_interval=1,
        rollout_global_dataset=True,
        update_weights_interval=2,
        eval_interval=1_000_000,
        skip_eval_before_train=False,
        concurrent_pretrain_eval=True,
        rollout_batch_size=1,
        n_samples_per_prompt=1,
        global_batch_size=1,
        wandb_always_use_train_step=True,
        use_wandb=False,
        use_tensorboard=False,
    )

    monkeypatch.setattr(train_async, "configure_logger", lambda: None)
    monkeypatch.setattr(
        train_async, "create_placement_groups", lambda _args: {"rollout": object()}
    )
    monkeypatch.setattr(
        train_async,
        "create_rollout_manager",
        lambda _args, _pg, *, wait_ready: (rollout_manager, 1),
    )
    monkeypatch.setattr(
        train_async,
        "create_training_models",
        lambda _args, _pgs, _manager, *, connect_rollout_manager: (
            events.append(("trainer-init", connect_rollout_manager))
            or (actor_model, None)
        ),
    )
    monkeypatch.setattr(
        train_async,
        "connect_training_models_to_rollout",
        lambda *_args: events.append(("trainer-rollout-wire", None)),
    )
    monkeypatch.setattr(train_async, "init_tracking", lambda _args: None)
    monkeypatch.setattr(
        train_async,
        "finish_tracking",
        lambda _args, *, raise_on_error=False: events.append(
            ("primary-finish", raise_on_error)
        ),
    )
    monkeypatch.setattr(
        train_async.logging_utils,
        "log",
        lambda _args, metrics, *, step_key: events.append(
            ("primary-log", step_key, dict(metrics))
        ),
    )
    monkeypatch.setattr(
        train_async,
        "write_final_eval_complete_marker",
        lambda *_args, **_kwargs: events.append(("final-eval-marker", None)),
    )
    monkeypatch.setattr(
        train_async,
        "write_training_complete_marker",
        lambda *_args, **_kwargs: events.append(("training-marker", None)),
    )
    monkeypatch.setattr(train_async.ray, "get", lambda value, **_kwargs: value)

    train_async.train(args)

    names = [event[0] for event in events]
    assert names.index("ready-submit") < names.index("trainer-init")
    assert names.index("trainer-init") < names.index("trainer-rollout-wire")
    assert names.index("generate") < names.index("state-save")
    assert names.index("state-save") < names.index("train")
    assert names.index("train") < names.index("pretrain-eval-wait")
    assert names.index("pretrain-eval-wait") < names.index("metrics-commit")
    assert names.index("train") < names.index("metrics-commit")
    assert names.index("metrics-commit") < names.index("model-save")
    assert names.index("train") < names.index("model-save")
    assert [event for event in events if event[0] == "weights"] == [
        ("weights", None),
        ("weights", 0),
    ]
    assert [event for event in events if event[0] == "eval"] == [
        ("eval", 0, False),
        ("eval", 0, True),
    ]
    final_weight_sync_index = max(
        index for index, event in enumerate(events) if event == ("weights", 0)
    )
    assert names.index("pretrain-eval-wait") < final_weight_sync_index
    final_eval_index = max(
        index for index, event in enumerate(events) if event == ("eval", 0, True)
    )
    assert final_weight_sync_index < final_eval_index
    final_eval_event_index = names.index("eval", names.index("eval") + 1)
    final_log_index = next(
        index
        for index, event in enumerate(events)
        if event[0] == "primary-log" and event[1] == "eval/train_step"
    )
    checked_finish_index = events.index(("primary-finish", True))
    assert final_eval_event_index < final_log_index
    assert final_log_index < checked_finish_index
    assert checked_finish_index < names.index("final-eval-marker")
    assert names.index("final-eval-marker") < names.index("training-marker")


def _resume_args(tmp_path, **overrides):
    values = dict(
        colocate=False,
        check_weight_update_equal=False,
        graceful_exit_at_unix_time=None,
        training_complete_marker=str(tmp_path / "TRAINING_COMPLETE"),
        final_eval_complete_marker=str(tmp_path / "FINAL_EVAL_COMPLETE"),
        final_eval_data_sha256="b" * 64,
        start_rollout_id=1,
        num_rollout=1,
        use_critic=False,
        num_critic_only_steps=0,
        save_interval=1,
        rollout_global_dataset=True,
        update_weights_interval=1,
        eval_interval=1_000_000,
        skip_eval_before_train=False,
        concurrent_pretrain_eval=True,
        rollout_batch_size=1,
        n_samples_per_prompt=1,
        global_batch_size=1,
        wandb_always_use_train_step=True,
        use_wandb=False,
        use_tensorboard=False,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def _patch_resume_runtime(monkeypatch, events, rollout_manager, actor_model):
    monkeypatch.setattr(train_async, "configure_logger", lambda: None)
    monkeypatch.setattr(
        train_async, "create_placement_groups", lambda _args: {"rollout": object()}
    )
    monkeypatch.setattr(
        train_async,
        "create_rollout_manager",
        lambda _args, _pg, *, wait_ready: (rollout_manager, 1),
    )
    monkeypatch.setattr(
        train_async,
        "create_training_models",
        lambda _args, _pgs, _manager, *, connect_rollout_manager: (
            actor_model,
            None,
        ),
    )
    monkeypatch.setattr(
        train_async,
        "connect_training_models_to_rollout",
        lambda *_args: events.append(("trainer-rollout-wire", None)),
    )
    monkeypatch.setattr(train_async, "init_tracking", lambda _args: None)
    monkeypatch.setattr(
        train_async,
        "finish_tracking",
        lambda _args, *, raise_on_error=False: None,
    )
    monkeypatch.setattr(train_async.ray, "get", lambda value, **_kwargs: value)


def test_final_checkpoint_resume_runs_only_missing_eval(monkeypatch, tmp_path):
    events = []
    rollout_manager = _RolloutManager(events)
    actor_model = _ActorModel(events)
    args = _resume_args(tmp_path)
    _patch_resume_runtime(monkeypatch, events, rollout_manager, actor_model)
    monkeypatch.setattr(
        train_async,
        "write_final_eval_complete_marker",
        lambda *_args, **_kwargs: events.append(("final-eval-marker", None)),
    )
    monkeypatch.setattr(
        train_async,
        "write_training_complete_marker",
        lambda *_args, **_kwargs: events.append(("training-marker", None)),
    )

    train_async.train(args)

    assert not [event for event in events if event[0] in {"generate", "train"}]
    assert [event for event in events if event[0] == "eval"] == [("eval", 0, True)]
    names = [event[0] for event in events]
    assert names.index("eval") < names.index("final-eval-marker")
    assert names.index("final-eval-marker") < names.index("training-marker")


def test_failed_final_eval_writes_no_completion_marker(monkeypatch, tmp_path):
    events = []
    rollout_manager = _RolloutManager(events)

    def fail_eval(rollout_id, completed_train_batch=True):
        events.append(("eval", rollout_id, completed_train_batch))
        raise RuntimeError("eval failed")

    rollout_manager.eval = _RemoteMethod(fail_eval)
    args = _resume_args(tmp_path)
    _patch_resume_runtime(monkeypatch, events, rollout_manager, _ActorModel(events))
    monkeypatch.setattr(
        train_async,
        "write_final_eval_complete_marker",
        lambda *_args, **_kwargs: events.append(("final-eval-marker", None)),
    )
    monkeypatch.setattr(
        train_async,
        "write_training_complete_marker",
        lambda *_args, **_kwargs: events.append(("training-marker", None)),
    )

    with pytest.raises(RuntimeError, match="eval failed"):
        train_async.train(args)

    assert ("final-eval-marker", None) not in events
    assert ("training-marker", None) not in events


def test_failed_primary_final_eval_flush_writes_no_completion_marker(
    monkeypatch, tmp_path
):
    events = []
    rollout_manager = _RolloutManager(events)
    args = _resume_args(tmp_path)
    _patch_resume_runtime(monkeypatch, events, rollout_manager, _ActorModel(events))
    monkeypatch.setattr(
        train_async,
        "finish_tracking",
        lambda _args, *, raise_on_error=False: (
            (_ for _ in ()).throw(RuntimeError("primary flush failed"))
            if raise_on_error
            else None
        ),
    )
    monkeypatch.setattr(
        train_async,
        "write_final_eval_complete_marker",
        lambda *_args, **_kwargs: events.append(("final-eval-marker", None)),
    )
    monkeypatch.setattr(
        train_async,
        "write_training_complete_marker",
        lambda *_args, **_kwargs: events.append(("training-marker", None)),
    )

    with pytest.raises(RuntimeError, match="primary flush failed"):
        train_async.train(args)

    assert ("eval", 0, True) in events
    assert ("final-eval-marker", None) not in events
    assert ("training-marker", None) not in events


def test_valid_final_eval_marker_skips_repeated_eval(monkeypatch, tmp_path):
    events = []
    args = _resume_args(tmp_path)
    write_final_eval_complete_marker(
        args.final_eval_complete_marker,
        final_rollout_id=0,
        model_iteration=0,
        num_rollout=1,
        eval_data_sha256=args.final_eval_data_sha256,
    )
    rollout_manager = _RolloutManager(events)
    _patch_resume_runtime(monkeypatch, events, rollout_manager, _ActorModel(events))
    monkeypatch.setattr(
        train_async,
        "write_final_eval_complete_marker",
        lambda *_args, **_kwargs: pytest.fail("valid marker must not be rewritten"),
    )
    monkeypatch.setattr(
        train_async,
        "write_training_complete_marker",
        lambda *_args, **_kwargs: events.append(("training-marker", None)),
    )

    train_async.train(args)

    assert not [event for event in events if event[0] == "eval"]
    assert ("training-marker", None) in events


def test_overshot_checkpoint_is_not_mislabeled_as_final_eval(monkeypatch, tmp_path):
    events = []
    args = _resume_args(tmp_path, start_rollout_id=2)
    rollout_manager = _RolloutManager(events)
    _patch_resume_runtime(monkeypatch, events, rollout_manager, _ActorModel(events))

    with pytest.raises(
        RuntimeError, match="beyond the configured final rollout boundary"
    ):
        train_async.train(args)

    assert not [event for event in events if event[0] == "eval"]


def test_dispose_timeout_terminates_rollout_manager(monkeypatch):
    events = []
    manager = _RolloutManager(events)

    def timeout_get(value, **kwargs):
        assert value is None
        assert kwargs["timeout"] == 12.0
        raise train_async.ray.exceptions.GetTimeoutError("timed out")

    monkeypatch.setenv("SLIME_DISPOSE_TIMEOUT_SECONDS", "12")
    monkeypatch.setattr(train_async.ray, "get", timeout_get)
    monkeypatch.setattr(
        train_async.ray,
        "kill",
        lambda actor, *, no_restart: events.append(("kill", actor, no_restart)),
    )

    train_async._dispose_rollout_manager(manager)

    assert events[0] == ("dispose", None)
    assert events[1] == ("kill", manager, True)


def test_failed_actor_batch_never_commits_prefetched_metrics(monkeypatch):
    events = []
    rollout_manager = _RolloutManager(events)

    class _FailingActor(_ActorModel):
        def async_train(self, rollout_id, _data, external_data=None):
            del external_data
            self._events.append(("train", rollout_id))
            raise RuntimeError("actor failed")

    args = SimpleNamespace(
        colocate=False,
        check_weight_update_equal=False,
        graceful_exit_at_unix_time=None,
        training_complete_marker=None,
        start_rollout_id=0,
        num_rollout=1,
        use_critic=False,
        num_critic_only_steps=0,
        save_interval=None,
        rollout_global_dataset=False,
        update_weights_interval=1,
        eval_interval=None,
        rollout_batch_size=1,
        n_samples_per_prompt=1,
        global_batch_size=1,
        wandb_always_use_train_step=True,
        use_wandb=False,
        use_tensorboard=False,
    )
    monkeypatch.setattr(train_async, "configure_logger", lambda: None)
    monkeypatch.setattr(
        train_async, "create_placement_groups", lambda _args: {"rollout": object()}
    )
    monkeypatch.setattr(
        train_async,
        "create_rollout_manager",
        lambda _args, _pg, *, wait_ready: (rollout_manager, 1),
    )
    monkeypatch.setattr(
        train_async,
        "create_training_models",
        lambda _args, _pgs, _manager, *, connect_rollout_manager: (
            _FailingActor(events),
            None,
        ),
    )
    monkeypatch.setattr(
        train_async,
        "connect_training_models_to_rollout",
        lambda *_args: events.append(("trainer-rollout-wire", None)),
    )
    monkeypatch.setattr(train_async, "init_tracking", lambda _args: None)
    monkeypatch.setattr(train_async.ray, "get", lambda value, **_kwargs: value)

    with pytest.raises(RuntimeError, match="actor failed"):
        train_async.train(args)

    assert ("metrics-commit", 0) not in events
