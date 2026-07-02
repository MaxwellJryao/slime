import threading

import pytest

import train_async
from slime.ray import rollout as rollout_module
from slime.ray.rollout import RolloutManager

NUM_GPUS = 0


@pytest.mark.parametrize("raw_timeout", ["nan", "inf", "-inf"])
def test_non_finite_dispose_timeout_uses_default(
    monkeypatch, caplog, raw_timeout
):
    dispose_ref = object()
    observed_timeouts = []

    class DisposeMethod:
        def remote(self):
            return dispose_ref

    class RolloutManagerStub:
        dispose = DisposeMethod()

    def get_dispose(ref, *, timeout):
        assert ref is dispose_ref
        observed_timeouts.append(timeout)

    monkeypatch.setenv("SLIME_DISPOSE_TIMEOUT_SECONDS", raw_timeout)
    monkeypatch.setattr(train_async.ray, "get", get_dispose)

    with caplog.at_level("WARNING"):
        train_async._dispose_rollout_manager(RolloutManagerStub())

    assert observed_timeouts == [60.0]
    assert (
        f"Invalid SLIME_DISPOSE_TIMEOUT_SECONDS={raw_timeout!r}; using 60s"
        in caplog.text
    )


def test_rollout_manager_disposes_custom_rollout_before_tracking(monkeypatch):
    events = []

    def generate_rollout(*_args, **_kwargs):
        raise AssertionError("not called")

    generate_rollout.dispose = lambda: events.append("rollout")

    manager_class = RolloutManager.__ray_metadata__.modified_class
    manager = object.__new__(manager_class)
    manager.generate_rollout = generate_rollout
    manager._health_monitors = []
    manager.args = object()
    monkeypatch.setattr(
        rollout_module.logging_utils,
        "finish_tracking",
        lambda _args: events.append("tracking"),
    )

    manager.dispose()

    assert events == ["rollout", "tracking"]


def _bare_rollout_manager():
    manager_class = RolloutManager.__ray_metadata__.modified_class
    manager = object.__new__(manager_class)
    manager._pretrain_eval_executor = None
    manager._pretrain_eval_future = None
    return manager


def test_pretrain_eval_tail_overlaps_first_actor_batch():
    manager = _bare_rollout_manager()
    eval_started = threading.Event()
    release_eval = threading.Event()

    def run_eval(_rollout_id, *, completed_train_batch):
        assert completed_train_batch is False
        eval_started.set()
        assert release_eval.wait(timeout=5)

    def generate(_rollout_id):
        assert eval_started.wait(timeout=5)
        return "rollout-0"

    manager.eval = run_eval
    manager.generate = generate

    # Rollout 0 is handed to the actor while the baseline tail is still live.
    assert manager.generate_with_pretrain_eval(0) == "rollout-0"
    assert manager._pretrain_eval_future is not None
    assert not manager._pretrain_eval_future.done()

    release_eval.set()
    manager.wait_pretrain_eval()
    assert manager._pretrain_eval_future is None
    assert manager._pretrain_eval_executor is None


def test_pretrain_eval_failure_surfaces_at_weight_sync_barrier():
    manager = _bare_rollout_manager()
    manager.generate = lambda _rollout_id: "rollout-0"

    def fail_eval(_rollout_id, *, completed_train_batch):
        assert completed_train_batch is False
        raise RuntimeError("baseline failed")

    manager.eval = fail_eval

    assert manager.generate_with_pretrain_eval(0) == "rollout-0"
    with pytest.raises(RuntimeError, match="baseline failed"):
        manager.wait_pretrain_eval()
    assert manager._pretrain_eval_future is None
    assert manager._pretrain_eval_executor is None


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
