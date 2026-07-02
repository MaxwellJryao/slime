import sys
from types import SimpleNamespace

import pytest
import requests

from slime.backends.sglang_utils import sglang_engine

NUM_GPUS = 0


class _FakeSession:
    def __init__(self, get):
        self.get = get

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False


class _FakeClock:
    def __init__(self):
        self.now = 0.0

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


def test_startup_timeout_env_preserves_long_default_and_allows_override(monkeypatch):
    monkeypatch.delenv(sglang_engine._SGLANG_STARTUP_TIMEOUT_ENV, raising=False)
    assert (
        sglang_engine._sglang_startup_timeout_seconds()
        == sglang_engine._DEFAULT_SGLANG_STARTUP_TIMEOUT_SECONDS
    )

    monkeypatch.setenv(sglang_engine._SGLANG_STARTUP_TIMEOUT_ENV, "42.5")
    assert sglang_engine._sglang_startup_timeout_seconds() == 42.5


@pytest.mark.parametrize("value", ["not-a-number", "0", "-1", "nan", "inf"])
def test_invalid_startup_timeout_env_uses_default(monkeypatch, value):
    monkeypatch.setenv(sglang_engine._SGLANG_STARTUP_TIMEOUT_ENV, value)
    assert (
        sglang_engine._sglang_startup_timeout_seconds()
        == sglang_engine._DEFAULT_SGLANG_STARTUP_TIMEOUT_SECONDS
    )


def test_wait_server_healthy_has_bounded_deadline_and_request_timeout(monkeypatch):
    clock = _FakeClock()
    request_timeouts = []

    def get(_url, *, headers, timeout):
        assert headers["Authorization"] == "Bearer secret"
        request_timeouts.append(timeout)
        raise requests.ConnectionError("not ready")

    monkeypatch.setattr(sglang_engine.requests, "Session", lambda: _FakeSession(get))
    monkeypatch.setattr(sglang_engine.time, "monotonic", clock.monotonic)
    monkeypatch.setattr(sglang_engine.time, "sleep", clock.sleep)

    with pytest.raises(TimeoutError, match=r"http://engine.*within 1s"):
        sglang_engine._wait_server_healthy(
            "http://engine",
            "secret",
            lambda: True,
            startup_timeout_seconds=1,
            health_request_timeout_seconds=10,
            poll_interval_seconds=0.25,
        )

    assert request_timeouts
    assert all(0 < timeout <= 1 for timeout in request_timeouts)
    assert clock.now == 1


def test_wait_server_healthy_reports_early_process_exit(monkeypatch):
    def get(_url, *, headers, timeout):
        return SimpleNamespace(status_code=503)

    monkeypatch.setattr(sglang_engine.requests, "Session", lambda: _FakeSession(get))

    with pytest.raises(RuntimeError, match="terminated unexpectedly"):
        sglang_engine._wait_server_healthy(
            "http://engine",
            "secret",
            lambda: False,
            startup_timeout_seconds=10,
            poll_interval_seconds=0,
        )


def test_launch_server_process_terminates_tree_and_reaps_on_startup_failure(monkeypatch):
    class FakeProcess:
        pid = 1234

        def __init__(self):
            self.alive = True
            self.started = False
            self.join_timeouts = []

        def start(self):
            self.started = True

        def is_alive(self):
            return self.alive

        def join(self, timeout):
            self.join_timeouts.append(timeout)

        def kill(self):
            self.alive = False

    process = FakeProcess()
    killed_pids = []

    def kill_process_tree(pid):
        killed_pids.append(pid)
        process.alive = False

    server_args = SimpleNamespace(
        encoder_only=False,
        host="127.0.0.1",
        node_rank=0,
        api_key="secret",
        url=lambda: "http://127.0.0.1:30000",
    )
    monkeypatch.setattr(sglang_engine.multiprocessing, "set_start_method", lambda *args, **kwargs: None)
    monkeypatch.setattr(sglang_engine.multiprocessing, "Process", lambda *args, **kwargs: process)
    # Keep this process-lifecycle unit test runnable on CPU-only hosts.  Importing
    # the real HTTP server eagerly initializes FlashInfer's CUDA runtime even
    # though the fake multiprocessing.Process never invokes its target.
    monkeypatch.setitem(
        sys.modules,
        "sglang.srt.entrypoints.http_server",
        SimpleNamespace(launch_server=lambda _server_args: None),
    )
    monkeypatch.setattr(sglang_engine, "kill_process_tree", kill_process_tree)
    monkeypatch.setattr(
        sglang_engine,
        "_wait_server_healthy",
        lambda **kwargs: (_ for _ in ()).throw(TimeoutError("startup timed out")),
    )

    with pytest.raises(TimeoutError, match="startup timed out"):
        sglang_engine.launch_server_process(server_args)

    assert process.started
    assert killed_pids == [1234]
    assert process.join_timeouts == [10]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
