from pathlib import Path
from types import SimpleNamespace

import pytest

from slime.ray import rollout as rollout_module

NUM_GPUS = 0


class _FakeRemoteMethod:
    def __init__(self, host: str) -> None:
        self.host = host

    def remote(self, *, start_port: int = 10000, consecutive: int = 1):
        del consecutive
        return self.host, start_port


class _FakeNodeIpRemoteMethod:
    def __init__(self, host: str) -> None:
        self.host = host

    def remote(self) -> str:
        return self.host


class _FakeEngine:
    def __init__(self, host: str) -> None:
        self._get_current_node_ip = _FakeNodeIpRemoteMethod(host)
        self._get_current_node_ip_and_free_port = _FakeRemoteMethod(host)


def _allocate_ports(monkeypatch, hosts: list[str]):
    monkeypatch.setenv("SLIME_EPHEMERAL_PORT_LOWER_BOUND", "9000")
    monkeypatch.setattr(rollout_module.ray, "get", lambda value: value)
    args = SimpleNamespace(
        num_gpus_per_node=8,
        rollout_num_gpus_per_engine=1,
        sglang_dp_size=1,
    )
    engines = [(rank, _FakeEngine(host)) for rank, host in enumerate(hosts)]
    return rollout_module._allocate_rollout_engine_addr_and_ports_normal(
        args=args,
        rollout_engines=engines,
        rank_offset=0,
        base_port=2048,
    )


def test_default_rollout_port_block_stays_below_ephemeral_range(monkeypatch) -> None:
    monkeypatch.delenv("SLIME_ROLLOUT_BASE_PORT", raising=False)
    monkeypatch.setenv("SLIME_EPHEMERAL_PORT_LOWER_BOUND", "9000")

    base = rollout_module._get_rollout_base_port()

    assert base == 2048
    assert 8 * 2 + 8 * (30 + 1) == 264
    assert rollout_module._ROLLOUT_PORT_BLOCK_SIZE == 320
    assert base + rollout_module._ROLLOUT_PORT_BLOCK_SIZE <= 9000


@pytest.mark.parametrize("base", [2047, 8681, 9000, 20000])
def test_rollout_port_block_rejects_unsafe_explicit_base(monkeypatch, base: int) -> None:
    monkeypatch.setenv("SLIME_EPHEMERAL_PORT_LOWER_BOUND", "9000")
    monkeypatch.setenv("SLIME_ROLLOUT_BASE_PORT", str(base))

    with pytest.raises(ValueError, match="320-port block"):
        rollout_module._get_rollout_base_port()


def test_rollout_port_block_accepts_last_safe_base(monkeypatch) -> None:
    monkeypatch.setenv("SLIME_EPHEMERAL_PORT_LOWER_BOUND", "9000")
    monkeypatch.setenv("SLIME_ROLLOUT_BASE_PORT", "8680")

    assert rollout_module._get_rollout_base_port() == 8680


@pytest.mark.parametrize(
    ("port", "consecutive"),
    [(2048, 1), (8680, 320), (8999, 1)],
)
def test_allocated_rollout_port_range_accepts_non_ephemeral_ports(
    port: int,
    consecutive: int,
) -> None:
    rollout_module._validate_allocated_rollout_port_range(port, consecutive, 9000)


@pytest.mark.parametrize(
    ("port", "consecutive"),
    [(2047, 1), (8999, 2), (9000, 1)],
)
def test_allocated_rollout_port_range_rejects_scan_into_ephemeral_ports(
    port: int,
    consecutive: int,
) -> None:
    with pytest.raises(RuntimeError, match="outside the non-ephemeral range"):
        rollout_module._validate_allocated_rollout_port_range(port, consecutive, 9000)


def test_rollout_port_validation_reads_kernel_range_file(monkeypatch, tmp_path: Path) -> None:
    port_range = tmp_path / "ip_local_port_range"
    port_range.write_text("9000 65000\n")
    monkeypatch.delenv("SLIME_EPHEMERAL_PORT_LOWER_BOUND", raising=False)
    monkeypatch.setenv("SLIME_IP_LOCAL_PORT_RANGE_PATH", str(port_range))

    assert rollout_module._get_ephemeral_port_lower_bound() == 9000


def test_rollout_port_validation_rejects_invalid_ephemeral_override(monkeypatch) -> None:
    monkeypatch.setenv("SLIME_EPHEMERAL_PORT_LOWER_BOUND", "not-a-port")

    with pytest.raises(ValueError, match="invalid ephemeral port lower bound"):
        rollout_module._get_ephemeral_port_lower_bound()


def test_full_node_rollout_port_mapping_is_unchanged(monkeypatch) -> None:
    addresses, cursors = _allocate_ports(
        monkeypatch,
        ["10.0.0.1"] * 8 + ["10.0.0.2"] * 8 + ["10.0.0.3"] * 8,
    )

    assert addresses[0] == {
        "host": "10.0.0.1",
        "port": 2048,
        "nccl_port": 2049,
        "dist_init_addr": "10.0.0.1:2064",
    }
    assert addresses[7] == {
        "host": "10.0.0.1",
        "port": 2062,
        "nccl_port": 2063,
        "dist_init_addr": "10.0.0.1:2281",
    }
    assert addresses[8] == {
        "host": "10.0.0.2",
        "port": 2048,
        "nccl_port": 2049,
        "dist_init_addr": "10.0.0.2:2064",
    }
    assert cursors == {
        "10.0.0.1": 2312,
        "10.0.0.2": 2312,
        "10.0.0.3": 2312,
    }


def test_fractional_actor_offset_uses_each_rollout_engines_actual_node(monkeypatch) -> None:
    # Four actor GPUs consume node A GPUs 0--3. The 28 rollout engines then
    # occupy A:4--7 followed by three complete eight-GPU nodes.
    expected_hosts = ["10.0.0.1"] * 4 + ["10.0.0.2"] * 8 + ["10.0.0.3"] * 8 + ["10.0.0.4"] * 8
    addresses, cursors = _allocate_ports(monkeypatch, expected_hosts)

    assert [addresses[rank]["host"] for rank in range(28)] == expected_hosts
    assert addresses[3]["port"] == 2054
    assert addresses[4]["host"] == "10.0.0.2"
    assert addresses[4]["port"] == 2048
    assert addresses[12]["host"] == "10.0.0.3"
    assert addresses[20]["host"] == "10.0.0.4"
    assert cursors == {
        "10.0.0.1": 2180,
        "10.0.0.2": 2312,
        "10.0.0.3": 2312,
        "10.0.0.4": 2312,
    }


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
