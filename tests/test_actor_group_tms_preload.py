from pathlib import Path

import pytest

from slime.ray.actor_group import _configure_tms_preload_env


def _fake_tms_install(tmp_path: Path, cuda_major: int) -> tuple[Path, Path]:
    binary = tmp_path / f"torch_memory_saver_hook_mode_preload_cu{cuda_major}.abi3.so"
    binary.touch()
    runtime_lib_dir = tmp_path / "nvidia" / f"cu{cuda_major}" / "lib"
    runtime_lib_dir.mkdir(parents=True)
    (runtime_lib_dir / f"libcudart.so.{cuda_major}").touch()
    return binary, runtime_lib_dir


def test_configure_tms_preload_uses_matching_cuda_runtime(tmp_path: Path) -> None:
    binary, runtime_lib_dir = _fake_tms_install(tmp_path, cuda_major=13)
    env = {"LD_LIBRARY_PATH": "/existing/one:/existing/two"}

    _configure_tms_preload_env(env, binary)

    assert env["LD_PRELOAD"] == str(binary.resolve())
    assert env["LD_LIBRARY_PATH"].split(":") == [
        str(runtime_lib_dir),
        "/existing/one",
        "/existing/two",
    ]
    assert env["TMS_INIT_ENABLE"] == "1"
    assert env["TMS_INIT_ENABLE_CPU_BACKUP"] == "1"


def test_configure_tms_preload_fails_before_ray_worker_retry_loop(tmp_path: Path) -> None:
    binary = tmp_path / "torch_memory_saver_hook_mode_preload_cu13.abi3.so"
    binary.touch()

    with pytest.raises(FileNotFoundError, match=r"libcudart\.so\.13"):
        _configure_tms_preload_env({}, binary)


def test_configure_tms_preload_resolves_production_binary(monkeypatch, tmp_path: Path) -> None:
    binary, runtime_lib_dir = _fake_tms_install(tmp_path, cuda_major=13)
    monkeypatch.setattr(
        "torch_memory_saver.utils.get_binary_path_from_package",
        lambda stem: binary if stem == "torch_memory_saver_hook_mode_preload" else None,
    )
    env = {}

    _configure_tms_preload_env(env)

    assert env["LD_PRELOAD"] == str(binary.resolve())
    assert env["LD_LIBRARY_PATH"].split(":")[0] == str(runtime_lib_dir)


def test_configure_tms_preload_supports_binary_inside_package(tmp_path: Path) -> None:
    package_dir = tmp_path / "site-packages" / "torch_memory_saver"
    package_dir.mkdir(parents=True)
    binary = package_dir / "torch_memory_saver_hook_mode_preload_cu13.abi3.so"
    binary.touch()
    runtime_lib_dir = tmp_path / "site-packages" / "nvidia" / "cu13" / "lib"
    runtime_lib_dir.mkdir(parents=True)
    (runtime_lib_dir / "libcudart.so.13").touch()
    env = {}

    _configure_tms_preload_env(env, binary)

    assert env["LD_LIBRARY_PATH"].split(":")[0] == str(runtime_lib_dir)
