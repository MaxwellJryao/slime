import logging
import pickle
import sys
import types
from unittest.mock import Mock

import pytest

from slime.backends.megatron_utils.compat import (
    ensure_checkpoint_enum_compat,
    warn_if_numpy_compatibility_is_unverified,
)


NUM_GPUS = 0


def test_checkpoint_enum_compat_installs_missing_capability() -> None:
    transformer_enums = types.ModuleType("fake_megatron.transformer.enums")

    assert ensure_checkpoint_enum_compat(transformer_enums) is True
    scope = transformer_enums.InferenceCudaGraphScope
    assert [member.name for member in scope] == ["none", "layer", "block"]
    assert [member.value for member in scope] == [1, 2, 3]
    assert scope.__module__ == transformer_enums.__name__


def test_checkpoint_enum_compat_supports_pickle_round_trip(monkeypatch) -> None:
    transformer_enums = sys.modules[__name__]
    monkeypatch.delattr(
        transformer_enums,
        "InferenceCudaGraphScope",
        raising=False,
    )

    assert ensure_checkpoint_enum_compat(transformer_enums) is True
    scope = transformer_enums.InferenceCudaGraphScope
    assert pickle.loads(pickle.dumps(scope.block)) is scope.block


def test_checkpoint_enum_compat_preserves_native_megatron_enum() -> None:
    native_scope = object()
    transformer_enums = types.ModuleType("fake_megatron.transformer.enums")
    transformer_enums.InferenceCudaGraphScope = native_scope

    assert ensure_checkpoint_enum_compat(transformer_enums) is False
    assert transformer_enums.InferenceCudaGraphScope is native_scope


def test_numpy_compatibility_warning_is_version_gated() -> None:
    logger = Mock(spec=logging.Logger)

    assert warn_if_numpy_compatibility_is_unverified("1.26.4", logger) is False
    logger.warning.assert_not_called()

    assert warn_if_numpy_compatibility_is_unverified("2.2.1", logger) is True
    logger.warning.assert_called_once()
    assert logger.warning.call_args.args[-1] == "2.2.1"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
