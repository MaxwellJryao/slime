"""Small compatibility shims for version-skewed Megatron environments."""

from __future__ import annotations

import enum
import logging
from types import ModuleType


def ensure_checkpoint_enum_compat(transformer_enums: ModuleType | None = None) -> bool:
    """Register a checkpoint enum absent from older Megatron releases.

    Returns ``True`` only when the compatibility enum was installed.  Passing a
    module explicitly keeps the capability check unit-testable without loading
    Megatron or initializing CUDA.
    """

    if transformer_enums is None:
        from megatron.core.transformer import enums as transformer_enums

    if hasattr(transformer_enums, "InferenceCudaGraphScope"):
        return False

    inference_scope = enum.Enum(
        "InferenceCudaGraphScope",
        {"none": 1, "layer": 2, "block": 3},
        module=transformer_enums.__name__,
        qualname="InferenceCudaGraphScope",
    )
    transformer_enums.InferenceCudaGraphScope = inference_scope
    return True


def warn_if_numpy_compatibility_is_unverified(
    version: str,
    logger: logging.Logger,
) -> bool:
    """Warn when running beyond Megatron's historically asserted NumPy 1.x."""

    if version.startswith("1."):
        return False
    logger.warning(
        "running Megatron initialization with NumPy %s; this compatibility path "
        "must be revalidated when the pinned Megatron version changes",
        version,
    )
    return True
