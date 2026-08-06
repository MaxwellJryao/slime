"""Minimal import stubs for CPU tests of ``MegatronTrainRayActor`` methods.

The CPU CI environment intentionally omits Megatron and torch-memory-saver.
Tests using this helper exercise actor methods that do not call the stubbed
training modules; the stubs only let Python construct the actor class.
"""

from __future__ import annotations

import importlib
import sys
import types
from types import SimpleNamespace

import _cp_dist_helpers  # noqa: F401  # Install the shared Megatron MPU stub.


_MISSING = object()


def _stub_module(name: str, **attributes) -> types.ModuleType | None:
    if name in sys.modules:
        return None
    module = types.ModuleType(name)
    for attribute_name, value in attributes.items():
        setattr(module, attribute_name, value)
    sys.modules[name] = module
    return module


def _unused(*_args, **_kwargs):
    return None


class _Unused:
    pass


def import_actor_module():
    """Import the actor with temporary dependency stubs, then remove them.

    The actor keeps direct references to the symbols it imported. Removing the
    stub modules afterward prevents unrelated tests from receiving a fake
    ``loss`` or ``model`` module through ``sys.modules`` or package attributes.
    """

    actor_module_name = "slime.backends.megatron_utils.actor"
    if actor_module_name in sys.modules:
        return sys.modules[actor_module_name]

    inserted = {}

    def insert(name, **attributes):
        module = _stub_module(name, **attributes)
        if module is not None:
            parent_name, _, attribute_name = name.rpartition(".")
            parent = sys.modules.get(parent_name)
            previous_attribute = getattr(parent, attribute_name, _MISSING) if parent is not None else _MISSING
            inserted[name] = (module, previous_attribute)

    insert("torch_memory_saver", torch_memory_saver=SimpleNamespace())
    insert(
        "megatron.core.packed_seq_params",
        PackedSeqParams=_Unused,
    )
    insert(
        "slime.backends.megatron_utils.checkpoint",
        is_release_checkpoint=_unused,
        load_checkpoint=_unused,
    )
    insert(
        "slime.backends.megatron_utils.hf_checkpoint_saver",
        save_hf_model_to_path=_unused,
    )
    insert(
        "slime.backends.megatron_utils.initialize",
        init=_unused,
        is_megatron_main_rank=_unused,
    )
    insert(
        "slime.backends.megatron_utils.logprob_guard",
        enforce_train_rollout_logprob_abs_diff=_unused,
    )
    insert(
        "slime.backends.megatron_utils.loss",
        compute_advantages_and_returns=_unused,
        get_log_probs_and_entropy=_unused,
        get_values=_unused,
    )
    insert(
        "slime.backends.megatron_utils.model",
        forward_only=_unused,
        initialize_model_and_optimizer=_unused,
        save=_unused,
        train=_unused,
    )
    insert(
        "slime.backends.megatron_utils.update_weight.common",
        named_params_and_buffers=_unused,
    )
    insert(
        "slime.backends.megatron_utils.update_weight.update_weight_from_disk",
        UpdateWeightFromDisk=_Unused,
    )
    insert(
        "slime.backends.megatron_utils.update_weight.update_weight_from_distributed",
        UpdateWeightFromDistributed=_Unused,
    )
    insert(
        "slime.backends.megatron_utils.update_weight.update_weight_from_tensor",
        UpdateWeightFromTensor=_Unused,
    )

    try:
        return importlib.import_module(actor_module_name)
    finally:
        for name, (module, previous_attribute) in reversed(inserted.items()):
            if sys.modules.get(name) is module:
                sys.modules.pop(name)
            parent_name, _, attribute_name = name.rpartition(".")
            parent = sys.modules.get(parent_name)
            if parent is None or getattr(parent, attribute_name, _MISSING) is not module:
                continue
            if previous_attribute is _MISSING:
                vars(parent).pop(attribute_name, None)
            else:
                setattr(parent, attribute_name, previous_attribute)
