"""Minimal import stubs for CPU tests of ``MegatronTrainRayActor`` methods.

The CPU CI environment intentionally omits Megatron and torch-memory-saver.
Tests using this helper exercise actor methods that do not call the stubbed
training modules; the stubs only let Python construct the actor class.
"""

from __future__ import annotations

import sys
import types
from types import SimpleNamespace

import _cp_dist_helpers  # noqa: F401  # Install the shared Megatron MPU stub.


def _stub_module(name: str, **attributes) -> None:
    module = types.ModuleType(name)
    for attribute_name, value in attributes.items():
        setattr(module, attribute_name, value)
    sys.modules.setdefault(name, module)


def _unused(*_args, **_kwargs):
    return None


class _Unused:
    pass


def install_actor_import_stubs() -> None:
    _stub_module("torch_memory_saver", torch_memory_saver=SimpleNamespace())
    _stub_module(
        "megatron.core.packed_seq_params",
        PackedSeqParams=_Unused,
    )
    _stub_module(
        "slime.backends.megatron_utils.checkpoint",
        is_release_checkpoint=_unused,
        load_checkpoint=_unused,
    )
    _stub_module(
        "slime.backends.megatron_utils.hf_checkpoint_saver",
        save_hf_model_to_path=_unused,
    )
    _stub_module(
        "slime.backends.megatron_utils.initialize",
        init=_unused,
        is_megatron_main_rank=_unused,
    )
    _stub_module(
        "slime.backends.megatron_utils.logprob_guard",
        enforce_train_rollout_logprob_abs_diff=_unused,
    )
    _stub_module(
        "slime.backends.megatron_utils.loss",
        compute_advantages_and_returns=_unused,
        get_log_probs_and_entropy=_unused,
        get_values=_unused,
    )
    _stub_module(
        "slime.backends.megatron_utils.model",
        forward_only=_unused,
        initialize_model_and_optimizer=_unused,
        save=_unused,
        train=_unused,
    )
    _stub_module(
        "slime.backends.megatron_utils.update_weight.common",
        named_params_and_buffers=_unused,
    )
    _stub_module(
        "slime.backends.megatron_utils.update_weight.update_weight_from_disk",
        UpdateWeightFromDisk=_Unused,
    )
    _stub_module(
        "slime.backends.megatron_utils.update_weight.update_weight_from_distributed",
        UpdateWeightFromDistributed=_Unused,
    )
    _stub_module(
        "slime.backends.megatron_utils.update_weight.update_weight_from_tensor",
        UpdateWeightFromTensor=_Unused,
    )
