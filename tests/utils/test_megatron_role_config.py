"""Unit tests for Megatron role config parsing and application."""

import sys
import tempfile
from argparse import Namespace
from types import ModuleType

import pytest
import yaml

_sglang_arguments_stub = ModuleType("slime.backends.sglang_utils.arguments")
_sglang_arguments_stub.sglang_parse_args = lambda: None
_sglang_arguments_stub.validate_args = lambda _args: None
sys.modules.setdefault(
    "slime.backends.sglang_utils.arguments",
    _sglang_arguments_stub,
)

from slime.utils.arguments import parse_megatron_role_args  # noqa: E402

if sys.modules.get("slime.backends.sglang_utils.arguments") is _sglang_arguments_stub:
    sys.modules.pop("slime.backends.sglang_utils.arguments")

NUM_GPUS = 0


def _write_yaml(data: dict) -> str:
    handle = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False)
    yaml.dump(data, handle)
    handle.flush()
    return handle.name


def _base_args(**overrides):
    args = dict(
        lr=2e-6,
        tensor_model_parallel_size=1,
        kl_coef=0.1,
        use_kl_loss=False,
        use_opd=True,
        opd_type="megatron",
        custom_advantage_function_path="slime.test.adv",
        untie_embeddings_and_output_weights=False,
        actor_num_nodes=1,
        actor_num_gpus_per_node=1,
        critic_num_nodes=1,
        critic_num_gpus_per_node=1,
        use_critic=False,
        megatron_config_path=None,
        only_train_params_name_list=None,
        freeze_params_name_list=None,
        start_rollout_id=None,
        rollout_global_dataset=False,
    )
    args.update(overrides)
    return Namespace(**args)


class TestMegatronRoleConfig:
    def test_parse_actor_and_critic_role_overrides(self):
        path = _write_yaml(
            {
                "megatron": [
                    {
                        "name": "default",
                        "role": "critic",
                        "overrides": {"lr": "1e-5", "tensor_model_parallel_size": 2},
                    },
                    {"name": "default", "role": "actor", "overrides": {"lr": "1e-6", "tensor_model_parallel_size": 4}},
                ]
            }
        )
        args = _base_args()

        actor_args = parse_megatron_role_args(args, path, role="actor")
        critic_args = parse_megatron_role_args(args, path, role="critic")

        assert actor_args.lr == 1e-6
        assert actor_args.tensor_model_parallel_size == 4
        assert actor_args.kl_coef == args.kl_coef
        assert actor_args.use_opd is args.use_opd

        assert critic_args.lr == 1e-5
        assert critic_args.tensor_model_parallel_size == 2
        assert critic_args.kl_coef == 0
        assert critic_args.use_opd is False
        assert critic_args.custom_advantage_function_path is None
        assert critic_args.untie_embeddings_and_output_weights is True

    def test_missing_role_inherits_base_args(self):
        path = _write_yaml(
            {
                "megatron": [
                    {"name": "default", "role": "actor", "overrides": {"lr": "1e-6"}},
                ]
            }
        )
        args = _base_args()

        critic_args = parse_megatron_role_args(args, path, role="critic")

        assert critic_args is not args
        assert critic_args.lr == args.lr
        assert critic_args.kl_coef == 0
        assert critic_args.use_opd is False

    def test_role_override_preserves_valid_parameter_pattern_list(self):
        path = _write_yaml(
            {
                "megatron": [
                    {
                        "name": "default",
                        "role": "critic",
                        "overrides": {
                            "freeze_params_name_list": [
                                r"(?:^|\.)self_attention(?:\.|$)",
                            ]
                        },
                    },
                ]
            }
        )

        critic_args = parse_megatron_role_args(_base_args(), path, role="critic")

        assert critic_args.freeze_params_name_list == [
            r"(?:^|\.)self_attention(?:\.|$)",
        ]

    @pytest.mark.parametrize(
        "field_name",
        ["only_train_params_name_list", "freeze_params_name_list"],
    )
    @pytest.mark.parametrize("invalid_value", ["attention", [], [""], [1]])
    def test_role_override_rejects_invalid_parameter_pattern_lists(
        self,
        field_name,
        invalid_value,
    ):
        path = _write_yaml(
            {
                "megatron": [
                    {
                        "name": "default",
                        "role": "critic",
                        "overrides": {field_name: invalid_value},
                    },
                ]
            }
        )

        with pytest.raises(ValueError, match=f"critic {field_name} must be a non-empty list"):
            parse_megatron_role_args(_base_args(), path, role="critic")

    def test_role_override_rechecks_mutual_exclusion_after_yaml(self):
        path = _write_yaml(
            {
                "megatron": [
                    {
                        "name": "default",
                        "role": "critic",
                        "overrides": {
                            "freeze_params_name_list": [r"(?:^|\.)embedding(?:\.|$)"],
                        },
                    },
                ]
            }
        )
        args = _base_args(
            only_train_params_name_list=[r"(?:^|\.)decoder(?:\.|$)"],
        )

        with pytest.raises(ValueError, match="cannot set both"):
            parse_megatron_role_args(args, path, role="critic")

    @pytest.mark.parametrize(
        "config",
        [
            {"critic": [{"name": "default", "overrides": {"lr": "1e-5"}}]},
            {"lr": "1e-5"},
            {},
        ],
    )
    def test_requires_top_level_megatron_key(self, config):
        path = _write_yaml(config)
        args = _base_args()

        with pytest.raises(AssertionError, match="top-level 'megatron' list"):
            parse_megatron_role_args(args, path, role="critic")

    def test_create_training_models_applies_actor_override_without_critic(self, monkeypatch):
        from slime.ray import placement_group as placement_group_module

        path = _write_yaml(
            {
                "megatron": [
                    {"name": "default", "role": "actor", "overrides": {"lr": "1e-6"}},
                ]
            }
        )
        args = _base_args(megatron_config_path=path, use_critic=False)

        class DummyModel:
            def __init__(self, model_args):
                self.args = model_args
                self.init_calls = []
                self.rollout_manager = None

            def async_init(self, model_args, role, with_ref=False, with_opd_teacher=False):
                self.args = model_args
                self.init_calls.append(
                    {
                        "args": model_args,
                        "role": role,
                        "with_ref": with_ref,
                        "with_opd_teacher": with_opd_teacher,
                    }
                )
                return [7]

            def set_rollout_manager(self, rollout_manager):
                self.rollout_manager = rollout_manager

        def fake_allocate_train_group(args, num_nodes, num_gpus_per_node, pg, role="actor"):
            return DummyModel(args)

        monkeypatch.setattr(placement_group_module, "allocate_train_group", fake_allocate_train_group)
        monkeypatch.setattr(placement_group_module.ray, "get", lambda value: value)

        actor_model, critic_model = placement_group_module.create_training_models(
            args,
            {"actor": None, "critic": None},
            object(),
        )

        assert critic_model is None
        assert actor_model.args.lr == 1e-6
        assert actor_model.init_calls[0]["args"].lr == 1e-6
        assert actor_model.init_calls[0]["role"] == "actor"
        assert args.start_rollout_id == 7


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
