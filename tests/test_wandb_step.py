from argparse import Namespace

import pytest

from slime.utils.metric_utils import set_wandb_step
from slime.utils import logging_utils, wandb_utils
from slime.utils.wandb_utils import (
    _DEFAULT_WANDB_FINISH_TIMEOUT_SECONDS,
    _compute_config_for_logging,
    _compute_secondary_config_for_logging,
    _init_wandb_common,
    _shared_writer_label,
    _wandb_run_name,
    _wandb_finish_timeout_seconds,
    define_logged_metric_axes,
)


NUM_GPUS = 0

_SPILOT_PAIR_ENV = {
    "SPILOT_SWEEP_ARM": "prctrl",
    "SPILOT_PROCESS_REWARD_PAIR_ID": "pair-20260717-r1",
    "SPILOT_PROCESS_REWARD_PAIR_ROLE": "control",
    "SPILOT_MATCHED_INVARIANTS_SHA256": "a" * 64,
    "SPILOT_PROCESS_REWARD_MODE": "terminal_broadcast",
}
_SPILOT_PAIR_CONFIG = {
    "spilot_sweep_arm": "prctrl",
    "spilot_process_reward_pair_id": "pair-20260717-r1",
    "spilot_process_reward_pair_role": "control",
    "spilot_matched_invariants_sha256": "a" * 64,
    "spilot_process_reward_mode": "terminal_broadcast",
}
_CREDENTIAL_ARG_NAMES = (
    "router_api_key",
    "router_control_plane_api_keys",
    "router_oracle_password",
    "sglang_admin_api_key",
    "sglang_api_key",
    "sglang_ssl_keyfile_password",
    "wandb_key",
)


def _args(*, always_use_train_step: bool) -> Namespace:
    return Namespace(
        wandb_always_use_train_step=always_use_train_step,
        rollout_batch_size=5,
        n_samples_per_prompt=8,
        global_batch_size=20,
    )


@pytest.mark.unit
def test_primary_wandb_config_includes_only_validated_spilot_metadata(monkeypatch):
    for name, value in _SPILOT_PAIR_ENV.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setenv("WANDB_API_KEY", "must-not-be-logged")
    monkeypatch.setenv("SPILOT_PROVIDER_API_KEY", "must-not-be-logged")

    config = _compute_config_for_logging(
        Namespace(use_critic=False, rank=0, wandb_key="argument-secret")
    )

    assert {key: config[key] for key in _SPILOT_PAIR_CONFIG} == _SPILOT_PAIR_CONFIG
    serialized = repr(config)
    assert "must-not-be-logged" not in serialized
    assert "WANDB_API_KEY" not in serialized
    assert "SPILOT_PROVIDER_API_KEY" not in serialized
    assert "argument-secret" not in serialized
    assert "wandb_key" not in config


@pytest.mark.unit
@pytest.mark.parametrize("role", [None, "actor", "critic"])
def test_secondary_wandb_config_includes_same_spilot_metadata(monkeypatch, role):
    for name, value in _SPILOT_PAIR_ENV.items():
        monkeypatch.setenv(name, value)

    config = _compute_secondary_config_for_logging(
        Namespace(rank=2, wandb_key="argument-secret"), role=role
    )

    assert {key: config[key] for key in _SPILOT_PAIR_CONFIG} == _SPILOT_PAIR_CONFIG
    if role == "critic":
        assert config["critic/rank"] == 2
        assert "rank" not in config
    else:
        assert config["rank"] == 2
    assert "argument-secret" not in repr(config)
    assert "wandb_key" not in config
    assert "critic/wandb_key" not in config


@pytest.mark.unit
def test_empty_spilot_metadata_is_omitted(monkeypatch):
    for name in _SPILOT_PAIR_ENV:
        monkeypatch.setenv(name, "  ")

    config = _compute_secondary_config_for_logging(Namespace(rank=1))

    assert not set(_SPILOT_PAIR_CONFIG).intersection(config)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"SPILOT_SWEEP_ARM": "bad arm"}, "safe experiment identifier"),
        ({"SPILOT_PROCESS_REWARD_PAIR_ROLE": "baseline"}, "control.*treatment"),
        ({"SPILOT_MATCHED_INVARIANTS_SHA256": "ABC"}, "64 lowercase"),
        ({"SPILOT_PROCESS_REWARD_MODE": "learned_prm"}, "terminal_broadcast.*cost_to_go"),
        ({"SPILOT_PROCESS_REWARD_PAIR_ID": ""}, "incomplete"),
    ],
)
def test_invalid_or_incomplete_spilot_metadata_fails_closed(
    monkeypatch, updates, message
):
    env = dict(_SPILOT_PAIR_ENV)
    env.update(updates)
    for name, value in env.items():
        monkeypatch.setenv(name, value)

    with pytest.raises(ValueError, match=message):
        _compute_config_for_logging(Namespace(use_critic=False))


@pytest.mark.unit
@pytest.mark.parametrize("credential_name", _CREDENTIAL_ARG_NAMES)
def test_credential_arguments_are_excluded_from_all_wandb_writer_configs(
    monkeypatch, credential_name
):
    for name, value in _SPILOT_PAIR_ENV.items():
        monkeypatch.setenv(name, value)
    args = Namespace(
        use_critic=False,
        input_key="prompt",
        reward_key="reward",
        max_tokens_per_gpu=8192,
        **{credential_name: "credential-must-not-be-logged"},
    )

    configs = (
        _compute_config_for_logging(args),
        _compute_secondary_config_for_logging(args, role="actor"),
        _compute_secondary_config_for_logging(args, role="critic"),
    )

    for config in configs:
        assert "credential-must-not-be-logged" not in repr(config)
        assert credential_name not in config
        assert f"critic/{credential_name}" not in config
    assert configs[0]["input_key"] == "prompt"
    assert configs[0]["reward_key"] == "reward"
    assert configs[0]["max_tokens_per_gpu"] == 8192


@pytest.mark.unit
def test_set_wandb_step_uses_rollout_axis_by_default():
    metrics = {"custom/reward_mean": 0.5}

    step_key = set_wandb_step(
        _args(always_use_train_step=False),
        metrics,
        rollout_id=3,
        default_step_key="rollout/step",
    )

    assert step_key == "rollout/step"
    assert metrics["rollout/step"] == 3
    assert "train/step" not in metrics


@pytest.mark.unit
def test_set_wandb_step_attaches_scaled_train_axis():
    metrics = {"custom/reward_mean": 0.5}

    step_key = set_wandb_step(
        _args(always_use_train_step=True),
        metrics,
        rollout_id=3,
        default_step_key="rollout/step",
    )

    assert step_key == "train/step"
    assert metrics["rollout/step"] == 6
    assert metrics["train/step"] == 6


@pytest.mark.unit
def test_set_wandb_step_aligns_eval_metrics_to_train_axis():
    metrics = {"eval/reward": 0.5}

    step_key = set_wandb_step(
        _args(always_use_train_step=True),
        metrics,
        rollout_id=4,
        default_step_key="eval/train_step",
        completed_train_batch=True,
    )

    assert step_key == "eval/train_step"
    assert metrics["eval/train_step"] == 9
    assert "train/step" not in metrics


@pytest.mark.unit
def test_pretrain_eval_metrics_start_at_train_step_zero():
    metrics = {"eval/reward": 0.5}

    step_key = set_wandb_step(
        _args(always_use_train_step=True),
        metrics,
        rollout_id=0,
        default_step_key="eval/train_step",
        completed_train_batch=False,
    )

    assert step_key == "eval/train_step"
    assert metrics["eval/train_step"] == 0
    assert "train/step" not in metrics


@pytest.mark.unit
def test_completed_rollout_metrics_use_last_train_step():
    metrics = {"custom/reward_mean": 0.5}

    step_key = set_wandb_step(
        _args(always_use_train_step=True),
        metrics,
        rollout_id=3,
        default_step_key="rollout/step",
        completed_train_batch=True,
    )

    assert step_key == "train/step"
    assert metrics["rollout/step"] == 7
    assert metrics["train/step"] == 7


@pytest.mark.unit
def test_logging_fails_fast_when_business_axis_is_missing(monkeypatch):
    args = _args(always_use_train_step=True)
    args.use_wandb = True
    args.use_tensorboard = False
    called = []
    monkeypatch.setattr(logging_utils.wandb, "log", lambda *_args, **_kwargs: called.append(True))

    with pytest.raises(KeyError, match="train/step"):
        logging_utils.log(args, {"custom/reward_mean": 0.5}, step_key="train/step")

    assert called == []


@pytest.mark.unit
def test_logging_rejects_non_train_axis_when_train_axis_is_required(monkeypatch):
    args = _args(always_use_train_step=True)
    args.use_wandb = True
    args.use_tensorboard = False
    called = []
    monkeypatch.setattr(logging_utils.wandb, "log", lambda *_args, **_kwargs: called.append(True))

    with pytest.raises(KeyError, match="requires business/timing records"):
        logging_utils.log(
            args,
            {"timing/rollout_time": 1.0, "rollout/step": 2},
            step_key="rollout/step",
        )

    assert called == []


@pytest.mark.unit
@pytest.mark.parametrize(
    ("always_use_train_step", "expected_rollout_axis", "expected_eval_axis"),
    [
        (False, "rollout/step", "eval/step"),
        (True, "train/step", "eval/train_step"),
    ],
)
def test_wandb_metric_definitions_use_explicit_axes(
    monkeypatch,
    always_use_train_step: bool,
    expected_rollout_axis: str,
    expected_eval_axis: str,
):
    definitions = []
    monkeypatch.delenv("GPU_MONITOR_PREFIX", raising=False)
    monkeypatch.delenv("GPU_MONITOR_NODE_ROLE", raising=False)
    monkeypatch.delenv("SLURM_NNODES", raising=False)
    monkeypatch.setattr(
        "slime.utils.wandb_utils.wandb.define_metric",
        lambda name, **kwargs: definitions.append((name, kwargs.get("step_metric"))),
    )

    _init_wandb_common(_args(always_use_train_step=always_use_train_step))

    assert definitions == [
        ("train/step", None),
        ("train/*", "train/step"),
        (
            "rollout/step",
            "train/step" if always_use_train_step else None,
        ),
        ("rollout/*", expected_rollout_axis),
        ("multi_turn/*", expected_rollout_axis),
        ("passrate/*", expected_rollout_axis),
        (
            "eval/train_step" if always_use_train_step else "eval/step",
            None,
        ),
        ("eval/*", expected_eval_axis),
        ("perf/*", expected_rollout_axis),
        ("timing/*", expected_rollout_axis),
        *(
            [("timing/eval/*", "eval/train_step")]
            if always_use_train_step
            else []
        ),
    ]
    assert all(name != "polar/*" for name, _ in definitions)


@pytest.mark.unit
def test_primary_predeclares_independent_gpu_node_axes(monkeypatch):
    definitions = []
    monkeypatch.setenv("GPU_MONITOR_PREFIX", "test_system")
    monkeypatch.setenv("GPU_MONITOR_NODE_ROLE", "rank")
    monkeypatch.setenv("SLURM_NNODES", "4")
    monkeypatch.setattr(
        "slime.utils.wandb_utils.wandb.define_metric",
        lambda name, **kwargs: definitions.append((name, kwargs.get("step_metric"))),
    )

    _init_wandb_common(_args(always_use_train_step=True))

    gpu_definitions = [item for item in definitions if item[0].startswith("test_system/")]
    assert gpu_definitions == [
        ("test_system/node_0/train_step", None),
        ("test_system/node_0/*", "test_system/node_0/train_step"),
        ("test_system/node_1/train_step", None),
        ("test_system/node_1/*", "test_system/node_1/train_step"),
        ("test_system/node_2/train_step", None),
        ("test_system/node_2/*", "test_system/node_2/train_step"),
        ("test_system/node_3/train_step", None),
        ("test_system/node_3/*", "test_system/node_3/train_step"),
    ]
    assert ("test_system/*", "train/step") not in definitions


@pytest.mark.unit
def test_primary_gpu_axes_follow_explicit_node_role(monkeypatch):
    definitions = []
    monkeypatch.setenv("GPU_MONITOR_PREFIX", "cluster_system")
    monkeypatch.setenv("GPU_MONITOR_NODE_ROLE", "worker")
    monkeypatch.setenv("SLURM_NNODES", "2")
    monkeypatch.setattr(
        "slime.utils.wandb_utils.wandb.define_metric",
        lambda name, **kwargs: definitions.append((name, kwargs.get("step_metric"))),
    )

    _init_wandb_common(_args(always_use_train_step=True))

    assert (
        "cluster_system/worker_node_0/*",
        "cluster_system/worker_node_0/train_step",
    ) in definitions
    assert (
        "cluster_system/worker_node_1/*",
        "cluster_system/worker_node_1/train_step",
    ) in definitions


@pytest.mark.unit
def test_primary_gpu_axes_infer_actor_and_rollout_nodes(monkeypatch):
    definitions = []
    args = _args(always_use_train_step=True)
    args.actor_num_nodes = 2
    monkeypatch.setenv("GPU_MONITOR_PREFIX", "cluster_system")
    monkeypatch.delenv("GPU_MONITOR_NODE_ROLE", raising=False)
    monkeypatch.setenv("SLURM_NNODES", "4")
    monkeypatch.setattr(
        "slime.utils.wandb_utils.wandb.define_metric",
        lambda name, **kwargs: definitions.append((name, kwargs.get("step_metric"))),
    )

    _init_wandb_common(args)

    assert (
        "cluster_system/actor_node_0/*",
        "cluster_system/actor_node_0/train_step",
    ) in definitions
    assert (
        "cluster_system/actor_node_1/*",
        "cluster_system/actor_node_1/train_step",
    ) in definitions
    assert (
        "cluster_system/rollout_node_2/*",
        "cluster_system/rollout_node_2/train_step",
    ) in definitions
    assert (
        "cluster_system/rollout_node_3/*",
        "cluster_system/rollout_node_3/train_step",
    ) in definitions


@pytest.mark.unit
def test_primary_wandb_protobuf_contains_all_axis_globs(monkeypatch, tmp_path):
    import wandb
    from wandb.proto import wandb_internal_pb2
    from wandb.sdk.internal.datastore import DataStore

    monkeypatch.setenv("WANDB_SILENT", "true")
    monkeypatch.setenv("GPU_MONITOR_PREFIX", "test_system")
    monkeypatch.setenv("GPU_MONITOR_NODE_ROLE", "rank")
    monkeypatch.setenv("SLURM_NNODES", "2")
    run = wandb.init(
        project="axis-protobuf-test",
        mode="offline",
        dir=str(tmp_path),
        settings=wandb.Settings(console="off", x_disable_stats=True),
    )
    _init_wandb_common(_args(always_use_train_step=True))
    run.finish()

    wandb_file = next(tmp_path.glob("wandb/*/run-*.wandb"))
    datastore = DataStore()
    datastore.open_for_scan(str(wandb_file))
    definitions = {}
    while (record_bytes := datastore.scan_data()) is not None:
        record = wandb_internal_pb2.Record()
        record.ParseFromString(record_bytes)
        if record.WhichOneof("record_type") != "metric":
            continue
        metric_name = record.metric.glob_name or record.metric.name
        definitions[metric_name] = record.metric.step_metric

    assert definitions["train/*"] == "train/step"
    assert definitions["eval/*"] == "eval/train_step"
    assert definitions["timing/*"] == "train/step"
    assert definitions["timing/eval/*"] == "eval/train_step"
    assert (
        definitions["test_system/node_0/*"]
        == "test_system/node_0/train_step"
    )
    assert (
        definitions["test_system/node_1/*"]
        == "test_system/node_1/train_step"
    )


@pytest.mark.unit
def test_every_concrete_metric_gets_an_exact_axis_definition(monkeypatch):
    definitions = []
    monkeypatch.setattr(
        "slime.utils.wandb_utils.wandb.define_metric",
        lambda name, **kwargs: definitions.append((name, kwargs.get("step_metric"))),
    )
    _init_wandb_common(_args(always_use_train_step=True))
    definitions.clear()

    business_metrics = {
        "train/step": 7,
        "custom/reward_mean": 0.5,
        "previously_unknown_namespace/value": 1.0,
        "_timestamp": 1234567890.0,
    }
    eval_metrics = {
        "eval/train_step": 0,
        "eval/tmax_holdout/reward_mean": 0.25,
        "eval/terminal_bench_2_1/reward_mean": 0.5,
        "eval/aggregate/reward_weighted_mean": 0.3677248677248677,
        "timing/eval/tmax_holdout/session_ms/e2e_mean": 123.0,
        "timing/eval/terminal_bench_2_1/session_ms/e2e_mean": 456.0,
    }
    define_logged_metric_axes(business_metrics, step_metric="train/step")
    define_logged_metric_axes(eval_metrics, step_metric="eval/train_step")
    # Repeated values must not generate an unbounded stream of definitions.
    define_logged_metric_axes(business_metrics, step_metric="train/step")
    define_logged_metric_axes(eval_metrics, step_metric="eval/train_step")

    assert definitions == [
        ("custom/reward_mean", "train/step"),
        ("previously_unknown_namespace/value", "train/step"),
        ("eval/tmax_holdout/reward_mean", "eval/train_step"),
        ("eval/terminal_bench_2_1/reward_mean", "eval/train_step"),
        ("eval/aggregate/reward_weighted_mean", "eval/train_step"),
        (
            "timing/eval/tmax_holdout/session_ms/e2e_mean",
            "eval/train_step",
        ),
        (
            "timing/eval/terminal_bench_2_1/session_ms/e2e_mean",
            "eval/train_step",
        ),
    ]


@pytest.mark.unit
def test_logging_defines_exact_axes_before_publishing(monkeypatch):
    args = _args(always_use_train_step=True)
    args.use_wandb = True
    args.use_tensorboard = False
    calls = []
    monkeypatch.setattr(
        logging_utils.wandb_utils,
        "define_logged_metric_axes",
        lambda metrics, *, step_metric: calls.append(
            ("define", tuple(metrics), step_metric)
        ),
    )
    monkeypatch.setattr(
        logging_utils.wandb,
        "log",
        lambda metrics: calls.append(("log", tuple(metrics))),
    )

    logging_utils.log(
        args,
        {"custom/reward_mean": 0.25, "train/step": 4},
        step_key="train/step",
    )

    assert calls == [
        ("define", ("custom/reward_mean", "train/step"), "train/step"),
        ("log", ("custom/reward_mean", "train/step")),
    ]


@pytest.mark.unit
def test_delayed_eval_axis_cannot_regress_canonical_train_step(monkeypatch):
    args = _args(always_use_train_step=True)
    args.use_wandb = True
    args.use_tensorboard = False
    published = []
    monkeypatch.setattr(
        logging_utils.wandb_utils,
        "define_logged_metric_axes",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(logging_utils.wandb, "log", lambda metrics: published.append(dict(metrics)))

    logging_utils.log(
        args,
        {"train/loss": 1.0, "train/step": 2},
        step_key="train/step",
    )
    logging_utils.log(
        args,
        {"eval/tmax_holdout": 0.25, "eval/train_step": 0},
        step_key="eval/train_step",
    )

    assert [row.get("train/step") for row in published] == [2, None]
    assert published[1]["eval/train_step"] == 0


@pytest.mark.unit
def test_eval_metric_on_canonical_axis_is_rejected(monkeypatch):
    args = _args(always_use_train_step=True)
    args.use_wandb = True
    args.use_tensorboard = False
    monkeypatch.setattr(logging_utils.wandb, "log", lambda _metrics: None)

    with pytest.raises(KeyError, match="evaluation metrics must use"):
        logging_utils.log(
            args,
            {"eval/tmax_holdout": 0.25, "train/step": 2},
            step_key="train/step",
        )


@pytest.mark.unit
def test_eval_axis_rejects_mixed_business_metrics(monkeypatch):
    args = _args(always_use_train_step=True)
    args.use_wandb = True
    args.use_tensorboard = False
    monkeypatch.setattr(logging_utils.wandb, "log", lambda _metrics: None)

    with pytest.raises(KeyError, match="must not mix non-evaluation"):
        logging_utils.log(
            args,
            {
                "eval/tmax_holdout": 0.25,
                "custom/reward_mean": 0.5,
                "eval/train_step": 0,
            },
            step_key="eval/train_step",
        )


@pytest.mark.unit
def test_rollout_buffer_metrics_include_train_step(monkeypatch):
    from slime_plugins.rollout_buffer.rollout_buffer_example import log_raw_info

    logged = []
    monkeypatch.setattr(
        "slime_plugins.rollout_buffer.rollout_buffer_example.logging_utils.log",
        lambda args, metrics, step_key: logged.append((metrics.copy(), step_key)),
    )
    args = _args(always_use_train_step=True)
    args.use_wandb = True
    args.use_tensorboard = False

    log_raw_info(
        args,
        [{"total_samples": 4, "avg_reward": 0.75}],
        rollout_id=3,
    )

    assert logged == [
        (
            {
                "rollout/no_filter/total_samples": 4,
                "rollout/no_filter/avg_reward": 0.75,
                "rollout/step": 6,
                "train/step": 6,
            },
            "train/step",
        )
    ]


@pytest.mark.unit
def test_wandb_finish_timeout_is_bounded_by_default(monkeypatch):
    monkeypatch.delenv("WANDB_FINISH_TIMEOUT", raising=False)

    assert _wandb_finish_timeout_seconds() == _DEFAULT_WANDB_FINISH_TIMEOUT_SECONDS


@pytest.mark.unit
@pytest.mark.parametrize("value", ["", "not-a-number", "0", "-1", "nan", "inf", "-inf"])
def test_invalid_wandb_finish_timeout_uses_bounded_default(monkeypatch, value):
    monkeypatch.setenv("WANDB_FINISH_TIMEOUT", value)

    assert _wandb_finish_timeout_seconds() == _DEFAULT_WANDB_FINISH_TIMEOUT_SECONDS


@pytest.mark.unit
def test_wandb_finish_timeout_can_be_overridden(monkeypatch):
    monkeypatch.setenv("WANDB_FINISH_TIMEOUT", "12.5")

    assert _wandb_finish_timeout_seconds() == 12.5


@pytest.mark.unit
def test_explicit_run_id_is_the_default_display_name(monkeypatch):
    monkeypatch.delenv("WANDB_NAME", raising=False)
    args = Namespace(wandb_run_id="stable-resumable-run-id")

    assert (
        _wandb_run_name(
            args,
            group="lambda-sweep",
            generated_name="lambda-sweep",
        )
        == "stable-resumable-run-id"
    )


@pytest.mark.unit
def test_wandb_name_override_remains_authoritative(monkeypatch):
    monkeypatch.setenv("WANDB_NAME", "human-readable-arm")
    args = Namespace(wandb_run_id="stable-resumable-run-id")

    assert (
        _wandb_run_name(
            args,
            group="lambda-sweep",
            generated_name="lambda-sweep",
        )
        == "human-readable-arm"
    )


@pytest.mark.unit
def test_shared_writer_labels_are_stable_and_role_specific():
    assert _shared_writer_label(primary=True) == "driver"
    assert _shared_writer_label(primary=False) == "rollout-manager"
    assert _shared_writer_label(primary=False, role="actor") == "trainer-actor"
    assert _shared_writer_label(primary=False, role="critic") == "trainer-critic"


@pytest.mark.unit
@pytest.mark.parametrize("mode", ["online", "offline"])
def test_secondary_wandb_writer_disables_console_capture(monkeypatch, mode):
    args = Namespace(
        use_wandb=True,
        wandb_run_id="run-id",
        wandb_mode=mode,
        wandb_key=None,
        wandb_host=None,
        wandb_team="team",
        wandb_project="project",
        wandb_dir=None,
    )
    initialized = []
    monkeypatch.setattr(wandb_utils.wandb, "Settings", lambda **kwargs: kwargs)
    monkeypatch.setattr(
        wandb_utils.wandb,
        "init",
        lambda **kwargs: initialized.append(kwargs),
    )
    monkeypatch.setattr(wandb_utils, "_init_wandb_common", lambda _args: None)

    wandb_utils.init_wandb_secondary(args)

    assert initialized[0]["settings"]["console"] == "off"
    if mode == "online":
        assert initialized[0]["settings"] == {
            "mode": "shared",
            "console": "off",
            "x_primary": False,
            "x_label": "rollout-manager",
            "x_update_finish_state": False,
            "x_server_side_derived_summary": True,
            "finish_timeout": _DEFAULT_WANDB_FINISH_TIMEOUT_SECONDS,
        }


@pytest.mark.unit
@pytest.mark.parametrize(
    ("role", "expected_label"),
    [("actor", "trainer-actor"), ("critic", "trainer-critic")],
)
def test_trainer_secondary_cannot_update_shared_run_finish_state(
    monkeypatch, role, expected_label
):
    args = Namespace(
        use_wandb=True,
        wandb_run_id="run-id",
        wandb_mode="online",
        wandb_key=None,
        wandb_host=None,
        wandb_team="team",
        wandb_project="project",
        wandb_dir=None,
    )
    initialized = []
    monkeypatch.setattr(wandb_utils.wandb, "Settings", lambda **kwargs: kwargs)
    monkeypatch.setattr(
        wandb_utils.wandb,
        "init",
        lambda **kwargs: initialized.append(kwargs),
    )
    monkeypatch.setattr(wandb_utils, "_init_wandb_common", lambda _args: None)

    wandb_utils.init_wandb_secondary(args, role=role)

    settings = initialized[0]["settings"]
    assert settings["x_label"] == expected_label
    assert settings["x_primary"] is False
    assert settings["x_update_finish_state"] is False


@pytest.mark.unit
def test_primary_online_writer_uses_server_summary_and_unique_label(monkeypatch):
    args = Namespace(
        use_wandb=True,
        wandb_run_id="run-id",
        wandb_mode="online",
        wandb_key=None,
        wandb_host=None,
        wandb_team="team",
        wandb_project="project",
        wandb_group="group",
        wandb_random_suffix=False,
        wandb_dir=None,
    )
    initialized = []
    monkeypatch.delenv("WANDB_NAME", raising=False)
    monkeypatch.setattr(wandb_utils.wandb, "Settings", lambda **kwargs: kwargs)
    monkeypatch.setattr(
        wandb_utils.wandb,
        "init",
        lambda **kwargs: initialized.append(kwargs),
    )
    monkeypatch.setattr(wandb_utils.wandb, "run", Namespace(id="run-id"))
    monkeypatch.setattr(wandb_utils, "_init_wandb_common", lambda _args: None)

    wandb_utils.init_wandb_primary(args)

    assert initialized[0]["name"] == "run-id"
    assert initialized[0]["group"] == "group"
    assert initialized[0]["settings"] == {
        "mode": "shared",
        "x_primary": True,
        "x_label": "driver",
        "x_server_side_derived_summary": True,
        "finish_timeout": _DEFAULT_WANDB_FINISH_TIMEOUT_SECONDS,
    }


@pytest.mark.unit
def test_concrete_metric_definitions_request_server_last_summary(monkeypatch):
    definitions = []
    monkeypatch.setattr(
        wandb_utils.wandb,
        "define_metric",
        lambda name, **kwargs: definitions.append((name, kwargs)),
    )
    _init_wandb_common(_args(always_use_train_step=True))
    definitions.clear()

    define_logged_metric_axes(
        {
            "train/step": 63,
            "rollout/accuracy_outcome_mean": 0.75,
            "rollout/total_cost_mean": 9.5,
        },
        step_metric="train/step",
    )

    assert definitions == [
        (
            "rollout/accuracy_outcome_mean",
            {"step_metric": "train/step", "summary": "last"},
        ),
        (
            "rollout/total_cost_mean",
            {"step_metric": "train/step", "summary": "last"},
        ),
    ]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
