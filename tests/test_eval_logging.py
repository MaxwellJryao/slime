from types import SimpleNamespace

import pytest

from slime.ray import rollout as rollout_module
from slime.rollout import sglang_rollout
from slime.utils.eval_config import build_eval_dataset_configs
from slime.utils.types import Sample

NUM_GPUS = 0


def _args() -> SimpleNamespace:
    return SimpleNamespace(
        custom_eval_rollout_log_function_path=None,
        log_passrate=False,
        wandb_always_use_train_step=True,
        rollout_batch_size=1,
        n_samples_per_prompt=1,
        global_batch_size=1,
    )


def test_legacy_eval_cli_minimum_is_applied_to_each_dataset() -> None:
    args = SimpleNamespace(
        min_eval_samples=32,
        n_samples_per_eval_prompt=2,
        n_samples_per_prompt=8,
        eval_temperature=0.2,
        rollout_temperature=1.0,
        eval_top_p=0.9,
        rollout_top_p=1.0,
        eval_top_k=None,
        rollout_top_k=-1,
        eval_max_response_len=4096,
        rollout_max_response_len=8192,
    )

    datasets = build_eval_dataset_configs(
        args,
        [{"name": "holdout", "path": "/tmp/holdout.jsonl"}],
        {},
    )

    assert datasets[0].min_eval_samples == 32


def test_empty_eval_rewards_are_zero_filled_and_do_not_abort(monkeypatch, caplog) -> None:
    logged = []
    monkeypatch.setattr(
        rollout_module.logging_utils,
        "log",
        lambda _args, metrics, *, step_key: logged.append((dict(metrics), step_key)),
    )

    with caplog.at_level("WARNING"):
        rollout_module._log_eval_rollout_data(
            rollout_id=4,
            args=_args(),
            data={
                "holdout": {
                    "rewards": [],
                    "all_rewards": [0.0, 0.0],
                    "truncated": [],
                    "samples": [],
                    "valid_count": 0,
                    "error_count": 2,
                }
            },
            extra_metrics={
                "eval/holdout/valid_count": 0.0,
                "eval/holdout/error_count": 2.0,
            },
            completed_train_batch=True,
        )

    assert len(logged) == 1
    metrics, step_key = logged[0]
    assert metrics["eval/holdout/valid_count"] == 0.0
    assert metrics["eval/holdout/error_count"] == 2.0
    assert metrics["eval/holdout/expected_count"] == 2.0
    assert metrics["eval/holdout/zero_filled_count"] == 2.0
    assert metrics["eval/holdout"] == 0.0
    assert metrics["eval/holdout/reward_mean"] == 0.0
    assert metrics["eval/train_step"] == 4
    assert "train/step" not in metrics
    assert step_key == "eval/train_step"
    assert "continued training" in caplog.text
    assert "errors=2" in caplog.text


def test_required_eval_logs_diagnostics_then_raises_even_with_custom_logger(
    monkeypatch, caplog
) -> None:
    custom_calls = []
    logged = []
    args = _args()
    args.custom_eval_rollout_log_function_path = "custom.eval_logger"
    monkeypatch.setattr(
        rollout_module,
        "load_function",
        lambda _path: lambda *call_args: custom_calls.append(call_args) or True,
    )
    monkeypatch.setattr(
        rollout_module.logging_utils,
        "log",
        lambda _args, metrics, *, step_key: logged.append((dict(metrics), step_key)),
    )

    with caplog.at_level("ERROR"), pytest.raises(
        RuntimeError, match=r"Required evaluation 4 is incomplete.*errors=2"
    ):
        rollout_module._log_eval_rollout_data(
            rollout_id=4,
            args=args,
            data={
                "holdout": {
                    "rewards": [],
                    "all_rewards": [0.0, 0.0],
                    "valid_count": 0,
                    "error_count": 2,
                }
            },
            completed_train_batch=True,
            require_complete=True,
        )

    assert len(custom_calls) == 1
    assert len(logged) == 1
    metrics, step_key = logged[0]
    assert metrics["eval/holdout/valid_count"] == 0.0
    assert metrics["eval/holdout/error_count"] == 2.0
    assert metrics["eval/holdout/reward_mean"] == 0.0
    assert step_key == "eval/train_step"
    assert "will not be marked complete" in caplog.text


def test_required_eval_rejects_an_empty_dataset_mapping(monkeypatch) -> None:
    logged = []
    monkeypatch.setattr(
        rollout_module.logging_utils,
        "log",
        lambda _args, metrics, *, step_key: logged.append((dict(metrics), step_key)),
    )

    with pytest.raises(RuntimeError, match="no evaluation datasets returned"):
        rollout_module._log_eval_rollout_data(
            rollout_id=9,
            args=_args(),
            data={},
            require_complete=True,
        )

    assert len(logged) == 1


def test_required_eval_accepts_errors_at_the_configured_valid_threshold(
    monkeypatch, caplog
) -> None:
    logged = []
    monkeypatch.setattr(
        rollout_module.logging_utils,
        "log",
        lambda _args, metrics, *, step_key: logged.append(dict(metrics)),
    )

    with caplog.at_level("WARNING"):
        result = rollout_module._log_eval_rollout_data(
            rollout_id=5,
            args=_args(),
            data={
                "holdout": {
                    "rewards": [1.0, 1.0],
                    "all_rewards": [1.0, 1.0, 0.0],
                    "valid_count": 2,
                    "error_count": 1,
                    "min_eval_samples": 2,
                }
            },
            require_complete=True,
        )

    assert result["eval/holdout/reward_mean"] == pytest.approx(2 / 3)
    assert result["eval/holdout/valid_count"] == 2.0
    assert result["eval/holdout/error_count"] == 1.0
    assert result["eval/holdout/min_valid_count"] == 2.0
    assert len(logged) == 1
    assert "accepted for final completion" in caplog.text


def test_partial_eval_logs_valid_and_error_counts_with_reward(monkeypatch) -> None:
    logged = []
    monkeypatch.setattr(rollout_module, "compute_metrics_from_samples", lambda *_args: {})
    monkeypatch.setattr(
        rollout_module.logging_utils,
        "log",
        lambda _args, metrics, *, step_key: logged.append(dict(metrics)),
    )
    sample = SimpleNamespace()

    result = rollout_module._log_eval_rollout_data(
        rollout_id=1,
        args=_args(),
        data={
            "holdout": {
                "rewards": [1.0],
                "all_rewards": [1.0, 0.0],
                "truncated": [False],
                "samples": [sample],
                "valid_count": 1,
                "error_count": 1,
            }
        },
        completed_train_batch=True,
    )

    assert result["eval/holdout"] == 0.5
    assert result["eval/holdout/reward_mean"] == 0.5
    assert result["eval/holdout/valid_count"] == 1.0
    assert result["eval/holdout/error_count"] == 1.0
    assert result["eval/holdout/expected_count"] == 2.0
    assert result["eval/holdout/zero_filled_count"] == 1.0
    assert len(logged) == 1


def test_terminal_bench_baseline_and_final_share_isolated_eval_train_axis() -> None:
    args = _args()
    args.rollout_batch_size = 24
    args.n_samples_per_prompt = 8
    args.global_batch_size = 64
    args.use_wandb = False
    args.use_tensorboard = False
    data = {
        "terminal_bench_2_1": {
            "rewards": [1.0, 0.0],
            "all_rewards": [1.0, 0.0],
            "truncated": [False, False],
            "samples": [],
            "valid_count": 2,
            "error_count": 0,
        }
    }
    extra_metrics = {
        "eval/terminal_bench_2_1/reward_mean": 0.5,
        "timing/eval/terminal_bench_2_1/session_ms/e2e_mean": 123.0,
    }

    baseline = rollout_module._log_eval_rollout_data(
        rollout_id=0,
        args=args,
        data=data,
        extra_metrics=extra_metrics,
        completed_train_batch=False,
    )
    final = rollout_module._log_eval_rollout_data(
        rollout_id=4,
        args=args,
        data=data,
        extra_metrics=extra_metrics,
        completed_train_batch=True,
    )

    assert baseline["eval/train_step"] == 0
    assert final["eval/train_step"] == 14
    for metrics in (baseline, final):
        assert "train/step" not in metrics
        assert metrics["eval/terminal_bench_2_1/reward_mean"] == 0.5
        assert (
            metrics["timing/eval/terminal_bench_2_1/session_ms/e2e_mean"]
            == 123.0
        )


def test_multiple_eval_datasets_log_on_one_eval_train_axis() -> None:
    args = _args()
    args.use_wandb = False
    args.use_tensorboard = False
    data = {
        "terminal_bench_2_1": {
            "rewards": [1.0],
            "samples": [],
            "valid_count": 1,
            "error_count": 0,
        },
        "secondary_holdout": {
            "rewards": [0.0],
            "samples": [],
            "valid_count": 1,
            "error_count": 0,
        },
    }
    extra_metrics = {
        "eval/terminal_bench_2_1/reward_mean": 1.0,
        "timing/eval/terminal_bench_2_1/session_ms/e2e_mean": 10.0,
        "eval/secondary_holdout/reward_mean": 0.0,
        "timing/eval/secondary_holdout/session_ms/e2e_mean": 20.0,
    }

    metrics = rollout_module._log_eval_rollout_data(
        rollout_id=2,
        args=args,
        data=data,
        extra_metrics=extra_metrics,
        completed_train_batch=True,
    )

    assert metrics["eval/train_step"] == 2
    assert "train/step" not in metrics
    assert set(extra_metrics).issubset(metrics)


def test_eval_below_minimum_zero_fills_logs_warning_and_continues(monkeypatch, caplog) -> None:
    logged = []
    monkeypatch.setattr(rollout_module, "compute_metrics_from_samples", lambda *_args: {})
    monkeypatch.setattr(
        rollout_module.logging_utils,
        "log",
        lambda _args, metrics, *, step_key: logged.append(dict(metrics)),
    )

    with caplog.at_level("WARNING"):
        rollout_module._log_eval_rollout_data(
            rollout_id=1,
            args=_args(),
            data={
                "holdout": {
                    "rewards": [1.0],
                    "all_rewards": [1.0, 0.0],
                    "truncated": [False],
                    "samples": [SimpleNamespace()],
                    "valid_count": 1,
                    "error_count": 1,
                    "min_eval_samples": 2,
                }
            },
            completed_train_batch=True,
        )

    assert logged[0]["eval/holdout"] == 0.5
    assert logged[0]["eval/holdout/reward_mean"] == 0.5
    assert logged[0]["eval/holdout/valid_count"] == 1.0
    assert logged[0]["eval/holdout/error_count"] == 1.0
    assert logged[0]["eval/holdout/expected_count"] == 2.0
    assert logged[0]["eval/holdout/min_valid_count"] == 2.0
    assert "required=2" in caplog.text
    assert "continued training" in caplog.text


def test_eval_does_not_double_fill_explicit_all_rewards(monkeypatch) -> None:
    logged = []
    monkeypatch.setattr(rollout_module, "compute_metrics_from_samples", lambda *_args: {})
    monkeypatch.setattr(
        rollout_module.logging_utils,
        "log",
        lambda _args, metrics, *, step_key: logged.append(dict(metrics)),
    )

    result = rollout_module._log_eval_rollout_data(
        rollout_id=3,
        args=_args(),
        data={
            "holdout": {
                "rewards": [1.0],
                "all_rewards": [1.0, 0.0],
                "samples": [SimpleNamespace()],
                "valid_count": 1,
                "error_count": 1,
            }
        },
    )

    assert result["eval/holdout"] == 0.5
    assert result["eval/holdout/expected_count"] == 2.0
    assert len(logged) == 1


@pytest.mark.asyncio
async def test_builtin_eval_isolates_task_and_reward_errors(monkeypatch, caplog) -> None:
    prompt_samples = [
        SimpleNamespace(metadata={}, index=-1, reward=None, status=Sample.Status.PENDING)
        for _ in range(3)
    ]
    monkeypatch.setattr(
        sglang_rollout,
        "Dataset",
        lambda **_kwargs: SimpleNamespace(samples=prompt_samples),
    )
    monkeypatch.setattr(sglang_rollout, "load_tokenizer", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(sglang_rollout, "load_processor", lambda *_args, **_kwargs: None)
    sglang_rollout.EVAL_PROMPT_DATASET.clear()

    async def generate_one(_args, sample, **_kwargs):
        if sample.index == 0:
            raise RuntimeError("inference unavailable")
        sample.status = Sample.Status.COMPLETED
        sample.reward = None if sample.index == 1 else 0.75
        return sample

    monkeypatch.setattr(sglang_rollout, "generate_and_rm", generate_one)
    args = SimpleNamespace(
        hf_checkpoint="model",
        multimodal_keys=None,
        apply_chat_template=False,
        apply_chat_template_kwargs=None,
        eval_max_prompt_len=128,
        rollout_stop=None,
        rollout_stop_token_ids=None,
        rollout_skip_special_tokens=True,
        sglang_enable_deterministic_inference=False,
        rollout_seed=1,
        eval_reward_key=None,
        reward_key=None,
        group_rm=False,
    )
    dataset_cfg = SimpleNamespace(
        name="holdout",
        path="/unused.jsonl",
        cache_key=("holdout",),
        multimodal_keys=None,
        apply_chat_template=None,
        apply_chat_template_kwargs=None,
        input_key="prompt",
        label_key=None,
        metadata_key=None,
        tool_key=None,
        temperature=0.0,
        top_p=1.0,
        top_k=-1,
        max_response_len=32,
        skip_special_tokens=None,
        no_stop_trim=None,
        repetition_penalty=None,
        n_samples_per_eval_prompt=1,
        custom_rm_path=None,
        custom_generate_function_path=None,
        min_eval_samples=3,
        inject_metadata=lambda metadata: dict(metadata or {}),
    )

    with caplog.at_level("WARNING"):
        result = await sglang_rollout.eval_rollout_single_dataset(
            args,
            rollout_id=0,
            dataset_cfg=dataset_cfg,
        )

    dataset = result["holdout"]
    assert dataset["rewards"] == [0.75]
    assert dataset["all_rewards"] == [0.75, 0.0, 0.0]
    assert dataset["valid_count"] == 1
    assert dataset["error_count"] == 2
    assert dataset["expected_count"] == 3
    assert "sample generation/reward failed" in caplog.text
    assert "has no usable reward" in caplog.text


def test_rollout_manager_eval_returns_logged_payload_for_primary_relay(
    monkeypatch,
) -> None:
    manager_cls = rollout_module.RolloutManager.__ray_metadata__.modified_class
    manager = manager_cls.__new__(manager_cls)
    args = _args()
    args.debug_train_only = False
    manager.args = args
    manager.eval_generate_rollout = object()
    manager.data_source = object()
    manager.health_monitoring_resume = lambda: None
    manager._save_debug_rollout_data = lambda *_args, **_kwargs: None
    eval_data = {"holdout": {"rewards": [1.0]}}
    monkeypatch.setattr(
        rollout_module,
        "call_rollout_fn",
        lambda *_args, **_kwargs: SimpleNamespace(
            data=eval_data,
            metrics={"eval/holdout/reward_mean": 1.0},
        ),
    )
    payload = {
        "eval/holdout/reward_mean": 1.0,
        "eval/train_step": 7,
    }
    observed_kwargs = {}

    def log_eval(*_args, **kwargs):
        observed_kwargs.update(kwargs)
        return payload

    monkeypatch.setattr(
        rollout_module,
        "_log_eval_rollout_data",
        log_eval,
    )

    result = manager.eval(rollout_id=7, require_complete=True)

    assert result is payload
    assert observed_kwargs["completed_train_batch"] is True
    assert observed_kwargs["require_complete"] is True


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
