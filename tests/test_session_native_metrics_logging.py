from types import SimpleNamespace

from slime.backends.megatron_utils import data as data_module
from slime.utils.session_native_gae_runtime import EXPECTED_METRICS, METRICS_FIELD


def test_seven_metrics_are_logged_once_on_their_canonical_axes(monkeypatch):
    metrics = {name: float(index) / 10.0 for index, name in enumerate(sorted(EXPECTED_METRICS))}
    rollout_data = {
        METRICS_FIELD: metrics,
        # Two optimizer steps in each rollout batch.
        "num_microbatches": [4, 4],
    }
    args = SimpleNamespace(
        wandb_always_use_train_step=False,
        use_wandb=False,
        use_tensorboard=False,
    )

    monkeypatch.setattr(data_module.mpu, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(data_module.mpu, "is_pipeline_last_stage", lambda: True)
    monkeypatch.setattr(
        data_module.mpu,
        "get_data_parallel_world_size",
        lambda **_kwargs: 2,
    )
    monkeypatch.setattr(
        data_module.mpu,
        "get_data_parallel_src_rank",
        lambda **_kwargs: 0,
    )
    monkeypatch.setattr(
        data_module.mpu,
        "get_data_parallel_group_gloo",
        lambda **_kwargs: object(),
    )
    monkeypatch.setattr(
        data_module,
        "gather_and_reduce_log_dict",
        lambda values, **_kwargs: dict(values),
    )
    emitted = []
    monkeypatch.setattr(
        data_module.logging_utils,
        "log",
        lambda _args, values, *, step_key: emitted.append((dict(values), step_key)),
    )

    data_module.log_session_native_gae_metrics(3, args, rollout_data)

    assert len(emitted) == 2
    rollout_payload, rollout_axis = emitted[0]
    train_payload, train_axis = emitted[1]
    assert rollout_axis == "rollout/step"
    assert rollout_payload["rollout/step"] == 3
    assert set(rollout_payload) == {
        "rollout/session_native_action_reward_mean",
        "rollout/session_native_action_reward_variance",
        "rollout/session_native_trainable_token_share",
        "rollout/step",
    }
    assert train_axis == "train/step"
    assert train_payload["train/step"] == 7
    assert set(train_payload) == {
        "train/session_native_action_advantage_mean",
        "train/session_native_action_advantage_variance",
        "train/session_native_critic_mae",
        "train/session_native_critic_explained_variance",
        "train/step",
    }
