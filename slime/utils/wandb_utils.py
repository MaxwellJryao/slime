import logging
import os
from copy import deepcopy

import wandb

logger = logging.getLogger(__name__)

_DEFAULT_WANDB_FINISH_TIMEOUT_SECONDS = 30.0
_defined_metric_axes: set[tuple[str, str]] = set()


def _wandb_finish_timeout_seconds() -> float:
    """Keep distributed workers from holding GPU allocations during W&B teardown."""
    raw = os.environ.get(
        "WANDB_FINISH_TIMEOUT",
        str(_DEFAULT_WANDB_FINISH_TIMEOUT_SECONDS),
    )
    try:
        timeout = float(raw)
    except ValueError:
        logger.warning(
            "Invalid WANDB_FINISH_TIMEOUT=%r; using %.0fs",
            raw,
            _DEFAULT_WANDB_FINISH_TIMEOUT_SECONDS,
        )
        return _DEFAULT_WANDB_FINISH_TIMEOUT_SECONDS
    if timeout <= 0:
        logger.warning(
            "WANDB_FINISH_TIMEOUT must be positive; using %.0fs",
            _DEFAULT_WANDB_FINISH_TIMEOUT_SECONDS,
        )
        return _DEFAULT_WANDB_FINISH_TIMEOUT_SECONDS
    return timeout


def _is_offline_mode(args) -> bool:
    """Detect whether W&B should run in offline mode.

    Priority order:
    1) args.wandb_mode if provided
    2) WANDB_MODE environment variable
    """
    if args.wandb_mode:
        return args.wandb_mode == "offline"
    return os.environ.get("WANDB_MODE") == "offline"


def init_wandb_primary(args):
    if not args.use_wandb:
        args.wandb_run_id = None
        return

    # Set W&B mode if specified (overrides WANDB_MODE env var)
    if args.wandb_mode:
        os.environ["WANDB_MODE"] = args.wandb_mode
        if args.wandb_mode == "offline":
            logger.info("W&B offline mode enabled. Data will be saved locally.")
        elif args.wandb_mode == "disabled":
            logger.info("W&B disabled mode enabled. No data will be logged.")
        elif args.wandb_mode == "online":
            logger.info("W&B online mode enabled. Data will be uploaded to cloud.")

    offline = _is_offline_mode(args)

    # Only perform explicit login when NOT offline
    if (not offline) and args.wandb_key is not None:
        wandb.login(key=args.wandb_key, host=args.wandb_host)

    # Prepare wandb init parameters
    # add random 6 length string with characters
    if args.wandb_random_suffix:
        group = args.wandb_group + "_" + wandb.util.generate_id()
        run_name = f"{group}-RANK_{args.rank}"
    else:
        group = args.wandb_group
        run_name = args.wandb_group

    # Prepare wandb init parameters
    init_kwargs = {
        "entity": args.wandb_team,
        "project": args.wandb_project,
        "group": group,
        "name": run_name,
        "config": _compute_config_for_logging(args),
    }
    if args.wandb_run_id is not None:
        init_kwargs["id"] = args.wandb_run_id
        init_kwargs["resume"] = os.environ.get("WANDB_RESUME", "allow")

    # Configure settings based on offline/online mode
    finish_timeout = _wandb_finish_timeout_seconds()
    if offline:
        init_kwargs["settings"] = wandb.Settings(
            mode="offline",
            finish_timeout=finish_timeout,
        )
    else:
        init_kwargs["settings"] = wandb.Settings(
            mode="shared",
            x_primary=True,
            finish_timeout=finish_timeout,
        )

    # Add custom directory if specified
    if args.wandb_dir:
        # Ensure directory exists to avoid backend crashes
        os.makedirs(args.wandb_dir, exist_ok=True)
        init_kwargs["dir"] = args.wandb_dir
        logger.info(f"W&B logs will be stored in: {args.wandb_dir}")

    wandb.init(**init_kwargs)

    _init_wandb_common(args)

    # Set wandb_run_id in args for easy access throughout the training process
    args.wandb_run_id = wandb.run.id


def _compute_config_for_logging(args):
    output = _args_to_config_dict(args)

    whitelist_env_vars = [
        "SLURM_JOB_ID",
        # We may insert more default values here, and may also allow users to configure a whitelist
    ]
    output["env_vars"] = {k: v for k, v in os.environ.items() if k in whitelist_env_vars}

    if getattr(args, "use_critic", False):
        critic_args = _get_role_args_for_logging(args, role="critic")
        output.update(_prefix_config_keys(_args_to_config_dict(critic_args), "critic"))

    return output


def _args_to_config_dict(args):
    return deepcopy(args.__dict__)


def _prefix_config_keys(config, prefix):
    return {f"{prefix}/{key}": value for key, value in config.items()}


def _get_role_args_for_logging(args, role):
    if getattr(args, "megatron_config_path", None) is None:
        return args

    from slime.utils.arguments import parse_megatron_role_args

    return parse_megatron_role_args(args, args.megatron_config_path, role=role)


def _compute_secondary_config_for_logging(args, role=None):
    config = _args_to_config_dict(args)
    if role == "critic":
        return _prefix_config_keys(config, "critic")
    return config


# https://docs.wandb.ai/guides/track/log/distributed-training/#track-all-processes-to-a-single-run
def init_wandb_secondary(args, role=None):
    wandb_run_id = getattr(args, "wandb_run_id", None)
    if wandb_run_id is None:
        return

    # Set W&B mode if specified (same as primary)
    if args.wandb_mode:
        os.environ["WANDB_MODE"] = args.wandb_mode

    offline = _is_offline_mode(args)

    if (not offline) and args.wandb_key is not None:
        wandb.login(key=args.wandb_key, host=args.wandb_host)

    # Configure settings based on offline/online mode
    if offline:
        settings_kwargs = dict(
            mode="offline",
            console="off",
            finish_timeout=_wandb_finish_timeout_seconds(),
        )
    else:
        settings_kwargs = dict(
            mode="shared",
            console="off",
            x_primary=False,
            x_update_finish_state=False,
            finish_timeout=_wandb_finish_timeout_seconds(),
        )

    init_kwargs = {
        "id": wandb_run_id,
        "entity": args.wandb_team,
        "project": args.wandb_project,
        "config": _compute_secondary_config_for_logging(args, role=role),
        "resume": "allow",
        "reinit": True,
        "settings": wandb.Settings(**settings_kwargs),
    }

    # Add custom directory if specified
    if args.wandb_dir:
        os.makedirs(args.wandb_dir, exist_ok=True)
        init_kwargs["dir"] = args.wandb_dir

    wandb.init(**init_kwargs)

    _init_wandb_common(args)


def _init_wandb_common(args):
    # A process may attach to more than one run over its lifetime.  Exact
    # metric definitions are run-local, so never carry this cache across
    # ``wandb.init`` calls.
    _defined_metric_axes.clear()

    rollout_step_metric = "train/step" if getattr(args, "wandb_always_use_train_step", False) else "rollout/step"
    eval_step_metric = "eval/train_step" if getattr(args, "wandb_always_use_train_step", False) else "eval/step"

    wandb.define_metric("train/step")
    wandb.define_metric("train/*", step_metric="train/step")
    if getattr(args, "wandb_always_use_train_step", False):
        wandb.define_metric("rollout/step", step_metric="train/step")
    else:
        wandb.define_metric("rollout/step")
    wandb.define_metric("rollout/*", step_metric=rollout_step_metric)
    wandb.define_metric("multi_turn/*", step_metric=rollout_step_metric)
    wandb.define_metric("passrate/*", step_metric=rollout_step_metric)
    wandb.define_metric("polar/*", step_metric=rollout_step_metric)
    _define_gpu_sidecar_metric_axes(args)
    if getattr(args, "wandb_always_use_train_step", False):
        wandb.define_metric("eval/train_step")
    else:
        wandb.define_metric("eval/step")
    wandb.define_metric("eval/*", step_metric=eval_step_metric)
    wandb.define_metric("perf/*", step_metric=rollout_step_metric)
    wandb.define_metric("timing/*", step_metric=rollout_step_metric)
    if getattr(args, "wandb_always_use_train_step", False):
        # Baseline evaluation can finish after later trainer records.  Give
        # all metrics in that delayed row an isolated axis so its model step 0
        # cannot make the canonical trainer series move backwards.
        wandb.define_metric("timing/eval/*", step_metric="eval/train_step")


def _define_gpu_sidecar_metric_axes(args) -> None:
    """Predeclare independent per-node GPU axes on the primary writer.

    Multiple sidecars must not publish the canonical ``train/step`` key into a
    shared run: their independently sampled rows can arrive out of order and
    make that canonical series move backwards.  Each node therefore owns one
    namespaced step metric while retaining the same numeric training step.
    """

    metric_prefix = os.environ.get("GPU_MONITOR_PREFIX")
    raw_num_nodes = os.environ.get("SLURM_NNODES")
    if not metric_prefix or not raw_num_nodes:
        return
    try:
        num_nodes = int(raw_num_nodes)
    except ValueError:
        logger.warning("Invalid SLURM_NNODES=%r; GPU metric axes will be sidecar-defined", raw_num_nodes)
        return
    if num_nodes <= 0:
        logger.warning("SLURM_NNODES must be positive; GPU metric axes will be sidecar-defined")
        return

    explicit_role = os.environ.get("GPU_MONITOR_NODE_ROLE") or None
    actor_num_nodes = int(getattr(args, "actor_num_nodes", 1))
    for node_rank in range(num_nodes):
        if num_nodes == 1:
            node_prefix = metric_prefix
        elif explicit_role == "rank":
            node_prefix = f"{metric_prefix}/node_{node_rank}"
        else:
            node_role = explicit_role or (
                "actor" if node_rank < actor_num_nodes else "rollout"
            )
            node_prefix = f"{metric_prefix}/{node_role}_node_{node_rank}"
        step_metric = f"{node_prefix}/train_step"
        wandb.define_metric(step_metric)
        wandb.define_metric(f"{node_prefix}/*", step_metric=step_metric)


def define_logged_metric_axes(metrics: dict, *, step_metric: str) -> None:
    """Bind every concrete user metric to the axis in its own history row.

    Prefix globs keep W&B's generated panels useful before the first value is
    logged, but exact definitions are deliberately emitted immediately before
    each metric's first value.  This removes any dependence on nested wildcard
    matching and prevents a broad ``perf/*`` or ``timing/*`` definition from
    choosing the wrong axis when those namespaces are produced by both trainer
    and rollout processes.
    """

    for metric_name in metrics:
        if metric_name == step_metric or metric_name.startswith("_"):
            continue
        cache_key = (metric_name, step_metric)
        if cache_key in _defined_metric_axes:
            continue
        wandb.define_metric(metric_name, step_metric=step_metric)
        _defined_metric_axes.add(cache_key)
