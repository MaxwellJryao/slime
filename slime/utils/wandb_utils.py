import json
import logging
import math
import os
import re
import secrets
import stat
from copy import deepcopy
from pathlib import Path

import wandb

logger = logging.getLogger(__name__)

_DEFAULT_WANDB_FINISH_TIMEOUT_SECONDS = 30.0
_defined_metric_axes: set[tuple[str, str]] = set()
_WANDB_PRIMARY_READY_SCHEMA = "slime.wandb-primary-ready/v1"
_WANDB_PRIMARY_READY_FILE_ENV = "WANDB_PRIMARY_READY_FILE"
_WANDB_PRIMARY_READY_TOKEN_ENV = "WANDB_PRIMARY_READY_TOKEN"
_WANDB_PRIMARY_READY_TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,255}")

# Deliberately map only reviewed, non-secret environment variables to stable
# W&B config keys.  Do not broaden this to an environment dump: distributed
# training environments routinely contain credentials and provider tokens.
_SPILOT_WANDB_CONFIG_ENV_ALLOWLIST = {
    "SPILOT_SWEEP_ARM": "spilot_sweep_arm",
    "SPILOT_PROCESS_REWARD_PAIR_ID": "spilot_process_reward_pair_id",
    "SPILOT_PROCESS_REWARD_PAIR_ROLE": "spilot_process_reward_pair_role",
    "SPILOT_MATCHED_INVARIANTS_SHA256": "spilot_matched_invariants_sha256",
    "SPILOT_PROCESS_REWARD_MODE": "spilot_process_reward_mode",
}
_SPILOT_SAFE_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,119}")
_SPILOT_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_SPILOT_PROCESS_REWARD_ROLES = frozenset({"control", "treatment"})
_SPILOT_PROCESS_REWARD_MODES = frozenset({"terminal_broadcast", "cost_to_go"})
_SPILOT_PAIRED_ARM_EXPECTATIONS = {
    "control": ("prctrl", "terminal_broadcast"),
    "treatment": ("prctg", "cost_to_go"),
}
_WANDB_CONFIG_SECRET_ARG_NAMES = frozenset(
    {
        "router_api_key",
        "router_control_plane_api_keys",
        "router_oracle_password",
        "sglang_admin_api_key",
        "sglang_api_key",
        "sglang_ssl_keyfile_password",
        "wandb_key",
    }
)


def _wandb_run_name(args, *, group: str, generated_name: str) -> str:
    """Choose a useful display name without changing the stable run identity.

    ``wandb_group`` is an experiment collection, not a run name.  Reusing it
    as the name makes every arm in a sweep indistinguishable in the UI.  An
    explicit ``WANDB_NAME`` remains authoritative; otherwise an explicit run
    ID is the most useful stable label for resumable jobs.
    """

    explicit_name = os.environ.get("WANDB_NAME", "").strip()
    if explicit_name:
        return explicit_name
    run_id = str(getattr(args, "wandb_run_id", "") or "").strip()
    if run_id:
        return run_id
    return generated_name or group


def _shared_writer_label(*, primary: bool, role: str | None = None) -> str:
    """Return a deterministic, role-unique label for W&B shared mode."""

    if primary:
        return "driver"
    if role:
        return f"trainer-{role}"
    return "rollout-manager"


def _primary_ready_marker_identity(
    args, *, group: str
) -> dict[str, str | int] | None:
    """Build the exact identity an auxiliary writer must observe.

    Fresh shared runs deliberately retain ``resume=never`` collision
    protection.  A local marker is therefore enabled only when the launcher
    provides both a destination and a per-allocation token.  It is not a
    substitute for W&B's collision check: it is published only after that
    check and all primary metric definitions have succeeded.
    """

    raw_path = os.environ.get(_WANDB_PRIMARY_READY_FILE_ENV, "")
    if not raw_path:
        return None
    path = Path(raw_path)
    if not path.is_absolute():
        raise ValueError(f"{_WANDB_PRIMARY_READY_FILE_ENV} must be absolute")

    token = os.environ.get(_WANDB_PRIMARY_READY_TOKEN_ENV, "")
    if _WANDB_PRIMARY_READY_TOKEN_RE.fullmatch(token) is None:
        raise ValueError(
            f"{_WANDB_PRIMARY_READY_TOKEN_ENV} must be a safe non-empty "
            "allocation token"
        )
    resume = os.environ.get("WANDB_RESUME", "allow")
    if resume != "never":
        raise ValueError(
            f"{_WANDB_PRIMARY_READY_FILE_ENV} requires WANDB_RESUME=never; "
            f"got {resume!r}"
        )

    run_id = str(getattr(args, "wandb_run_id", "") or "")
    project = str(getattr(args, "wandb_project", "") or "")
    entity = str(getattr(args, "wandb_team", "") or "")
    if not entity or not run_id or not project or not group:
        raise ValueError(
            "primary-ready marker requires W&B entity, run, project, and group "
            "identities"
        )
    return {
        "schema_version": 1,
        "schema": _WANDB_PRIMARY_READY_SCHEMA,
        "entity": entity,
        "project": project,
        "group": group,
        "run_id": run_id,
        "launch_token": token,
        "primary_label": _shared_writer_label(primary=True),
        "resume": resume,
    }


def _prepare_primary_ready_marker(identity: dict[str, str | int] | None) -> Path | None:
    """Fail before network initialization if the marker destination is stale."""

    if identity is None:
        return None
    path = Path(os.environ[_WANDB_PRIMARY_READY_FILE_ENV])
    try:
        parent_info = path.parent.lstat()
    except FileNotFoundError as exc:
        raise ValueError("primary-ready marker parent does not exist") from exc
    if (
        path.parent.resolve(strict=True) != path.parent
        or stat.S_ISLNK(parent_info.st_mode)
        or not stat.S_ISDIR(parent_info.st_mode)
        or parent_info.st_uid != os.getuid()
        or stat.S_IMODE(parent_info.st_mode) & 0o077
    ):
        raise ValueError(
            "primary-ready marker parent must be a private user-owned "
            "non-symlink directory"
        )
    try:
        path.lstat()
    except FileNotFoundError:
        return path
    raise FileExistsError(
        f"refusing to reuse stale primary-ready marker destination: {path}"
    )


def _write_all(fd: int, payload: bytes) -> None:
    offset = 0
    while offset < len(payload):
        written = os.write(fd, payload[offset:])
        if written <= 0:
            raise OSError("short write while publishing W&B primary-ready marker")
        offset += written


def _publish_primary_ready_marker(
    path: Path | None,
    identity: dict[str, str | int] | None,
    *,
    actual_run_id: str,
) -> None:
    """Atomically install one fsynced, no-replace primary ownership marker."""

    if path is None or identity is None:
        return
    if actual_run_id != identity["run_id"]:
        raise RuntimeError(
            "W&B returned a run ID that disagrees with the primary-ready identity"
        )

    payload = (
        json.dumps(identity, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")
    temporary = path.parent / (
        f".{path.name}.tmp-{os.getpid()}-{secrets.token_hex(8)}"
    )
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = os.open(temporary, flags, 0o600)
    installed = False
    try:
        os.fchmod(fd, 0o600)
        _write_all(fd, payload)
        os.fsync(fd)
        os.close(fd)
        fd = -1
        # Hard-link installation is atomic and, unlike os.replace, never
        # overwrites a stale/tampered destination. The temp inode was fully
        # written and fsynced before it became visible at the final path.
        os.link(temporary, path, follow_symlinks=False)
        installed = True
        temporary.unlink()
        directory_fd = os.open(
            path.parent,
            os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC,
        )
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if fd >= 0:
            os.close(fd)
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
    if not installed:
        raise RuntimeError("W&B primary-ready marker was not installed")
    marker_info = path.lstat()
    if (
        not stat.S_ISREG(marker_info.st_mode)
        or stat.S_IMODE(marker_info.st_mode) != 0o600
        or marker_info.st_uid != os.getuid()
    ):
        raise RuntimeError(
            "published W&B primary-ready marker failed its mode/owner audit"
        )


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
    if not math.isfinite(timeout) or timeout <= 0:
        logger.warning(
            "WANDB_FINISH_TIMEOUT must be finite and positive; using %.0fs",
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

    # Prepare wandb init parameters
    # add random 6 length string with characters
    if args.wandb_random_suffix:
        group = args.wandb_group + "_" + wandb.util.generate_id()
        generated_name = f"{group}-RANK_{args.rank}"
    else:
        group = args.wandb_group
        generated_name = args.wandb_group
    run_name = _wandb_run_name(
        args,
        group=group,
        generated_name=generated_name,
    )
    marker_identity = _primary_ready_marker_identity(args, group=group)
    marker_path = _prepare_primary_ready_marker(marker_identity)

    # Reject a stale/tampered marker locally before any W&B network call.
    if (not offline) and args.wandb_key is not None:
        wandb.login(key=args.wandb_key, host=args.wandb_host)

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
            x_label=_shared_writer_label(primary=True),
            # Shared writers each maintain a local summary snapshot.  Let the
            # backend derive automatic summaries from the merged history so a
            # late worker heartbeat cannot replace newer business metrics with
            # the snapshot it loaded at startup.
            x_server_side_derived_summary=True,
            finish_timeout=finish_timeout,
        )

    # Add custom directory if specified
    if args.wandb_dir:
        # Ensure directory exists to avoid backend crashes
        os.makedirs(args.wandb_dir, exist_ok=True)
        init_kwargs["dir"] = args.wandb_dir
        logger.info(f"W&B logs will be stored in: {args.wandb_dir}")

    primary_run = wandb.init(**init_kwargs)

    _init_wandb_common(args)

    if marker_path is not None:
        actual_run_id = str(getattr(primary_run, "id", "") or "")
        if not actual_run_id and wandb.run is not None:
            actual_run_id = str(getattr(wandb.run, "id", "") or "")
        _publish_primary_ready_marker(
            marker_path,
            marker_identity,
            actual_run_id=actual_run_id,
        )
        logger.info("Published W&B primary-ready marker: %s", marker_path)

    # Set wandb_run_id in args for easy access throughout the training process
    args.wandb_run_id = wandb.run.id


def _compute_config_for_logging(args):
    output = _args_to_config_dict(args)

    whitelist_env_vars = [
        "SLURM_JOB_ID",
        # We may insert more default values here, and may also allow users to configure a whitelist
    ]
    output["env_vars"] = {k: v for k, v in os.environ.items() if k in whitelist_env_vars}
    output.update(_spilot_wandb_config_from_env())

    if getattr(args, "use_critic", False):
        critic_args = _get_role_args_for_logging(args, role="critic")
        output.update(_prefix_config_keys(_args_to_config_dict(critic_args), "critic"))

    return output


def _args_to_config_dict(args):
    config = deepcopy(args.__dict__)
    for name in _WANDB_CONFIG_SECRET_ARG_NAMES:
        config.pop(name, None)
    return config


def _prefix_config_keys(config, prefix):
    return {f"{prefix}/{key}": value for key, value in config.items()}


def _get_role_args_for_logging(args, role):
    if getattr(args, "megatron_config_path", None) is None:
        return args

    from slime.utils.arguments import parse_megatron_role_args

    return parse_megatron_role_args(args, args.megatron_config_path, role=role)


def _spilot_wandb_config_from_env() -> dict[str, str]:
    """Return validated, explicitly allowlisted SPilot experiment metadata."""

    values: dict[str, str] = {}
    for env_name, config_name in _SPILOT_WANDB_CONFIG_ENV_ALLOWLIST.items():
        raw = os.environ.get(env_name)
        if raw is None or not raw.strip():
            continue
        if raw != raw.strip():
            raise ValueError(f"{env_name} must not contain surrounding whitespace")
        values[config_name] = raw

    for key in ("spilot_sweep_arm", "spilot_process_reward_pair_id"):
        value = values.get(key)
        if value is not None and _SPILOT_SAFE_ID_RE.fullmatch(value) is None:
            raise ValueError(f"invalid {key}: expected a safe experiment identifier")

    role = values.get("spilot_process_reward_pair_role")
    if role is not None and role not in _SPILOT_PROCESS_REWARD_ROLES:
        raise ValueError(
            "spilot_process_reward_pair_role must be 'control' or 'treatment'"
        )

    matched_hash = values.get("spilot_matched_invariants_sha256")
    if matched_hash is not None and _SPILOT_SHA256_RE.fullmatch(matched_hash) is None:
        raise ValueError(
            "spilot_matched_invariants_sha256 must be 64 lowercase hexadecimal characters"
        )

    mode = values.get("spilot_process_reward_mode")
    if mode is not None and mode not in _SPILOT_PROCESS_REWARD_MODES:
        raise ValueError(
            "spilot_process_reward_mode must be 'terminal_broadcast' or 'cost_to_go'"
        )

    pair_keys = {
        "spilot_process_reward_pair_id",
        "spilot_process_reward_pair_role",
        "spilot_matched_invariants_sha256",
    }
    present_pair_keys = pair_keys.intersection(values)
    if present_pair_keys and present_pair_keys != pair_keys:
        missing = ", ".join(sorted(pair_keys - present_pair_keys))
        raise ValueError(f"incomplete SPilot process-reward pair metadata; missing: {missing}")
    if present_pair_keys:
        if "spilot_sweep_arm" not in values or "spilot_process_reward_mode" not in values:
            raise ValueError(
                "paired SPilot process-reward metadata requires sweep arm and reward mode"
            )
        expected_arm, expected_mode = _SPILOT_PAIRED_ARM_EXPECTATIONS[role]
        if (
            values["spilot_sweep_arm"] != expected_arm
            or values["spilot_process_reward_mode"] != expected_mode
        ):
            raise ValueError(
                "SPilot process-reward pair role, sweep arm, and reward mode disagree"
            )

    return values


def _compute_secondary_config_for_logging(args, role=None):
    config = _args_to_config_dict(args)
    if role == "critic":
        config = _prefix_config_keys(config, "critic")
    config.update(_spilot_wandb_config_from_env())
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
            x_label=_shared_writer_label(primary=False, role=role),
            x_update_finish_state=False,
            x_server_side_derived_summary=True,
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

    # In a resumed shared run, writer-local `_step` and arrival order are not
    # model progress.  Keep the monotonic business axes useful in the run
    # summary even when workers finish or reconnect out of order.
    wandb.define_metric("train/step", summary="max")
    wandb.define_metric("train/*", step_metric="train/step")
    if getattr(args, "wandb_always_use_train_step", False):
        wandb.define_metric("rollout/step", step_metric="train/step", summary="max")
    else:
        wandb.define_metric("rollout/step", summary="max")
    wandb.define_metric("rollout/*", step_metric=rollout_step_metric)
    wandb.define_metric("multi_turn/*", step_metric=rollout_step_metric)
    wandb.define_metric("passrate/*", step_metric=rollout_step_metric)
    _define_gpu_sidecar_metric_axes(args)
    if getattr(args, "wandb_always_use_train_step", False):
        wandb.define_metric("eval/train_step", summary="max")
    else:
        wandb.define_metric("eval/step", summary="max")
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
        # Explicit ``last`` aggregation is evaluated from history by the
        # server-side summary reducer configured for shared online runs.
        wandb.define_metric(metric_name, step_metric=step_metric, summary="last")
        _defined_metric_axes.add(cache_key)
