import logging

import wandb

from . import wandb_utils
from .tensorboard_utils import _TensorboardAdapter

_LOGGER_CONFIGURED = False


# ref: SGLang
def configure_logger(prefix: str = ""):
    global _LOGGER_CONFIGURED
    if _LOGGER_CONFIGURED:
        return

    _LOGGER_CONFIGURED = True

    logging.basicConfig(
        level=logging.INFO,
        format=f"[%(asctime)s{prefix}] %(filename)s:%(lineno)d - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )


def init_tracking(args, primary: bool = True, **kwargs):
    if primary:
        wandb_utils.init_wandb_primary(args, **kwargs)
    else:
        wandb_utils.init_wandb_secondary(args, **kwargs)


def finish_tracking(
    args,
    *,
    raise_on_error: bool = False,
    exit_code: int | None = None,
):
    """Finish the primary/secondary tracking client.

    Most shutdown paths keep the historical best-effort behavior.  A caller
    that is about to publish an external completion marker can opt into
    ``raise_on_error`` to propagate exceptions detected by the client.  W&B
    can report some transport failures only as warnings, so a successful
    return is a flush attempt, not proof of server-side persistence.  When
    supplied, ``exit_code`` is forwarded to W&B's primary run-state update;
    shared secondary writers are configured not to update that state.
    """
    if not args.use_wandb:
        return
    try:
        if wandb.run is not None:
            wandb.finish(exit_code=exit_code)
    except Exception:
        logging.getLogger(__name__).exception("Failed to finish wandb run")
        if raise_on_error:
            raise


def finish_distributed_tracking(
    args,
    actor_model,
    critic_model,
    *,
    finish_rollout_tracking,
    raise_on_primary_error: bool = False,
):
    """Close a shared W&B run in secondary-to-primary order.

    Trainer failures are isolated so every remaining writer gets a close
    attempt.  The rollout callback is kept between trainer secondaries and the
    primary because RolloutManager owns its own secondary.  A rollout teardown
    exception is re-raised after the primary close attempt and is not masked
    by a simultaneous primary failure.
    """

    for role, model in (("actor", actor_model), ("critic", critic_model)):
        if model is None:
            continue
        try:
            model.finish_tracking()
        except Exception:
            logging.getLogger(__name__).exception(
                "Failed to finish %s trainer tracking; continuing shared-run shutdown",
                role,
            )

    rollout_failure = None
    try:
        finish_rollout_tracking()
    except Exception as exc:
        rollout_failure = (exc, exc.__traceback__)

    try:
        finish_tracking(
            args,
            raise_on_error=raise_on_primary_error,
            # A failure while disposing non-telemetry rollout state makes the
            # process fail below, so the primary must not report a successful
            # cloud terminal state first. Trainer telemetry failures are
            # intentionally isolated above and retain exit code zero.
            exit_code=1 if rollout_failure is not None else 0,
        )
    except Exception:
        if rollout_failure is None:
            raise
        logging.getLogger(__name__).exception(
            "Primary tracking finish also failed after rollout teardown failed"
        )

    if rollout_failure is not None:
        exc, traceback = rollout_failure
        raise exc.with_traceback(traceback)


# TODO further refactor, e.g. put TensorBoard init to the "init" part
def log(args, metrics, step_key: str):
    if step_key not in metrics:
        raise KeyError(f"logging step key {step_key!r} is missing from metrics; attach an explicit business axis before logging")
    if getattr(args, "wandb_always_use_train_step", False):
        if step_key not in {"train/step", "eval/train_step"}:
            raise KeyError("wandb_always_use_train_step requires business/timing records to use a training-step axis")
        payload_keys = {
            key
            for key in metrics
            if key not in {"train/step", "eval/train_step"}
            and not key.startswith("_")
        }
        eval_keys = {
            key
            for key in payload_keys
            if key.startswith("eval/") or key.startswith("timing/eval/")
        }
        if eval_keys and step_key != "eval/train_step":
            raise KeyError("evaluation metrics must use the isolated 'eval/train_step' axis")
        if step_key == "eval/train_step" and payload_keys - eval_keys:
            raise KeyError("evaluation records must not mix non-evaluation business metrics")
        if step_key == "eval/train_step" and "train/step" in metrics:
            raise KeyError("evaluation records must not publish canonical 'train/step'; delayed eval can make it move backwards")

    if args.use_wandb:
        wandb_utils.define_logged_metric_axes(metrics, step_metric=step_key)
        wandb.log(metrics)

    if args.use_tensorboard:
        metrics_except_step = {k: v for k, v in metrics.items() if k != step_key}
        _TensorboardAdapter(args).log(data=metrics_except_step, step=metrics[step_key])
