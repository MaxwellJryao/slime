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


def finish_tracking(args, *, raise_on_error: bool = False):
    """Finish the primary/secondary tracking client.

    Most shutdown paths keep the historical best-effort behavior.  A caller
    that is about to publish an external completion marker can opt into
    ``raise_on_error`` to propagate exceptions detected by the client.  W&B
    can report some transport failures only as warnings, so a successful
    return is a flush attempt, not proof of server-side persistence.
    """
    if not args.use_wandb:
        return
    try:
        if wandb.run is not None:
            wandb.finish()
    except Exception:
        logging.getLogger(__name__).exception("Failed to finish wandb run")
        if raise_on_error:
            raise


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
