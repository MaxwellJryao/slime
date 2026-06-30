import logging
from argparse import Namespace
from collections.abc import Callable
from copy import deepcopy

from slime.utils import logging_utils
from slime.utils.metric_utils import set_wandb_step
from slime.utils.timer import Timer

logger = logging.getLogger(__name__)


def log_perf_data_raw(
    rollout_id: int,
    args: Namespace,
    is_primary_rank: bool,
    compute_total_fwd_flops: Callable,
    extra_metrics: dict | None = None,
) -> None:
    timer_instance = Timer()
    log_dict_raw = deepcopy(timer_instance.log_dict())
    timer_instance.reset()

    if not is_primary_rank:
        return

    log_dict = {f"timing/{key}_time": val for key, val in log_dict_raw.items()}
    if extra_metrics:
        log_dict.update(extra_metrics)

    if ("timing/actor_train_time" in log_dict) and (compute_total_fwd_flops is not None):
        total_fwd_flops = compute_total_fwd_flops(seq_lens=timer_instance.seq_lens)

        if "timing/log_probs_time" in log_dict:
            log_dict["perf/log_probs_tflops"] = total_fwd_flops / log_dict["timing/log_probs_time"]

        if "timing/ref_log_probs_time" in log_dict:
            log_dict["perf/ref_log_probs_tflops"] = total_fwd_flops / log_dict["timing/ref_log_probs_time"]

        if log_dict["timing/actor_train_time"] > 0:
            log_dict["perf/actor_train_tflops"] = 3 * total_fwd_flops / log_dict["timing/actor_train_time"]
            log_dict["perf/actor_train_tok_per_s"] = sum(timer_instance.seq_lens) / log_dict["timing/actor_train_time"]

    if "timing/train_wait_time" in log_dict and "timing/train_time" in log_dict:
        total_time = log_dict["timing/train_wait_time"] + log_dict["timing/train_time"]
        if total_time > 0:
            log_dict["timing/step_time"] = total_time
            log_dict["perf/wait_time_ratio"] = log_dict["timing/train_wait_time"] / total_time

    logger.info(f"perf {rollout_id}: {log_dict}")

    step_key = set_wandb_step(
        args,
        log_dict,
        rollout_id,
        default_step_key="rollout/step",
        completed_train_batch=True,
    )
    logging_utils.log(args, log_dict, step_key=step_key)
