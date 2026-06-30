"""Fail-fast validation for trainer/rollout log-probability parity."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
import torch.distributed as dist
from megatron.core import mpu

from slime.utils.types import RolloutBatch

from .cp_utils import slice_log_prob_with_cp


def _collective_device(rollout_data: RolloutBatch) -> torch.device:
    """Return a device accepted by the active distributed backend."""

    if torch.cuda.is_available():
        return torch.device("cuda", torch.cuda.current_device())
    for key in ("log_probs", "rollout_log_probs", "loss_masks", "tokens"):
        values = rollout_data.get(key)
        if isinstance(values, Sequence):
            for value in values:
                if isinstance(value, torch.Tensor):
                    return value.device
    return torch.device("cpu")


def enforce_train_rollout_logprob_abs_diff(
    args: Any,
    rollout_data: RolloutBatch,
    *,
    rollout_id: int,
) -> float | None:
    """Abort all actor ranks when trainer/rollout log-probs are incompatible.

    The check intentionally runs after the trainer's whole-rollout log-prob
    forward and before the first policy-loss backward.  Unlike the reporting
    reducer, this computes a true loss-mask-weighted token mean, so its value
    does not depend on dynamic micro-batch packing or the number of traces
    emitted by one rollout.

    Only pipeline-last ranks own the forward results.  Every actor rank still
    participates in one world all-reduce; this both aggregates DP/CP shards and
    gives every PP/TP rank the same pass/fail decision before training starts.
    TP duplicates cancel between numerator and denominator.
    """

    threshold = getattr(args, "max_train_rollout_logprob_abs_diff", None)
    if threshold is None:
        return None

    device = _collective_device(rollout_data)
    # [masked absolute-error sum, mask weight, malformed rank count,
    #  non-finite token count]. Float64 keeps the aggregate stable for long
    # agent traces and is supported by both NCCL and Gloo.
    stats = torch.zeros(4, dtype=torch.float64, device=device)
    local_detail = ""

    if mpu.is_pipeline_last_stage(ignore_virtual=True):
        train_log_probs = rollout_data.get("log_probs")
        rollout_log_probs = rollout_data.get("rollout_log_probs")
        loss_masks = rollout_data.get("loss_masks")
        total_lengths = rollout_data.get("total_lengths")
        response_lengths = rollout_data.get("response_lengths")

        fields = (train_log_probs, rollout_log_probs, loss_masks, total_lengths, response_lengths)
        if any(value is None for value in fields):
            stats[2] = 1
            local_detail = "required log_probs/rollout_log_probs/loss_masks fields are missing"
        elif len({len(value) for value in fields}) != 1:
            stats[2] = 1
            local_detail = "trainer/rollout/mask sample counts differ"
        else:
            for sample_index, (train_lp, rollout_lp, loss_mask, total_length, response_length) in enumerate(
                zip(
                    train_log_probs,
                    rollout_log_probs,
                    loss_masks,
                    total_lengths,
                    response_lengths,
                    strict=True,
                )
            ):
                if not all(isinstance(value, torch.Tensor) for value in (train_lp, rollout_lp, loss_mask)):
                    stats[2] = 1
                    local_detail = f"sample {sample_index} contains a non-tensor log-probability or mask"
                    break
                local_mask = slice_log_prob_with_cp(loss_mask, total_length, response_length)
                if train_lp.numel() != rollout_lp.numel() or train_lp.numel() != local_mask.numel():
                    stats[2] = 1
                    local_detail = (
                        f"sample {sample_index} length mismatch: trainer={train_lp.numel()} "
                        f"rollout={rollout_lp.numel()} mask={local_mask.numel()}"
                    )
                    break

                diff = (train_lp.float() - rollout_lp.float()).abs()
                mask = local_mask.to(device=diff.device, dtype=torch.float32)
                selected = mask > 0
                if selected.any():
                    finite = torch.isfinite(diff)
                    stats[3] += (selected & ~finite).sum().to(dtype=torch.float64, device=device)
                    safe_diff = torch.where(finite, diff, torch.zeros_like(diff))
                    stats[0] += (safe_diff * mask).sum().to(dtype=torch.float64, device=device)
                    stats[1] += mask.sum().to(dtype=torch.float64, device=device)

    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)

    malformed_ranks = int(stats[2].item())
    nonfinite_tokens = int(stats[3].item())
    mask_weight = stats[1].item()
    if malformed_ranks or nonfinite_tokens or mask_weight <= 0:
        detail = f"; local detail: {local_detail}" if local_detail else ""
        raise RuntimeError(
            "trainer/rollout log-probability guard could not compute a finite masked mean "
            f"before backward (rollout_id={rollout_id}, malformed_ranks={malformed_ranks}, "
            f"nonfinite_tokens={nonfinite_tokens}, mask_weight={mask_weight:g}){detail}"
        )

    mean_abs_diff = stats[0].item() / mask_weight
    if mean_abs_diff > threshold:
        raise RuntimeError(
            "trainer/rollout log-probability mismatch exceeded the fail-fast threshold before backward: "
            f"rollout_id={rollout_id}, masked_mean_abs_diff={mean_abs_diff:.6g}, "
            f"threshold={threshold:.6g}. Check model architecture/weight conversion, tokenizer and "
            "response-token alignment. In fully-async training this also includes genuine policy lag; "
            "raise the explicit threshold only after verifying that lag is expected."
        )
    return mean_abs_diff
