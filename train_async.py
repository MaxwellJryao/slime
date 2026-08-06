import logging
import math
import os
import time

import ray

from slime.ray.placement_group import (
    connect_training_models_to_rollout,
    create_placement_groups,
    create_rollout_manager,
    create_training_models,
)
from slime.ray.rollout import commit_rollout_metrics_from_journal
from slime.utils import logging_utils
from slime.utils.arguments import parse_args
from slime.utils.logging_utils import configure_logger, init_tracking
from slime.utils.misc import should_run_periodic_action
from slime.utils.startup_timing import (
    elapsed_seconds_from_env,
    launcher_startup_metrics,
)
from slime.utils.training_lifecycle import (
    final_eval_complete_marker_matches,
    graceful_exit_due,
    write_final_eval_complete_marker,
    write_training_complete_marker,
)


logger = logging.getLogger(__name__)

_DEFAULT_DISPOSE_TIMEOUT_SECONDS = 60.0


def _attach_startup_step(args, metrics: dict[str, float]) -> str:
    """Attach both axes and return the one configured for ``timing/*``."""

    num_steps_per_rollout = (
        args.rollout_batch_size * args.n_samples_per_prompt // args.global_batch_size
    )
    start_rollout_id = int(args.start_rollout_id or 0)
    first_train_step = start_rollout_id * num_steps_per_rollout
    metrics["train/step"] = first_train_step
    metrics["rollout/step"] = (
        first_train_step
        if getattr(args, "wandb_always_use_train_step", False)
        else start_rollout_id
    )
    return (
        "train/step"
        if getattr(args, "wandb_always_use_train_step", False)
        else "rollout/step"
    )


def _dispose_rollout_manager(rollout_manager) -> None:
    raw_timeout = os.environ.get(
        "SLIME_DISPOSE_TIMEOUT_SECONDS",
        str(_DEFAULT_DISPOSE_TIMEOUT_SECONDS),
    )
    try:
        timeout = float(raw_timeout)
    except ValueError:
        timeout = None

    if timeout is None or not math.isfinite(timeout) or timeout <= 0:
        logger.warning(
            "Invalid SLIME_DISPOSE_TIMEOUT_SECONDS=%r; using %.0fs",
            raw_timeout,
            _DEFAULT_DISPOSE_TIMEOUT_SECONDS,
        )
        timeout = _DEFAULT_DISPOSE_TIMEOUT_SECONDS

    dispose_ref = rollout_manager.dispose.remote()
    try:
        ray.get(dispose_ref, timeout=timeout)
    except ray.exceptions.GetTimeoutError:
        # The durable checkpoint is already complete at this point. Do not let
        # a telemetry or service shutdown bug hold an entire GPU allocation
        # until Slurm's wall-time limit.
        logger.error(
            "Rollout manager dispose exceeded %.0fs; terminating the actor",
            timeout,
        )
        ray.kill(rollout_manager, no_restart=True)


def _relay_final_eval_metrics_to_primary(args, metrics) -> None:
    """Relay a RolloutManager eval payload through the primary writer.

    RolloutManager keeps its secondary writer for backwards compatibility and
    for ordinary evals.  The final eval is also relayed to the driver so its
    critical row gets a second publication path before the durable final eval
    marker is created.  The primary stays open until ordered shutdown has
    closed trainer and rollout-manager secondaries; the marker also stores the
    payload for later recovery if W&B reports only a warning during teardown.
    """

    if not isinstance(metrics, dict):
        raise RuntimeError(
            "Final evaluation returned no metric payload for the primary tracking writer"
        )
    metrics = dict(metrics)
    step_key = (
        "eval/train_step"
        if getattr(args, "wandb_always_use_train_step", False)
        else "eval/step"
    )
    if step_key not in metrics:
        raise RuntimeError(
            f"Final evaluation metric payload is missing its {step_key!r} axis"
        )
    logging_utils.log(args, metrics, step_key=step_key)


def _update_actor_weights_with_generation_barrier(
    rollout_manager,
    actor_model,
    *,
    rollout_id: int | None = None,
):
    """Freeze custom rollout traffic across an actor -> SGLang weight sync.

    Waiting for ``RolloutManager.generate`` is not sufficient for persistent
    fully-async rollout functions: their background worker may still have
    model requests in flight.  An optional custom-rollout hook can freeze new
    traffic and drain existing traffic before the ordinary SGLang
    pause/flush/update/continue sequence begins.

    A failed pause prevents the weight update.  A failed weight update leaves
    the custom rollout paused (fail-closed); resuming could otherwise admit
    requests while SGLang is still paused or has only partially updated.
    """

    generation_paused = ray.get(
        rollout_manager.pause_generation_for_weight_update.remote()
    )
    update_succeeded = False
    try:
        result = actor_model.update_weights(rollout_id=rollout_id)
        update_succeeded = True
        return result
    finally:
        if generation_paused and update_succeeded:
            # actor_model.update_weights returns only after every SGLang engine
            # has completed continue_generation, so traffic cannot enter early.
            ray.get(rollout_manager.resume_generation_after_weight_update.remote())
        elif generation_paused:
            logger.error(
                "Actor weight update failed; leaving custom rollout generation "
                "paused because serving state may be incomplete"
            )


# The framework supports other asynchronous approaches such as fully async (which is shown in examples/full_async).
def train(args):
    assert not args.colocate, "Colocation is not supported for async training."
    train_process_started_at = time.perf_counter()
    configure_logger()
    # allocate the GPUs
    startup_metrics = launcher_startup_metrics()
    placement_group_started_at = time.perf_counter()
    pgs = create_placement_groups(args)
    startup_metrics["timing/startup_placement_group_time"] = (
        time.perf_counter() - placement_group_started_at
    )
    init_tracking(args)

    # create the rollout manager, with sglang engines inside.
    # need to initialize rollout manager first to calculate num_rollout
    parallel_model_init_started_at = time.perf_counter()
    rollout_manager_started_at = time.perf_counter()
    rollout_manager, num_rollout_per_epoch = create_rollout_manager(
        args,
        pgs["rollout"],
        wait_ready=bool(
            args.check_weight_update_equal or args.offload_rollout
        ),
    )
    # Submit the explicit engine health/router-registration barrier before
    # trainer initialization.  Ray can then load the disjoint rollout and
    # trainer GPU sets concurrently instead of leaving either side idle.
    rollout_ready_ref = rollout_manager.ready.remote()
    startup_metrics["timing/startup_rollout_manager_dispatch_time"] = (
        time.perf_counter() - rollout_manager_started_at
    )

    # create the actor and critic models
    trainer_model_init_started_at = time.perf_counter()
    actor_model, critic_model = create_training_models(
        args,
        pgs,
        rollout_manager,
        connect_rollout_manager=False,
    )
    startup_metrics["timing/startup_trainer_model_init_time"] = (
        time.perf_counter() - trainer_model_init_started_at
    )

    rollout_startup_metrics = ray.get(rollout_ready_ref)
    rollout_ready_unix_ns = time.time_ns()
    if isinstance(rollout_startup_metrics, dict):
        startup_metrics.update(rollout_startup_metrics)
    ray_submission_to_rollout_ready = elapsed_seconds_from_env(
        "SLIME_RAY_JOB_SUBMIT_UNIX_NS",
        now_ns=rollout_ready_unix_ns,
    )
    if ray_submission_to_rollout_ready is not None:
        startup_metrics["timing/startup_ray_submission_to_rollout_ready_time"] = (
            ray_submission_to_rollout_ready
        )
    startup_metrics["timing/startup_parallel_model_initialization_time"] = (
        time.perf_counter() - parallel_model_init_started_at
    )

    trainer_rollout_wiring_started_at = time.perf_counter()
    connect_training_models_to_rollout(
        args,
        actor_model,
        critic_model,
        rollout_manager,
    )
    # A previous allocation may have died after checkpoint N became durable
    # but while commit_rollout_metrics(N) was queued behind generate(N+1).
    # Replay only journals covered by the loaded checkpoint before new
    # generation starts; speculative journals beyond that frontier stay
    # uncommitted and will be replaced when their rollout is regenerated.
    recovered_rollout_metrics = ray.get(
        rollout_manager.recover_rollout_metrics.remote(args.start_rollout_id - 1)
    )
    if recovered_rollout_metrics:
        logger.info(
            "Recovered rollout metrics for checkpointed batches %s",
            recovered_rollout_metrics,
        )
    startup_metrics["timing/startup_trainer_rollout_wiring_time"] = (
        time.perf_counter() - trainer_rollout_wiring_started_at
    )

    # The initial sync needs healthy, registered rollout engines.
    initial_weight_sync_started_at = time.perf_counter()
    _update_actor_weights_with_generation_barrier(
        rollout_manager,
        actor_model,
    )
    startup_metrics["timing/startup_initial_weight_sync_time"] = (
        time.perf_counter() - initial_weight_sync_started_at
    )
    startup_metrics["timing/startup_train_process_to_ready_time"] = (
        time.perf_counter() - train_process_started_at
    )
    startup_step_key = _attach_startup_step(args, startup_metrics)
    logging_utils.log(args, startup_metrics, step_key=startup_step_key)

    if args.check_weight_update_equal:
        ray.get(rollout_manager.check_weights.remote(action="compare"))

    # Keep a fixed-set baseline at train/step=0.  The synchronous driver has
    # always supported this, but train_async previously emitted only periodic
    # post-update evals, making a before/after learning comparison impossible.
    pretrain_eval_due = False
    if args.eval_interval is not None:
        if args.num_rollout == 0:
            ray.get(
                rollout_manager.eval.remote(
                    rollout_id=0,
                    completed_train_batch=False,
                )
            )
        elif args.start_rollout_id == 0 and not args.skip_eval_before_train:
            pretrain_eval_due = True
            if not getattr(args, "concurrent_pretrain_eval", False):
                ray.get(
                    rollout_manager.eval.remote(
                        rollout_id=0,
                        completed_train_batch=False,
                    )
                )

        elif (
            getattr(args, "eval_resumed_checkpoint_before_train", False)
            and not args.skip_eval_before_train
            and 0 < args.start_rollout_id < args.num_rollout
        ):
            # Numeric checkpoint N resumes at rollout N+1. Evaluate the
            # already-synced checkpoint weights synchronously before any
            # generation is queued, and label the row as the completed
            # checkpoint rollout rather than the next untrained rollout.
            resumed_checkpoint_id = args.start_rollout_id - 1
            logger.info(
                "Running fixed evaluation for resumed checkpoint %s before rollout %s",
                resumed_checkpoint_id,
                args.start_rollout_id,
            )
            ray.get(
                rollout_manager.eval.remote(
                    resumed_checkpoint_id,
                    completed_train_batch=True,
                )
            )
    graceful_exit_deadline = getattr(args, "graceful_exit_at_unix_time", None)
    training_complete_marker = getattr(args, "training_complete_marker", None)
    final_eval_complete_marker = getattr(args, "final_eval_complete_marker", None)
    final_eval_data_sha256 = getattr(args, "final_eval_data_sha256", None)
    final_rollout_id = args.num_rollout - 1
    require_final_eval_marker = (
        final_eval_complete_marker is not None
        and args.eval_interval is not None
        and args.num_rollout > 0
    )
    if require_final_eval_marker and (
        not isinstance(final_eval_data_sha256, str)
        or len(final_eval_data_sha256) != 64
        or any(
            character not in "0123456789abcdef" for character in final_eval_data_sha256
        )
    ):
        raise ValueError(
            "--final-eval-complete-marker requires a lowercase 64-character --final-eval-data-sha256"
        )
    final_eval_marker_valid = (
        require_final_eval_marker
        and final_eval_complete_marker_matches(
            final_eval_complete_marker,
            final_rollout_id=final_rollout_id,
            model_iteration=final_rollout_id,
            num_rollout=args.num_rollout,
            eval_data_sha256=final_eval_data_sha256,
        )
    )
    pending_final_eval_metrics = None
    if graceful_exit_deadline is not None:
        logger.info(
            "Graceful checkpoint deadline is Unix timestamp %.0f",
            graceful_exit_deadline,
        )

    # A final model checkpoint is committed before its fixed-set evaluation.
    # If the allocation dies during that evaluation, checkpoint restore sets
    # start_rollout_id == num_rollout. Run only the missing eval against the
    # already-loaded final weights; do not generate or train another batch.
    if require_final_eval_marker and args.start_rollout_id > args.num_rollout:
        raise RuntimeError(
            f"Loaded checkpoint resumes at rollout {args.start_rollout_id}, beyond the configured final rollout boundary {args.num_rollout}; refusing to label the newer model as the requested final evaluation"
        )
    if (
        require_final_eval_marker
        and args.start_rollout_id == args.num_rollout
        and not final_eval_marker_valid
        and not graceful_exit_due(graceful_exit_deadline)
    ):
        logger.info(
            "Final checkpoint %s is loaded but its eval marker is absent; running eval-only recovery",
            final_rollout_id,
        )
        final_eval_metrics = ray.get(
            rollout_manager.eval.remote(
                final_rollout_id,
                require_complete=True,
            )
        )
        _relay_final_eval_metrics_to_primary(args, final_eval_metrics)
        pending_final_eval_metrics = final_eval_metrics
        final_eval_marker_valid = True

    # async train loop.
    completed_all_rollouts = True
    exit_before_first_rollout = False
    rollout_data_next_future = None
    rollout_metrics_ack_future = None
    if args.start_rollout_id < args.num_rollout:
        if graceful_exit_due(graceful_exit_deadline):
            completed_all_rollouts = False
            exit_before_first_rollout = True
            logger.info("Graceful deadline was reached before the next rollout started")
        else:
            if pretrain_eval_due and getattr(args, "concurrent_pretrain_eval", False):
                rollout_data_next_future = (
                    rollout_manager.generate_with_pretrain_eval.remote(
                        args.start_rollout_id
                    )
                )
            else:
                rollout_data_next_future = rollout_manager.generate.remote(
                    args.start_rollout_id
                )
    for rollout_id in range(args.start_rollout_id, args.num_rollout):
        if exit_before_first_rollout:
            break
        # Sync the last generation
        if rollout_data_next_future is not None:
            rollout_data_curr_ref = ray.get(rollout_data_next_future)
        if rollout_metrics_ack_future is not None:
            # The acknowledgement was queued behind the generation just
            # consumed above.  It only releases actor memory; telemetry was
            # already durably emitted by the driver.
            ray.get(rollout_metrics_ack_future)
            rollout_metrics_ack_future = None

        save_due = should_run_periodic_action(
            rollout_id, args.save_interval, num_rollout_per_epoch, args.num_rollout
        )
        # The model checkpoint tracker is the commit marker for a resumable
        # rollout. Persist the matching data-source snapshot before either
        # launching the next prefetch or saving the model, so a kill can leave
        # only an ignored orphan data snapshot, never a model pointer whose
        # exact rollout state is absent. With a graceful deadline enabled we
        # do this tiny snapshot every iteration because the deadline is tested
        # only after the in-flight training step completes.
        if args.rollout_global_dataset and (
            save_due or graceful_exit_deadline is not None
        ):
            ray.get(rollout_manager.save.remote(rollout_id))

        # Start the next rollout early, except once the graceful deadline has
        # arrived. Queuing another synchronous RolloutManager.generate call
        # would otherwise put dispose behind a minutes-long actor method and
        # leave the rollout GPUs idle during teardown.
        deadline_reached_before_train = graceful_exit_due(graceful_exit_deadline)
        if rollout_id + 1 < args.num_rollout and not deadline_reached_before_train:
            rollout_data_next_future = rollout_manager.generate.remote(rollout_id + 1)

        if args.use_critic:
            actor_trains_this_step = rollout_id >= args.num_critic_only_steps
            value_refs = critic_model.async_train(rollout_id, rollout_data_curr_ref)
            if actor_trains_this_step:
                ray.get(
                    actor_model.async_train(
                        rollout_id, rollout_data_curr_ref, external_data=value_refs
                    )
                )
            else:
                ray.get(value_refs)
        else:
            ray.get(actor_model.async_train(rollout_id, rollout_data_curr_ref))

        graceful_exit = graceful_exit_due(graceful_exit_deadline)
        if save_due or graceful_exit:
            # RolloutManager is a single ordered actor.  The speculative
            # generate(N+1) submitted above can therefore sit in front of
            # wait_pretrain_eval/commit_rollout_metrics for minutes.  Commit
            # checkpoint N before either of those calls so a long prefetch
            # cannot delay persistence of the successfully trained batch.
            #
            # The matching data-source snapshot was fsynced before prefetch
            # submission.  Force completion even with --async-save so the
            # model tracker is a durable commit marker when this block
            # returns.  Checkpoint I/O still overlaps generate(N+1), which is
            # already running on the disjoint rollout resources.
            if (not args.use_critic) or rollout_id >= args.num_critic_only_steps:
                actor_model.save_model(rollout_id, force_sync=True)
            if args.use_critic:
                critic_model.save_model(rollout_id, force_sync=True)

        # The fsynced journal was created before generate() returned.  Emit it
        # from the driver now instead of entering the RolloutManager FIFO,
        # where generate(N+1) may run for minutes.  The actor acknowledgement
        # is intentionally fire-and-overlap here and is drained only after the
        # already-needed generation wait.
        metrics_committed_from_journal = commit_rollout_metrics_from_journal(
            args,
            rollout_id,
        )
        if metrics_committed_from_journal:
            rollout_metrics_ack_future = (
                rollout_manager.acknowledge_rollout_metrics.remote(rollout_id)
            )

        if pretrain_eval_due and getattr(args, "concurrent_pretrain_eval", False):
            # Rollout 0 no longer waits for the long tail of the fixed
            # baseline, so the first actor step can use otherwise-idle trainer
            # GPUs.  Join the baseline now, before the update_weights block
            # below can expose post-train weights to any of its later turns.
            ray.get(rollout_manager.wait_pretrain_eval.remote())
            pretrain_eval_due = False

        # Rollout N+1 may already be prefetched, but only batch N has now been
        # consumed successfully by the actor.  Commit its reward/service/perf
        # metrics here so an untrained speculative batch never reaches W&B.
        # Runs without a checkpoint directory retain the legacy in-actor path.
        if not metrics_committed_from_journal:
            ray.get(rollout_manager.commit_rollout_metrics.remote(rollout_id))

        if graceful_exit:
            completed_all_rollouts = False
            if rollout_data_next_future is not None:
                ray.cancel(rollout_data_next_future)
                rollout_data_next_future = None
            logger.info(
                "Graceful deadline reached after rollout %s; checkpoint is complete",
                rollout_id,
            )
            break

        if (rollout_id + 1) % args.update_weights_interval == 0:
            # Join the actor method first; the lifecycle hook below then freezes
            # any persistent fully-async worker traffic that outlives it.
            rollout_data_curr_ref = (
                ray.get(x) if (x := rollout_data_next_future) is not None else None
            )
            rollout_data_next_future = None
            if rollout_metrics_ack_future is not None:
                ray.get(rollout_metrics_ack_future)
                rollout_metrics_ack_future = None
            _update_actor_weights_with_generation_barrier(
                rollout_manager,
                actor_model,
                rollout_id=rollout_id,
            )

        if should_run_periodic_action(
            rollout_id,
            args.eval_interval,
            num_rollout_per_epoch,
            args.num_rollout,
        ):
            if (
                rollout_id == final_rollout_id
                and (rollout_id + 1) % args.update_weights_interval != 0
            ):
                # The actor checkpoint above contains this final update, but
                # SGLang may still serve the last periodic sync. Fixed final
                # eval must observe exactly the checkpointed actor weights.
                _update_actor_weights_with_generation_barrier(
                    rollout_manager,
                    actor_model,
                    rollout_id=rollout_id,
                )
            eval_metrics = ray.get(
                rollout_manager.eval.remote(
                    rollout_id,
                    require_complete=rollout_id == final_rollout_id,
                )
            )
            if require_final_eval_marker and rollout_id == final_rollout_id:
                _relay_final_eval_metrics_to_primary(args, eval_metrics)
                pending_final_eval_metrics = eval_metrics
                final_eval_marker_valid = True

    if completed_all_rollouts:
        if rollout_metrics_ack_future is not None:
            ray.get(rollout_metrics_ack_future)
            rollout_metrics_ack_future = None
        if require_final_eval_marker and not final_eval_marker_valid:
            raise RuntimeError(
                "All rollouts are checkpointed, but the required final evaluation did not complete"
            )
    logging_utils.finish_distributed_tracking(
        args,
        actor_model,
        critic_model,
        finish_rollout_tracking=lambda: _dispose_rollout_manager(
            rollout_manager
        ),
        # Preserve the final-eval marker contract: detectable primary finish
        # failures must surface before the recoverable marker is published.
        raise_on_primary_error=pending_final_eval_metrics is not None,
    )
    if pending_final_eval_metrics is not None:
        write_final_eval_complete_marker(
            final_eval_complete_marker,
            final_rollout_id=final_rollout_id,
            model_iteration=final_rollout_id,
            num_rollout=args.num_rollout,
            eval_data_sha256=final_eval_data_sha256,
            metrics=pending_final_eval_metrics,
            primary_tracking_flush_attempted=bool(args.use_wandb),
        )
    if completed_all_rollouts:
        write_training_complete_marker(
            training_complete_marker, num_rollout=args.num_rollout
        )


def main():
    args = parse_args()
    try:
        train(args)
    except BaseException:
        # Preserve the original training exception/exit status while making
        # the shared W&B primary terminal state unambiguously unsuccessful.
        try:
            logging_utils.finish_tracking(args, exit_code=1)
        except BaseException:
            # Never let a sidecar/service teardown error mask the failure that
            # actually terminated training.
            logger.exception("Primary tracking teardown also failed")
        raise


if __name__ == "__main__":
    main()
