import logging

import ray

from slime.ray.placement_group import create_placement_groups, create_rollout_manager, create_training_models
from slime.utils.arguments import parse_args
from slime.utils.logging_utils import configure_logger, finish_tracking, init_tracking
from slime.utils.misc import should_run_periodic_action
from slime.utils.training_lifecycle import graceful_exit_due, write_training_complete_marker


logger = logging.getLogger(__name__)


# The framework supports other asynchronous approaches such as fully async (which is shown in examples/full_async).
def train(args):
    assert not args.colocate, "Colocation is not supported for async training."
    configure_logger()
    # allocate the GPUs
    pgs = create_placement_groups(args)
    init_tracking(args)

    # create the rollout manager, with sglang engines inside.
    # need to initialize rollout manager first to calculate num_rollout
    rollout_manager, num_rollout_per_epoch = create_rollout_manager(args, pgs["rollout"])

    # create the actor and critic models
    actor_model, critic_model = create_training_models(args, pgs, rollout_manager)

    # Always push actor weights to rollout once weights are loaded.
    actor_model.update_weights()

    if args.check_weight_update_equal:
        ray.get(rollout_manager.check_weights.remote(action="compare"))

    graceful_exit_deadline = getattr(args, "graceful_exit_at_unix_time", None)
    training_complete_marker = getattr(args, "training_complete_marker", None)
    if graceful_exit_deadline is not None:
        logger.info("Graceful checkpoint deadline is Unix timestamp %.0f", graceful_exit_deadline)

    # async train loop.
    completed_all_rollouts = True
    exit_before_first_rollout = False
    rollout_data_next_future = None
    if args.start_rollout_id < args.num_rollout:
        if graceful_exit_due(graceful_exit_deadline):
            completed_all_rollouts = False
            exit_before_first_rollout = True
            logger.info("Graceful deadline was reached before the next rollout started")
        else:
            rollout_data_next_future = rollout_manager.generate.remote(args.start_rollout_id)
    for rollout_id in range(args.start_rollout_id, args.num_rollout):
        if exit_before_first_rollout:
            break
        # Sync the last generation
        if rollout_data_next_future is not None:
            rollout_data_curr_ref = ray.get(rollout_data_next_future)

        # Start the next rollout early.
        if rollout_id + 1 < args.num_rollout:
            rollout_data_next_future = rollout_manager.generate.remote(rollout_id + 1)

        if args.use_critic:
            actor_trains_this_step = rollout_id >= args.num_critic_only_steps
            value_refs = critic_model.async_train(rollout_id, rollout_data_curr_ref)
            if actor_trains_this_step:
                ray.get(actor_model.async_train(rollout_id, rollout_data_curr_ref, external_data=value_refs))
            else:
                ray.get(value_refs)
        else:
            ray.get(actor_model.async_train(rollout_id, rollout_data_curr_ref))

        graceful_exit = graceful_exit_due(graceful_exit_deadline)
        save_due = should_run_periodic_action(
            rollout_id, args.save_interval, num_rollout_per_epoch, args.num_rollout
        )
        if save_due or graceful_exit:
            if (not args.use_critic) or rollout_id >= args.num_critic_only_steps:
                actor_model.save_model(
                    rollout_id,
                    force_sync=graceful_exit or rollout_id == args.num_rollout - 1,
                )
            if args.use_critic:
                critic_model.save_model(
                    rollout_id,
                    force_sync=graceful_exit or rollout_id == args.num_rollout - 1,
                )
            if args.rollout_global_dataset:
                ray.get(rollout_manager.save.remote(rollout_id))

        if graceful_exit:
            completed_all_rollouts = False
            logger.info("Graceful deadline reached after rollout %s; checkpoint is complete", rollout_id)
            break

        if (rollout_id + 1) % args.update_weights_interval == 0:
            # sync generate before update weights to prevent update weight in the middle of generation
            rollout_data_curr_ref = ray.get(x) if (x := rollout_data_next_future) is not None else None
            rollout_data_next_future = None
            actor_model.update_weights()

        if should_run_periodic_action(rollout_id, args.eval_interval, num_rollout_per_epoch):
            ray.get(rollout_manager.eval.remote(rollout_id))

    if completed_all_rollouts:
        write_training_complete_marker(training_complete_marker, num_rollout=args.num_rollout)
    ray.get(rollout_manager.dispose.remote())
    finish_tracking(args)


if __name__ == "__main__":
    args = parse_args()
    train(args)
