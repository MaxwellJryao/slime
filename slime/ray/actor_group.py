import logging
import os
from pathlib import Path
from time import perf_counter

import ray
from ray.util.placement_group import PlacementGroup
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from slime.ray.utils import NOSET_VISIBLE_DEVICES_ENV_VARS_LIST, add_default_ray_env_vars
from slime.utils import logging_utils
from slime.utils.metric_utils import set_wandb_step

logger = logging.getLogger(__name__)


def _configure_tms_preload_env(env_vars: dict[str, str], dynlib_path: str | os.PathLike[str] | None = None) -> None:
    """Configure a Ray worker to preload the TMS binary matching Torch's CUDA."""
    if dynlib_path is None:
        from torch_memory_saver.utils import get_binary_path_from_package

        dynlib_path = get_binary_path_from_package("torch_memory_saver_hook_mode_preload")

    dynlib_path = Path(dynlib_path).resolve()
    try:
        cuda_major = dynlib_path.name.rsplit("_cu", 1)[1].split(".", 1)[0]
    except IndexError as exc:
        raise RuntimeError(f"Cannot determine CUDA major from TorchMemorySaver binary: {dynlib_path}") from exc
    if not cuda_major.isdigit():
        raise RuntimeError(f"Invalid CUDA major in TorchMemorySaver binary: {dynlib_path}")

    site_packages = dynlib_path.parent
    runtime_candidates = (
        site_packages / "nvidia" / f"cu{cuda_major}" / "lib",
        site_packages / "nvidia" / "cuda_runtime" / "lib",
        site_packages / "nvidia" / "cuda_runtime" / "lib64",
    )
    runtime_lib_dir = next(
        (
            path
            for path in runtime_candidates
            if (path / f"libcudart.so.{cuda_major}").is_file()
        ),
        None,
    )
    if runtime_lib_dir is None:
        raise FileNotFoundError(
            f"Cannot find libcudart.so.{cuda_major} required by {dynlib_path}; "
            f"searched: {', '.join(map(str, runtime_candidates))}"
        )

    existing_ld_library_path = env_vars.get("LD_LIBRARY_PATH", os.environ.get("LD_LIBRARY_PATH", ""))
    ld_library_entries = [entry for entry in existing_ld_library_path.split(os.pathsep) if entry]
    runtime_lib_dir_str = str(runtime_lib_dir)
    if runtime_lib_dir_str in ld_library_entries:
        ld_library_entries.remove(runtime_lib_dir_str)
    ld_library_entries.insert(0, runtime_lib_dir_str)

    env_vars["LD_PRELOAD"] = str(dynlib_path)
    env_vars["LD_LIBRARY_PATH"] = os.pathsep.join(ld_library_entries)
    env_vars["TMS_INIT_ENABLE"] = "1"
    env_vars["TMS_INIT_ENABLE_CPU_BACKUP"] = "1"
    logger.info(
        "Configured TorchMemorySaver preload binary=%s cuda_runtime_lib=%s",
        dynlib_path,
        runtime_lib_dir,
    )


class RayTrainGroup:
    """
    A group of ray actors
    Functions start with 'async' should return list of object refs

    Args:
        args (Namespace): Arguments for the actor group.
        num_nodes (int): Number of nodes for this actor group.
        num_gpus_per_node (int): Number of gpus for this actor group.
        pg (PlacementGroup, optional): Placement group to schedule actor on.
            If none, create new placement group automatically. Defaults to None.
        num_gpus_per_actor (float, optional): Number of gpus allocated for each actor.
            If < 1.0, multiple models can share same gpu. Defaults to 1.
        resources (Dict[str, float], optional): Custom resources to allocate for each actor.
            See https://docs.ray.io/en/latest/ray-core/scheduling/resources.html
        num_resources_per_node (int, optional): Number of custom resources to allocate for each node.
            See https://docs.ray.io/en/latest/ray-core/scheduling/resources.html
    """

    def __init__(
        self,
        args,
        num_nodes,
        num_gpus_per_node,
        pg: tuple[PlacementGroup, list[int], list[int]],
        num_gpus_per_actor: float = 1,
        role: str = "actor",
        actor_cls=None,
    ) -> None:
        self.args = args
        self._num_nodes = num_nodes
        self._num_gpus_per_node = num_gpus_per_node
        self.role = role
        self._actor_cls = actor_cls

        # Allocate the GPUs for actors w/o instantiating them
        self._allocate_gpus_for_actor(pg, num_gpus_per_actor)

    def _allocate_gpus_for_actor(self, pg, num_gpus_per_actor):
        world_size = self._num_nodes * self._num_gpus_per_node

        # Use placement group to lock resources for models of same type
        assert pg is not None
        pg, reordered_bundle_indices, _reordered_gpu_ids = pg

        env_vars = {
            # because sglang will always set NCCL_CUMEM_ENABLE to 0
            # we need also set it to 0 to prevent nccl error.
            "NCCL_CUMEM_ENABLE": os.environ.get("NCCL_CUMEM_ENABLE", "0"),
            "NVTE_FP8_BLOCK_SCALING_FP32_SCALES": os.environ.get("NVTE_FP8_BLOCK_SCALING_FP32_SCALES", "1"),
            **{name: "1" for name in NOSET_VISIBLE_DEVICES_ENV_VARS_LIST},
            **self.args.train_env_vars,
        }

        if self.args.offload_train and self.args.train_backend == "megatron":
            _configure_tms_preload_env(env_vars)

        # We cannot do routing replay for critic.
        if self.args.use_routing_replay and self.role == "actor":
            env_vars["ENABLE_ROUTING_REPLAY"] = "1"

        if self._actor_cls is None:
            from slime.backends.megatron_utils.actor import MegatronTrainRayActor

            actor_impl = MegatronTrainRayActor
        else:
            actor_impl = self._actor_cls

        actor_options = {
            "num_gpus": 1,
            "runtime_env": {"env_vars": add_default_ray_env_vars(env_vars)},
        }
        if getattr(self.args, "rollout_data_transport", "object-store") == "nixl":
            actor_options["enable_tensor_transport"] = True
        TrainRayActor = ray.remote(**actor_options)(actor_impl)

        # Create worker actors
        self._actor_handlers = []
        master_addr, master_port = None, None
        for rank in range(world_size):
            actor = TrainRayActor.options(
                num_cpus=num_gpus_per_actor,
                num_gpus=num_gpus_per_actor,
                scheduling_strategy=PlacementGroupSchedulingStrategy(
                    placement_group=pg,
                    placement_group_bundle_index=reordered_bundle_indices[rank],
                ),
            ).remote(world_size, rank, master_addr, master_port)
            if rank == 0:
                master_addr, master_port = ray.get(actor.get_master_addr_and_port.remote())
            self._actor_handlers.append(actor)

    def async_init(self, args, role, with_ref=False, with_opd_teacher=False):
        """
        Allocate GPU resourced and initialize model, optimzier, local ckpt, etc.
        """
        self.args = args
        return [actor.init.remote(args, role, with_ref=with_ref, with_opd_teacher=with_opd_teacher) for actor in self._actor_handlers]

    def async_train(self, rollout_id, rollout_data_ref, external_data=None):
        """Do one rollout training. Returns a list of Ray refs (one per worker).

        For critics, each ref resolves to ``{"values": [cpu tensors...]}`` (or ``{}``
        for non-last-PP-stage workers). Actor refs resolve to ``None``.

        ``external_data`` may be a list (one item per worker) or a single dict
        broadcast to all workers.
        """
        if isinstance(external_data, list):
            assert len(external_data) == len(self._actor_handlers)
            return [actor.train.remote(rollout_id, rollout_data_ref, external_data=ed) for actor, ed in zip(self._actor_handlers, external_data, strict=False)]
        return [actor.train.remote(rollout_id, rollout_data_ref, external_data=external_data) for actor in self._actor_handlers]

    def save_model(self, rollout_id, force_sync=False):
        """Save actor model and attribute elapsed time to the producing step."""
        started = perf_counter()
        rank_metrics = ray.get([actor.save_model.remote(rollout_id, force_sync=force_sync) for actor in self._actor_handlers])
        metrics = {"timing/save_model_time": perf_counter() - started}
        for candidate in rank_metrics:
            if candidate:
                metrics.update(candidate)
                break
        step_key = set_wandb_step(
            self.args,
            metrics,
            rollout_id,
            default_step_key="rollout/step",
            completed_train_batch=True,
        )
        logging_utils.log(self.args, metrics, step_key=step_key)
        return rank_metrics

    def update_weights(self, rollout_id: int | None = None):
        """Broadcast weights and attribute its end-to-end time to this rollout.

        ``rollout_id=None`` is the initial model seed before any optimizer
        step.  It is intentionally measured only in logs and not attached to
        train/step 0.  Later syncs are logged immediately at the final optimizer
        step that produced the weights, rather than leaking into the next
        rollout's timer flush.
        """
        started = perf_counter()
        rank_metrics = ray.get([actor.update_weights.remote() for actor in self._actor_handlers])
        elapsed = perf_counter() - started

        if rollout_id is None:
            logger.info("Initial actor-to-rollout weight sync completed in %.3fs", elapsed)
        else:
            metrics: dict[str, float] = {"timing/update_weights_time": elapsed}
            for candidate in rank_metrics:
                if candidate:
                    metrics.update(candidate)
                    break
            step_key = set_wandb_step(
                self.args,
                metrics,
                rollout_id,
                default_step_key="rollout/step",
                completed_train_batch=True,
            )
            logging_utils.log(self.args, metrics, step_key=step_key)
        return rank_metrics

    def onload(self):
        return ray.get([actor.wake_up.remote() for actor in self._actor_handlers])

    def offload(self):
        return ray.get([actor.sleep.remote() for actor in self._actor_handlers])

    def clear_memory(self):
        return ray.get([actor.clear_memory.remote() for actor in self._actor_handlers])

    def set_rollout_manager(self, rollout_manager):
        return ray.get([actor.set_rollout_manager.remote(rollout_manager) for actor in self._actor_handlers])
