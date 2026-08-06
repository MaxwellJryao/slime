import dataclasses
import itertools
import logging
import multiprocessing
import os
import random
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import ray
import torch
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
from sglang.srt.constants import GPU_MEMORY_TYPE_CUDA_GRAPH, GPU_MEMORY_TYPE_KV_CACHE, GPU_MEMORY_TYPE_WEIGHTS

from slime.backends.sglang_utils.external import start_external_rollout_servers
from slime.backends.sglang_utils.sglang_config import ModelConfig, ServerGroupConfig, SglangConfig
from slime.backends.sglang_utils.sglang_engine import SGLangEngine
from slime.rollout.base_types import call_rollout_fn
from slime.utils import logging_utils
from slime.utils.dp_schedule import build_dp_schedule
from slime.utils.health_monitor import RolloutHealthMonitor
from slime.utils.http_utils import _wrap_ipv6, find_available_port, get_host_info, init_http_client
from slime.utils.logging_utils import configure_logger, init_tracking
from slime.utils.metric_utils import compute_pass_rate, compute_statistics, dict_add_prefix, set_wandb_step
from slime.utils.misc import Box, group_by, load_function
from slime.utils.types import Sample

from ..utils.metric_utils import has_repetition
from .rollout_validation import validate_server_group_gpu_indices
from .utils import NOSET_VISIBLE_DEVICES_ENV_VARS_LIST, Lock, add_default_ray_env_vars

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

_DEFAULT_ROLLOUT_BASE_PORT = 2048
_ROLLOUT_PORT_BLOCK_SIZE = 320
_EPHEMERAL_PORT_LOWER_FALLBACK = 32768
_ROLLOUT_METRICS_JOURNAL_VERSION = 1
_ROLLOUT_METRICS_JOURNAL_DIR = "rollout_metrics_journal"


@dataclasses.dataclass(slots=True)
class _PendingRolloutLog:
    samples: Any
    extra_metrics: dict[str, Any] | None
    rollout_time: float
    prepared_log_dict: dict[str, Any] | None = None


def _rollout_metrics_journal_paths(
    args,
    rollout_id: int,
    *,
    checkpoint_root: str | os.PathLike[str] | None = None,
) -> tuple[Path, Path] | None:
    """Return pending/emitted journal paths when checkpointing is configured."""

    root = checkpoint_root if checkpoint_root is not None else getattr(args, "save", None)
    if root is None:
        return None
    journal_dir = Path(root) / "rollout" / _ROLLOUT_METRICS_JOURNAL_DIR
    stem = f"rollout_{int(rollout_id):07d}"
    return journal_dir / f"{stem}.pending.pt", journal_dir / f"{stem}.emitted"


def _fsync_directory(path: Path) -> None:
    directory_fd = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _atomic_torch_save(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f"{path.name}.tmp.{os.getpid()}.{threading.get_ident()}"
    )
    try:
        with open(temporary, "wb") as stream:
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _atomic_write_rollout_metrics_marker(path: Path, rollout_id: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f"{path.name}.tmp.{os.getpid()}.{threading.get_ident()}"
    )
    try:
        with open(temporary, "w", encoding="utf-8") as stream:
            stream.write(
                f"version={_ROLLOUT_METRICS_JOURNAL_VERSION}\n"
                f"rollout_id={int(rollout_id)}\n"
            )
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _load_rollout_metrics_journal(
    pending_path: Path,
    expected_rollout_id: int,
) -> _PendingRolloutLog:
    payload = torch.load(pending_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise RuntimeError(f"invalid rollout metrics journal payload: {pending_path}")
    if payload.get("version") != _ROLLOUT_METRICS_JOURNAL_VERSION:
        raise RuntimeError(
            f"unsupported rollout metrics journal version in {pending_path}: "
            f"{payload.get('version')!r}"
        )
    if payload.get("rollout_id") != expected_rollout_id:
        raise RuntimeError(
            f"rollout metrics journal id mismatch in {pending_path}: "
            f"expected {expected_rollout_id}, got {payload.get('rollout_id')!r}"
        )
    pending = payload.get("pending")
    if not isinstance(pending, _PendingRolloutLog):
        raise RuntimeError(f"invalid pending rollout metrics in {pending_path}")
    return pending


def _mark_rollout_metrics_journal_emitted(
    rollout_id: int,
    pending_path: Path,
    emitted_path: Path,
) -> None:
    if not pending_path.is_file():
        raise FileNotFoundError(
            f"cannot mark missing rollout metrics payload emitted: {pending_path}"
        )
    _atomic_write_rollout_metrics_marker(emitted_path, rollout_id)


def commit_rollout_metrics_from_journal(args, rollout_id: int) -> bool:
    """Commit a fsynced rollout metric payload without entering the actor FIFO.

    Returns ``False`` when checkpoint journaling is unavailable, allowing the
    caller to use the legacy in-actor path.  The compact payload is retained
    after emission as a local durable audit record, while an emitted marker
    makes retries locally idempotent.  As with any external tracker, a process
    loss after the tracker accepts a row but before the local marker fsync may
    replay that same business step once; it cannot silently lose the record.
    """

    paths = _rollout_metrics_journal_paths(args, rollout_id)
    if paths is None:
        return False
    pending_path, emitted_path = paths
    if emitted_path.exists():
        if not pending_path.is_file():
            raise FileNotFoundError(
                f"rollout metrics emitted marker has no durable payload for "
                f"batch {rollout_id}: {pending_path}"
            )
        return True
    if not pending_path.is_file():
        raise FileNotFoundError(
            f"missing durable rollout metrics journal for batch {rollout_id}: "
            f"{pending_path}"
        )
    pending = _load_rollout_metrics_journal(pending_path, rollout_id)
    _emit_pending_rollout_log(rollout_id, args, pending)
    _mark_rollout_metrics_journal_emitted(
        rollout_id,
        pending_path,
        emitted_path,
    )
    return True


def _get_ephemeral_port_lower_bound() -> int:
    """Return the first client port the kernel may allocate automatically."""
    override = os.environ.get("SLIME_EPHEMERAL_PORT_LOWER_BOUND")
    if override is not None:
        raw_lower = override
        source = "SLIME_EPHEMERAL_PORT_LOWER_BOUND"
    else:
        path = Path(
            os.environ.get(
                "SLIME_IP_LOCAL_PORT_RANGE_PATH",
                "/proc/sys/net/ipv4/ip_local_port_range",
            )
        )
        try:
            raw_lower = path.read_text().split()[0]
        except (OSError, IndexError):
            raw_lower = str(_EPHEMERAL_PORT_LOWER_FALLBACK)
        source = str(path)

    try:
        lower = int(raw_lower)
    except ValueError as exc:
        raise ValueError(f"invalid ephemeral port lower bound from {source}: {raw_lower!r}") from exc
    if not 1024 <= lower <= 65535:
        raise ValueError(f"invalid ephemeral port lower bound from {source}: {lower}")
    return lower


def _get_rollout_base_port() -> int:
    raw_port = os.environ.get("SLIME_ROLLOUT_BASE_PORT", str(_DEFAULT_ROLLOUT_BASE_PORT))
    try:
        base_port = int(raw_port)
    except ValueError as exc:
        raise ValueError(f"SLIME_ROLLOUT_BASE_PORT must be an integer, got {raw_port!r}") from exc

    # An eight-engine TP=1 node consumes 264 ports. Keep the complete reserved
    # block below the live kernel ephemeral range: merely probing a port is not
    # enough because SGLang binds much later, after outbound sockets may have
    # claimed it.
    ephemeral_lower = _get_ephemeral_port_lower_bound()
    max_safe_base = ephemeral_lower - _ROLLOUT_PORT_BLOCK_SIZE
    if max_safe_base < _DEFAULT_ROLLOUT_BASE_PORT:
        raise ValueError(f"ephemeral ports begin at {ephemeral_lower}; no {_ROLLOUT_PORT_BLOCK_SIZE}-port rollout block fits at or above {_DEFAULT_ROLLOUT_BASE_PORT}")
    if not _DEFAULT_ROLLOUT_BASE_PORT <= base_port <= max_safe_base:
        raise ValueError(f"SLIME_ROLLOUT_BASE_PORT={base_port} is unsafe: its {_ROLLOUT_PORT_BLOCK_SIZE}-port block must fit below the ephemeral range beginning at {ephemeral_lower} (base must be between {_DEFAULT_ROLLOUT_BASE_PORT} and {max_safe_base})")
    return base_port


def _validate_allocated_rollout_port_range(port: int, consecutive: int, ephemeral_lower: int) -> None:
    """Reject a free-port scan that escaped into the ephemeral range."""
    if consecutive < 1:
        raise ValueError(f"consecutive must be positive, got {consecutive}")
    if port < _DEFAULT_ROLLOUT_BASE_PORT or port + consecutive > ephemeral_lower:
        raise RuntimeError(f"rollout port allocation [{port}, {port + consecutive - 1}] is outside the non-ephemeral range [{_DEFAULT_ROLLOUT_BASE_PORT}, {ephemeral_lower - 1}]; choose another allocation-scoped base port")


_ROLLOUT_DATA_TENSOR_DTYPES = {
    "tokens": torch.long,
    "loss_masks": torch.int,
    "rollout_log_probs": torch.float32,
    "rollout_top_p_token_ids": torch.int32,
    "rollout_top_p_token_offsets": torch.int32,
    "teacher_log_probs": torch.float32,
    "rollout_routed_experts": None,
}

_SGLANG_REQUEST_PERF_FIELDS = (
    ("request/e2e_latency", "e2e_latency"),
    ("request/queue_time", "queue_time"),
    ("decode/throughput", "decode_throughput"),
)
_SGLANG_PREFILL_PERF_FIELDS = (
    ("prefill/bootstrap_queue_duration", "pd_prefill_bootstrap_queue_duration"),
    ("prefill/bootstrap_duration", "pd_prefill_bootstrap_duration"),
    ("prefill/alloc_wait_duration", "pd_prefill_alloc_wait_duration"),
    ("prefill/forward_duration", "pd_prefill_forward_duration"),
    ("prefill/transfer_queue_duration", "pd_prefill_transfer_queue_duration"),
    ("prefill/transfer_speed_gb_s", "pd_transfer_speed_gb_s"),
    ("prefill/transfer_total_mb", "pd_transfer_total_mb"),
    ("prefill/retry_count", "pd_prefill_retry_count"),
)
_SGLANG_DECODE_PERF_FIELDS = (
    ("decode/prealloc_duration", "pd_decode_prealloc_duration"),
    ("decode/bootstrap_duration", "pd_decode_bootstrap_duration"),
    ("decode/alloc_wait_duration", "pd_decode_alloc_wait_duration"),
    ("decode/transfer_duration", "pd_decode_transfer_duration"),
    ("decode/forward_duration", "pd_decode_forward_duration"),
)


def _cpu_tensor(value, dtype: torch.dtype | None = None) -> torch.Tensor:
    if isinstance(value, np.ndarray) and not value.flags.writeable:
        value = value.copy()
    tensor = torch.as_tensor(value, dtype=dtype) if dtype is not None else torch.as_tensor(value)
    return tensor.detach().cpu().contiguous()


def _tensorize_rollout_data_for_training(rollout_data: dict[str, Any]) -> None:
    for key, dtype in _ROLLOUT_DATA_TENSOR_DTYPES.items():
        if key in rollout_data:
            rollout_data[key] = [_cpu_tensor(value, dtype=dtype) for value in rollout_data[key]]

    if "multimodal_train_inputs" in rollout_data:
        rollout_data["multimodal_train_inputs"] = [({key: _cpu_tensor(value) if isinstance(value, (np.ndarray, torch.Tensor)) else value for key, value in mm_dict.items()} if mm_dict is not None else None) for mm_dict in rollout_data["multimodal_train_inputs"]]

    if "rollout_mask_sums" in rollout_data:
        rollout_data["rollout_mask_sums"] = _cpu_tensor(
            rollout_data["rollout_mask_sums"],
            dtype=torch.float32,
        )


@dataclasses.dataclass
class ServerGroup:
    """A group of homogeneous SGLang engines with the same configuration.

    All engines in a group share the same tp_size / nodes_per_engine / pg.
    A RolloutServer may contain multiple ServerGroups (e.g. prefill vs decode
    in PD disaggregation).
    """

    args: Any
    pg: Any  # (placement_group, reordered_bundle_indices, reordered_gpu_ids)
    all_engines: list
    num_gpus_per_engine: int
    num_new_engines: int
    worker_type: str = "regular"  # "regular", "prefill", "decode", or "placeholder"
    rank_offset: int = 0  # cumulative engine count before this group
    gpu_offset: int = 0  # cumulative GPU count before this group
    sglang_overrides: dict = dataclasses.field(default_factory=dict)
    needs_offload: bool = False  # True when this group's GPUs overlap with megatron
    model_path: str | None = None  # checkpoint path for update_weights_from_disk
    router_ip: str | None = None
    router_port: int | None = None

    @property
    def nodes_per_engine(self):
        return max(1, self.num_gpus_per_engine // self.args.num_gpus_per_node)

    @property
    def engines(self):
        """Node-0 engines only (for multi-node serving)."""
        return self.all_engines[:: self.nodes_per_engine]

    def start_engines(self, port_cursors: dict[str, int] | None = None) -> tuple[list, dict[str, int]]:
        """Create Ray actors, allocate ports, and fire ``engine.init()`` without waiting.

        Returns ``(init_handles, port_cursors)`` where *init_handles* is a list
        of Ray ObjectRefs and *port_cursors* maps node address → next free port.
        The caller should ``ray.get()`` on the handles to block until the
        engines are healthy, and pass *port_cursors* to the next server group
        so that different groups on the same node don't race for ports.

        Placeholder groups (worker_type="placeholder") skip engine creation entirely.
        """
        if port_cursors is None:
            port_cursors = {}
        if self.args.debug_train_only or self.worker_type == "placeholder":
            self.num_new_engines = 0
            return [], port_cursors

        num_gpu_per_engine = min(self.num_gpus_per_engine, self.args.num_gpus_per_node)

        pg, reordered_bundle_indices, reordered_gpu_ids = self.pg
        validate_server_group_gpu_indices(
            worker_type=self.worker_type,
            gpu_offset=self.gpu_offset,
            num_gpus_per_engine=self.num_gpus_per_engine,
            num_gpu_per_engine=num_gpu_per_engine,
            num_engines=len(self.all_engines),
            num_available_gpus=len(reordered_gpu_ids),
            rollout_num_gpus=self.args.rollout_num_gpus,
            rollout_num_gpus_per_engine=self.args.rollout_num_gpus_per_engine,
        )

        RolloutRayActor = ray.remote(SGLangEngine)

        rollout_engines = []
        for i in range(len(self.all_engines)):
            if self.all_engines[i] is not None:
                continue

            global_rank = self.rank_offset + i
            num_gpus = 0.2
            num_cpus = num_gpus

            # Get the base GPU ID from placement group using gpu_offset.
            gpu_index = self.gpu_offset + i * num_gpu_per_engine
            base_gpu_id = int(reordered_gpu_ids[gpu_index])

            scheduling_strategy = PlacementGroupSchedulingStrategy(
                placement_group=pg,
                placement_group_capture_child_tasks=True,
                placement_group_bundle_index=reordered_bundle_indices[gpu_index],
            )

            env_vars = {name: "1" for name in NOSET_VISIBLE_DEVICES_ENV_VARS_LIST} | {
                key: os.environ.get(key, default_val)
                for key, default_val in {
                    "SGLANG_JIT_DEEPGEMM_PRECOMPILE": "true",
                    "SGLANG_JIT_DEEPGEMM_FAST_WARMUP": "true",
                    "SGL_DISABLE_TP_MEMORY_INBALANCE_CHECK": "true",
                    "SGLANG_DISABLE_TP_MEMORY_INBALANCE_CHECK": "true",
                    "SGLANG_MEMORY_SAVER_CUDA_GRAPH": "true",
                    "SGLANG_BATCH_INVARIANT_OPS_ENABLE_MM_FALLBACK_VARIANT": "true",
                    "SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION": "false",
                    "SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE": "false",
                    "SLIME_ENABLE_PROFILING": "true",
                }.items()
            }
            rollout_engine = RolloutRayActor.options(
                num_cpus=num_cpus,
                num_gpus=num_gpus,
                scheduling_strategy=scheduling_strategy,
                runtime_env={
                    "env_vars": add_default_ray_env_vars(env_vars),
                },
            ).remote(
                self.args,
                rank=global_rank,
                worker_type=self.worker_type,
                base_gpu_id=base_gpu_id,
                sglang_overrides=self.sglang_overrides,
                num_gpus_per_engine=self.num_gpus_per_engine,
            )

            rollout_engines.append((global_rank, rollout_engine))
            self.all_engines[i] = rollout_engine

        self.num_new_engines = len(rollout_engines)

        if self.num_new_engines == 0:
            return [], port_cursors

        # Compute base_port from the maximum cursor across all nodes that
        # this group's engines may land on (conservative: just use global max).
        base_port = max(port_cursors.values()) if port_cursors else _get_rollout_base_port()
        addr_and_ports, port_cursors = _allocate_rollout_engine_addr_and_ports_normal(
            args=self.args,
            rollout_engines=rollout_engines,
            worker_type=self.worker_type,
            num_gpus_per_engine=self.num_gpus_per_engine,
            rank_offset=self.rank_offset,
            base_port=base_port,
        )

        init_handles = [
            engine.init.remote(
                **(addr_and_ports[rank]),
                router_ip=self.router_ip,
                router_port=self.router_port,
            )
            for rank, engine in rollout_engines
        ]
        return init_handles, port_cursors

    def offload(self):
        """Fire release_memory_occupation on all engines (non-blocking).

        Returns a list of Ray ObjectRefs.  Skipped for groups that do not
        overlap with megatron GPUs (``needs_offload=False``).
        """
        if not self.needs_offload:
            return []
        return [engine.release_memory_occupation.remote() for engine in self.engines if engine is not None]

    def onload(self, tags: list[str] | None = None):
        """Fire resume_memory_occupation on all engines (non-blocking).

        Returns a list of Ray ObjectRefs.  Skipped for groups that do not
        overlap with megatron GPUs (``needs_offload=False``).
        """
        if not self.needs_offload:
            return []
        return [engine.resume_memory_occupation.remote(tags=tags) for engine in self.engines if engine is not None]

    def onload_weights_from_disk(self):
        """Reload weights from ``model_path`` for non-updatable groups.

        Used instead of ``resume_memory_occupation(tags=[WEIGHTS])`` so that
        CPU memory is not consumed by offloaded weight copies.
        """
        if not self.needs_offload or not self.model_path:
            return []
        return [engine.update_weights_from_disk.remote(self.model_path) for engine in self.engines if engine is not None]


@dataclasses.dataclass
class RolloutServer:
    """A model served behind a shared router, with one or more server groups.

    Each RolloutServer represents one model deployed behind a single router.
    A server may contain multiple ServerGroups with different
    ``num_gpus_per_engine`` (e.g. prefill TP=2, decode TP=4).
    """

    server_groups: list[ServerGroup]
    router_ip: str | None = None
    router_port: int | None = None
    model_name: str = "default"
    update_weights: bool = True

    @property
    def engines(self):
        """All node-0 engines across all groups (placeholder groups contribute nothing)."""
        return [e for g in self.server_groups for e in g.engines]

    @property
    def all_engines(self):
        """All engines (including non-node-0) across all groups."""
        return [e for g in self.server_groups for e in g.all_engines]

    @property
    def num_new_engines(self):
        return sum(g.num_new_engines for g in self.server_groups)

    @num_new_engines.setter
    def num_new_engines(self, value):
        for g in self.server_groups:
            g.num_new_engines = value

    @property
    def engine_gpu_counts(self) -> list[int]:
        """Per-engine GPU count for all node-0 engines, parallel to ``engines``."""
        return [g.num_gpus_per_engine for g in self.server_groups for _ in g.engines]

    @property
    def engine_gpu_offsets(self) -> list[int]:
        """Per-engine GPU offset for all node-0 engines, parallel to ``engines``.

        Accounts for placeholder groups that occupy GPU slots without creating engines.
        """
        offsets = []
        for g in self.server_groups:
            for j in range(len(g.engines)):
                offsets.append(g.gpu_offset + j * g.num_gpus_per_engine)
        return offsets

    @property
    def nodes_per_engine(self):
        """Nodes per engine.  Only valid when all active groups share the same value."""
        values = {g.nodes_per_engine for g in self.server_groups if g.worker_type != "placeholder"}
        if len(values) != 1:
            raise ValueError(f"Heterogeneous nodes_per_engine across groups: {values}")
        return values.pop()

    def recover(self):
        """Recover dead engines across all active groups, overlapping init."""
        # Record dead indices per group before starting.
        dead_per_group = [[i for i, engine in enumerate(g.all_engines) if engine is None] for g in self.server_groups]

        # Start all groups concurrently.
        all_handles = []
        port_cursors: dict[str, int] = {}
        for g in self.server_groups:
            handles, port_cursors = g.start_engines(port_cursors)
            all_handles.extend(handles)
        if all_handles:
            ray.get(all_handles)

        # Post-recovery: offload then onload weights for newly created engines.
        release_handles = []
        updatable_new_engines = []
        non_updatable_groups_engines: list[tuple[str, list]] = []
        for g, dead_indices in zip(self.server_groups, dead_per_group, strict=True):
            logger.info(f"Recovered {g.num_new_engines} dead rollout engines (worker_type={g.worker_type})")
            assert g.num_new_engines == len(dead_indices), "num_new_engines does not match dead_indices length"
            if g.needs_offload and dead_indices:
                new_engines = [g.all_engines[i] for i in dead_indices]
                release_handles.extend(engine.release_memory_occupation.remote() for engine in new_engines)
                if self.update_weights:
                    updatable_new_engines.extend(new_engines)
                elif g.model_path:
                    non_updatable_groups_engines.append((g.model_path, new_engines))

        if release_handles:
            ray.get(release_handles)
            # Resume GPU memory for all engines that need offload.
            all_resume_engines = updatable_new_engines[:]
            for _model_path, engines in non_updatable_groups_engines:
                all_resume_engines.extend(engines)
            if all_resume_engines:
                ray.get([engine.resume_memory_occupation.remote(tags=[GPU_MEMORY_TYPE_WEIGHTS]) for engine in all_resume_engines])

    def offload(self):
        """Release memory occupation across all groups (concurrent)."""
        handles = []
        for g in self.server_groups:
            handles.extend(g.offload())
        return ray.get(handles) if handles else []

    def onload(self, tags: list[str] | None = None):
        """Resume memory occupation across all groups (concurrent)."""
        handles = []
        for g in self.server_groups:
            handles.extend(g.onload(tags))
        return ray.get(handles) if handles else []

    def onload_weights(self):
        """Restore weights for offloaded groups.

        All groups resume from CPU cache via ``resume_memory_occupation``.
        For updatable servers, weights will be overwritten by
        ``update_weights`` shortly after.  For non-updatable servers the
        CPU backup already contains the correct (unchanged) weights.
        """
        handles = []
        for g in self.server_groups:
            if not g.needs_offload:
                continue
            handles.extend(g.onload(tags=[GPU_MEMORY_TYPE_WEIGHTS]))
        return ray.get(handles) if handles else []

    def onload_kv(self):
        """Resume KV cache and CUDA graphs for offloaded groups."""
        handles = []
        for g in self.server_groups:
            handles.extend(g.onload(tags=[GPU_MEMORY_TYPE_KV_CACHE, GPU_MEMORY_TYPE_CUDA_GRAPH]))
        return ray.get(handles) if handles else []


@ray.remote
class RolloutManager:
    """The class to run rollout and convert rollout data to training data."""

    def __init__(self, args, pg):
        remote_init_started_at = time.perf_counter()
        configure_logger()

        self.pg = pg
        self.args = args

        rollout_init_handles: list[Any] = []
        engines_started_at = time.perf_counter()
        if self.args.debug_train_only:
            self.servers: dict[str, Any] = {}
        else:
            init_http_client(args)
            self.servers, rollout_init_handles = start_rollout_servers(args, pg)

        data_source_cls = load_function(self.args.data_source_path)
        self.data_source = data_source_cls(args)

        self.generate_rollout = load_function(self.args.rollout_function_path)
        self.eval_generate_rollout = load_function(self.args.eval_function_path)
        # Optional custom-rollout lifecycle around actor -> serving weight
        # updates.  A persistent/fully-async rollout implementation can keep
        # issuing model requests after generate() has returned, so the driver
        # needs a stronger boundary than waiting for the top-level Ray future.
        # The custom function owns the mechanism (for example, freeze + drain
        # an inference proxy fleet); RolloutManager owns strict pairing.
        self._weight_update_generation_paused = False
        self.custom_reward_post_process_func = None
        if self.args.custom_reward_post_process_path is not None:
            self.custom_reward_post_process_func = load_function(self.args.custom_reward_post_process_path)
        self.custom_convert_samples_to_train_data_func = None
        if self.args.custom_convert_samples_to_train_data_path is not None:
            self.custom_convert_samples_to_train_data_func = load_function(self.args.custom_convert_samples_to_train_data_path)
        logger.info(f"import {self.args.rollout_function_path} as generate_rollout function.")
        logger.info(f"import {self.args.eval_function_path} as eval_generate_rollout function.")

        init_tracking(args, primary=False)
        # Generation is speculative in train_async: rollout N+1 can finish
        # while actor N is still training.  Hold its business metrics until
        # the driver confirms that the corresponding actor batch succeeded.
        self._pending_rollout_logs: dict[int, _PendingRolloutLog] = {}
        self._pending_rollout_log_paths: dict[int, Path] = {}
        # The fixed pre-train eval may have a much longer tail than rollout 0.
        # Keep it alive after rollout 0 is handed to the actor so the first
        # training step can overlap that tail.  The driver calls
        # ``wait_pretrain_eval`` before any updated actor weights are synced to
        # serving, preserving a true initial-policy baseline.
        self._pretrain_eval_executor: ThreadPoolExecutor | None = None
        self._pretrain_eval_future: Future[Any] | None = None
        self.rollout_engine_lock = Lock.options(
            num_cpus=1,
            num_gpus=0,
            runtime_env={"env_vars": add_default_ray_env_vars()},
        ).remote()
        self.rollout_id = -1

        self._health_monitors = []
        self._health_monitors_started = False
        self._ci_fault_injection_pending = False

        # Engine init ObjectRefs keep running after this constructor returns.
        # ``ready()`` is the explicit, idempotent health/router-registration
        # barrier.  Keeping the wait out of __init__ lets the async driver load
        # trainer and rollout models concurrently on their disjoint GPUs.
        self._rollout_init_handles = rollout_init_handles
        self._engines_started_at = engines_started_at
        self._engines_ready = False
        self._startup_timing = {
            "timing/startup_rollout_manager_remote_init_time": (time.perf_counter() - remote_init_started_at),
        }

    def set_start_rollout_id(self, start_rollout_id: int) -> int:
        """Synchronize the checkpoint-derived rollout cursor before generation.

        ``RolloutManager`` is constructed before trainer checkpoint loading so
        rollout-engine and trainer initialization can overlap. Ray therefore
        serializes an args snapshot whose ``start_rollout_id`` can still be
        ``None``. The driver learns the durable cursor from the trainer later;
        copy that derived value into the actor before loading rollout data or
        generating the first resumed batch.
        """

        normalized = int(start_rollout_id)
        if normalized < 0:
            raise ValueError("start_rollout_id must be non-negative")
        if self.rollout_id != -1:
            raise RuntimeError(
                "start_rollout_id cannot be changed after rollout generation starts"
            )

        current = getattr(self.args, "start_rollout_id", None)
        if current is not None and int(current) != normalized:
            raise RuntimeError(
                "RolloutManager start_rollout_id conflicts with the trainer checkpoint: "
                f"actor={int(current)}, trainer={normalized}"
            )
        self.args.start_rollout_id = normalized
        logger.info("Synchronized RolloutManager start_rollout_id=%d", normalized)
        return normalized

    def ready(self) -> dict[str, float]:
        """Wait once for every engine health check and router registration."""

        if not self._engines_ready:
            if self._rollout_init_handles:
                ray.get(self._rollout_init_handles)
                self._startup_timing["timing/startup_sglang_router_engines_ready_time"] = time.perf_counter() - self._engines_started_at
            self._rollout_init_handles = []
            self._engines_ready = True

        # A monitor must never inspect engines while their normal startup
        # health checks are still in flight: it could misclassify loading as a
        # crash and restart a healthy engine. Start monitors once, only after
        # the engine/router barrier above has succeeded.
        if not self._health_monitors_started:
            if not self.args.debug_train_only and self.args.use_fault_tolerance:
                for srv in self.servers.values():
                    for group in srv.server_groups:
                        monitor = RolloutHealthMonitor(group, self.args)
                        monitor.start()
                        self._health_monitors.append(monitor)
                self._ci_fault_injection_pending = self.args.ci_test
            self._health_monitors_started = True

        return dict(self._startup_timing)

    def _try_ci_fault_injection(self):
        """Try to inject fault during generate (when health monitor is running)."""
        if not self._ci_fault_injection_pending:
            return

        # Only inject fault once
        self._ci_fault_injection_pending = False

        if self.server and self.server.server_groups and self.server.server_groups[0].all_engines and self.server.server_groups[0].all_engines[0]:
            logger.info("CI Fault Injection: Simulating crash on engine 0 during generate")
            try:
                # This will cause the ray actor to exit
                self.server.server_groups[0].all_engines[0].simulate_crash.remote()
                # Wait for health monitor to detect the crash and mark engine as None
                # health_check_interval + health_check_timeout + buffer
                wait_time = self.args.rollout_health_check_interval + self.args.rollout_health_check_timeout + 5
                logger.info(f"CI Fault Injection: Waiting {wait_time}s for health monitor to detect crash")
                time.sleep(wait_time)
            except Exception as e:
                logger.warning(f"CI Fault Injection failed: {e}")

    def dispose(self):
        dispose_rollout = getattr(self.generate_rollout, "dispose", None)
        try:
            if callable(dispose_rollout):
                dispose_rollout()
        finally:
            pending_logs = getattr(self, "_pending_rollout_logs", None)
            if pending_logs:
                logger.info(
                    "Discarding uncommitted rollout metrics for speculative batches %s",
                    sorted(pending_logs),
                )
                pending_logs.clear()
                getattr(self, "_pending_rollout_log_paths", {}).clear()
            for monitor in self._health_monitors:
                monitor.stop()
            logging_utils.finish_tracking(self.args)

    def _weight_update_generation_hooks(self):
        """Return the optional, strictly paired custom-rollout hooks."""

        pause = getattr(self.generate_rollout, "pause_for_weight_update", None)
        resume = getattr(self.generate_rollout, "resume_after_weight_update", None)
        pause = pause if callable(pause) else None
        resume = resume if callable(resume) else None
        if (pause is None) != (resume is None):
            raise RuntimeError(
                "Custom rollout weight-update lifecycle is incomplete: "
                "pause_for_weight_update and resume_after_weight_update must "
                "both be callable or both be absent"
            )
        return pause, resume

    def pause_generation_for_weight_update(self) -> bool:
        """Freeze/drain custom rollout traffic before SGLang is paused.

        Returns ``True`` when a custom hook was applied and therefore needs a
        matching resume.  Hook failures propagate before any weight mutation.
        The hook itself must roll back any partial freeze if it raises.
        """

        pause, _ = self._weight_update_generation_hooks()
        if pause is None:
            return False
        if self._weight_update_generation_paused:
            raise RuntimeError("Rollout generation is already paused for a weight update")

        pause(self.args)
        self._weight_update_generation_paused = True
        return True

    def resume_generation_after_weight_update(self) -> bool:
        """Resume a custom rollout only after SGLang has continued generation.

        State is cleared only after the hook succeeds, so a failed resume is
        fail-closed and can be diagnosed or retried without admitting traffic.
        """

        _, resume = self._weight_update_generation_hooks()
        if resume is None:
            return False
        if not self._weight_update_generation_paused:
            raise RuntimeError("Rollout generation is not paused for a weight update")

        resume(self.args)
        self._weight_update_generation_paused = False
        return True

    @property
    def server(self) -> Any | None:
        """Default server (first model).  For backward compatibility."""
        if not self.servers:
            return None
        return next(iter(self.servers.values()))

    def _get_updatable_server(self) -> Any | None:
        """Return the server with ``update_weights=True``.

        When multiple updatable servers exist, returns the first one
        (multi-model weight update is not yet supported).
        """
        for srv in self.servers.values():
            if srv.update_weights:
                return srv
        return None

    @property
    def rollout_engines(self):
        """All node-0 engines across all servers / models."""
        return [e for srv in self.servers.values() for e in srv.engines]

    def get_updatable_engines_and_lock(self):
        """Return engines eligible for weight updates.

        Returns engines from the first model that has
        ``update_weights=True``.  Frozen models (reference, reward,
        etc.) are automatically excluded.
        """
        srv = self._get_updatable_server()
        engines = srv.engines if srv else []
        gpu_counts = srv.engine_gpu_counts if srv else []
        gpu_offsets = srv.engine_gpu_offsets if srv else []
        num_new = srv.num_new_engines if srv else 0
        return engines, self.rollout_engine_lock, num_new, gpu_counts, gpu_offsets

    def get_num_rollout_per_epoch(self):
        assert self.args.rollout_global_dataset
        return len(self.data_source) // self.args.rollout_batch_size

    def generate(self, rollout_id):
        start_time = time.perf_counter()
        self.rollout_id = rollout_id
        self.health_monitoring_resume()
        if self.args.ci_test and self.args.use_fault_tolerance and rollout_id >= 2:
            self._try_ci_fault_injection()
        data, metrics = self._get_rollout_data(rollout_id=rollout_id)
        self._save_debug_rollout_data(data, rollout_id=rollout_id, evaluation=False)
        samples = data
        rollout_time = time.perf_counter() - start_time
        if self.args.debug_rollout_only:
            # if debug rollout only, we don't convert samples to train data and directly return
            _log_rollout_data(
                rollout_id,
                self.args,
                data,
                metrics,
                rollout_time,
            )
            return
        conversion_started = time.perf_counter()
        data = self._convert_samples_to_train_data(data)
        conversion_time = time.perf_counter() - conversion_started
        split_started = time.perf_counter()
        train_data = self._split_train_data_by_dp(data)
        split_time = time.perf_counter() - split_started
        generate_e2e_time = time.perf_counter() - start_time
        metrics = dict(metrics or {})
        metrics.update(
            {
                "timing/rollout_to_train_data_time": conversion_time,
                "timing/dp_split_time": split_time,
                "timing/generate_e2e_time": generate_e2e_time,
            }
        )
        pending_log = _PendingRolloutLog(
            samples=samples,
            extra_metrics=metrics,
            rollout_time=rollout_time,
        )
        if rollout_id in self._pending_rollout_logs:
            raise RuntimeError(f"rollout metrics for batch {rollout_id} are already pending")
        journal_path = self._persist_pending_rollout_log(rollout_id, pending_log)
        self._pending_rollout_logs[rollout_id] = pending_log
        if journal_path is not None:
            self._pending_rollout_log_paths[rollout_id] = journal_path
        return train_data

    def generate_with_pretrain_eval(self, rollout_id):
        """Start the fixed baseline and return as soon as rollout 0 is ready.

        Evaluation does not consume the training data source.  Its future is
        retained until ``wait_pretrain_eval`` so actor training can overlap a
        slow baseline tail without allowing a weight update in the middle of
        that evaluation. Keeping this opt-in at the driver preserves
        compatibility with custom rollout functions that do not support
        concurrent train/eval HTTP traffic.
        """

        if self._pretrain_eval_future is not None:
            raise RuntimeError("a concurrent pre-train eval is already pending")
        logger.info("Running pre-train eval concurrently with rollout %s", rollout_id)
        executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="slime-pretrain")
        self._pretrain_eval_executor = executor
        # Submit eval first so its fixed sessions enter the shared runtime
        # queue before the larger fully-async training window.
        self._pretrain_eval_future = executor.submit(
            self.eval,
            rollout_id,
            completed_train_batch=False,
        )
        rollout_future = executor.submit(self.generate, rollout_id)
        try:
            return rollout_future.result()
        except BaseException:
            # The driver cannot make progress without rollout 0.  Drain the
            # paired eval thread before surfacing the generation failure so no
            # background work outlives the actor method unexpectedly.
            try:
                self.wait_pretrain_eval()
            except BaseException:
                logger.exception("Concurrent pre-train eval also failed")
            raise

    def wait_pretrain_eval(self) -> None:
        """Join a pending baseline before serving weights can be updated."""

        future = self._pretrain_eval_future
        executor = self._pretrain_eval_executor
        if future is None:
            return
        try:
            future.result()
        finally:
            self._pretrain_eval_future = None
            self._pretrain_eval_executor = None
            if executor is not None:
                executor.shutdown(wait=True, cancel_futures=False)

    def _persist_pending_rollout_log(
        self,
        rollout_id: int,
        pending: _PendingRolloutLog,
    ) -> Path | None:
        """Fsync speculative telemetry before its train data leaves the actor."""

        paths = _rollout_metrics_journal_paths(self.args, rollout_id)
        if paths is None:
            return None
        pending_path, emitted_path = paths
        if (
            pending.prepared_log_dict is None
            and getattr(self.args, "custom_rollout_log_function_path", None) is None
            and not getattr(self.args, "load_debug_rollout_data", False)
        ):
            pending.prepared_log_dict = _prepare_rollout_log_dict(
                self.args,
                pending.samples,
                pending.extra_metrics,
                pending.rollout_time,
            )
        # A rolled-back step may be generated again after a newer, uncheckpointed
        # attempt logged it.  The new payload supersedes that old emitted marker.
        try:
            emitted_path.unlink()
        except FileNotFoundError:
            pass
        journal_pending = pending
        if pending.prepared_log_dict is not None:
            # Default logging needs only this compact numeric payload.  Avoid
            # retaining prompts, responses, and token arrays in every durable
            # telemetry record.  Custom log hooks keep the original samples
            # because their data dependencies are intentionally opaque.
            journal_pending = _PendingRolloutLog(
                samples=None,
                extra_metrics=None,
                rollout_time=pending.rollout_time,
                prepared_log_dict=dict(pending.prepared_log_dict),
            )
        payload = {
            "version": _ROLLOUT_METRICS_JOURNAL_VERSION,
            "rollout_id": int(rollout_id),
            "pending": journal_pending,
        }
        _atomic_torch_save(payload, pending_path)
        return pending_path

    def _mark_rollout_metrics_emitted(
        self,
        rollout_id: int,
        pending_path: Path | None,
    ) -> None:
        if pending_path is None:
            return
        emitted_path = pending_path.with_suffix("").with_suffix(".emitted")
        _mark_rollout_metrics_journal_emitted(
            rollout_id,
            pending_path,
            emitted_path,
        )

    def commit_rollout_metrics(self, rollout_id: int) -> None:
        """Emit metrics only after the driver confirms actor-train success."""
        pending = self._pending_rollout_logs.get(rollout_id)
        if pending is None:
            raise KeyError(f"no uncommitted rollout metrics for batch {rollout_id}")
        _emit_pending_rollout_log(rollout_id, self.args, pending)
        pending_paths = getattr(self, "_pending_rollout_log_paths", {})
        pending_path = pending_paths.get(rollout_id)
        self._mark_rollout_metrics_emitted(rollout_id, pending_path)
        del self._pending_rollout_logs[rollout_id]
        pending_paths.pop(rollout_id, None)

    def acknowledge_rollout_metrics(self, rollout_id: int) -> None:
        """Release actor memory after the driver committed its journal."""

        pending = self._pending_rollout_logs.get(rollout_id)
        if pending is None:
            raise KeyError(f"no uncommitted rollout metrics for batch {rollout_id}")
        pending_path = getattr(self, "_pending_rollout_log_paths", {}).get(
            rollout_id
        )
        if pending_path is None:
            raise RuntimeError(
                f"rollout metrics for batch {rollout_id} have no durable journal"
            )
        emitted_path = pending_path.with_suffix("").with_suffix(".emitted")
        if not emitted_path.is_file():
            raise RuntimeError(
                f"rollout metrics for batch {rollout_id} were not durably emitted"
            )
        del self._pending_rollout_logs[rollout_id]
        self._pending_rollout_log_paths.pop(rollout_id, None)

    def recover_rollout_metrics(self, checkpoint_rollout_id: int) -> list[int]:
        """Replay telemetry whose successful train is proven by a checkpoint.

        The model checkpoint tracker is the acceptance commit marker.  A
        journal for a later speculative rollout is intentionally left pending
        because that batch was not made resumable and will be generated again.
        """

        recovered: list[int] = []
        if not hasattr(self, "_pending_rollout_log_paths"):
            self._pending_rollout_log_paths = {}
        roots = []
        for candidate in (getattr(self.args, "load", None), getattr(self.args, "save", None)):
            if candidate is not None and candidate not in roots:
                roots.append(candidate)
        for checkpoint_root in roots:
            paths = _rollout_metrics_journal_paths(
                self.args,
                0,
                checkpoint_root=checkpoint_root,
            )
            if paths is None:
                continue
            journal_dir = paths[0].parent
            if not journal_dir.is_dir():
                continue
            for emitted_path in sorted(journal_dir.glob("rollout_*.emitted")):
                stem = emitted_path.name.removeprefix("rollout_").removesuffix(
                    ".emitted"
                )
                try:
                    emitted_rollout_id = int(stem)
                except ValueError as exc:
                    raise RuntimeError(
                        f"invalid rollout metrics marker filename: {emitted_path}"
                    ) from exc
                pending_path = emitted_path.with_suffix(".pending.pt")
                if (
                    emitted_rollout_id <= checkpoint_rollout_id
                    and not pending_path.is_file()
                ):
                    raise FileNotFoundError(
                        "rollout metrics emitted marker has no durable payload "
                        f"for checkpointed batch {emitted_rollout_id}: {pending_path}"
                    )
            for pending_path in sorted(journal_dir.glob("rollout_*.pending.pt")):
                stem = pending_path.name.removeprefix("rollout_").removesuffix(
                    ".pending.pt"
                )
                try:
                    rollout_id = int(stem)
                except ValueError as exc:
                    raise RuntimeError(
                        f"invalid rollout metrics journal filename: {pending_path}"
                    ) from exc
                if rollout_id > checkpoint_rollout_id or rollout_id in recovered:
                    continue
                emitted_path = pending_path.with_suffix("").with_suffix(".emitted")
                if emitted_path.exists():
                    continue
                pending = _load_rollout_metrics_journal(pending_path, rollout_id)
                if rollout_id in self._pending_rollout_logs:
                    raise RuntimeError(
                        f"rollout metrics for batch {rollout_id} are already pending"
                    )
                self._pending_rollout_logs[rollout_id] = pending
                self._pending_rollout_log_paths[rollout_id] = pending_path
                logger.info(
                    "Replaying crash-recovered rollout metrics for checkpoint %s",
                    rollout_id,
                )
                self.commit_rollout_metrics(rollout_id)
                recovered.append(rollout_id)
        return recovered

    def eval(
        self,
        rollout_id,
        *,
        completed_train_batch: bool = True,
        require_complete: bool = False,
    ):
        if self.args.debug_train_only:
            # if debug train only, we don't generate evaluation data
            return
        self.health_monitoring_resume()

        result = call_rollout_fn(self.eval_generate_rollout, self.args, rollout_id, self.data_source, evaluation=True)
        data = result.data
        self._save_debug_rollout_data(data, rollout_id=rollout_id, evaluation=True)
        return _log_eval_rollout_data(
            rollout_id,
            self.args,
            data,
            result.metrics,
            completed_train_batch=completed_train_batch,
            require_complete=require_complete,
        )

    def save(self, rollout_id):
        self.data_source.save(rollout_id)

    def load(self, rollout_id=None):
        self.data_source.load(rollout_id)

    def offload(self):
        self.health_monitoring_pause()
        for srv in self.servers.values():
            srv.offload()

    def onload(self, tags: list[str] | None = None):
        for srv in self.servers.values():
            srv.onload(tags)

    def onload_weights(self):
        for srv in self.servers.values():
            srv.onload_weights()

    def onload_kv(self):
        for srv in self.servers.values():
            srv.onload_kv()

    def recover_updatable_engines(self):
        """Restart any dead rollout engines and update num_new_engines for update_weights detection.

        Recovers the updatable model (the one that receives weight
        updates from training).
        """
        self.health_monitoring_pause()
        srv = self._get_updatable_server()
        if self.rollout_id == -1 or srv is None:
            engines = srv.engines if srv else []
            gpu_counts = srv.engine_gpu_counts if srv else []
            gpu_offsets = srv.engine_gpu_offsets if srv else []
            return engines, self.rollout_engine_lock, (srv.num_new_engines if srv else 0), gpu_counts, gpu_offsets

        srv.recover()
        return (
            srv.engines,
            self.rollout_engine_lock,
            srv.num_new_engines,
            srv.engine_gpu_counts,
            srv.engine_gpu_offsets,
        )

    def clear_updatable_num_new_engines(self):
        # when fault tolerance is not enabled, we need to manually clear num_new_engines after update_weights
        srv = self._get_updatable_server()
        if srv:
            srv.num_new_engines = 0

    def health_monitoring_pause(self) -> None:
        for monitor in self._health_monitors:
            monitor.pause()

    def health_monitoring_resume(self) -> None:
        for monitor in self._health_monitors:
            monitor.resume()

    def check_weights(self, action: str):
        return ray.get([engine.check_weights.remote(action=action) for engine in self.rollout_engines])

    def _get_rollout_data(self, rollout_id):
        if self.args.load_debug_rollout_data:
            data = torch.load(
                self.args.load_debug_rollout_data.format(rollout_id=rollout_id),
                weights_only=False,
            )["samples"]
            data = [Sample.from_dict(sample) for sample in data]
            if (ratio := self.args.load_debug_rollout_data_subsample) is not None:
                original_num_rows = len(data)
                rough_subsample_num_rows = int(original_num_rows * ratio)
                data = data[: rough_subsample_num_rows // 2] + data[-rough_subsample_num_rows // 2 :]
                logger.info(f"Subsample loaded debug rollout data using {ratio=} and change num rows {original_num_rows} -> {len(data)}")
            metrics = None
        else:
            data = call_rollout_fn(self.generate_rollout, self.args, rollout_id, self.data_source, evaluation=False)
            metrics = data.metrics
            data = data.samples
            # Enforce the rollout_id contract before flattening: any list[Sample]
            # encountered in the nested output must have rollout_id set on every
            # element. Default rollouts inherit it from the data source; compact /
            # subagent paths that split one rollout into N training samples must
            # set the same rollout_id on every sibling so the loss reducer counts
            # the rollout once instead of N times.
            _validate_rollout_id_annotated(data)
            # flatten the data if it is a list of lists
            while isinstance(data[0], list):
                data = list(itertools.chain.from_iterable(data))

        return data, metrics

    def _save_debug_rollout_data(self, data, rollout_id, evaluation: bool):
        # TODO to be refactored (originally Buffer._set_data)
        if (path_template := self.args.save_debug_rollout_data) is not None:
            path = Path(path_template.format(rollout_id=("eval_" if evaluation else "") + str(rollout_id)))
            logger.info(f"Save debug rollout data to {path}")
            path.parent.mkdir(parents=True, exist_ok=True)

            # TODO may improve the format
            if evaluation:
                dump_data = dict(samples=[sample.to_dict() for dataset_name, info in data.items() for sample in info["samples"]])
            else:
                dump_data = dict(
                    samples=[sample.to_dict() for sample in data],
                )

            torch.save(dict(rollout_id=rollout_id, **dump_data), path)

    def _post_process_rewards(self, samples: list[Sample] | list[list[Sample]]):
        if self.custom_reward_post_process_func is not None:
            return self.custom_reward_post_process_func(self.args, samples)

        raw_rewards = [sample.get_reward_value(self.args) for sample in samples]
        if self.args.advantage_estimator in ["grpo", "gspo", "cispo", "reinforce_plus_plus_baseline"] and self.args.rewards_normalization:
            # group norm
            rewards = torch.tensor(raw_rewards, dtype=torch.float)
            if rewards.shape[-1] == self.args.n_samples_per_prompt * self.args.rollout_batch_size:
                rewards = rewards.reshape(-1, self.args.n_samples_per_prompt)
            else:
                # when samples count are not equal in each group
                rewards = rewards.view(-1, rewards.shape[-1])
            mean = rewards.mean(dim=-1, keepdim=True)
            rewards = rewards - mean

            if self.args.advantage_estimator in ["grpo", "gspo", "cispo"] and self.args.grpo_std_normalization:
                std = rewards.std(dim=-1, keepdim=True)
                rewards = rewards / (std + 1e-6)

            return raw_rewards, rewards.flatten().tolist()

        return raw_rewards, raw_rewards

    def _convert_samples_to_train_data(self, samples: list[Sample] | list[list[Sample]]):
        """
        Convert inference generated samples to training data.
        """
        if self.custom_convert_samples_to_train_data_func is not None:
            return self.custom_convert_samples_to_train_data_func(self.args, samples)

        raw_rewards, rewards = self._post_process_rewards(samples)

        assert len(raw_rewards) == len(samples)
        assert len(rewards) == len(samples)

        rollout_ids = [sample.rollout_id for sample in samples]
        existed_rollout_id_values = set(rid for rid in rollout_ids if rid is not None)
        tmp_id = 0
        for i in range(len(rollout_ids)):
            if rollout_ids[i] is None:
                while tmp_id in existed_rollout_id_values:
                    tmp_id += 1
                rollout_ids[i] = tmp_id
                existed_rollout_id_values.add(tmp_id)

        train_data = {
            "tokens": [sample.tokens for sample in samples],
            "response_lengths": [sample.response_length for sample in samples],
            # some reward model, e.g. remote rm, may return multiple rewards,
            # we could use key to select the reward.
            "rewards": rewards,
            "raw_reward": raw_rewards,
            "truncated": [1 if sample.status == Sample.Status.TRUNCATED else 0 for sample in samples],
            "sample_indices": [sample.index for sample in samples],
            "rollout_ids": rollout_ids,
        }

        # loss mask
        # TODO: compress the loss mask
        loss_masks = []
        for sample in samples:
            # always instantiate loss_mask if not provided
            if sample.loss_mask is None:
                sample.loss_mask = [1] * sample.response_length

            assert len(sample.loss_mask) == sample.response_length, f"loss mask length {len(sample.loss_mask)} != response length {sample.response_length}"
            if sample.remove_sample:
                sample.loss_mask = [0] * sample.response_length
            loss_masks.append(sample.loss_mask)
        train_data["loss_masks"] = loss_masks

        # Per-rollout aggregate, precomputed at the step level (where we can
        # see every sample of every rollout) and broadcast per-sample so the
        # per-mb loss reducer uses the correct whole-rollout denominator even
        # when a rollout's samples land in different micro-batches (first-fit
        # packing can split a rollout across mbs):
        #
        #   ``rollout_mask_sums[i]`` — sum of loss-mask totals over every
        #   sample in sample i's rollout. Used as the reducer's denominator
        #   so summing partial contributions across mbs yields one
        #   token-weighted mean per rollout.
        rollout_id_list = train_data["rollout_ids"]
        mask_sums_per_sample = [sum(m) for m in loss_masks]
        rollout_total_mask: dict[int, int] = {}
        for rid, ms in zip(rollout_id_list, mask_sums_per_sample, strict=True):
            rollout_total_mask[rid] = rollout_total_mask.get(rid, 0) + ms
        train_data["rollout_mask_sums"] = [rollout_total_mask[rid] for rid in rollout_id_list]

        # Overwrite raw_reward when available. Mixed-source batches may only
        # populate this field for a subset of samples (e.g. SWE but not code).
        if any(sample.metadata and "raw_reward" in sample.metadata for sample in samples):
            train_data["raw_reward"] = [sample.metadata["raw_reward"] if sample.metadata and "raw_reward" in sample.metadata else sample.reward for sample in samples]

        # For rollout buffer
        if samples[0].metadata and "round_number" in samples[0].metadata:
            train_data["round_number"] = [sample.metadata["round_number"] for sample in samples]

        # Add rollout log probabilities for off-policy correction
        if samples[0].rollout_log_probs is not None:
            train_data["rollout_log_probs"] = [sample.rollout_log_probs for sample in samples]

        if samples[0].rollout_top_p_token_ids is not None:
            for sample in samples:
                assert sample.rollout_top_p_token_ids is not None
                assert sample.rollout_top_p_token_offsets is not None
                assert len(sample.rollout_top_p_token_offsets) == sample.response_length + 1, f"top-p token offsets length {len(sample.rollout_top_p_token_offsets)} != response length + 1 {sample.response_length + 1}"
                offset_end = int(sample.rollout_top_p_token_offsets[-1])
                assert offset_end == len(sample.rollout_top_p_token_ids), f"top-p token offsets[-1] {offset_end} != token ids length {len(sample.rollout_top_p_token_ids)}"
            train_data["rollout_top_p_token_ids"] = [sample.rollout_top_p_token_ids for sample in samples]
            train_data["rollout_top_p_token_offsets"] = [sample.rollout_top_p_token_offsets for sample in samples]

        if samples[0].rollout_routed_experts is not None:
            train_data["rollout_routed_experts"] = [sample.rollout_routed_experts for sample in samples]

        if any(sample.train_metadata is not None for sample in samples):
            train_data["metadata"] = [sample.train_metadata for sample in samples]

        if any(sample.multimodal_train_inputs is not None for sample in samples):
            train_data["multimodal_train_inputs"] = [sample.multimodal_train_inputs for sample in samples]

        if samples[0].teacher_log_probs is not None:
            train_data["teacher_log_probs"] = [sample.teacher_log_probs for sample in samples]

        return train_data

    def set_train_parallel_config(self, config: dict):
        self.train_parallel_config = config

    def _split_train_data_by_dp(self, data):
        """Compute the DP/mbs schedule and package each rank's rollout_data
        into a Ray Box. The schedule itself is computed by
        :func:`build_dp_schedule` so it stays unit-testable without Ray/sglang.

        Step split is by rollout id (``samples[i].rollout_id``, falling back
        to ``samples[i].index``); each step holds exactly
        ``args.global_batch_size`` rollouts so the training-step count per
        rollout is fixed at ``rollout_batch_size * n_samples_per_prompt //
        global_batch_size`` regardless of how many training samples each
        rollout produced.
        """
        dp_size = self.train_parallel_config["dp_size"]
        total_lengths = [len(t) for t in data["tokens"]]
        data["total_lengths"] = total_lengths

        partitions, micro_batch_indices, num_microbatches, global_batch_sizes = build_dp_schedule(
            self.args,
            self.train_parallel_config,
            total_lengths,
            global_batch_size=self.args.global_batch_size,
            rollout_indices=data["rollout_ids"],
            trainable_samples=[any(mask) for mask in data["loss_masks"]],
        )

        # Package per-rank rollout_data
        rollout_data_refs = []
        for r in range(dp_size):
            partition = partitions[r]
            rollout_data = {"partition": partition}
            for key in [
                "tokens",
                "multimodal_train_inputs",
                "response_lengths",
                "rewards",
                "truncated",
                "loss_masks",
                "round_number",
                "metadata",
                "sample_indices",
                "rollout_ids",
                "rollout_mask_sums",
                "rollout_log_probs",
                "rollout_top_p_token_ids",
                "rollout_top_p_token_offsets",
                "rollout_routed_experts",
                "prompt",
                "teacher_log_probs",
            ]:
                if key not in data:
                    continue
                rollout_data[key] = [data[key][j] for j in partition]
            # keys that need to be splited at train side
            for key in ["raw_reward", "total_lengths"]:
                if key not in data:
                    continue
                rollout_data[key] = data[key]
            rollout_data["global_batch_sizes"] = global_batch_sizes
            rollout_data["num_microbatches"] = num_microbatches
            rollout_data["micro_batch_indices"] = micro_batch_indices[r]
            _tensorize_rollout_data_for_training(rollout_data)
            transport = getattr(self.args, "rollout_data_transport", "object-store")
            if transport == "nixl":
                rollout_data_refs.append(Box(ray.put(rollout_data, _tensor_transport="nixl")))
            elif transport == "object-store":
                rollout_data_refs.append(Box(ray.put(rollout_data)))
            else:
                raise ValueError(f"Unsupported rollout data transport: {transport!r}")
        return rollout_data_refs


def _validate_rollout_id_annotated(node, depth=0):
    """Walk the rollout function's nested output and validate ``rollout_id`` only
    when a compact / subagent pattern is detected.

    "Compact" = the rollout function wraps multiple training samples from one
    rollout execution into a ``list[Sample]``. In slime's convention the
    default rollout shape is ``list[list[Sample]]`` (depth-2: prompt × rollout)
    so its leaf ``list[Sample]`` lands at depth 1 and we skip validation,
    preserving backward compatibility. A compact rollout adds a third level:
    ``list[list[list[Sample]]]`` (prompt × rollout × samples-from-one-rollout),
    so the leaf ``list[Sample]`` lands at depth ≥ 2. At that point we require
    every sibling to carry a non-None ``rollout_id`` and to share the same
    value, so the loss reducer counts the rollout once instead of N times.
    """
    if isinstance(node, Sample):
        return
    assert isinstance(node, list), f"unexpected rollout output node type: {type(node).__name__}"
    if node and isinstance(node[0], Sample):
        if depth >= 2 and len(node) > 1:
            rids = [s.rollout_id for s in node]
            missing = [i for i, r in enumerate(rids) if r is None]
            assert not missing, f"Compact rollout returned {len(node)} samples but rollout_id is unset on positions {missing}. Set Sample.rollout_id on every sibling so the loss reducer can aggregate them as one rollout instead of N."
            assert len(set(rids)) == 1, f"Sibling samples from one compact rollout must share rollout_id; got {rids}."
        return
    for item in node:
        _validate_rollout_id_annotated(item, depth + 1)


def _allocate_rollout_engine_addr_and_ports_normal(
    *,
    args,
    rollout_engines,
    worker_type="regular",
    num_gpus_per_engine=None,
    rank_offset=0,
    base_port=15000,
):
    # get ports
    # there are 4 ports we need to allocate
    # 1. server port
    # 2. nccl port
    # 3. dist_init_addr port
    # 4. other ports for dp_attention, which is of size 4 + dp_size
    _gpus_per_engine = num_gpus_per_engine or args.rollout_num_gpus_per_engine
    addr_and_ports: dict[int, dict] = {}
    ephemeral_lower = _get_ephemeral_port_lower_bound()

    # Track per-node port cursors so that different server groups (called
    # sequentially) never race for the same ports on a given node.
    node_port_cursor: dict[str, int] = {}

    # A non-colocated rollout does not necessarily begin at a physical-node
    # boundary. For example, a four-GPU actor in a 4x8-GPU allocation leaves
    # GPUs 4--7 on its node for rollout, followed by three complete rollout
    # nodes. Deriving the node from ``local_rank // num_gpus_per_node`` assigns
    # the actor node's address to the first four engines on the next node. Ask
    # every Ray actor where it actually landed and group by that address.
    engine_locations = ray.get([engine._get_current_node_ip.remote() for _, engine in rollout_engines])
    engines_by_host: dict[str, list[tuple[int, Any]]] = {}
    for (rank, engine), host in zip(rollout_engines, engine_locations, strict=True):
        engines_by_host.setdefault(host, []).append((rank, engine))

    def get_port(host: str, engine, consecutive: int = 1) -> int:
        # The validated base reserves a block below the live kernel ephemeral
        # range. Ray worker ports are skipped by the free-port probe when they
        # overlap this low range.
        start_port = node_port_cursor.get(host, base_port)
        actual_host, port = ray.get(
            engine._get_current_node_ip_and_free_port.remote(
                start_port=start_port,
                consecutive=consecutive,
            )
        )
        if actual_host != host:
            raise RuntimeError(f"rollout engine moved from {host} to {actual_host} while allocating ports")
        _validate_allocated_rollout_port_range(port, consecutive, ephemeral_lower)
        node_port_cursor[host] = port + consecutive
        return port

    for host, engines_on_host in engines_by_host.items():
        for rank, engine in engines_on_host:
            addr_and_ports[rank] = {
                "host": host,
                "port": get_port(host, engine),
                "nccl_port": get_port(host, engine),
            }
            if worker_type == "prefill":
                addr_and_ports[rank]["disaggregation_bootstrap_port"] = get_port(host, engine)

    if _gpus_per_engine > args.num_gpus_per_node:
        nodes_per_engine = _gpus_per_engine // args.num_gpus_per_node
        engines_by_rank = {rank: (engine, host) for host, engines in engines_by_host.items() for rank, engine in engines}
        for rank, (engine, host) in engines_by_rank.items():
            local_rank = rank - rank_offset
            if local_rank % nodes_per_engine != 0:
                continue
            dist_init_addr = f"{host}:{get_port(host, engine, 30 + args.sglang_dp_size)}"
            for component_rank in range(rank, rank + nodes_per_engine):
                if component_rank in addr_and_ports:
                    addr_and_ports[component_rank]["dist_init_addr"] = dist_init_addr
    else:
        for host, engines_on_host in engines_by_host.items():
            for rank, engine in engines_on_host:
                addr_and_ports[rank]["dist_init_addr"] = f"{host}:{get_port(host, engine, 30 + args.sglang_dp_size)}"

    for i, _ in rollout_engines:
        for key in ["port", "nccl_port", "dist_init_addr"]:
            assert key in addr_and_ports[i], f"Engine {i} {key} is not set."
        logger.info(f"Ports for engine {i}: {addr_and_ports[i]}")

    return addr_and_ports, node_port_cursor


def _start_router(args, *, has_pd_disaggregation: bool = False, force_new: bool = False) -> tuple[str, int]:
    """Start sglang_router and return (router_ip, router_port).

    If ``args.sglang_router_ip`` is already set (e.g. by the user) and
    ``force_new`` is False, skip launching and return the existing values.
    When ``force_new`` is True (multi-model), always allocate a fresh port.
    """
    if not force_new and args.sglang_router_ip is not None:
        return args.sglang_router_ip, args.sglang_router_port

    router_ip = _wrap_ipv6(get_host_info()[1])
    if force_new:
        router_port = find_available_port(random.randint(3000, 4000))
    else:
        router_port = args.sglang_router_port
        if router_port is None:
            router_port = find_available_port(random.randint(3000, 4000))

    from sglang_router.launch_router import RouterArgs

    from slime.utils.http_utils import run_router

    router_args = RouterArgs.from_cli_args(args, use_router_prefix=True)
    router_args.host = router_ip
    router_args.port = router_port
    router_args.prometheus_port = find_available_port(random.randint(4000, 5000))
    router_args.log_level = "warn"
    router_args.request_timeout_secs = args.sglang_router_request_timeout_secs

    if has_pd_disaggregation:
        router_args.pd_disaggregation = True
        # Disable circuit breaker to prevent RDMA transfer timeouts from
        # marking decode workers as dead. Timeouts are transient (PCIe
        # contention under high load) and do not indicate a dead server.
        router_args.disable_circuit_breaker = True

    # We will not use the health check from router.
    router_args.disable_health_check = True

    logger.info(f"Launch router with args: {router_args}")

    process = multiprocessing.Process(
        target=run_router,
        args=(router_args,),
    )
    process.daemon = True  # Set the process as a daemon
    process.start()
    # Wait 3 seconds
    time.sleep(3)
    assert process.is_alive()
    logger.info(f"Router launched at {router_ip}:{router_port}, Prometheus port: {router_args.prometheus_port}")
    return router_ip, router_port


def _compute_rollout_offset(args) -> int:
    """Offset (in PG bundle slots) where rollout GPUs start."""
    if args.debug_train_only or args.debug_rollout_only or args.colocate:
        return 0
    offset = args.actor_num_nodes * args.actor_num_gpus_per_node
    return offset


def _compute_megatron_num_gpus(args) -> int:
    """Total number of megatron (actor + critic) GPU slots in the placement group."""
    if args.debug_rollout_only:
        return 0
    num = args.actor_num_nodes * args.actor_num_gpus_per_node
    return num


def start_rollout_servers(args, pg) -> tuple[dict[str, Any], list[Any]]:
    """Start rollout servers without waiting for final engine initialization.

    Each model defined in the sglang config gets its own router and set
    of server groups.  Server groups within a model may have different
    ``num_gpus_per_engine`` (e.g. for PD disaggregation where prefill
    and decode use different TP sizes).

    Returns ``(servers, init_handles)`` where servers maps model name to
    ``RolloutServer`` and init_handles contains pending ``engine.init`` refs.

    Note: ``init_http_client`` should be called separately before this,
    as the HTTP client is shared across all servers.
    """
    if args.rollout_external:
        return start_external_rollout_servers(args, start_router=_start_router)

    config = _resolve_sglang_config(args)

    servers: dict[str, RolloutServer] = {}
    pending_init_handles: list[Any] = []
    gpu_offset = 0
    engine_offset = 0

    # Compute megatron GPU range for per-group offload decisions.
    rollout_pg_offset = _compute_rollout_offset(args)
    megatron_num_gpus = _compute_megatron_num_gpus(args)

    for model_idx, model_cfg in enumerate(config.models):
        model_cfg.resolve(args)

        has_pd = model_cfg.has_pd_disaggregation
        router_ip, router_port = _start_router(args, has_pd_disaggregation=has_pd, force_new=(model_idx > 0))

        # Write back for backward compat (first model only).
        if model_idx == 0:
            args.sglang_router_ip = router_ip
            args.sglang_router_port = router_port

        server_groups: list[ServerGroup] = []
        port_cursors: dict[str, int] = {}

        has_epd = model_cfg.has_encoder_disaggregation

        def _make_group(group_cfg, router_ip, router_port, overrides_extra=None):
            nonlocal engine_offset, gpu_offset
            gpus_per_engine = group_cfg.num_gpus_per_engine
            num_gpu_per_engine_local = min(gpus_per_engine, args.num_gpus_per_node)
            num_engines = group_cfg.num_gpus // num_gpu_per_engine_local

            group_abs_start = rollout_pg_offset + gpu_offset
            needs_offload = args.offload_rollout and group_abs_start < megatron_num_gpus
            overrides = dict(group_cfg.overrides)
            if overrides_extra:
                for k, v in overrides_extra.items():
                    overrides.setdefault(k, v)
            if args.offload_rollout and not needs_offload:
                overrides.setdefault("enable_memory_saver", False)
            logger.info(f"Engine group '{group_cfg.worker_type}' gpu_offset={gpu_offset} (abs={group_abs_start}): needs_offload={needs_offload}")

            group = ServerGroup(
                args=args,
                pg=pg,
                all_engines=[None] * num_engines if group_cfg.worker_type != "placeholder" else [],
                num_gpus_per_engine=gpus_per_engine,
                num_new_engines=0,
                worker_type=group_cfg.worker_type,
                rank_offset=engine_offset,
                gpu_offset=gpu_offset,
                sglang_overrides=overrides,
                needs_offload=needs_offload,
                model_path=overrides.get("model_path", args.hf_checkpoint),
                router_ip=router_ip,
                router_port=router_port,
            )
            engine_offset += num_engines
            gpu_offset += group_cfg.num_gpus
            return group

        if has_epd:
            # --- Phase 1: start encoder groups, wait, collect URLs ---
            # Encoder URLs are injected into the non-encoder workers' server args,
            # so this phase must stay synchronous even though final LLM init is deferred.
            encoder_urls: list[str] = []
            for group_cfg in model_cfg.server_groups:
                if group_cfg.worker_type != "encoder":
                    continue
                group = _make_group(group_cfg, router_ip, router_port)
                handles, port_cursors = group.start_engines(port_cursors)
                if handles:
                    ray.get(handles)
                urls = ray.get([e.get_url.remote() for e in group.engines])
                encoder_urls.extend(u for u in urls if u is not None)
                server_groups.append(group)

            logger.info(f"EPD phase 1 done: collected {len(encoder_urls)} encoder URLs: {encoder_urls}")

            # --- Phase 2: start non-encoder groups, injecting encoder URLs into
            # language-only LLM workers. Prefill groups use this for full EPD,
            # while regular groups allow encoder/LLM split without PD.
            non_encoder_handles: list = []
            for group_cfg in model_cfg.server_groups:
                if group_cfg.worker_type == "encoder":
                    continue
                overrides_extra = {}
                if encoder_urls and group_cfg.worker_type in ("prefill", "regular"):
                    overrides_extra["language_only"] = True
                    overrides_extra["encoder_urls"] = encoder_urls
                group = _make_group(group_cfg, router_ip, router_port, overrides_extra=overrides_extra)
                handles, port_cursors = group.start_engines(port_cursors)
                non_encoder_handles.extend(handles)
                server_groups.append(group)

            pending_init_handles.extend(non_encoder_handles)
        else:
            # No EPD — start all groups in one pass (original path).
            all_init_handles: list = []
            for group_cfg in model_cfg.server_groups:
                group = _make_group(group_cfg, router_ip, router_port)
                handles, port_cursors = group.start_engines(port_cursors)
                all_init_handles.extend(handles)
                server_groups.append(group)

            pending_init_handles.extend(all_init_handles)

        servers[model_cfg.name] = RolloutServer(
            server_groups=server_groups,
            router_ip=router_ip,
            router_port=router_port,
            model_name=model_cfg.name,
            update_weights=model_cfg.update_weights,
        )

    # Expose per-model router info for custom rollout functions.
    args.sglang_model_routers = {name: (srv.router_ip, srv.router_port) for name, srv in servers.items()}

    return servers, pending_init_handles


def _resolve_sglang_config(args) -> SglangConfig:
    """Build a SglangConfig from args, choosing the right source."""
    if getattr(args, "sglang_config", None) is not None:
        config = SglangConfig.from_yaml(args.sglang_config)
        # Validate total GPUs match.
        expected = args.rollout_num_gpus
        actual = config.total_num_gpus
        assert actual == expected, f"sglang_config total GPUs ({actual}) != rollout_num_gpus ({expected})"
        return config

    if args.rollout_num_gpus == 0:
        return SglangConfig(models=[ModelConfig(name="default", server_groups=[])])

    if args.prefill_num_servers is not None:
        return SglangConfig.from_prefill_num_servers(args)

    # Default: single regular group.
    return SglangConfig(
        models=[
            ModelConfig(
                name="default",
                server_groups=[ServerGroupConfig(worker_type="regular", num_gpus=args.rollout_num_gpus)],
            )
        ]
    )


def _log_eval_rollout_data(
    rollout_id,
    args,
    data,
    extra_metrics: dict[str, Any] | None = None,
    *,
    completed_train_batch: bool = True,
    require_complete: bool = False,
):
    if args.custom_eval_rollout_log_function_path is not None:
        custom_log_func = load_function(args.custom_eval_rollout_log_function_path)
        custom_log_handled = custom_log_func(rollout_id, args, data, extra_metrics)
        if custom_log_handled and not require_complete:
            return

        # A strict final eval must still produce the canonical metric payload
        # and run completeness validation. A custom logger is an additional
        # publication path, not a way to suppress lifecycle invariants.

    log_dict = dict(extra_metrics or {})
    incomplete_datasets: list[str] = []
    completion_failures: list[str] = []
    for key in data.keys():
        dataset = data[key]
        rewards = list(dataset.get("rewards") or [])
        all_rewards = list(dataset.get("all_rewards") or rewards)
        valid_count = int(dataset.get("valid_count", len(rewards)))
        error_count = int(dataset.get("error_count", max(0, len(all_rewards) - valid_count)))
        expected_count = max(valid_count + error_count, len(rewards), len(all_rewards))
        # Evaluation errors are unsuccessful attempts, not samples that should
        # disappear from the denominator. Some rollout functions already put
        # zero placeholders in ``all_rewards``; older/custom functions expose
        # only the trusted rewards plus ``error_count``. Normalize both forms
        # to one reward per expected attempt without changing the diagnostic
        # valid/error counts.
        if len(all_rewards) == expected_count:
            accounted_rewards = all_rewards
        else:
            accounted_rewards = list(rewards)
            accounted_rewards.extend([0.0] * (expected_count - len(accounted_rewards)))
        min_eval_samples = dataset.get("min_eval_samples")
        if min_eval_samples is not None:
            min_eval_samples = int(min_eval_samples)
        log_dict[f"eval/{key}/valid_count"] = float(valid_count)
        log_dict[f"eval/{key}/error_count"] = float(error_count)
        log_dict[f"eval/{key}/expected_count"] = float(expected_count)
        log_dict[f"eval/{key}/zero_filled_count"] = float(error_count)
        if min_eval_samples is not None:
            log_dict[f"eval/{key}/min_valid_count"] = float(min_eval_samples)
        if accounted_rewards:
            reward_mean = sum(accounted_rewards) / len(accounted_rewards)
            log_dict[f"eval/{key}"] = reward_mean
            # Custom rollout functions commonly publish this spelling in
            # ``extra_metrics``. Override their valid-only mean so every view
            # of the dataset uses the same zero-filled denominator.
            log_dict[f"eval/{key}/reward_mean"] = reward_mean
        minimum = "" if min_eval_samples is None else f", required={min_eval_samples}"
        details = (
            f"{key} (valid={valid_count}, errors={error_count}, "
            f"expected={expected_count}{minimum})"
        )
        if error_count or not expected_count or (min_eval_samples is not None and valid_count < min_eval_samples):
            incomplete_datasets.append(details)
        completion_failed = (
            valid_count < min_eval_samples
            if min_eval_samples is not None
            else not expected_count or error_count > 0
        )
        if completion_failed:
            completion_failures.append(details)
        if samples := dataset.get("samples"):
            log_dict |= dict_add_prefix(compute_metrics_from_samples(args, samples), f"eval/{key}/")
        truncated = list(dataset.get("truncated") or dataset.get("all_truncated") or [])
        if truncated:
            if len(truncated) < expected_count:
                truncated.extend([False] * (expected_count - len(truncated)))
            log_dict[f"eval/{key}-truncated_ratio"] = sum(truncated) / len(truncated)
        if args.log_passrate and accounted_rewards:
            log_dict |= dict_add_prefix(
                compute_pass_rate(
                    flat_rewards=accounted_rewards,
                    group_size=args.n_samples_per_eval_prompt,
                ),
                f"eval/{key}-",
            )

    if not data:
        incomplete_datasets.append("no evaluation datasets returned")
        completion_failures.append("no evaluation datasets returned")

    logger.info(f"eval {rollout_id}: {log_dict}")

    step_key = set_wandb_step(
        args,
        log_dict,
        rollout_id,
        default_step_key=(
            "eval/train_step"
            if getattr(args, "wandb_always_use_train_step", False)
            else "eval/step"
        ),
        completed_train_batch=completed_train_batch,
    )
    logging_utils.log(args, log_dict, step_key=step_key)

    if require_complete and completion_failures:
        failure_details = ", ".join(completion_failures)
        logger.error(
            "Required evaluation %s did not meet its completion threshold and will not be marked complete: %s",
            rollout_id,
            failure_details,
        )
        raise RuntimeError(
            f"Required evaluation {rollout_id} is incomplete: {failure_details}"
        )
    if incomplete_datasets:
        diagnostic_details = ", ".join(incomplete_datasets)
        outcome = (
            "accepted for final completion because every configured minimum was met"
            if require_complete
            else "continued training"
        )
        logger.warning(
            "Evaluation %s had incomplete dataset(s), counted every missing/error sample as reward 0 and %s: %s",
            rollout_id,
            outcome,
            diagnostic_details,
        )

    return log_dict


def _log_rollout_data(
    rollout_id,
    args,
    samples,
    rollout_extra_metrics,
    rollout_time,
    *,
    completed_train_batch: bool = False,
):
    if args.custom_rollout_log_function_path is not None:
        custom_log_func = load_function(args.custom_rollout_log_function_path)
        if custom_log_func(rollout_id, args, samples, rollout_extra_metrics, rollout_time):
            return

    if args.load_debug_rollout_data:
        return

    log_dict = _prepare_rollout_log_dict(
        args,
        samples,
        rollout_extra_metrics,
        rollout_time,
    )
    _emit_rollout_log_dict(
        rollout_id,
        args,
        log_dict,
        completed_train_batch=completed_train_batch,
    )


def _prepare_rollout_log_dict(
    args,
    samples,
    rollout_extra_metrics,
    rollout_time,
) -> dict[str, Any]:
    """Compute the compact, journal-safe business/performance payload."""

    log_dict = {**(rollout_extra_metrics or {})}
    log_dict |= dict_add_prefix(compute_metrics_from_samples(args, samples), "rollout/")
    perf_time = _resolve_rollout_perf_time(rollout_extra_metrics, rollout_time)
    log_dict |= _prefix_rollout_performance_metrics(compute_perf_metrics_from_samples(args, samples, perf_time))
    log_dict["timing/handoff_time"] = rollout_time
    return log_dict


def _emit_rollout_log_dict(
    rollout_id: int,
    args,
    log_dict: dict[str, Any],
    *,
    completed_train_batch: bool,
) -> None:
    logger.info(f"perf {rollout_id}: {log_dict}")
    step_key = set_wandb_step(
        args,
        log_dict,
        rollout_id,
        default_step_key="rollout/step",
        completed_train_batch=completed_train_batch,
    )
    logging_utils.log(args, log_dict, step_key=step_key)


def _emit_pending_rollout_log(
    rollout_id: int,
    args,
    pending: _PendingRolloutLog,
) -> None:
    if pending.prepared_log_dict is not None:
        _emit_rollout_log_dict(
            rollout_id,
            args,
            dict(pending.prepared_log_dict),
            completed_train_batch=True,
        )
        return
    _log_rollout_data(
        rollout_id,
        args,
        pending.samples,
        pending.extra_metrics,
        pending.rollout_time,
        completed_train_batch=True,
    )


def _resolve_rollout_perf_time(rollout_extra_metrics, handoff_time):
    """Use actual async service time when generation was drained from a buffer."""
    metrics = rollout_extra_metrics or {}
    for metric_name in ("timing/service_window", "timing/service_time_max"):
        service_time = metrics.get(metric_name)
        if service_time is None:
            continue
        try:
            service_time = float(service_time)
        except (TypeError, ValueError):
            logger.warning("Ignoring invalid %s=%r", metric_name, service_time)
            continue
        if not np.isfinite(service_time) or service_time <= 0:
            logger.warning("Ignoring invalid %s=%r", metric_name, service_time)
            continue
        return service_time
    return handoff_time


def _prefix_rollout_performance_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    """Separate duration metrics from rates, throughput, and counters."""
    prefixed: dict[str, Any] = {}
    for name, value in metrics.items():
        path_parts = name.split("/")
        is_timing = any(part.endswith(("_time", "_duration", "_latency")) for part in path_parts)
        namespace = "timing" if is_timing else "perf"
        prefixed[f"{namespace}/{name}"] = value
    return prefixed


def compute_metrics_from_samples(args, samples):
    response_lengths = [sample.effective_response_length for sample in samples]

    log_dict = {}
    log_dict |= dict_add_prefix(compute_statistics(response_lengths), "response_len/")
    log_dict |= _compute_zero_std_metrics(args, samples)
    log_dict |= _compute_spec_metrics(args, samples)
    log_dict |= _compute_prefix_cache_metrics(args, samples)
    log_dict |= _compute_reward_cat_metrics(args, samples)
    log_dict |= _compute_top_p_kept_vocab_metrics(args, samples)
    log_dict["repetition_frac"] = np.mean([int(has_repetition(s.response)) for s in samples]).item()
    log_dict["truncated_ratio"] = np.mean([int(s.status == Sample.Status.TRUNCATED) for s in samples]).item()
    return log_dict


def compute_perf_metrics_from_samples(args, samples, rollout_time):
    non_generation_time = [sample.non_generation_time for sample in samples]

    log_dict = {}
    log_dict["rollout_time"] = rollout_time
    if max(non_generation_time) > 0:
        log_dict |= dict_add_prefix(compute_statistics(non_generation_time), "non_generation_time/")

    def token_perf(response_lengths, non_generation_time, key=""):
        max_response_length = max(response_lengths)
        if args.rollout_num_gpus:
            log_dict[f"{key}tokens_per_gpu_per_sec"] = sum(response_lengths) / rollout_time / args.rollout_num_gpus
        log_dict[f"longest_{key}sample_tokens_per_sec"] = max_response_length / rollout_time

        if max(non_generation_time) == 0:
            return

        non_generation_time = [t for t, length in zip(non_generation_time, response_lengths, strict=True) if length == max_response_length]
        mean_non_generation_time = sum(non_generation_time) / len(non_generation_time)

        log_dict[f"longest_{key}sample_non_generation_time"] = mean_non_generation_time
        log_dict[f"longest_{key}sample_tokens_per_sec_without_non_generation"] = max_response_length / (rollout_time - mean_non_generation_time)

    token_perf([sample.response_length for sample in samples], non_generation_time, key="")
    token_perf([sample.effective_response_length for sample in samples], non_generation_time, key="effective_")
    log_dict |= _compute_sglang_request_perf_metrics(samples)

    return log_dict


def _compute_sglang_request_perf_metrics(all_samples: list[Sample]):
    attrs_by_request = list(_iter_sglang_generate_attrs(all_samples))
    if not attrs_by_request:
        return {}

    values_by_metric: dict[str, list[float]] = {}
    profiled_request_count = 0

    def add_value(metric_key: str, source_key: str, attrs: dict) -> bool:
        value = attrs.get(source_key)
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not np.isfinite(value):
            return False
        values_by_metric.setdefault(metric_key, []).append(float(value))
        return True

    for attrs in attrs_by_request:
        request_has_perf = False

        for metric_key, source_key in _SGLANG_REQUEST_PERF_FIELDS:
            request_has_perf |= add_value(metric_key, source_key, attrs)

        for metric_key, source_key in _SGLANG_PREFILL_PERF_FIELDS:
            request_has_perf |= add_value(metric_key, source_key, attrs)

        for metric_key, source_key in _SGLANG_DECODE_PERF_FIELDS:
            request_has_perf |= add_value(metric_key, source_key, attrs)

        if request_has_perf:
            profiled_request_count += 1

    metrics: dict[str, float] = {}
    for key, values in values_by_metric.items():
        if not values:
            continue
        metrics |= dict_add_prefix(compute_statistics(values), f"{key}/")

    return metrics


def _iter_sglang_generate_attrs(all_samples: list[Sample]):
    for sample in all_samples:
        trace = getattr(sample, "trace", None)
        if not isinstance(trace, dict):
            continue
        for event in trace.get("events") or []:
            if event.get("type") != "span_end" or event.get("name") != "sglang_generate":
                continue
            attrs = event.get("attrs")
            if isinstance(attrs, dict):
                yield attrs


def _compute_zero_std_metrics(args, all_samples: list[Sample]):
    # only compute in GRPO-like algorithms where one prompt has multiple responses
    if args.advantage_estimator == "ppo":
        return {}

    def _is_zero_std(samples: list[Sample]):
        rewards = [sample.get_reward_value(args) for sample in samples]
        return len(rewards) == 0 or all(rewards[0] == r for r in rewards)

    all_sample_groups = group_by(all_samples, lambda s: s.group_index)
    interesting_sample_groups = [g for g in all_sample_groups.values() if _is_zero_std(g)]

    interesting_rewards = [str(round(g[0].get_reward_value(args), 1)) for g in interesting_sample_groups]

    return {f"zero_std/count_{reward}": len(items) for reward, items in group_by(interesting_rewards).items()}


def _compute_top_p_kept_vocab_metrics(args, all_samples: list[Sample]):
    total_kept = 0
    total_tokens = 0
    for sample in all_samples:
        offsets = sample.rollout_top_p_token_offsets
        if offsets is None or sample.response_length == 0:
            continue
        offsets = torch.as_tensor(offsets, dtype=torch.int64)
        if offsets.numel() == 0:
            continue
        assert offsets.numel() == sample.response_length + 1, f"top-p token offsets length {offsets.numel()} != response length + 1 {sample.response_length + 1}"
        if sample.remove_sample:
            continue
        if sample.loss_mask is None:
            total_kept += int(offsets[-1] - offsets[0])
            total_tokens += sample.response_length
            continue
        loss_mask = torch.as_tensor(sample.loss_mask, dtype=torch.bool, device=offsets.device)
        assert loss_mask.numel() == sample.response_length, f"loss mask length {loss_mask.numel()} != response length {sample.response_length}"
        total_kept += int(torch.diff(offsets)[loss_mask].sum())
        total_tokens += int(loss_mask.sum())
    if total_tokens == 0:
        return {}
    return {"top_p_kept_vocab_per_token": total_kept / total_tokens}


def _compute_spec_metrics(args, all_samples: list[Sample]):
    if getattr(args, "sglang_speculative_algorithm", None) is None:
        return {}
    num_samples = len(all_samples)
    metrics = {}
    metrics["spec_accept_rate"] = sum(sample.spec_info.spec_accept_rate for sample in all_samples) / num_samples
    metrics["spec_accept_length"] = sum(sample.spec_info.spec_accept_length for sample in all_samples) / num_samples
    return metrics


def _compute_prefix_cache_metrics(args, all_samples: list[Sample]):
    num_samples = len(all_samples)
    metrics = {}
    total_cached_tokens = sum(sample.prefix_cache_info.cached_tokens for sample in all_samples)
    total_prompt_tokens = sum(sample.prefix_cache_info.total_prompt_tokens for sample in all_samples)

    metrics["prefix_cache_hit_rate"] = total_cached_tokens / total_prompt_tokens if total_prompt_tokens > 0 else 0.0
    metrics["avg_cached_tokens_per_sample"] = total_cached_tokens / num_samples
    return metrics


def _compute_reward_cat_metrics(args, all_samples: list[Sample]):
    reward_cat_key = args.log_reward_category
    if reward_cat_key is None:
        return {}

    samples_of_reward_cat = group_by(all_samples, lambda s: s.reward[reward_cat_key])

    return {f"error_cat/{reward_cat}": len(s) / len(all_samples) for reward_cat, s in samples_of_reward_cat.items()}
