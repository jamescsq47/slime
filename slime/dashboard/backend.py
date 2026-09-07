from __future__ import annotations

import logging
import re
import time
import uuid
from typing import Any

from slime.dashboard import COLLECTOR_NAME_PREFIX
from slime.dashboard.logging_utils import RateLimitedWarner

logger = logging.getLogger(__name__)

_handle = None
_is_primary = False
_trace_sink = None
_warner = RateLimitedWarner(logger)


def _enabled(args) -> bool:
    return bool(getattr(args, "use_slime_dashboard", False))


def init_dashboard(args, primary: bool = True) -> None:
    global _handle, _is_primary
    if not _enabled(args):
        return
    try:
        import ray

        if primary:
            from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

            from slime.dashboard.collector import DashboardCollector

            _is_primary = True
            suffix = getattr(args, "wandb_run_id", None) or uuid.uuid4().hex[:10]
            safe_suffix = re.sub(r"[^A-Za-z0-9_-]", "_", str(suffix))
            collector_name = f"{COLLECTOR_NAME_PREFIX}_{safe_suffix}"
            args.slime_dashboard_collector_name = collector_name
            config = {
                "directory": args.slime_dashboard_dir,
                "run_name": getattr(args, "wandb_group", None) or "slime-run",
                "start_ts": time.time(),
                "flush_interval": args.slime_dashboard_flush_interval,
                "gpu_sample_interval": args.slime_dashboard_gpu_sample_interval,
                "sglang_scrape_interval": args.slime_dashboard_sglang_scrape_interval,
                "max_buffered_records": args.slime_dashboard_max_buffered_records,
                "args": _args_snapshot(args),
            }
            actor = ray.remote(DashboardCollector)
            _handle = actor.options(
                name=collector_name,
                scheduling_strategy=NodeAffinitySchedulingStrategy(
                    node_id=ray.get_runtime_context().get_node_id(), soft=False
                ),
            ).remote(config)
            ray.get(_handle.ping.remote())
            _handle.start.remote(_handle)
            logger.info(
                "Slime dashboard telemetry: %s (view with python -m slime.dashboard.serve --dashboard-dir %s)",
                args.slime_dashboard_dir,
                args.slime_dashboard_dir,
            )
            return
        _handle = _resolve_handle(args)
    except Exception:
        _warner.warn("Slime dashboard initialization failed; training will continue without dashboard telemetry")
        _handle = None


def register_rollout_manager(args, router_addr: str | None) -> None:
    global _handle, _trace_sink
    if not _enabled(args):
        return
    try:
        if _handle is None:
            _handle = _resolve_handle(args)
        if _handle is None:
            return
        _handle.set_router.remote(router_addr)
        if _trace_sink is None:
            from slime.dashboard.hooks import TraceEventSink
            from slime.utils.trace_utils import add_trace_event_sink

            _trace_sink = TraceEventSink(_handle)
            add_trace_event_sink(_trace_sink)
    except Exception:
        _warner.warn("Slime dashboard rollout hooks could not be attached")


def log_metrics(metrics: dict, step: int | None = None, step_key: str | None = None) -> None:
    if _handle is None:
        return
    try:
        record = {
            "ts": time.time(),
            "step": step,
            "step_key": step_key,
            "metrics": _scalars_only(metrics),
        }
        _handle.push.remote("metrics", record)
    except Exception:
        _warner.warn("Slime dashboard metric push failed; dropping this record")


def get_sglang_summary(timeout: float = 2.0) -> dict[str, float]:
    """Return the latest already-scraped SGLang aggregate for live logging."""
    if _handle is None:
        return {}
    try:
        import ray

        return ray.get(_handle.get_sglang_summary.remote(), timeout=timeout)
    except Exception:
        _warner.warn("Slime dashboard SGLang summary is temporarily unavailable")
        return {}


def finish_dashboard() -> None:
    global _handle, _is_primary, _trace_sink
    if _trace_sink is not None:
        try:
            from slime.utils.trace_utils import remove_trace_event_sink

            remove_trace_event_sink(_trace_sink)
            _trace_sink.flush()
        except Exception:
            pass
        _trace_sink = None
    if _handle is not None and _is_primary:
        try:
            import ray

            ray.get(_handle.shutdown.remote(), timeout=30)
            ray.kill(_handle, no_restart=True)
        except Exception:
            _warner.warn("Slime dashboard final flush was incomplete")
    _handle = None
    _is_primary = False


def current_handle():
    return _handle


def _resolve_handle(args):
    name = getattr(args, "slime_dashboard_collector_name", None)
    if not name:
        return None
    try:
        import ray

        return ray.get_actor(name)
    except Exception:
        _warner.warn("Slime dashboard collector %s was not found; telemetry is disabled in this process", name)
        return None


def _args_snapshot(args) -> dict[str, Any]:
    keys = (
        "wandb_group",
        "actor_num_nodes",
        "actor_num_gpus_per_node",
        "rollout_num_gpus",
        "rollout_num_gpus_per_engine",
        "rollout_batch_size",
        "over_sampling_batch_size",
        "adaptive_group_oversampling",
        "adaptive_group_oversampling_min_groups",
        "adaptive_group_oversampling_running_threshold",
        "adaptive_group_oversampling_queue_threshold",
        "adaptive_group_oversampling_expansion_kv_threshold",
        "adaptive_group_oversampling_window_seconds",
        "adaptive_group_oversampling_cooldown_seconds",
        "adaptive_group_oversampling_post_resume_expansion_grace_seconds",
        "adaptive_group_oversampling_recovery_seconds",
        "adaptive_group_oversampling_step_groups",
        "adaptive_group_oversampling_pressure_queue_threshold",
        "adaptive_group_oversampling_pressure_kv_threshold",
        "adaptive_group_oversampling_hard_max_engine_kv_threshold",
        "adaptive_group_oversampling_hard_engine_queue_threshold",
        "adaptive_group_oversampling_hard_pressure_kv_threshold",
        "adaptive_group_oversampling_hard_pressure_seconds",
        "adaptive_group_oversampling_hard_step_groups",
        "decoupled_gpu_tool_scheduling",
        "gpu_generation_slots",
        "terminal_live_session_limit",
        "terminal_concurrent_resets",
        "inflight_group_soft_limit",
        "max_inflight_groups",
        "n_samples_per_prompt",
        "global_batch_size",
        "update_weights_interval",
        "fully_async_buffer_policy",
        "fully_async_version_window",
        "hf_checkpoint",
        "colocate",
        "partial_rollout",
        "partial_rollout_tool_handoff",
    )
    return {key: getattr(args, key) for key in keys if hasattr(args, key)}


def _scalars_only(metrics: dict) -> dict:
    output = {}
    for key, value in metrics.items():
        if isinstance(value, (str, bool, int, float)):
            output[key] = value
        elif hasattr(value, "item"):
            try:
                output[key] = value.item()
            except Exception:
                continue
    return output
