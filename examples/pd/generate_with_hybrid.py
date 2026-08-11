"""Dispatch mixed serving samples to the Retool or BrowseComp agent."""

import os
from typing import Any

from slime.utils.types import Sample


def _int_env(name: str, default: int) -> int:
    return int(os.getenv(name, str(default)))


async def generate_unified(
    args: Any, sample: Sample, sampling_params: dict[str, Any]
) -> Sample:
    task_type = "math"
    if isinstance(sample.metadata, dict):
        task_type = sample.metadata.get("task_type", "math")
    if task_type == "math":
        from generate_with_retool import generate as retool_generate

        params = dict(sampling_params)
        cap = _int_env("MIXED_RETOOL_MAX_RESPONSE_LEN", 8192)
        params["max_new_tokens"] = min(params.get("max_new_tokens") or cap, cap)
        result = await retool_generate(args, sample, params)
        result.code_call_count = getattr(result, "tool_call_count", 0)
        result.search_call_count = 0
        return result
    if task_type == "qa":
        from browsecomp_agent import generate as browsecomp_generate

        params = dict(sampling_params)
        cap = _int_env("MIXED_BROWSECOMP_MAX_RESPONSE_LEN", 36864)
        params["max_new_tokens"] = min(params.get("max_new_tokens") or cap, cap)
        result = await browsecomp_generate(args, sample, params)
        tool_stats = result.metadata.get("tool_stats", {}) if isinstance(result.metadata, dict) else {}
        result.search_call_count = int(tool_stats.get("search", 0))
        result.code_call_count = 0
        return result
    raise ValueError(f"Unknown task_type: {task_type}. Must be math or qa.")
