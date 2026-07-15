"""Unified generation/reward dispatch for mixed Retool + BrowseComp training."""

import os

from typing import Any

from slime.utils.types import Sample


def _int_env(name: str, default: int) -> int:
    return int(os.getenv(name, str(default)))


async def generate_unified(args, sample: Sample, sampling_params) -> Sample:
    """Unified generation function dispatching to task-specific generators.

    For math tasks, delegates to generate_with_retool.generate().
    For QA tasks, delegates to browsecomp_agent.generate().
    """

    # Determine task type from sample metadata
    task_type = "math"
    if isinstance(sample.metadata, dict):
        task_type = sample.metadata.get("task_type", "math")

    if task_type == "math":
        from generate_with_retool_advisor import generate as retool_generate

        math_sampling_params = dict(sampling_params)
        math_sampling_params["max_new_tokens"] = min(
            math_sampling_params.get("max_new_tokens") or _int_env("MIXED_RETOOL_MAX_RESPONSE_LEN", 8192),
            _int_env("MIXED_RETOOL_MAX_RESPONSE_LEN", 8192),
        )
        result = await retool_generate(args, sample, math_sampling_params)
        result.code_call_count = getattr(result, "tool_call_count", 0)
        result.search_call_count = 0
        return result
    elif task_type == "qa":
        from browsecomp_agent import generate as browsecomp_generate

        qa_sampling_params = dict(sampling_params)
        qa_sampling_params["max_new_tokens"] = min(
            qa_sampling_params.get("max_new_tokens") or _int_env("MIXED_BROWSECOMP_MAX_RESPONSE_LEN", 36864),
            _int_env("MIXED_BROWSECOMP_MAX_RESPONSE_LEN", 36864),
        )
        result = await browsecomp_generate(args, sample, qa_sampling_params)
        tool_stats = result.metadata.get("tool_stats", {}) if isinstance(result.metadata, dict) else {}
        result.search_call_count = int(tool_stats.get("search", 0))
        result.code_call_count = 0
        return result
    else:
        raise ValueError(f"Unknown task_type: {task_type}. Must be 'math' or 'qa'.")


async def reward_func_unified(args, sample: Sample, task_type: str = None, **kwargs) -> Any:
    """Unified reward function dispatching to task-specific reward functions.

    For math tasks, delegates to generate_with_retool.reward_func().
    For QA tasks, delegates to browsecomp_rm.reward_func().
    """

    if not isinstance(sample, Sample):
        raise TypeError("Sample must be an instance of Sample class.")

    # Get task_type from sample.metadata if not provided
    if task_type is None:
        metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
        task_type = metadata.get("task_type", "math")

    if task_type == "math":
        from generate_with_retool_advisor import reward_func as retool_reward

        return await retool_reward(args, sample, **kwargs)
    elif task_type == "qa":
        from browsecomp_rm import reward_func as browsecomp_reward

        return {"score": await browsecomp_reward(args, sample, **kwargs)}
    else:
        raise ValueError(f"Unknown task_type: {task_type}. Must be 'math' or 'qa'.")
