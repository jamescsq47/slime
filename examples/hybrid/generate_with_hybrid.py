# Unified generation function dispatching to task-specific generators

from typing import Any

from slime.utils.types import Sample


async def generate_unified(args, sample: Sample, sampling_params) -> Sample:
    """Unified generation function dispatching to task-specific generators.

    For math tasks, delegates to generate_with_retool.generate().
    For QA tasks, delegates to generate_with_search.generate().
    """

    # Determine task type from sample metadata
    task_type = "math"
    if isinstance(sample.metadata, dict):
        task_type = sample.metadata.get("task_type", "math")

    if task_type == "math":
        from generate_with_retool import generate as retool_generate

        result = await retool_generate(args, sample, sampling_params)
        result.code_call_count = getattr(result, "tool_call_count", 0)
        result.search_call_count = 0
        return result
    elif task_type == "qa":
        from generate_with_search import generate as search_generate

        result = await search_generate(args, sample, sampling_params)
        result.search_call_count = getattr(result, "tool_call_count", 0)
        result.code_call_count = 0
        return result
    else:
        raise ValueError(f"Unknown task_type: {task_type}. Must be 'math' or 'qa'.")


async def reward_func_unified(args, sample: Sample, task_type: str = None, **kwargs) -> Any:
    """Unified reward function dispatching to task-specific reward functions.

    For math tasks, delegates to generate_with_retool.reward_func().
    For QA tasks, delegates to generate_with_search.reward_func().
    """

    if not isinstance(sample, Sample):
        raise TypeError("Sample must be an instance of Sample class.")

    # Get task_type from sample.metadata if not provided
    if task_type is None:
        metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
        task_type = metadata.get("task_type", "math")

    if task_type == "math":
        from generate_with_retool import reward_func as retool_reward

        return await retool_reward(args, sample, **kwargs)
    elif task_type == "qa":
        from generate_with_search import reward_func as search_reward

        return await search_reward(args, sample, **kwargs)
    else:
        raise ValueError(f"Unknown task_type: {task_type}. Must be 'math' or 'qa'.")
