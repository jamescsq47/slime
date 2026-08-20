"""Inference adapter for the BrowseComp search agent."""

from __future__ import annotations

from typing import Any

from slime.utils.types import Sample


async def generate(
    args: Any, sample: Sample, sampling_params: dict[str, Any]
) -> Sample:
    # The search service remains an explicitly managed external process.  A
    # harness import never starts a GPU worker or downloads a corpus.
    from .agent import generate as generate_impl

    result = await generate_impl(args, sample, sampling_params)
    tool_stats = result.metadata.get("tool_stats", {}) if isinstance(result.metadata, dict) else {}
    result.search_call_count = int(tool_stats.get("search", 0))
    result.code_call_count = 0
    return result
