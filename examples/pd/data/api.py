"""Small, inference-only interface implemented by every dataset harness."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Awaitable, Callable

if TYPE_CHECKING:
    from slime.utils.types import Sample

    from .config import DatasetSpec


@dataclass(frozen=True)
class LoadContext:
    """Shared objects needed while converting source rows into ``Sample`` objects."""

    args: Any
    tokenizer: Any
    processor: Any


Loader = Callable[[LoadContext, "DatasetSpec"], list["Sample"]]
Generator = Callable[[Any, "Sample", dict[str, Any]], Awaitable["Sample"]]


@dataclass(frozen=True)
class HarnessSpec:
    """One dataset adapter plus the agent loop used to execute its samples."""

    name: str
    load_samples: Loader
    generate: Generator
    default_max_response_tokens: int
    tools: tuple[str, ...] = ()
    required_services: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    async def run(
        self,
        args: Any,
        sample: "Sample",
        sampling_params: dict[str, Any],
        *,
        options: dict[str, Any] | None = None,
    ) -> "Sample":
        options = options or {}
        cap = int(options.get("max_response_tokens", self.default_max_response_tokens))
        params = dict(sampling_params)
        params["max_new_tokens"] = min(params.get("max_new_tokens") or cap, cap)
        return await self.generate(args, sample, params)
