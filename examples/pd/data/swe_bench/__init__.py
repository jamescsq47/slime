"""SWE-bench dataset adapter and official mini-SWE-agent harness."""

from data.api import HarnessSpec

from .loader import load_samples
from .mini_harness import generate


HARNESS = HarnessSpec(
    name="swe_bench",
    load_samples=load_samples,
    generate=generate,
    default_max_response_tokens=36864,
    tools=("shell",),
    metadata={
        "agent": "official mini-SWE-agent",
        "environment": "per-trajectory Docker container",
    },
)

__all__ = ["HARNESS"]
