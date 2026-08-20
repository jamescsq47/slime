"""Terminal-Bench 2 dataset adapter and persistent-shell harness."""

from data.api import HarnessSpec

from .harness import generate
from .loader import load_samples


HARNESS = HarnessSpec(
    name="terminal_bench",
    load_samples=load_samples,
    generate=generate,
    default_max_response_tokens=36864,
    tools=("shell",),
    required_services=("tbench2_env",),
)

__all__ = ["HARNESS"]
