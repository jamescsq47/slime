"""Retool math dataset adapter and code-interpreter harness."""

from data.api import HarnessSpec

from .harness import generate
from .loader import load_samples


HARNESS = HarnessSpec(
    name="retool",
    load_samples=load_samples,
    generate=generate,
    default_max_response_tokens=8192,
    tools=("code_interpreter",),
)

__all__ = ["HARNESS"]
