"""BFCL V4 Web Search with keyless, live web tools (baseline-only for now)."""

from data.api import HarnessSpec
from .loader import load_samples


async def generate(args, sample, sampling_params):
    from .harness import generate as implementation
    return await implementation(args, sample, sampling_params)


HARNESS = HarnessSpec(
    name="bfcl_web_search", load_samples=load_samples, generate=generate,
    default_max_response_tokens=32768,
    tools=("search_engine_query", "fetch_url_content"),
    metadata={"serving_modes": ["colocated"], "official_bfcl_score": False},
)
