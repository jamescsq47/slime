from data.api import HarnessSpec
from .loader import load_samples


async def generate(args, sample, sampling_params):
    from .harness import generate as run
    return await run(args, sample, sampling_params)


HARNESS = HarnessSpec(name="sciagentgym", load_samples=load_samples, generate=generate,
                      default_max_response_tokens=32768,
                      metadata={"serving_modes": ["colocated"], "official_score": False})
