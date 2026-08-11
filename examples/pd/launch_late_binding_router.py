"""CLI entry point for the experimental late-binding PD router."""

import logging
import sys

import sglang_router.mini_lb as mini_lb_module
from sglang_router.launch_router import parse_router_args

from late_binding_router import LateBindingMiniLoadBalancer


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = parse_router_args(sys.argv[1:])
    args.mini_lb = True
    router = LateBindingMiniLoadBalancer(args)
    # The FastAPI handlers in sglang_router.mini_lb resolve this module-global.
    mini_lb_module.lb = router
    router.start()


if __name__ == "__main__":
    main()

