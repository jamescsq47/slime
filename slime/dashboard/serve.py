from __future__ import annotations

import argparse
from pathlib import Path

from slime.dashboard.server import make_app


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve a live or completed Slime dashboard telemetry directory")
    parser.add_argument("--dashboard-dir", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7788)
    args = parser.parse_args()

    directory = Path(args.dashboard_dir)
    if not directory.is_dir():
        parser.error(f"dashboard directory not found: {directory}")

    import uvicorn

    uvicorn.run(make_app(directory), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
