from __future__ import annotations

from pathlib import Path

from slime.dashboard.reader import DashboardReader


def make_app(directory: str | Path):
    from fastapi import FastAPI, Query
    from fastapi.responses import FileResponse
    from fastapi.staticfiles import StaticFiles

    reader = DashboardReader(directory)
    static_dir = Path(__file__).with_name("static")
    app = FastAPI(title="Slime Dashboard", docs_url=None, redoc_url=None)

    @app.get("/api/health")
    def health():
        return {"ok": True, "directory": str(reader.directory)}

    @app.get("/api/snapshot")
    def snapshot(
        minutes: float = Query(30.0, ge=1.0, le=240.0),
        raw_engine: bool = Query(False),
    ):
        return reader.snapshot(minutes, aggregate_engine=not raw_engine)

    app.mount("/static", StaticFiles(directory=static_dir), name="static")

    @app.get("/")
    def index():
        return FileResponse(static_dir / "index.html")

    return app
