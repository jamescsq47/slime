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

    @app.middleware("http")
    async def disable_dashboard_asset_cache(request, call_next):
        response = await call_next(request)
        if request.url.path == "/" or request.url.path.startswith("/static/"):
            response.headers["Cache-Control"] = "no-cache"
        return response

    @app.get("/api/health")
    def health():
        return {"ok": True, "directory": str(reader.directory)}

    @app.get("/api/snapshot")
    def snapshot(
        minutes: float = Query(30.0, ge=1.0, le=240.0),
        raw_engine: bool = Query(False),
        raw_trace: bool = Query(False),
    ):
        return reader.snapshot(minutes, aggregate_engine=not raw_engine, include_raw_trace=raw_trace)

    app.mount("/static", StaticFiles(directory=static_dir), name="static")

    @app.get("/")
    def index():
        return FileResponse(static_dir / "index.html")

    return app
