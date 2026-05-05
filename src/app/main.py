from __future__ import annotations

from fastapi import FastAPI

from app.api.routes.jobs import router as jobs_router
from app.api.routes.measurements import router as measurements_router
from app.core.paths import ensure_runtime_dirs


def create_app() -> FastAPI:
    app = FastAPI(
        title="Gym Body Measurement Backend",
        description="FastAPI wrapper for point-cloud body measurement scripts.",
        version="0.1.0",
    )

    @app.on_event("startup")
    async def startup() -> None:
        ensure_runtime_dirs()

    @app.get("/health", tags=["health"])
    async def health() -> dict[str, object]:
        from app.api.routes.health import get_health_payload

        return get_health_payload()

    app.include_router(measurements_router)
    app.include_router(jobs_router)
    return app


app = create_app()
