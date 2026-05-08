from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI

from src.app.api.measurement import router as measurements_router
from src.app.api.task import router as task_router
from src.app.utils.database import init_db


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    await init_db()
    yield


app = FastAPI(lifespan=lifespan)
app.include_router(measurements_router)
app.include_router(task_router)


@app.get("/health", tags=["health"])
async def health() -> dict[str, object]:
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)