from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import FileResponse

from app.services.artifact_service import resolve_artifact_path
from app.services.job_service import delete_job_files, read_job


router = APIRouter(prefix="/api/v1/jobs", tags=["jobs"])


@router.get("/{job_id}")
async def get_job(job_id: str) -> dict[str, object]:
    return read_job(job_id)


@router.get("/{job_id}/artifacts/{measurement}/{filename}")
async def get_artifact(job_id: str, measurement: str, filename: str) -> FileResponse:
    return FileResponse(resolve_artifact_path(job_id, measurement, filename))


@router.delete("/{job_id}", status_code=204)
async def delete_job(job_id: str) -> None:
    delete_job_files(job_id)
