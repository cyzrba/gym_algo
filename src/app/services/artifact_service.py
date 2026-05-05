from __future__ import annotations

from pathlib import Path

from fastapi import HTTPException

from app.core.constants import MEASUREMENTS
from app.core.paths import job_dir
from app.utils.file_utils import ensure_child_path


def list_artifacts(output_dir: Path) -> list[str]:
    if not output_dir.exists():
        return []
    return sorted(path.name for path in output_dir.iterdir() if path.is_file())


def resolve_artifact_path(job_id: str, measurement: str, filename: str) -> Path:
    if measurement not in MEASUREMENTS:
        raise HTTPException(status_code=404, detail=f"Unknown measurement: {measurement}")

    base_dir = (job_dir(job_id) / measurement).resolve()
    target = (base_dir / filename).resolve()
    ensure_child_path(base_dir, target)
    if not target.exists() or not target.is_file():
        raise HTTPException(status_code=404, detail=f"Artifact not found: {filename}")
    return target
