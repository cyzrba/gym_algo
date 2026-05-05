from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from typing import Any, Literal

from fastapi import BackgroundTasks, HTTPException

from app.core.paths import IMG_DIR, RESULT_DIR, SEG_MODEL_PATH, job_dir, job_json_path
from app.schemas.common import SavedInputs
from app.schemas.measurements import MeasurementParams


def now_epoch() -> float:
    return round(time.time(), 3)


def read_job(job_id: str) -> dict[str, Any]:
    path = job_json_path(job_id)
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")
    return json.loads(path.read_text(encoding="utf-8"))


def write_job(job: dict[str, Any]) -> None:
    path = job_json_path(str(job["job_id"]))
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(".json.tmp")
    tmp_path.write_text(json.dumps(job, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp_path.replace(path)


def build_job_payload(
    *,
    job_id: str,
    input_mode: Literal["front_back", "single"],
    selected_measurements: list[str],
    pose_model_path: Path,
    saved_inputs: SavedInputs,
    params: MeasurementParams,
) -> dict[str, Any]:
    return {
        "job_id": job_id,
        "status": "queued",
        "created_at": now_epoch(),
        "started_at": None,
        "finished_at": None,
        "duration_seconds": None,
        "input_mode": input_mode,
        "measurements": selected_measurements,
        "pose_model": str(pose_model_path),
        "seg_model": {"path": str(SEG_MODEL_PATH), "exists": SEG_MODEL_PATH.exists(), "reserved": True},
        "input_dir": str(IMG_DIR / job_id),
        "result_dir": str(job_dir(job_id)),
        "inputs": {key: str(value) for key, value in saved_inputs.paths.items()},
        "params": {
            "arm_side": params.arm_side,
            "leg_side": params.leg_side,
            "rgb_width": params.rgb_width,
            "rgb_height": params.rgb_height,
            "rgb_format": params.rgb_format,
            "depth_width": params.depth_width,
            "depth_height": params.depth_height,
            "depth_dtype": params.depth_dtype,
            "depth_endian": params.depth_endian,
            "depth_scale": params.depth_scale,
            "depth_window": params.depth_window,
        },
        "module_results": {},
        "summary": {},
        "summary_mm": {},
        "error": None,
    }


def enqueue_job(background_tasks: BackgroundTasks, job: dict[str, Any], process_job) -> dict[str, Any]:
    write_job(job)
    background_tasks.add_task(process_job, str(job["job_id"]))
    return {
        "job_id": job["job_id"],
        "status": "queued",
        "measurements": job["measurements"],
        "job_url": f"/api/v1/jobs/{job['job_id']}",
    }


def delete_job_files(job_id: str) -> None:
    path = job_json_path(job_id)
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")
    shutil.rmtree(IMG_DIR / job_id, ignore_errors=True)
    shutil.rmtree(RESULT_DIR / job_id, ignore_errors=True)
