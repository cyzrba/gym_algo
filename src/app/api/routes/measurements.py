from __future__ import annotations

import uuid
from typing import Annotated, Literal

from fastapi import APIRouter, BackgroundTasks, File, Form, UploadFile

from app.core.paths import ensure_runtime_dirs
from app.core.validation import parse_measurements, resolve_pose_model, validate_measurement_params
from app.schemas.measurements import MeasurementParams
from app.services.input_service import save_archive_upload, save_folder_uploads, save_inputs
from app.services.job_service import build_job_payload, enqueue_job
from app.services.measurement_runner import process_job


router = APIRouter(prefix="/api/v1/measurements", tags=["measurements"])


def build_params(
    *,
    arm_side: str,
    leg_side: str,
    rgb_width: int,
    rgb_height: int,
    rgb_format: str,
    depth_width: int,
    depth_height: int,
    depth_dtype: str,
    depth_endian: str,
    depth_scale: float,
    depth_window: int,
) -> MeasurementParams:
    return MeasurementParams(
        arm_side=arm_side,
        leg_side=leg_side,
        rgb_width=rgb_width,
        rgb_height=rgb_height,
        rgb_format=rgb_format,
        depth_width=depth_width,
        depth_height=depth_height,
        depth_dtype=depth_dtype,
        depth_endian=depth_endian,
        depth_scale=depth_scale,
        depth_window=depth_window,
    )


@router.post("", status_code=202)
async def create_measurement_job(
    background_tasks: BackgroundTasks,
    measurements: Annotated[str, Form()] = "all",
    input_mode: Annotated[Literal["front_back", "single"], Form()] = "front_back",
    arm_side: Annotated[Literal["left", "right", "both"], Form()] = "both",
    leg_side: Annotated[Literal["left", "right", "both"], Form()] = "both",
    pose_model: Annotated[str | None, Form()] = None,
    rgb_width: Annotated[int, Form()] = 640,
    rgb_height: Annotated[int, Form()] = 480,
    rgb_format: Annotated[str, Form()] = "rgb8",
    depth_width: Annotated[int, Form()] = 640,
    depth_height: Annotated[int, Form()] = 480,
    depth_dtype: Annotated[str, Form()] = "uint16",
    depth_endian: Annotated[str, Form()] = "little",
    depth_scale: Annotated[float, Form()] = 0.001,
    depth_window: Annotated[int, Form()] = 5,
    front_rgb: Annotated[UploadFile | None, File()] = None,
    front_depth: Annotated[UploadFile | None, File()] = None,
    front_ply: Annotated[UploadFile | None, File()] = None,
    back_rgb: Annotated[UploadFile | None, File()] = None,
    back_depth: Annotated[UploadFile | None, File()] = None,
    back_ply: Annotated[UploadFile | None, File()] = None,
    rgb: Annotated[UploadFile | None, File()] = None,
    depth: Annotated[UploadFile | None, File()] = None,
    ply: Annotated[UploadFile | None, File()] = None,
) -> dict[str, object]:
    ensure_runtime_dirs()
    selected_measurements = parse_measurements(measurements)
    pose_model_path = resolve_pose_model(pose_model)
    validate_measurement_params(
        rgb_format=rgb_format,
        depth_dtype=depth_dtype,
        depth_endian=depth_endian,
        depth_scale=depth_scale,
        depth_window=depth_window,
    )
    params = build_params(
        arm_side=arm_side,
        leg_side=leg_side,
        rgb_width=rgb_width,
        rgb_height=rgb_height,
        rgb_format=rgb_format,
        depth_width=depth_width,
        depth_height=depth_height,
        depth_dtype=depth_dtype,
        depth_endian=depth_endian,
        depth_scale=depth_scale,
        depth_window=depth_window,
    )

    job_id = uuid.uuid4().hex
    saved_inputs = await save_inputs(
        job_id=job_id,
        input_mode=input_mode,
        front_rgb=front_rgb,
        front_depth=front_depth,
        front_ply=front_ply,
        back_rgb=back_rgb,
        back_depth=back_depth,
        back_ply=back_ply,
        rgb=rgb,
        depth=depth,
        ply=ply,
    )
    job = build_job_payload(
        job_id=job_id,
        input_mode=input_mode,
        selected_measurements=selected_measurements,
        pose_model_path=pose_model_path,
        saved_inputs=saved_inputs,
        params=params,
    )
    return enqueue_job(background_tasks, job, process_job)


@router.post("/archive", status_code=202)
async def create_measurement_job_from_archive(
    background_tasks: BackgroundTasks,
    archive: Annotated[UploadFile, File()],
    measurements: Annotated[str, Form()] = "all",
    arm_side: Annotated[Literal["left", "right", "both"], Form()] = "both",
    leg_side: Annotated[Literal["left", "right", "both"], Form()] = "both",
    pose_model: Annotated[str | None, Form()] = None,
    rgb_width: Annotated[int, Form()] = 640,
    rgb_height: Annotated[int, Form()] = 480,
    rgb_format: Annotated[str, Form()] = "rgb8",
    depth_width: Annotated[int, Form()] = 640,
    depth_height: Annotated[int, Form()] = 480,
    depth_dtype: Annotated[str, Form()] = "uint16",
    depth_endian: Annotated[str, Form()] = "little",
    depth_scale: Annotated[float, Form()] = 0.001,
    depth_window: Annotated[int, Form()] = 5,
) -> dict[str, object]:
    ensure_runtime_dirs()
    selected_measurements = parse_measurements(measurements)
    pose_model_path = resolve_pose_model(pose_model)
    validate_measurement_params(
        rgb_format=rgb_format,
        depth_dtype=depth_dtype,
        depth_endian=depth_endian,
        depth_scale=depth_scale,
        depth_window=depth_window,
    )
    params = build_params(
        arm_side=arm_side,
        leg_side=leg_side,
        rgb_width=rgb_width,
        rgb_height=rgb_height,
        rgb_format=rgb_format,
        depth_width=depth_width,
        depth_height=depth_height,
        depth_dtype=depth_dtype,
        depth_endian=depth_endian,
        depth_scale=depth_scale,
        depth_window=depth_window,
    )

    job_id = uuid.uuid4().hex
    saved_inputs = await save_archive_upload(job_id=job_id, archive=archive)
    job = build_job_payload(
        job_id=job_id,
        input_mode="front_back",
        selected_measurements=selected_measurements,
        pose_model_path=pose_model_path,
        saved_inputs=saved_inputs,
        params=params,
    )
    return enqueue_job(background_tasks, job, process_job)


@router.post("/folder", status_code=202)
async def create_measurement_job_from_folder(
    background_tasks: BackgroundTasks,
    files: Annotated[list[UploadFile], File()],
    measurements: Annotated[str, Form()] = "all",
    arm_side: Annotated[Literal["left", "right", "both"], Form()] = "both",
    leg_side: Annotated[Literal["left", "right", "both"], Form()] = "both",
    pose_model: Annotated[str | None, Form()] = None,
    rgb_width: Annotated[int, Form()] = 640,
    rgb_height: Annotated[int, Form()] = 480,
    rgb_format: Annotated[str, Form()] = "rgb8",
    depth_width: Annotated[int, Form()] = 640,
    depth_height: Annotated[int, Form()] = 480,
    depth_dtype: Annotated[str, Form()] = "uint16",
    depth_endian: Annotated[str, Form()] = "little",
    depth_scale: Annotated[float, Form()] = 0.001,
    depth_window: Annotated[int, Form()] = 5,
) -> dict[str, object]:
    ensure_runtime_dirs()
    selected_measurements = parse_measurements(measurements)
    pose_model_path = resolve_pose_model(pose_model)
    validate_measurement_params(
        rgb_format=rgb_format,
        depth_dtype=depth_dtype,
        depth_endian=depth_endian,
        depth_scale=depth_scale,
        depth_window=depth_window,
    )
    params = build_params(
        arm_side=arm_side,
        leg_side=leg_side,
        rgb_width=rgb_width,
        rgb_height=rgb_height,
        rgb_format=rgb_format,
        depth_width=depth_width,
        depth_height=depth_height,
        depth_dtype=depth_dtype,
        depth_endian=depth_endian,
        depth_scale=depth_scale,
        depth_window=depth_window,
    )

    job_id = uuid.uuid4().hex
    saved_inputs = await save_folder_uploads(job_id=job_id, files=files)
    job = build_job_payload(
        job_id=job_id,
        input_mode="front_back",
        selected_measurements=selected_measurements,
        pose_model_path=pose_model_path,
        saved_inputs=saved_inputs,
        params=params,
    )
    return enqueue_job(background_tasks, job, process_job)
