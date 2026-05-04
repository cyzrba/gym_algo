from __future__ import annotations

import asyncio
import json
import shutil
import os
import subprocess
import sys
import time
import uuid
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Literal

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse


ROOT_DIR = Path(__file__).resolve().parent
IMG_DIR = ROOT_DIR / "img"
RESULT_DIR = ROOT_DIR / "result"
POSE_MODEL_PATH = ROOT_DIR / "yolo26n-pose.pt"
SEG_MODEL_PATH = ROOT_DIR / "yolo26n-seg.pt"

MEASUREMENTS = ("arm", "shoulder", "leg", "waist")
MEASUREMENT_SCRIPTS = {
    "arm": ROOT_DIR / "arm_pointcloud.py",
    "shoulder": ROOT_DIR / "shoulder_pointcloud.py",
    "leg": ROOT_DIR / "leg_pointcloud.py",
    "waist": ROOT_DIR / "waist_pointcloud.py",
}

TASK_SEMAPHORE = asyncio.Semaphore(1)

app = FastAPI(
    title="Gym Body Measurement Backend",
    description="FastAPI wrapper for point-cloud body measurement scripts.",
    version="0.1.0",
)


@dataclass(frozen=True)
class SavedInputs:
    input_mode: Literal["front_back", "single"]
    paths: dict[str, Path]


def ensure_runtime_dirs() -> None:
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)


def job_dir(job_id: str) -> Path:
    return RESULT_DIR / job_id


def job_json_path(job_id: str) -> Path:
    return job_dir(job_id) / "job.json"


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


def parse_measurements(raw: str) -> list[str]:
    values = [part.strip().lower() for part in raw.split(",") if part.strip()]
    if not values or values == ["all"] or "all" in values:
        return list(MEASUREMENTS)

    invalid = sorted(set(values) - set(MEASUREMENTS))
    if invalid:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported measurements: {', '.join(invalid)}. Supported: all,{','.join(MEASUREMENTS)}",
        )
    return list(dict.fromkeys(values))


def resolve_pose_model(raw_pose_model: str | None) -> Path:
    if raw_pose_model is not None:
        raw_pose_model = raw_pose_model.strip()

    if not raw_pose_model or raw_pose_model == "string":
        model_path = POSE_MODEL_PATH
    else:
        candidate = Path(raw_pose_model)
        model_path = candidate if candidate.is_absolute() else ROOT_DIR / candidate

    if not model_path.exists():
        raise HTTPException(status_code=400, detail=f"Pose model not found: {model_path}")
    return model_path.resolve()


def validate_choice(name: str, value: str, allowed: set[str]) -> str:
    if value not in allowed:
        raise HTTPException(status_code=400, detail=f"{name} must be one of: {', '.join(sorted(allowed))}")
    return value


def upload_target_path(base_dir: Path, field_name: str, upload: UploadFile) -> Path:
    suffix = Path(upload.filename or "").suffix
    return base_dir / f"{field_name}{suffix}"


def safe_relative_path(raw_path: str) -> Path:
    cleaned = raw_path.replace("\\", "/")
    parts = [
        part
        for part in cleaned.split("/")
        if part and part not in {".", ".."} and not part.endswith(":")
    ]
    if not parts:
        raise HTTPException(status_code=400, detail="Invalid upload filename")
    return Path(*parts)


async def save_upload(upload: UploadFile, target_path: Path) -> Path:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    with target_path.open("wb") as output:
        while True:
            chunk = await upload.read(1024 * 1024)
            if not chunk:
                break
            output.write(chunk)
    await upload.close()
    return target_path


def require_upload(name: str, upload: UploadFile | None) -> UploadFile:
    if upload is None:
        raise HTTPException(status_code=400, detail=f"Missing required file: {name}")
    return upload


async def save_inputs(
    *,
    job_id: str,
    input_mode: Literal["front_back", "single"],
    front_rgb: UploadFile | None,
    front_depth: UploadFile | None,
    front_ply: UploadFile | None,
    back_rgb: UploadFile | None,
    back_depth: UploadFile | None,
    back_ply: UploadFile | None,
    rgb: UploadFile | None,
    depth: UploadFile | None,
    ply: UploadFile | None,
) -> SavedInputs:
    input_dir = IMG_DIR / job_id

    if input_mode == "front_back":
        uploads = {
            "front_rgb": require_upload("front_rgb", front_rgb),
            "front_depth": require_upload("front_depth", front_depth),
            "front_ply": require_upload("front_ply", front_ply),
            "back_rgb": require_upload("back_rgb", back_rgb),
            "back_depth": require_upload("back_depth", back_depth),
            "back_ply": require_upload("back_ply", back_ply),
        }
    else:
        uploads = {
            "rgb": require_upload("rgb", rgb),
            "depth": require_upload("depth", depth),
            "ply": require_upload("ply", ply),
        }

    paths: dict[str, Path] = {}
    for field_name, upload in uploads.items():
        paths[field_name] = await save_upload(upload, upload_target_path(input_dir, field_name, upload))

    return SavedInputs(input_mode=input_mode, paths=paths)


async def save_folder_uploads(*, job_id: str, files: list[UploadFile]) -> SavedInputs:
    if not files:
        raise HTTPException(status_code=400, detail="Missing uploaded folder files")

    source_dir = IMG_DIR / job_id / "source"
    for upload in files:
        relative_path = safe_relative_path(upload.filename or "")
        await save_upload(upload, source_dir / relative_path)

    return infer_front_back_inputs(source_dir)


async def save_archive_upload(*, job_id: str, archive: UploadFile) -> SavedInputs:
    suffix = Path(archive.filename or "").suffix.lower()
    if suffix != ".zip":
        raise HTTPException(status_code=400, detail="Only .zip archives are supported")

    input_dir = IMG_DIR / job_id
    archive_path = await save_upload(archive, input_dir / "source.zip")
    extract_dir = input_dir / "source"
    extract_zip_safely(archive_path, extract_dir)
    return infer_front_back_inputs(extract_dir)


def extract_zip_safely(archive_path: Path, extract_dir: Path) -> None:
    extract_dir.mkdir(parents=True, exist_ok=True)
    base_dir = extract_dir.resolve()
    with zipfile.ZipFile(archive_path) as archive:
        for member in archive.infolist():
            if member.is_dir():
                continue
            relative_path = safe_relative_path(member.filename)
            target_path = (base_dir / relative_path).resolve()
            if Path(os.path.commonpath([base_dir, target_path])) != base_dir:
                raise HTTPException(status_code=400, detail=f"Unsafe archive member: {member.filename}")
            target_path.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member) as src, target_path.open("wb") as dst:
                shutil.copyfileobj(src, dst)


def infer_front_back_inputs(source_dir: Path) -> SavedInputs:
    discovered: dict[str, dict[str, Path]] = {"front": {}, "back": {}}
    for path in source_dir.rglob("*"):
        if not path.is_file():
            continue
        role = infer_input_role(path)
        side = infer_input_side(path)
        if side and role and role not in discovered[side]:
            discovered[side][role] = path

    required = {
        "front_rgb": discovered["front"].get("rgb"),
        "front_depth": discovered["front"].get("depth"),
        "front_ply": discovered["front"].get("ply"),
        "back_rgb": discovered["back"].get("rgb"),
        "back_depth": discovered["back"].get("depth"),
        "back_ply": discovered["back"].get("ply"),
    }
    missing = [name for name, path in required.items() if path is None]
    if missing:
        raise HTTPException(
            status_code=400,
            detail=(
                "Could not infer required front/back files. "
                f"Missing: {', '.join(missing)}. Expected folders like front/rgb.raw, "
                "front/depth.raw, front/*.ply, back/rgb.raw, back/depth.raw, back/*.ply."
            ),
        )

    return SavedInputs(
        input_mode="front_back",
        paths={name: path for name, path in required.items() if path is not None},
    )


def infer_input_side(path: Path) -> str | None:
    tokens = [part.lower() for part in path.parts]
    if "front" in tokens or any("front" in token or "正面" in token for token in tokens):
        return "front"
    if "back" in tokens or any("back" in token or "背面" in token or "后面" in token for token in tokens):
        return "back"
    return None


def infer_input_role(path: Path) -> str | None:
    name = path.name.lower()
    stem = path.stem.lower()
    suffix = path.suffix.lower()

    if suffix == ".ply":
        return "ply"
    if suffix == ".raw":
        if "depth" in stem or "dep" in stem or "深度" in stem:
            return "depth"
        if "rgb" in stem or "color" in stem or "colour" in stem or "彩色" in stem:
            return "rgb"
    if "depth" in name and suffix in {".png", ".jpg", ".jpeg", ".tif", ".tiff"}:
        return "depth"
    if ("rgb" in name or "color" in name or "colour" in name) and suffix in {".png", ".jpg", ".jpeg"}:
        return "rgb"
    return None


def build_common_script_args(
    *,
    saved_inputs: SavedInputs,
    output_dir: Path,
    pose_model: Path,
    rgb_width: int,
    rgb_height: int,
    rgb_format: str,
    depth_width: int,
    depth_height: int,
    depth_dtype: str,
    depth_endian: str,
    depth_scale: float,
    depth_window: int,
) -> list[str]:
    args: list[str] = []
    if saved_inputs.input_mode == "front_back":
        args.extend(
            [
                "--front-rgb",
                str(saved_inputs.paths["front_rgb"]),
                "--front-depth",
                str(saved_inputs.paths["front_depth"]),
                "--front-ply",
                str(saved_inputs.paths["front_ply"]),
                "--back-rgb",
                str(saved_inputs.paths["back_rgb"]),
                "--back-depth",
                str(saved_inputs.paths["back_depth"]),
                "--back-ply",
                str(saved_inputs.paths["back_ply"]),
            ]
        )
    else:
        args.extend(
            [
                "--rgb",
                str(saved_inputs.paths["rgb"]),
                "--depth",
                str(saved_inputs.paths["depth"]),
                "--ply",
                str(saved_inputs.paths["ply"]),
            ]
        )

    args.extend(
        [
            "--pose-model",
            str(pose_model),
            "--output-dir",
            str(output_dir),
            "--rgb-width",
            str(rgb_width),
            "--rgb-height",
            str(rgb_height),
            "--rgb-format",
            rgb_format,
            "--depth-width",
            str(depth_width),
            "--depth-height",
            str(depth_height),
            "--depth-dtype",
            depth_dtype,
            "--depth-endian",
            depth_endian,
            "--depth-window",
            str(depth_window),
        ]
    )

    if output_dir.name != "shoulder":
        args.extend(["--depth-scale", str(depth_scale)])

    return args


def build_script_args(
    *,
    measurement: str,
    saved_inputs: SavedInputs,
    output_dir: Path,
    pose_model: Path,
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
) -> list[str]:
    args = build_common_script_args(
        saved_inputs=saved_inputs,
        output_dir=output_dir,
        pose_model=pose_model,
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
    if measurement == "arm":
        args.extend(["--arm-side", arm_side])
    elif measurement == "leg":
        args.extend(["--leg-side", leg_side])
    return args


async def run_measurement_script(
    *,
    measurement: str,
    script_args: list[str],
    output_dir: Path,
) -> dict[str, Any]:
    script_path = MEASUREMENT_SCRIPTS[measurement]
    output_dir.mkdir(parents=True, exist_ok=True)

    started_at = now_epoch()
    completed = await asyncio.to_thread(
        subprocess.run,
        [sys.executable, str(script_path), *script_args],
        cwd=str(ROOT_DIR),
        env={**os.environ, "MPLCONFIGDIR": str((ROOT_DIR / ".mplconfig").resolve())},
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    finished_at = now_epoch()

    summary_path = output_dir / "pose_joints.json"

    result: dict[str, Any] = {
        "status": "succeeded" if completed.returncode == 0 else "failed",
        "returncode": completed.returncode,
        "started_at": started_at,
        "finished_at": finished_at,
        "duration_seconds": round(finished_at - started_at, 3),
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "output_dir": str(output_dir),
        "summary_json": str(summary_path) if summary_path.exists() else None,
        "artifacts": list_artifacts(output_dir),
        "measurement_summary": {},
        "measurement_summary_mm": {},
    }

    if completed.returncode != 0:
        result["error"] = f"{measurement} script failed with return code {completed.returncode}"
        return result

    if not summary_path.exists():
        result["status"] = "failed"
        result["error"] = f"{measurement} did not produce pose_joints.json"
        return result

    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    measurement_summary = payload.get("measurement_summary", {})
    result["measurement_summary"] = measurement_summary
    result["measurement_summary_mm"] = convert_meters_summary_to_mm(measurement_summary)
    return result


def convert_meters_summary_to_mm(summary: dict[str, Any]) -> dict[str, float]:
    converted: dict[str, float] = {}
    for key, value in summary.items():
        if not key.endswith("_m"):
            continue
        try:
            numeric_value = float(value)
        except (TypeError, ValueError):
            continue
        converted[f"{key[:-2]}_mm"] = round(numeric_value * 1000, 3)
    return converted


def list_artifacts(output_dir: Path) -> list[str]:
    if not output_dir.exists():
        return []
    return sorted(path.name for path in output_dir.iterdir() if path.is_file())


async def process_job(job_id: str) -> None:
    async with TASK_SEMAPHORE:
        job = read_job(job_id)
        job["status"] = "running"
        job["started_at"] = now_epoch()
        write_job(job)

        try:
            for measurement in job["measurements"]:
                output_dir = job_dir(job_id) / measurement
                script_args = build_script_args(
                    measurement=measurement,
                    saved_inputs=SavedInputs(
                        input_mode=job["input_mode"],
                        paths={key: Path(value) for key, value in job["inputs"].items()},
                    ),
                    output_dir=output_dir,
                    pose_model=Path(job["pose_model"]),
                    arm_side=job["params"]["arm_side"],
                    leg_side=job["params"]["leg_side"],
                    rgb_width=job["params"]["rgb_width"],
                    rgb_height=job["params"]["rgb_height"],
                    rgb_format=job["params"]["rgb_format"],
                    depth_width=job["params"]["depth_width"],
                    depth_height=job["params"]["depth_height"],
                    depth_dtype=job["params"]["depth_dtype"],
                    depth_endian=job["params"]["depth_endian"],
                    depth_scale=job["params"]["depth_scale"],
                    depth_window=job["params"]["depth_window"],
                )
                job["module_results"][measurement] = {
                    "status": "running",
                    "output_dir": str(output_dir),
                }
                write_job(job)

                module_result = await run_measurement_script(
                    measurement=measurement,
                    script_args=script_args,
                    output_dir=output_dir,
                )
                job["module_results"][measurement] = module_result
                job["summary"][measurement] = module_result.get("measurement_summary", {})
                job["summary_mm"][measurement] = module_result.get("measurement_summary_mm", {})
                write_job(job)

                if module_result["status"] != "succeeded":
                    job["status"] = "failed"
                    job["error"] = module_result.get("error", f"{measurement} failed")
                    break
            else:
                job["status"] = "succeeded"

        except Exception as exc:  # noqa: BLE001 - persist unexpected worker errors for the API caller.
            job["status"] = "failed"
            job["error"] = f"{type(exc).__name__}: {exc}"
        finally:
            job["finished_at"] = now_epoch()
            if job.get("started_at"):
                job["duration_seconds"] = round(job["finished_at"] - job["started_at"], 3)
            write_job(job)


@app.on_event("startup")
async def startup() -> None:
    ensure_runtime_dirs()


@app.get("/health")
async def health() -> dict[str, Any]:
    ensure_runtime_dirs()
    return {
        "status": "ok",
        "root_dir": str(ROOT_DIR),
        "input_dir": str(IMG_DIR),
        "result_dir": str(RESULT_DIR),
        "measurements": list(MEASUREMENTS),
        "scripts": {name: path.exists() for name, path in MEASUREMENT_SCRIPTS.items()},
        "models": {
            "pose": {"path": str(POSE_MODEL_PATH), "exists": POSE_MODEL_PATH.exists()},
            "seg": {"path": str(SEG_MODEL_PATH), "exists": SEG_MODEL_PATH.exists(), "reserved": True},
        },
    }


def build_job_payload(
    *,
    job_id: str,
    input_mode: Literal["front_back", "single"],
    selected_measurements: list[str],
    pose_model_path: Path,
    saved_inputs: SavedInputs,
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
            "arm_side": arm_side,
            "leg_side": leg_side,
            "rgb_width": rgb_width,
            "rgb_height": rgb_height,
            "rgb_format": rgb_format,
            "depth_width": depth_width,
            "depth_height": depth_height,
            "depth_dtype": depth_dtype,
            "depth_endian": depth_endian,
            "depth_scale": depth_scale,
            "depth_window": depth_window,
        },
        "module_results": {},
        "summary": {},
        "summary_mm": {},
        "error": None,
    }


def enqueue_job(background_tasks: BackgroundTasks, job: dict[str, Any]) -> dict[str, Any]:
    write_job(job)
    background_tasks.add_task(process_job, str(job["job_id"]))
    return {
        "job_id": job["job_id"],
        "status": "queued",
        "measurements": job["measurements"],
        "job_url": f"/api/v1/jobs/{job['job_id']}",
    }


@app.post("/api/v1/measurements", status_code=202)
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
) -> dict[str, Any]:
    ensure_runtime_dirs()
    selected_measurements = parse_measurements(measurements)
    pose_model_path = resolve_pose_model(pose_model)
    validate_choice("rgb_format", rgb_format, {"rgb8", "bgr8", "gray8"})
    validate_choice("depth_dtype", depth_dtype, {"uint16", "uint8", "float32"})
    validate_choice("depth_endian", depth_endian, {"little", "big"})
    if depth_window < 1 or depth_window % 2 == 0:
        raise HTTPException(status_code=400, detail="depth_window must be a positive odd integer")
    if depth_scale <= 0:
        raise HTTPException(status_code=400, detail="depth_scale must be > 0")

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
    return enqueue_job(background_tasks, job)


@app.post("/api/v1/measurements/archive", status_code=202)
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
) -> dict[str, Any]:
    ensure_runtime_dirs()
    selected_measurements = parse_measurements(measurements)
    pose_model_path = resolve_pose_model(pose_model)
    validate_choice("rgb_format", rgb_format, {"rgb8", "bgr8", "gray8"})
    validate_choice("depth_dtype", depth_dtype, {"uint16", "uint8", "float32"})
    validate_choice("depth_endian", depth_endian, {"little", "big"})
    if depth_window < 1 or depth_window % 2 == 0:
        raise HTTPException(status_code=400, detail="depth_window must be a positive odd integer")
    if depth_scale <= 0:
        raise HTTPException(status_code=400, detail="depth_scale must be > 0")

    job_id = uuid.uuid4().hex
    saved_inputs = await save_archive_upload(job_id=job_id, archive=archive)
    job = build_job_payload(
        job_id=job_id,
        input_mode="front_back",
        selected_measurements=selected_measurements,
        pose_model_path=pose_model_path,
        saved_inputs=saved_inputs,
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
    return enqueue_job(background_tasks, job)


@app.post("/api/v1/measurements/folder", status_code=202)
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
) -> dict[str, Any]:
    ensure_runtime_dirs()
    selected_measurements = parse_measurements(measurements)
    pose_model_path = resolve_pose_model(pose_model)
    validate_choice("rgb_format", rgb_format, {"rgb8", "bgr8", "gray8"})
    validate_choice("depth_dtype", depth_dtype, {"uint16", "uint8", "float32"})
    validate_choice("depth_endian", depth_endian, {"little", "big"})
    if depth_window < 1 or depth_window % 2 == 0:
        raise HTTPException(status_code=400, detail="depth_window must be a positive odd integer")
    if depth_scale <= 0:
        raise HTTPException(status_code=400, detail="depth_scale must be > 0")

    job_id = uuid.uuid4().hex
    saved_inputs = await save_folder_uploads(job_id=job_id, files=files)
    job = build_job_payload(
        job_id=job_id,
        input_mode="front_back",
        selected_measurements=selected_measurements,
        pose_model_path=pose_model_path,
        saved_inputs=saved_inputs,
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
    return enqueue_job(background_tasks, job)


@app.get("/api/v1/jobs/{job_id}")
async def get_job(job_id: str) -> dict[str, Any]:
    return read_job(job_id)


@app.get("/api/v1/jobs/{job_id}/artifacts/{measurement}/{filename}")
async def get_artifact(job_id: str, measurement: str, filename: str) -> FileResponse:
    if measurement not in MEASUREMENTS:
        raise HTTPException(status_code=404, detail=f"Unknown measurement: {measurement}")

    base_dir = (job_dir(job_id) / measurement).resolve()
    target = (base_dir / filename).resolve()

    if Path(os.path.commonpath([base_dir, target])) != base_dir:
        raise HTTPException(status_code=400, detail="Invalid artifact path")
    if not target.exists() or not target.is_file():
        raise HTTPException(status_code=404, detail=f"Artifact not found: {filename}")

    return FileResponse(target)


@app.delete("/api/v1/jobs/{job_id}", status_code=204)
async def delete_job(job_id: str) -> None:
    job_path = job_json_path(job_id)
    if not job_path.exists():
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")
    shutil.rmtree(IMG_DIR / job_id, ignore_errors=True)
    shutil.rmtree(job_dir(job_id), ignore_errors=True)
