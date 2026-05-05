from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from app.core.constants import MEASUREMENT_MODULES
from app.core.paths import MPLCONFIG_DIR, ROOT_DIR, SRC_DIR, job_dir
from app.schemas.common import SavedInputs
from app.services.artifact_service import list_artifacts
from app.services.job_service import now_epoch, read_job, write_job


TASK_SEMAPHORE = asyncio.Semaphore(1)


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
    module_name = MEASUREMENT_MODULES[measurement]
    output_dir.mkdir(parents=True, exist_ok=True)

    started_at = now_epoch()
    env = {
        **os.environ,
        "MPLCONFIGDIR": str(MPLCONFIG_DIR.resolve()),
        "PYTHONPATH": merge_pythonpath(str(SRC_DIR), os.environ.get("PYTHONPATH")),
    }
    completed = await asyncio.to_thread(
        subprocess.run,
        [sys.executable, "-m", module_name, *script_args],
        cwd=str(ROOT_DIR),
        env=env,
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


def merge_pythonpath(src_path: str, existing: str | None) -> str:
    if not existing:
        return src_path
    return os.pathsep.join([src_path, existing])


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
