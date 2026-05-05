from __future__ import annotations

from pathlib import Path

from fastapi import HTTPException

from app.core.constants import DEPTH_DTYPES, DEPTH_ENDIANS, MEASUREMENTS, RGB_FORMATS
from app.core.paths import POSE_MODEL_PATH, ROOT_DIR


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


def validate_measurement_params(
    *,
    rgb_format: str,
    depth_dtype: str,
    depth_endian: str,
    depth_scale: float,
    depth_window: int,
) -> None:
    validate_choice("rgb_format", rgb_format, RGB_FORMATS)
    validate_choice("depth_dtype", depth_dtype, DEPTH_DTYPES)
    validate_choice("depth_endian", depth_endian, DEPTH_ENDIANS)
    if depth_window < 1 or depth_window % 2 == 0:
        raise HTTPException(status_code=400, detail="depth_window must be a positive odd integer")
    if depth_scale <= 0:
        raise HTTPException(status_code=400, detail="depth_scale must be > 0")


def validate_choice(name: str, value: str, allowed: set[str]) -> str:
    if value not in allowed:
        raise HTTPException(status_code=400, detail=f"{name} must be one of: {', '.join(sorted(allowed))}")
    return value
