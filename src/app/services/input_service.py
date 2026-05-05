from __future__ import annotations

import shutil
import zipfile
from pathlib import Path

from fastapi import HTTPException, UploadFile

from app.core.paths import IMG_DIR
from app.schemas.common import SavedInputs
from app.utils.file_utils import ensure_child_path, safe_relative_path, save_upload, upload_target_path


def require_upload(name: str, upload: UploadFile | None) -> UploadFile:
    if upload is None:
        raise HTTPException(status_code=400, detail=f"Missing required file: {name}")
    return upload


async def save_inputs(
    *,
    job_id: str,
    input_mode: str,
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

    return SavedInputs(input_mode=input_mode, paths=paths)  # type: ignore[arg-type]


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
    with zipfile.ZipFile(archive_path) as archive:
        for member in archive.infolist():
            if member.is_dir():
                continue
            relative_path = safe_relative_path(member.filename)
            target_path = extract_dir / relative_path
            ensure_child_path(extract_dir, target_path)
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
