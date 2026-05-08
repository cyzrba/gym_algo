from __future__ import annotations
import shutil
import zipfile
import os
from pathlib import Path
from src.app.schemas.common import ArchiveRead
from fastapi import HTTPException, UploadFile

# 解压zip文件夹到指定目录
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

def collect_front_back_inputs(source_dir: Path) -> ArchiveRead:
    """扫描目录并收集前后视角所需的 RGB/Depth/PLY 输入文件。"""
    discovered: dict[str, dict[str, Path]] = {"front": {}, "back": {}}
    for path in source_dir.rglob("*"):
        if not path.is_file():
            continue
        # 根据路径关键词判断是 front 还是 back 视角。
        side = detect_input_side(path)
        # 根据文件名和后缀判断是 rgb/depth/ply 哪一类输入。
        role = detect_input_file_role(path)
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

    return ArchiveRead(
        paths={name: path for name, path in required.items() if path is not None},
    )


def detect_input_side(path: Path) -> str | None:
    """从路径文本中识别输入所属视角（front/back）。"""
    tokens = [part.lower() for part in path.parts]
    if "front" in tokens or any("front" in token or "正面" in token for token in tokens):
        return "front"
    if "back" in tokens or any("back" in token or "背面" in token or "后面" in token for token in tokens):
        return "back"
    return None


def detect_input_file_role(path: Path) -> str | None:
    """从文件名与后缀识别输入类型（rgb/depth/ply）。"""
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


def ensure_child_path(base_dir: Path, target_path: Path) -> None:
    base = base_dir.resolve()
    target = target_path.resolve()
    if Path(os.path.commonpath([base, target])) != base:
        raise HTTPException(status_code=400, detail="Invalid path")


def upload_target_path(base_dir: Path, field_name: str, upload: UploadFile) -> Path:
    suffix = Path(upload.filename or "").suffix
    return base_dir / f"{field_name}{suffix}"


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
