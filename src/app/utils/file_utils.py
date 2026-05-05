from __future__ import annotations

import os
from pathlib import Path

from fastapi import HTTPException, UploadFile


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
