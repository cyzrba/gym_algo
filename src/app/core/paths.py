from __future__ import annotations

from pathlib import Path


SRC_DIR = Path(__file__).resolve().parents[2]
ROOT_DIR = SRC_DIR.parent
IMG_DIR = ROOT_DIR / "img"
RESULT_DIR = ROOT_DIR / "result"
MPLCONFIG_DIR = ROOT_DIR / ".mplconfig"
POSE_MODEL_PATH = ROOT_DIR / "yolo26n-pose.pt"
SEG_MODEL_PATH = ROOT_DIR / "yolo26n-seg.pt"


def ensure_runtime_dirs() -> None:
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    MPLCONFIG_DIR.mkdir(parents=True, exist_ok=True)


def job_dir(job_id: str) -> Path:
    return RESULT_DIR / job_id


def job_json_path(job_id: str) -> Path:
    return job_dir(job_id) / "job.json"
