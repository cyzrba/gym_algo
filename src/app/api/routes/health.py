from __future__ import annotations

from app.core.constants import MEASUREMENTS, MEASUREMENT_MODULES
from app.core.paths import IMG_DIR, POSE_MODEL_PATH, RESULT_DIR, ROOT_DIR, SEG_MODEL_PATH, ensure_runtime_dirs


def get_health_payload() -> dict[str, object]:
    ensure_runtime_dirs()
    return {
        "status": "ok",
        "root_dir": str(ROOT_DIR),
        "input_dir": str(IMG_DIR),
        "result_dir": str(RESULT_DIR),
        "measurements": list(MEASUREMENTS),
        "scripts": {name: True for name in MEASUREMENT_MODULES},
        "models": {
            "pose": {"path": str(POSE_MODEL_PATH), "exists": POSE_MODEL_PATH.exists()},
            "seg": {"path": str(SEG_MODEL_PATH), "exists": SEG_MODEL_PATH.exists(), "reserved": True},
        },
    }
