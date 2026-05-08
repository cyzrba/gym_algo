from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from fastapi import UploadFile

from src.algo import BodyMeasurement, Config
from src.algo.model import pose_model
from src.app.schemas import MeasurementResult
from src.app.utils.file_utils import collect_front_back_inputs, extract_zip_safely


async def process_measurement(archive: UploadFile) -> MeasurementResult:
    suffix = Path(archive.filename or "").suffix.lower()
    if suffix != ".zip":
        raise ValueError("Only .zip archives are supported")

    tmp_dir = Path(tempfile.mkdtemp(prefix="measurement_"))
    try:
        archive_path = tmp_dir / "source.zip"
        with archive_path.open("wb") as f:
            while True:
                chunk = await archive.read(1024 * 1024)
                if not chunk:
                    break
                f.write(chunk)

        extract_dir = tmp_dir / "source"
        extract_zip_safely(archive_path, extract_dir)
        inputs = collect_front_back_inputs(extract_dir)

        config = Config(
            front_rgb=inputs.paths["front_rgb"],
            front_depth=inputs.paths["front_depth"],
            front_ply=inputs.paths["front_ply"],
            back_rgb=inputs.paths["back_rgb"],
            back_depth=inputs.paths["back_depth"],
            back_ply=inputs.paths["back_ply"],
            output_dir=tmp_dir / "result",
        )

        body = BodyMeasurement(config, pose_model.model)
        result = body.run()
        return result
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
