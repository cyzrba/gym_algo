from __future__ import annotations

import logging

from src.algo.arm import ArmMeasurement
from src.algo.base import LowConfidenceError, MeasurementBase
from src.algo.leg import LegMeasurement
from src.algo.shoulder import ShoulderMeasurement
from src.algo.waist import WaistMeasurement
from src.app.schemas.measurements import MeasurementResult

logger = logging.getLogger(__name__)


class BodyMeasurement(MeasurementBase):
    """全身测量调度：统一加载数据后分发给各子算法，避免重复IO和推理。"""

    def run(self) -> MeasurementResult:
        views = self.resolve_input_views()
        self.validate_input_view_paths(views)

        view_data_list = [
            self.load_view_data(rgb_path, depth_path, ply_path)
            for _, rgb_path, depth_path, ply_path in views
        ]

        arm_result = self._safe_aggregate("arm", ArmMeasurement, view_data_list)
        leg_result = self._safe_aggregate("leg", LegMeasurement, view_data_list)
        waist_result = self._safe_aggregate("waist", WaistMeasurement, view_data_list)
        shoulder_result = self._safe_aggregate("shoulder", ShoulderMeasurement, view_data_list)

        result = MeasurementResult(
            arm=arm_result,
            leg=leg_result,
            waist=waist_result,
            shoulder=shoulder_result,
        )
        self.emit_schema(result)
        return result

    def _safe_aggregate(self, part_name: str, cls: type[MeasurementBase], view_data_list: list[dict]):
        """Run a sub-algorithm's aggregate, returning None for the whole part on LowConfidenceError."""
        try:
            instance = cls(self.config, self.pose_model)
            return instance.aggregate(view_data_list)
        except LowConfidenceError as exc:
            logger.warning("Body %s skipped: %s", part_name, exc)
            return None
