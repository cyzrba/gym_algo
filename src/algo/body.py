from __future__ import annotations

from src.algo.arm import ArmMeasurement
from src.algo.base import MeasurementBase
from src.algo.leg import LegMeasurement
from src.algo.shoulder import ShoulderMeasurement
from src.algo.waist import WaistMeasurement
from src.app.schemas.measurements import MeasurementResult


class BodyMeasurement(MeasurementBase):
    """全身测量调度：统一加载数据后分发给各子算法，避免重复IO和推理。"""

    def run(self) -> MeasurementResult:
        views = self.resolve_input_views()
        self.validate_input_view_paths(views)

        view_data_list = [
            self.load_view_data(rgb_path, depth_path, ply_path)
            for _, rgb_path, depth_path, ply_path in views
        ]

        arm = ArmMeasurement(self.config, self.pose_model)
        leg = LegMeasurement(self.config, self.pose_model)
        waist = WaistMeasurement(self.config, self.pose_model)
        shoulder = ShoulderMeasurement(self.config, self.pose_model)

        result = MeasurementResult(
            arm=arm.aggregate(view_data_list),
            leg=leg.aggregate(view_data_list),
            waist=waist.aggregate(view_data_list),
            shoulder=shoulder.aggregate(view_data_list),
        )
        self.emit_schema(result)
        return result
