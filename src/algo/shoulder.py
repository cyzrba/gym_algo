from __future__ import annotations

import numpy as np

from src.algo.base import MeasurementBase
from src.app.schemas.measurements import ShoulderMeasurement as ShoulderSchema


class ShoulderMeasurement(MeasurementBase):
    """肩宽测量：通过左右肩关键点的3D欧氏距离计算。"""

    def run(self) -> ShoulderSchema:
        views = self.resolve_input_views()
        self.validate_input_view_paths(views)
        view_data_list = [
            self.load_view_data(rgb_path, depth_path, ply_path)
            for _, rgb_path, depth_path, ply_path in views
        ]
        result = self.aggregate(view_data_list)
        self.emit_schema(result)
        return result

    def aggregate(self, view_data_list: list[dict]) -> ShoulderSchema:
        """接收已加载的多视角数据，返回肩宽测量结果。"""
        widths: list[float] = []
        for view_data in view_data_list:
            width = self._measure_shoulder_width(
                points_xyz=view_data["points_xyz"],
                rgb_shape=view_data["rgb_shape"],
                cloud_shape=view_data["cloud_shape"],
                joint_map=view_data["joint_map"],
            )
            widths.append(width)

        mean_width = round(float(np.mean(widths)), 6) if widths else None
        return ShoulderSchema(shoulder_width=mean_width)

    def _measure_shoulder_width(
        self,
        *,
        points_xyz: np.ndarray,
        rgb_shape: tuple[int, int],
        cloud_shape: tuple[int, int],
        joint_map: dict[str, dict[str, object]],
    ) -> float:
        """计算左右肩3D距离（米）。"""
        cfg = self.config

        left_joint = self.pick_joint(joint_map, "left_shoulder", min_confidence=cfg.shoulder_min_confidence)
        right_joint = self.pick_joint(joint_map, "right_shoulder", min_confidence=cfg.shoulder_min_confidence)

        left_xyz, _, _ = self.sample_joint_xyz_from_point_cloud(
            points_xyz=points_xyz,
            joint_uv_rgb=self.joint_to_uv(left_joint),
            rgb_shape=rgb_shape,
            cloud_shape=cloud_shape,
            window_size=cfg.depth_window,
        )
        right_xyz, _, _ = self.sample_joint_xyz_from_point_cloud(
            points_xyz=points_xyz,
            joint_uv_rgb=self.joint_to_uv(right_joint),
            rgb_shape=rgb_shape,
            cloud_shape=cloud_shape,
            window_size=cfg.depth_window,
        )

        return float(np.linalg.norm(left_xyz - right_xyz))
