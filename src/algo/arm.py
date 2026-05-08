from __future__ import annotations

import numpy as np

from src.algo.base import MeasurementBase
from src.app.schemas.measurements import ArmMeasurement as ArmSchema


class ArmMeasurement(MeasurementBase):
    """手臂围度测量：通过点云切片提取可见弧长。"""

    def run(self) -> ArmSchema:
        views = self.resolve_input_views()
        self.validate_input_view_paths(views)
        view_data_list = [
            self.load_view_data(rgb_path, depth_path, ply_path)
            for _, rgb_path, depth_path, ply_path in views
        ]
        result = self.aggregate(view_data_list)
        self.emit_schema(result)
        return result

    def aggregate(self, view_data_list: list[dict]) -> ArmSchema:
        """接收已加载的多视角数据，返回手臂测量结果。"""
        requested_sides = (
            ["left", "right"] if self.config.arm_side == "both" else [self.config.arm_side]
        )
        totals: dict[str, float] = {side: 0.0 for side in requested_sides}

        for view_data in view_data_list:
            for side in requested_sides:
                arc_length = self._measure_arm_side(
                    arm_side=side,
                    points_xyz=view_data["points_xyz"],
                    rgb_shape=view_data["rgb_shape"],
                    cloud_shape=view_data["cloud_shape"],
                    joint_map=view_data["joint_map"],
                )
                totals[side] += arc_length

        return ArmSchema(
            left_arc=round(totals["left"], 6) if "left" in totals else None,
            right_arc=round(totals["right"], 6) if "right" in totals else None,
        )

    def _measure_arm_side(
        self,
        *,
        arm_side: str,
        points_xyz: np.ndarray,
        rgb_shape: tuple[int, int],
        cloud_shape: tuple[int, int],
        joint_map: dict[str, dict[str, object]],
    ) -> float:
        """测量单侧手臂的可见弧长（米）。"""
        cfg = self.config
        shoulder_name = f"{arm_side}_shoulder"
        elbow_name = f"{arm_side}_elbow"

        shoulder_joint = self.pick_joint(joint_map, shoulder_name, min_confidence=0.20)
        elbow_joint = self.pick_joint(joint_map, elbow_name, min_confidence=0.20)

        shoulder_uv = self.joint_to_uv(shoulder_joint)
        elbow_uv = self.joint_to_uv(elbow_joint)

        shoulder_xyz, _, _ = self._sample_limb_endpoint_xyz(
            points_xyz=points_xyz,
            endpoint_uv_rgb=shoulder_uv,
            fallback_uv_rgb=elbow_uv,
            rgb_shape=rgb_shape,
            cloud_shape=cloud_shape,
            window_size=cfg.depth_window,
            endpoint_name=shoulder_name,
        )
        elbow_xyz, _, _ = self._sample_limb_endpoint_xyz(
            points_xyz=points_xyz,
            endpoint_uv_rgb=elbow_uv,
            fallback_uv_rgb=shoulder_uv,
            rgb_shape=rgb_shape,
            cloud_shape=cloud_shape,
            window_size=cfg.depth_window,
            endpoint_name=elbow_name,
        )

        depth_high_m = max(float(shoulder_xyz[2]), float(elbow_xyz[2])) + cfg.depth_margin_m
        filtered_points, _ = self.filter_point_cloud_by_max_depth(
            points_xyz, depth_high_m=depth_high_m
        )

        center_xyz, u_axis, v_axis, w_axis, _ = self.build_local_frame(
            shoulder_xyz=shoulder_xyz,
            elbow_xyz=elbow_xyz,
            center_ratio=cfg.arm_center_ratio,
        )

        _, slice_local, _ = self.extract_perpendicular_slice(
            filtered_points,
            origin_xyz=center_xyz,
            u_axis=u_axis,
            v_axis=v_axis,
            w_axis=w_axis,
            slice_half_thickness_m=cfg.slice_half_thickness_m,
            slice_radius_m=cfg.slice_radius_m,
        )

        filtered_slice_local, _ = self.filter_deviating_slice_points(
            slice_local,
            iqr_scale=cfg.deviation_iqr_scale,
            min_keep_ratio=cfg.deviation_min_keep_ratio,
            min_keep_points=cfg.deviation_min_points,
        )

        visible_arc_uv, _ = self.extract_visible_arc_uv(filtered_slice_local)
        return self.compute_polyline_length(visible_arc_uv)

    def _sample_limb_endpoint_xyz(
        self,
        *,
        points_xyz: np.ndarray,
        endpoint_uv_rgb: np.ndarray,
        fallback_uv_rgb: np.ndarray,
        rgb_shape: tuple[int, int],
        cloud_shape: tuple[int, int],
        window_size: int,
        endpoint_name: str,
    ) -> tuple[np.ndarray, tuple[int, int], dict]:
        """尝试从精确像素采样3D坐标，失败则沿向对端偏移逐步重试。"""
        endpoint = self._clip_uv_to_shape(endpoint_uv_rgb, rgb_shape)
        fallback = self._clip_uv_to_shape(fallback_uv_rgb, rgb_shape)
        ratios = [0.0, 0.03, 0.06, 0.10, 0.15, 0.20, 0.28, 0.36]
        failures: list[str] = []

        for ratio in ratios:
            candidate_uv = endpoint + (fallback - endpoint) * float(ratio)
            candidate_uv = self._clip_uv_to_shape(candidate_uv, rgb_shape)
            try:
                xyz, cloud_uv, meta = self.sample_joint_xyz_from_point_cloud(
                    points_xyz=points_xyz,
                    joint_uv_rgb=candidate_uv,
                    rgb_shape=rgb_shape,
                    cloud_shape=cloud_shape,
                    window_size=window_size,
                )
                out_meta: dict = dict(meta)
                out_meta.update({
                    "endpoint_name": endpoint_name,
                    "fallback_ratio": float(ratio),
                    "fallback_attempts": int(len(failures) + 1),
                })
                return xyz, cloud_uv, out_meta
            except RuntimeError as exc:
                failures.append(str(exc))

        raise RuntimeError(
            f"No valid point-cloud XYZ for {endpoint_name}; "
            f"attempts failed: {' | '.join(failures[-3:])}"
        )

    @staticmethod
    def _clip_uv_to_shape(uv: np.ndarray, shape_hw: tuple[int, int]) -> np.ndarray:
        h, w = shape_hw
        clipped = np.asarray(uv, dtype=np.float64).copy()
        clipped[0] = float(np.clip(clipped[0], 0, max(w - 1, 0)))
        clipped[1] = float(np.clip(clipped[1], 0, max(h - 1, 0)))
        return clipped
