from __future__ import annotations

import logging

import numpy as np

from src.algo.base import LowConfidenceError, MeasurementBase
from src.app.schemas.measurements import LegMeasurement as LegSchema

logger = logging.getLogger(__name__)


class LegMeasurement(MeasurementBase):
    """大腿围度测量：复用手臂切片流程，额外增加髋中线过滤分离左右腿。"""

    def run(self) -> LegSchema:
        views = self.resolve_input_views()
        self.validate_input_view_paths(views)
        view_data_list = [
            self.load_view_data(rgb_path, depth_path, ply_path)
            for _, rgb_path, depth_path, ply_path in views
        ]
        result = self.aggregate(view_data_list)
        self.emit_schema(result)
        return result

    def aggregate(self, view_data_list: list[dict]) -> LegSchema:
        """接收已加载的多视角数据，返回腿部测量结果。"""
        requested_sides = (
            ["left", "right"] if self.config.leg_side == "both" else [self.config.leg_side]
        )
        totals: dict[str, float | None] = {side: 0.0 for side in requested_sides}

        for view_data in view_data_list:
            for side in requested_sides:
                if totals[side] is None:
                    continue
                arc_length = self._measure_leg_side(
                    leg_side=side,
                    points_xyz=view_data["points_xyz"],
                    rgb_shape=view_data["rgb_shape"],
                    cloud_shape=view_data["cloud_shape"],
                    joint_map=view_data["joint_map"],
                )
                if arc_length is None:
                    totals[side] = None
                else:
                    totals[side] += arc_length

        return LegSchema(
            left_arc=round(totals["left"], 6) if totals.get("left") is not None else None,
            right_arc=round(totals["right"], 6) if totals.get("right") is not None else None,
        )

    def _measure_leg_side(
        self,
        *,
        leg_side: str,
        points_xyz: np.ndarray,
        rgb_shape: tuple[int, int],
        cloud_shape: tuple[int, int],
        joint_map: dict[str, dict[str, object]],
    ) -> float | None:
        """测量单侧大腿的可见弧长（米）。置信度不足时返回 None。"""
        try:
            return self._measure_leg_side_inner(
                leg_side=leg_side,
                points_xyz=points_xyz,
                rgb_shape=rgb_shape,
                cloud_shape=cloud_shape,
                joint_map=joint_map,
            )
        except LowConfidenceError as exc:
            logger.warning("Leg %s skipped: %s", leg_side, exc)
            return None

    def _measure_leg_side_inner(
        self,
        *,
        leg_side: str,
        points_xyz: np.ndarray,
        rgb_shape: tuple[int, int],
        cloud_shape: tuple[int, int],
        joint_map: dict[str, dict[str, object]],
    ) -> float:
        """测量单侧大腿的可见弧长（米）。"""
        cfg = self.config
        hip_name = f"{leg_side}_hip"
        knee_name = f"{leg_side}_knee"
        opposite_hip_name = "right_hip" if leg_side == "left" else "left_hip"

        hip_joint = self.pick_joint(joint_map, hip_name, min_confidence=cfg.hip_min_confidence)
        knee_joint = self.pick_joint(joint_map, knee_name, min_confidence=cfg.knee_min_confidence)
        opposite_hip_joint = self.pick_joint(joint_map, opposite_hip_name, min_confidence=cfg.hip_min_confidence)

        hip_uv = self.joint_to_uv(hip_joint)
        knee_uv = self.joint_to_uv(knee_joint)
        opposite_hip_uv = self.joint_to_uv(opposite_hip_joint)

        hip_xyz, _, _ = self._sample_limb_endpoint_xyz(
            points_xyz=points_xyz,
            endpoint_uv_rgb=hip_uv,
            fallback_uv_rgb=knee_uv,
            rgb_shape=rgb_shape,
            cloud_shape=cloud_shape,
            window_size=cfg.depth_window,
            endpoint_name=hip_name,
        )
        knee_xyz, _, _ = self._sample_limb_endpoint_xyz(
            points_xyz=points_xyz,
            endpoint_uv_rgb=knee_uv,
            fallback_uv_rgb=hip_uv,
            rgb_shape=rgb_shape,
            cloud_shape=cloud_shape,
            window_size=cfg.depth_window,
            endpoint_name=knee_name,
        )
        opposite_hip_xyz, _, _ = self._sample_limb_endpoint_xyz(
            points_xyz=points_xyz,
            endpoint_uv_rgb=opposite_hip_uv,
            fallback_uv_rgb=hip_uv,
            rgb_shape=rgb_shape,
            cloud_shape=cloud_shape,
            window_size=cfg.depth_window,
            endpoint_name=opposite_hip_name,
        )

        depth_high_m = max(float(hip_xyz[2]), float(knee_xyz[2])) + cfg.leg_depth_margin_m
        filtered_points, _ = self.filter_point_cloud_by_max_depth(
            points_xyz, depth_high_m=depth_high_m
        )

        center_xyz, u_axis, v_axis, w_axis, _ = self.build_local_frame(
            shoulder_xyz=hip_xyz,
            elbow_xyz=knee_xyz,
            center_ratio=cfg.thigh_center_ratio,
        )

        slice_world, slice_local, _ = self.extract_perpendicular_slice(
            filtered_points,
            origin_xyz=center_xyz,
            u_axis=u_axis,
            v_axis=v_axis,
            w_axis=w_axis,
            slice_half_thickness_m=cfg.leg_slice_half_thickness_m,
            slice_radius_m=cfg.leg_slice_radius_m,
        )

        slice_world, slice_local = self._filter_by_hip_midline(
            slice_world=slice_world,
            slice_local=slice_local,
            hip_xyz=hip_xyz,
            opposite_hip_xyz=opposite_hip_xyz,
            leg_side=leg_side,
            margin_m=cfg.hip_midline_margin_m,
        )

        filtered_slice_local, _ = self.filter_deviating_slice_points(
            slice_local,
            iqr_scale=cfg.leg_deviation_iqr_scale,
            min_keep_ratio=cfg.leg_deviation_min_keep_ratio,
            min_keep_points=cfg.leg_deviation_min_points,
        )

        visible_arc_uv, _ = self.extract_visible_arc_uv(filtered_slice_local)
        return self.compute_polyline_length(visible_arc_uv)

    def _filter_by_hip_midline(
        self,
        *,
        slice_world: np.ndarray,
        slice_local: np.ndarray,
        hip_xyz: np.ndarray,
        opposite_hip_xyz: np.ndarray,
        leg_side: str,
        margin_m: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """通过髋中线将当前侧大腿的点云与对侧分离。"""
        if len(slice_world) == 0:
            return slice_world, slice_local

        midpoint = (np.asarray(hip_xyz, dtype=np.float64) + np.asarray(opposite_hip_xyz, dtype=np.float64)) / 2.0
        hip_dir = np.asarray(hip_xyz, dtype=np.float64) - np.asarray(opposite_hip_xyz, dtype=np.float64)
        hip_dir_norm = float(np.linalg.norm(hip_dir))
        if hip_dir_norm < 1e-6:
            return slice_world, slice_local

        hip_axis = hip_dir / hip_dir_norm

        offsets = np.asarray(slice_world, dtype=np.float64) - midpoint
        projections = offsets @ hip_axis

        if leg_side == "left":
            keep = projections >= -margin_m
        else:
            keep = projections <= margin_m

        kept_world = np.asarray(slice_world[keep], dtype=np.float64)
        kept_local = np.asarray(slice_local[keep], dtype=np.float64)

        if len(kept_world) < self.config.leg_deviation_min_points:
            return slice_world, slice_local

        return kept_world, kept_local

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
