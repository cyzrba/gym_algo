from __future__ import annotations

import cv2
import numpy as np
from scipy.spatial import cKDTree

from src.algo.base import MeasurementBase
from src.app.schemas.measurements import WaistMeasurement as WaistSchema


class WaistMeasurement(MeasurementBase):
    """腰部围度测量：水平切片 + kNN过滤 + XZ平面可见弧线。"""

    def run(self) -> WaistSchema:
        views = self.resolve_input_views()
        self.validate_input_view_paths(views)
        view_data_list = [
            self.load_view_data(rgb_path, depth_path, ply_path)
            for _, rgb_path, depth_path, ply_path in views
        ]
        result = self.aggregate(view_data_list)
        self.emit_schema(result)
        return result

    def aggregate(self, view_data_list: list[dict]) -> WaistSchema:
        """接收已加载的多视角数据，返回腰部测量结果。"""
        total_arc: float = 0.0

        for view_data in view_data_list:
            arc_length = self._measure_waist(
                points_xyz=view_data["points_xyz"],
                rgb_shape=view_data["rgb_shape"],
                cloud_shape=view_data["cloud_shape"],
                joint_map=view_data["joint_map"],
            )
            total_arc += arc_length

        return WaistSchema(waist_arc=round(total_arc, 6))

    def _measure_waist(
        self,
        *,
        points_xyz: np.ndarray,
        rgb_shape: tuple[int, int],
        cloud_shape: tuple[int, int],
        joint_map: dict[str, dict[str, object]],
    ) -> float:
        """测量腰部的可见弧长（米）。"""
        cfg = self.config

        left_shoulder = self.pick_joint(joint_map, "left_shoulder", min_confidence=0.20)
        right_shoulder = self.pick_joint(joint_map, "right_shoulder", min_confidence=0.20)
        left_hip = self.pick_joint(joint_map, "left_hip", min_confidence=0.20)
        right_hip = self.pick_joint(joint_map, "right_hip", min_confidence=0.20)

        shoulder_mid_uv = (self.joint_to_uv(left_shoulder) + self.joint_to_uv(right_shoulder)) / 2.0
        hip_mid_uv = (self.joint_to_uv(left_hip) + self.joint_to_uv(right_hip)) / 2.0
        waist_center_uv = hip_mid_uv + (shoulder_mid_uv - hip_mid_uv) * cfg.waist_up_ratio

        waist_center_xyz, _, _ = self.sample_joint_xyz_from_point_cloud(
            points_xyz=points_xyz,
            joint_uv_rgb=waist_center_uv,
            rgb_shape=rgb_shape,
            cloud_shape=cloud_shape,
            window_size=cfg.depth_window,
        )

        depth_high_m = float(waist_center_xyz[2]) + cfg.waist_depth_margin_m
        filtered_points, _ = self.filter_point_cloud_by_max_depth(
            points_xyz, depth_high_m=depth_high_m
        )

        slice_points = self._extract_waist_slice(
            filtered_points,
            waist_center_xyz=waist_center_xyz,
            half_height_m=cfg.waist_half_height_m,
            radius_m=cfg.waist_radius_m,
        )

        if not cfg.disable_robust_waist_filter:
            slice_points = self._knn_filter(
                slice_points,
                k=cfg.waist_filter_k,
                std_ratio=cfg.waist_filter_std_ratio,
                min_keep_ratio=cfg.visible_keep_ratio,
            )

        visible_arc = self._extract_visible_arc_xz(slice_points)
        return self.compute_polyline_length(visible_arc)

    def _extract_waist_slice(
        self,
        points_xyz: np.ndarray,
        *,
        waist_center_xyz: np.ndarray,
        half_height_m: float,
        radius_m: float,
    ) -> np.ndarray:
        """取腰部中心附近Y方向薄层 + XZ平面半径范围的切片。"""
        points = np.asarray(points_xyz, dtype=np.float64)
        valid = np.isfinite(points).all(axis=1) & (points[:, 2] > 0)
        points = points[valid]

        center_y = float(waist_center_xyz[1])
        center_x = float(waist_center_xyz[0])
        center_z = float(waist_center_xyz[2])

        y_keep = np.abs(points[:, 1] - center_y) <= half_height_m

        dx = points[:, 0] - center_x
        dz = points[:, 2] - center_z
        xz_dist = np.sqrt(dx * dx + dz * dz)
        xz_keep = xz_dist <= radius_m

        return np.asarray(points[y_keep & xz_keep], dtype=np.float64)

    def _knn_filter(
        self,
        points_xyz: np.ndarray,
        *,
        k: int,
        std_ratio: float,
        min_keep_ratio: float,
    ) -> np.ndarray:
        """kNN离群点过滤：移除平均邻居距离过大的点。"""
        if len(points_xyz) < k + 1:
            return points_xyz

        xz = points_xyz[:, [0, 2]]
        tree = cKDTree(xz)
        dists, _ = tree.query(xz, k=k + 1)
        mean_dists = dists[:, 1:].mean(axis=1)

        median_dist = float(np.median(mean_dists))
        std_dist = float(np.std(mean_dists))
        threshold = median_dist + std_ratio * std_dist

        keep = mean_dists <= threshold
        if np.count_nonzero(keep) < len(points_xyz) * min_keep_ratio:
            return points_xyz

        return np.asarray(points_xyz[keep], dtype=np.float64)

    def _extract_visible_arc_xz(self, points_xyz: np.ndarray) -> np.ndarray:
        """在XZ平面上提取可见弧线（Z均值较小侧 = 离相机近）。"""
        points = np.asarray(points_xyz, dtype=np.float64)
        points = points[np.isfinite(points).all(axis=1)]

        if len(points) < 2:
            return np.empty((0, 2), dtype=np.float64)

        xz = points[:, [0, 2]]
        xz_unique = np.unique(np.round(xz, 6), axis=0).astype(np.float64)

        if len(xz_unique) < 3:
            order = np.argsort(xz_unique[:, 0])
            return np.asarray(xz_unique[order], dtype=np.float64)

        hull = cv2.convexHull(xz_unique.astype(np.float32)).reshape(-1, 2).astype(np.float64)
        if len(hull) < 3:
            order = np.argsort(xz_unique[:, 0])
            return np.asarray(xz_unique[order], dtype=np.float64)

        left_idx = int(np.argmin(hull[:, 0]))
        right_idx = int(np.argmax(hull[:, 0]))
        if left_idx == right_idx:
            left_idx = int(np.argmin(hull[:, 1]))
            right_idx = int(np.argmax(hull[:, 1]))

        path_a, path_b = self.hull_path_between(hull, left_idx, right_idx)

        mean_z_a = float(path_a[:, 1].mean())
        mean_z_b = float(path_b[:, 1].mean())
        arc = path_a if mean_z_a <= mean_z_b else path_b

        if len(arc) >= 2 and arc[0, 0] > arc[-1, 0]:
            arc = arc[::-1]

        return np.asarray(arc, dtype=np.float64)
