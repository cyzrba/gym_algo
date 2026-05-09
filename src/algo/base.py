from __future__ import annotations

import json

from plyfile import PlyData
from abc import ABC, abstractmethod
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from .config import Config
from src.app.utils.constants import COCO_KEYPOINT_NAMES


class LowConfidenceError(RuntimeError):
    """Raised when a joint's detection confidence is below the required threshold."""

    def __init__(self, joint_name: str, confidence: float, min_confidence: float):
        self.joint_name = joint_name
        self.confidence = confidence
        self.min_confidence = min_confidence
        super().__init__(
            f"Joint confidence too low for {joint_name}: "
            f"{confidence:.4f} < {min_confidence:.4f}"
        )


class MeasurementBase(ABC):
    """测量算法基类：集中配置、模型与通用工具方法。"""

    def __init__(self, config: Config, model: YOLO):
        """初始化共享配置与姿态模型。"""
        self.config = config
        self.pose_model = model

    @abstractmethod
    def run(self) -> None:
        """子类入口：组装检测流程并输出对应 schema。"""
        pass

    def emit_schema(self, result: object) -> None:
        """将测量结果 schema 以 JSON 格式输出到控制台。"""
        data = result.model_dump() if hasattr(result, "model_dump") else result
        print(json.dumps(data, ensure_ascii=False, indent=2))

    def resolve_input_views(self) -> list[tuple[str, Path, Path, Path]]:
        """从配置解析前后视角输入路径。"""
        cfg = self.config
        if not (cfg.front_rgb and cfg.front_depth and cfg.front_ply and cfg.back_rgb and cfg.back_depth and cfg.back_ply):
            raise RuntimeError("front/back RGB, depth, ply must all be provided in Config.")
        return [
            ("front", cfg.front_rgb, cfg.front_depth, cfg.front_ply),
            ("back", cfg.back_rgb, cfg.back_depth, cfg.back_ply),
        ]

    def validate_input_view_paths(self, input_views: list[tuple[str, Path, Path, Path]]) -> None:
        """校验每个视角的 RGB/Depth/PLY 文件是否存在。"""
        for view_name, rgb_path, depth_path, ply_path in input_views:
            if not rgb_path.exists():
                raise FileNotFoundError(f"[{view_name}] RGB input not found: {rgb_path}")
            if not depth_path.exists():
                raise FileNotFoundError(f"[{view_name}] Depth input not found: {depth_path}")
            if not ply_path.exists():
                raise FileNotFoundError(f"[{view_name}] Point cloud input not found: {ply_path}")

    def load_view_data(self, rgb_path: Path, depth_path: Path, ply_path: Path) -> dict:
        """加载单视角数据并执行姿态检测，返回统一字典。"""
        cfg = self.config
        image_bgr = MeasurementBase.load_rgb_image(
            rgb_path,
            width=int(cfg.rgb_width),
            height=int(cfg.rgb_height),
            rgb_format=str(cfg.rgb_format),
        )
        if image_bgr is None:
            raise RuntimeError(f"Failed to read RGB image: {rgb_path}")
        depth_map_raw = MeasurementBase.load_depth_map(
            depth_path,
            width=int(cfg.depth_width),
            height=int(cfg.depth_height),
            depth_dtype=str(cfg.depth_dtype),
            endian=str(cfg.depth_endian),
        )
        rgb_h, rgb_w = image_bgr.shape[:2]
        depth_h, depth_w = depth_map_raw.shape[:2]
        points_xyz, _ = MeasurementBase.load_point_cloud(ply_path)
        joints, detection_meta = MeasurementBase.detect_pose_joints(image_bgr, self.pose_model)
        return {
            "image_bgr": image_bgr,
            "depth_map_raw": depth_map_raw,
            "points_xyz": points_xyz,
            "rgb_shape": (rgb_h, rgb_w),
            "cloud_shape": (depth_h, depth_w),
            "joints": joints,
            "joint_map": MeasurementBase.joints_list_to_map(joints),
            "detection_meta": detection_meta,
        }

    # ------------------------------------------------------------------
    # 静态工具方法
    # ------------------------------------------------------------------

    @staticmethod
    def load_point_cloud(ply_path: Path) -> tuple[np.ndarray, np.ndarray | None]:
        """读取 PLY 点云并返回点坐标与可选颜色。"""
        ply = PlyData.read(str(ply_path))
        if "vertex" not in ply:
            raise RuntimeError(f"PLY 中没有 vertex 元素: {ply_path}")

        vertex = ply["vertex"].data
        names = set(vertex.dtype.names or ())
        if not {"x", "y", "z"}.issubset(names):
            raise RuntimeError(f"PLY 顶点缺少 x/y/z 字段: {ply_path}")

        points = np.column_stack([vertex["x"], vertex["y"], vertex["z"]]).astype(np.float64)
        if len(points) == 0:
            raise RuntimeError(f"点云为空或读取失败: {ply_path}")

        colors: np.ndarray | None = None
        color_candidates = [("red", "green", "blue"), ("r", "g", "b")]
        for r_name, g_name, b_name in color_candidates:
            if {r_name, g_name, b_name}.issubset(names):
                colors = np.column_stack(
                    [vertex[r_name], vertex[g_name], vertex[b_name]]
                ).astype(np.float32)
                break

        return points, colors

    @staticmethod
    def load_rgb_image(path: Path, *, width: int, height: int, rgb_format: str) -> np.ndarray | None:
        """读取 RGB 图像，支持常规图片与 raw 格式。"""
        suffix = path.suffix.lower()
        if suffix == ".raw":
            channels = 1 if rgb_format == "gray8" else 3
            raw = np.fromfile(path, dtype=np.uint8)
            expected = width * height * channels
            if raw.size != expected:
                raise RuntimeError(f"RGB raw size mismatch: got {raw.size}, expected {expected}.")
            if channels == 1:
                img = raw.reshape(height, width)
                return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            img = raw.reshape(height, width, 3)
            if rgb_format == "rgb8":
                return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            return img.copy()

        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if image is None:
            return None
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        return image

    @staticmethod
    def load_depth_map(
        path: Path,
        *,
        width: int,
        height: int,
        depth_dtype: str,
        endian: str,
    ) -> np.ndarray:
        """读取深度图，支持 npy/raw/常规深度图。"""
        suffix = path.suffix.lower()
        if suffix == ".npy":
            return np.asarray(np.load(path))
        if suffix == ".raw":
            dtype_map = {"uint16": np.uint16, "uint8": np.uint8, "float32": np.float32}
            dtype = np.dtype(dtype_map[depth_dtype])
            if dtype.itemsize > 1:
                dtype = dtype.newbyteorder("<" if endian == "little" else ">")
            raw = np.fromfile(path, dtype=dtype)
            expected = width * height
            if raw.size != expected:
                raise RuntimeError(f"Depth raw size mismatch: got {raw.size}, expected {expected}.")
            return raw.reshape(height, width)

        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise RuntimeError(f"Failed to read depth image: {path}")
        if image.ndim == 3:
            image = image[:, :, 0]
        return np.asarray(image)

    @staticmethod
    def prepare_depth_uv(rgb_uv: np.ndarray, rgb_shape: tuple[int, int], depth_shape: tuple[int, int]) -> np.ndarray:
        """将 RGB 坐标映射到深度图坐标系。"""
        rgb_h, rgb_w = rgb_shape
        depth_h, depth_w = depth_shape
        if (rgb_h, rgb_w) == (depth_h, depth_w):
            return rgb_uv.astype(np.float64)
        return np.array(
            [float(rgb_uv[0]) * depth_w / rgb_w, float(rgb_uv[1]) * depth_h / rgb_h],
            dtype=np.float64,
        )

    @staticmethod
    def sample_depth_median(
        depth_map_raw: np.ndarray,
        uv: np.ndarray,
        window_size: int,
    ) -> tuple[float, tuple[int, int]]:
        """在深度图窗口内采样中位数深度。"""
        if window_size < 1 or window_size % 2 == 0:
            raise RuntimeError("--depth-window must be a positive odd integer.")
        h, w = depth_map_raw.shape[:2]
        u = int(round(float(uv[0])))
        v = int(round(float(uv[1])))
        radius = window_size // 2
        left = max(0, u - radius)
        right = min(w, u + radius + 1)
        top = max(0, v - radius)
        bottom = min(h, v + radius + 1)
        patch = np.asarray(depth_map_raw[top:bottom, left:right]).astype(np.float64)
        valid = patch[np.isfinite(patch) & (patch > 0)]
        if len(valid) == 0:
            raise RuntimeError(f"No valid depth near pixel ({u}, {v}).")
        return float(np.median(valid)), (u, v)

    @staticmethod
    def sample_joint_xyz_from_point_cloud(
        *,
        points_xyz: np.ndarray,
        joint_uv_rgb: np.ndarray,
        rgb_shape: tuple[int, int],
        cloud_shape: tuple[int, int],
        window_size: int,
    ) -> tuple[np.ndarray, tuple[int, int], dict[str, int | str]]:
        """在有组织点云中采样关节点 XYZ。"""
        if window_size < 1 or window_size % 2 == 0:
            raise RuntimeError("--depth-window must be a positive odd integer.")

        cloud_h, cloud_w = cloud_shape
        expected_count = cloud_h * cloud_w
        if len(points_xyz) != expected_count:
            raise RuntimeError(
                f"Point cloud must be organized as {cloud_w}x{cloud_h}; "
                f"got {len(points_xyz)} points, expected {expected_count}."
            )

        joint_uv_cloud = MeasurementBase.prepare_depth_uv(joint_uv_rgb, rgb_shape, cloud_shape)
        u = int(round(float(joint_uv_cloud[0])))
        v = int(round(float(joint_uv_cloud[1])))
        if not (0 <= u < cloud_w and 0 <= v < cloud_h):
            raise RuntimeError(f"Joint pixel ({u}, {v}) is outside the point cloud image bounds.")

        points_grid = np.asarray(points_xyz, dtype=np.float64).reshape(cloud_h, cloud_w, 3)
        exact_xyz = np.asarray(points_grid[v, u], dtype=np.float64)
        exact_valid = bool(np.isfinite(exact_xyz).all() and exact_xyz[2] > 0)
        if exact_valid:
            return exact_xyz, (u, v), {
                "sample_method": "exact_pixel",
                "valid_sample_count": 1,
            }

        radius = window_size // 2
        left = max(0, u - radius)
        right = min(cloud_w, u + radius + 1)
        top = max(0, v - radius)
        bottom = min(cloud_h, v + radius + 1)
        patch = np.asarray(points_grid[top:bottom, left:right], dtype=np.float64).reshape(-1, 3)
        valid = np.isfinite(patch).all(axis=1) & (patch[:, 2] > 0)
        valid_points = patch[valid]
        if len(valid_points) == 0:
            raise RuntimeError(f"No valid point-cloud XYZ near pixel ({u}, {v}).")

        return np.median(valid_points, axis=0).astype(np.float64), (u, v), {
            "sample_method": "window_median",
            "valid_sample_count": int(len(valid_points)),
        }

    @staticmethod
    def filter_point_cloud_by_max_depth(
        points_xyz: np.ndarray,
        *,
        depth_high_m: float,
    ) -> tuple[np.ndarray, dict[str, int | float]]:
        """按最大深度阈值过滤点云。"""
        valid = np.isfinite(points_xyz).all(axis=1) & (points_xyz[:, 2] > 0)
        valid_points = np.asarray(points_xyz[valid], dtype=np.float64)
        keep = valid_points[:, 2] <= float(depth_high_m)
        filtered = np.asarray(valid_points[keep], dtype=np.float64)
        return filtered, {
            "input_count": int(len(points_xyz)),
            "finite_positive_count": int(len(valid_points)),
            "output_count": int(len(filtered)),
            "depth_high_m": float(depth_high_m),
        }

    @staticmethod
    def build_local_frame(
        *,
        shoulder_xyz: np.ndarray,
        elbow_xyz: np.ndarray,
        center_ratio: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
        """基于两端点构建局部坐标系。"""
        shoulder = np.asarray(shoulder_xyz, dtype=np.float64)
        elbow = np.asarray(elbow_xyz, dtype=np.float64)
        arm_vec = elbow - shoulder
        arm_length_m = float(np.linalg.norm(arm_vec))
        if arm_length_m < 1e-6:
            raise RuntimeError("Shoulder and elbow 3D points are too close to define an arm axis.")

        w_axis = arm_vec / arm_length_m
        center = shoulder + arm_vec * float(np.clip(center_ratio, 0.0, 1.0))

        camera_depth_axis = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        u_axis = camera_depth_axis - w_axis * float(np.dot(camera_depth_axis, w_axis))
        if float(np.linalg.norm(u_axis)) < 1e-6:
            u_axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
            u_axis = u_axis - w_axis * float(np.dot(u_axis, w_axis))
        u_axis = u_axis / max(float(np.linalg.norm(u_axis)), 1e-9)

        v_axis = np.cross(w_axis, u_axis)
        v_axis = v_axis / max(float(np.linalg.norm(v_axis)), 1e-9)
        return center.astype(np.float64), u_axis.astype(np.float64), v_axis.astype(np.float64), w_axis.astype(np.float64), arm_length_m

    @staticmethod
    def map_points_to_local_frame(
        points_xyz: np.ndarray,
        *,
        origin_xyz: np.ndarray,
        u_axis: np.ndarray,
        v_axis: np.ndarray,
        w_axis: np.ndarray,
    ) -> np.ndarray:
        """将世界坐标点映射到局部坐标系。"""
        if len(points_xyz) == 0:
            return np.empty((0, 3), dtype=np.float64)
        rel = np.asarray(points_xyz, dtype=np.float64) - np.asarray(origin_xyz, dtype=np.float64)
        return np.column_stack([rel @ u_axis, rel @ v_axis, rel @ w_axis]).astype(np.float64)

    @staticmethod
    def extract_perpendicular_slice(
        points_xyz: np.ndarray,
        *,
        origin_xyz: np.ndarray,
        u_axis: np.ndarray,
        v_axis: np.ndarray,
        w_axis: np.ndarray,
        slice_half_thickness_m: float,
        slice_radius_m: float,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, int | float]]:
        """提取垂直于主轴的点云切片。"""
        points = np.asarray(points_xyz, dtype=np.float64)
        valid = np.isfinite(points).all(axis=1) & (points[:, 2] > 0)
        valid_points = points[valid]
        local = MeasurementBase.map_points_to_local_frame(
            valid_points,
            origin_xyz=origin_xyz,
            u_axis=u_axis,
            v_axis=v_axis,
            w_axis=w_axis,
        )
        axial_abs = np.abs(local[:, 2])
        radial = np.linalg.norm(local[:, :2], axis=1)
        keep = axial_abs <= float(slice_half_thickness_m)
        if slice_radius_m > 0:
            keep &= radial <= float(slice_radius_m)

        slice_world = np.asarray(valid_points[keep], dtype=np.float64)
        slice_local = np.asarray(local[keep], dtype=np.float64)
        return slice_world, slice_local, {
            "input_count": int(len(points_xyz)),
            "finite_positive_count": int(len(valid_points)),
            "output_count": int(len(slice_world)),
            "slice_half_thickness_m": float(slice_half_thickness_m),
            "slice_radius_m": float(slice_radius_m),
            "max_abs_axis_m": 0.0 if len(slice_local) == 0 else float(np.max(np.abs(slice_local[:, 2]))),
            "max_radial_m": 0.0 if len(slice_local) == 0 else float(np.max(np.linalg.norm(slice_local[:, :2], axis=1))),
        }

    @staticmethod
    def compute_polyline_length(polyline_xy: np.ndarray) -> float:
        """计算折线总长度。"""
        points = np.asarray(polyline_xy, dtype=np.float64)
        if len(points) < 2:
            return 0.0
        diffs = np.diff(points, axis=0)
        return float(np.linalg.norm(diffs, axis=1).sum())

    @staticmethod
    def filter_deviating_slice_points(
        slice_points_local: np.ndarray,
        *,
        iqr_scale: float,
        min_keep_ratio: float,
        min_keep_points: int,
    ) -> tuple[np.ndarray, dict[str, int | float | str]]:
        """使用稳健统计过滤切片离群点。"""
        points = np.asarray(slice_points_local, dtype=np.float64)
        finite = np.isfinite(points).all(axis=1)
        points = points[finite]
        if len(points) < max(4, int(min_keep_points)):
            return points, {
                "status": "skip_small_input",
                "input_count": int(len(slice_points_local)),
                "finite_count": int(len(points)),
                "output_count": int(len(points)),
                "removed_count": 0,
                "keep_ratio": 1.0 if len(points) > 0 else 0.0,
                "iqr_scale": float(iqr_scale),
            }

        uv = points[:, :2]
        center = np.median(uv, axis=0)
        q25, q75 = np.quantile(uv, [0.25, 0.75], axis=0)
        iqr = np.maximum(q75 - q25, 1e-6)
        robust_sigma = np.maximum(iqr / 1.349, 1e-6)
        normalized_radius = np.linalg.norm((uv - center) / robust_sigma, axis=1)
        r25, r75 = np.quantile(normalized_radius, [0.25, 0.75])
        radius_iqr = max(float(r75 - r25), 1e-6)
        threshold = float(r75 + float(iqr_scale) * radius_iqr)
        keep = normalized_radius <= threshold

        kept_count = int(np.count_nonzero(keep))
        min_required = max(int(min_keep_points), int(np.ceil(len(points) * float(min_keep_ratio))))
        if kept_count < min_required:
            return points, {
                "status": "reverted_low_support",
                "input_count": int(len(slice_points_local)),
                "finite_count": int(len(points)),
                "output_count": int(len(points)),
                "removed_count": 0,
                "candidate_kept_count": kept_count,
                "candidate_keep_ratio": float(kept_count / max(len(points), 1)),
                "min_required_count": int(min_required),
                "keep_ratio": 1.0,
                "iqr_scale": float(iqr_scale),
                "threshold_normalized_radius": float(threshold),
            }

        filtered = np.asarray(points[keep], dtype=np.float64)
        return filtered, {
            "status": "ok",
            "input_count": int(len(slice_points_local)),
            "finite_count": int(len(points)),
            "output_count": int(len(filtered)),
            "removed_count": int(len(points) - len(filtered)),
            "keep_ratio": float(len(filtered) / max(len(points), 1)),
            "iqr_scale": float(iqr_scale),
            "center_u_m": float(center[0]),
            "center_v_m": float(center[1]),
            "iqr_u_m": float(iqr[0]),
            "iqr_v_m": float(iqr[1]),
            "threshold_normalized_radius": float(threshold),
        }

    @staticmethod
    def hull_path_between(hull_uv: np.ndarray, start_idx: int, end_idx: int) -> tuple[np.ndarray, np.ndarray]:
        """提取凸包两点间的双向路径。"""
        n = len(hull_uv)
        path_a = [hull_uv[start_idx]]
        idx = int(start_idx)
        while idx != int(end_idx):
            idx = (idx + 1) % n
            path_a.append(hull_uv[idx])

        path_b = [hull_uv[start_idx]]
        idx = int(start_idx)
        while idx != int(end_idx):
            idx = (idx - 1 + n) % n
            path_b.append(hull_uv[idx])

        return np.asarray(path_a, dtype=np.float64), np.asarray(path_b, dtype=np.float64)

    @staticmethod
    def extract_visible_arc_uv(slice_points_local: np.ndarray) -> tuple[np.ndarray, dict[str, int | float | str]]:
        """从局部切片中提取可见弧线。"""
        points = np.asarray(slice_points_local, dtype=np.float64)
        points = points[np.isfinite(points).all(axis=1)]
        uv = points[:, :2]
        if len(uv) < 2:
            return np.empty((0, 2), dtype=np.float64), {
                "status": "skip_small_input",
                "method": "none",
                "point_count": int(len(uv)),
                "arc_point_count": 0,
            }

        uv_unique = np.unique(np.round(uv, 6), axis=0).astype(np.float64)
        if len(uv_unique) < 3:
            order = np.argsort(uv_unique[:, 1])
            arc = np.asarray(uv_unique[order], dtype=np.float64)
            return arc, {
                "status": "fallback_few_unique_points",
                "method": "sort_by_v",
                "point_count": int(len(uv)),
                "unique_point_count": int(len(uv_unique)),
                "arc_point_count": int(len(arc)),
            }

        hull = cv2.convexHull(uv_unique.astype(np.float32)).reshape(-1, 2).astype(np.float64)
        if len(hull) < 3:
            order = np.argsort(uv_unique[:, 1])
            arc = np.asarray(uv_unique[order], dtype=np.float64)
            return arc, {
                "status": "fallback_small_hull",
                "method": "sort_by_v",
                "point_count": int(len(uv)),
                "unique_point_count": int(len(uv_unique)),
                "hull_point_count": int(len(hull)),
                "arc_point_count": int(len(arc)),
            }

        low_v_idx = int(np.argmin(hull[:, 1]))
        high_v_idx = int(np.argmax(hull[:, 1]))
        if low_v_idx == high_v_idx:
            low_v_idx = int(np.argmin(hull[:, 0]))
            high_v_idx = int(np.argmax(hull[:, 0]))

        path_a, path_b = MeasurementBase.hull_path_between(hull, low_v_idx, high_v_idx)
        mean_u_a = float(path_a[:, 0].mean())
        mean_u_b = float(path_b[:, 0].mean())
        arc = path_a if mean_u_a <= mean_u_b else path_b
        selected_path = "path_a" if mean_u_a <= mean_u_b else "path_b"

        if len(arc) >= 2 and arc[0, 1] > arc[-1, 1]:
            arc = arc[::-1]

        return np.asarray(arc, dtype=np.float64), {
            "status": "ok",
            "method": "convex_hull_visible_path",
            "visible_side_rule": "lower_mean_u_is_closer_to_camera",
            "selected_path": selected_path,
            "point_count": int(len(uv)),
            "unique_point_count": int(len(uv_unique)),
            "hull_point_count": int(len(hull)),
            "arc_point_count": int(len(arc)),
            "mean_u_path_a_m": mean_u_a,
            "mean_u_path_b_m": mean_u_b,
        }

    @staticmethod
    def choose_largest_detection_index(boxes_xyxy: np.ndarray) -> int:
        """选择面积最大的检测框索引。"""
        if len(boxes_xyxy) == 0:
            raise RuntimeError("No person detection was found.")
        wh = np.maximum(boxes_xyxy[:, 2:4] - boxes_xyxy[:, 0:2], 0.0)
        areas = wh[:, 0] * wh[:, 1]
        return int(np.argmax(areas))

    @staticmethod
    def detect_pose_joints(image_bgr: np.ndarray, pose_model: YOLO) -> tuple[list[dict[str, object]], dict[str, object]]:
        """执行姿态检测并返回关节与元信息。"""
        results = pose_model.predict(image_bgr, verbose=False)
        if not results:
            raise RuntimeError("YOLO Pose did not return any result.")

        result = results[0]
        if result.boxes is None or len(result.boxes) == 0 or result.keypoints is None:
            raise RuntimeError("YOLO Pose did not detect a person with keypoints.")

        boxes = result.boxes.xyxy.detach().cpu().numpy()
        person_idx = MeasurementBase.choose_largest_detection_index(boxes)
        keypoints_xy = result.keypoints.xy.detach().cpu().numpy()[person_idx]
        if result.keypoints.conf is not None:
            keypoints_conf = result.keypoints.conf.detach().cpu().numpy()[person_idx]
        else:
            keypoints_conf = np.ones(len(COCO_KEYPOINT_NAMES), dtype=np.float32)

        joints: list[dict[str, object]] = []
        for idx, name in enumerate(COCO_KEYPOINT_NAMES):
            conf = float(keypoints_conf[idx]) if idx < len(keypoints_conf) else 1.0
            joints.append(
                {
                    "name": name,
                    "x": float(keypoints_xy[idx, 0]),
                    "y": float(keypoints_xy[idx, 1]),
                    "confidence": conf,
                    "visible": bool(conf >= 0.20),
                }
            )

        detection_meta = {
            "selected_person_index": int(person_idx),
            "selected_box_xyxy": np.round(boxes[person_idx], 3).tolist(),
            "person_count": int(len(boxes)),
        }
        return joints, detection_meta

    @staticmethod
    def joints_list_to_map(joints: list[dict[str, object]]) -> dict[str, dict[str, object]]:
        """将关节列表转为按名称索引的字典。"""
        return {str(item["name"]): item for item in joints}

    @staticmethod
    def pick_joint(
        joint_map: dict[str, dict[str, object]],
        joint_name: str,
        *,
        min_confidence: float,
    ) -> dict[str, object]:
        """按名称选取关节并校验置信度。"""
        joint = joint_map.get(joint_name)
        if joint is None:
            raise RuntimeError(f"Joint not found: {joint_name}")
        confidence = float(joint.get("confidence", 0.0))
        if confidence < min_confidence:
            raise LowConfidenceError(joint_name, confidence, min_confidence)
        return joint

    @staticmethod
    def joint_to_uv(joint: dict[str, object]) -> np.ndarray:
        """将关节字典转换为 uv 坐标。"""
        return np.array([float(joint["x"]), float(joint["y"])], dtype=np.float64)



