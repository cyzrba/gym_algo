from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial import cKDTree
from ultralytics import YOLO

from arm_pointcloud import (
    DEFAULT_POSE_MODEL_PATH,
    compute_polyline_length,
    detect_pose_joints,
    joint_to_uv,
    joints_list_to_map,
    load_depth_map,
    load_rgb_image,
    pick_joint,
    sample_joint_xyz_from_point_cloud,
)
from pointcloud import load_point_cloud


DEFAULT_OUTPUT_DIR = Path("result") / "waist_pointcloud"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Measure waist visible arc lengths from front/back RGB+Depth+PLY inputs. "
            "The waist center is estimated from shoulder and hip midpoints."
        )
    )
    parser.add_argument("--rgb", type=Path, default=None, help="Single-view RGB image path, kept for compatibility.")
    parser.add_argument("--depth", type=Path, default=None, help="Single-view depth image path, kept for compatibility.")
    parser.add_argument("--ply", type=Path, default=None, help="Single-view point cloud .ply path, kept for compatibility.")
    parser.add_argument("--front-rgb", type=Path, default=None, help="Front-view RGB image path.")
    parser.add_argument("--front-depth", type=Path, default=None, help="Front-view depth image path.")
    parser.add_argument("--front-ply", type=Path, default=None, help="Front-view point cloud .ply path.")
    parser.add_argument("--back-rgb", type=Path, default=None, help="Back-view RGB image path.")
    parser.add_argument("--back-depth", type=Path, default=None, help="Back-view depth image path.")
    parser.add_argument("--back-ply", type=Path, default=None, help="Back-view point cloud .ply path.")
    parser.add_argument("--pose-model", type=Path, default=DEFAULT_POSE_MODEL_PATH, help="YOLO pose model path.")
    parser.add_argument("--depth-scale", type=float, default=0.001, help="Depth unit scale to meters.")
    parser.add_argument("--depth-window", type=int, default=5, help="Odd median window for point-cloud sampling.")
    parser.add_argument(
        "--waist-up-ratio",
        type=float,
        default=0.20,
        help="Waist center ratio from hip midpoint upward toward shoulder midpoint.",
    )
    parser.add_argument(
        "--waist-half-height-m",
        type=float,
        default=0.04,
        help="Waist slab half height around the estimated waist center, in meters.",
    )
    parser.add_argument(
        "--waist-radius-m",
        type=float,
        default=0.35,
        help="Maximum X/Z-plane distance from the waist center kept in the slice, in meters.",
    )
    parser.add_argument(
        "--visible-keep-ratio",
        type=float,
        default=0.65,
        help="Nearest-depth ratio kept from each view before extracting the visible waist arc.",
    )
    parser.add_argument(
        "--disable-robust-waist-filter",
        action="store_true",
        help="Disable kNN outlier filtering on waist slice points.",
    )
    parser.add_argument("--waist-filter-k", type=int, default=14, help="k for kNN waist outlier filtering.")
    parser.add_argument(
        "--waist-filter-std-ratio",
        type=float,
        default=2.2,
        help="Outlier threshold = median(knn_dist) + std_ratio * std(knn_dist).",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Output directory.")
    parser.add_argument("--rgb-width", type=int, default=640, help="Required when RGB input is .raw.")
    parser.add_argument("--rgb-height", type=int, default=480, help="Required when RGB input is .raw.")
    parser.add_argument(
        "--rgb-format",
        choices=["rgb8", "bgr8", "gray8"],
        default="rgb8",
        help="Raw RGB pixel format.",
    )
    parser.add_argument("--depth-width", type=int, default=640, help="Required when depth input is .raw.")
    parser.add_argument("--depth-height", type=int, default=480, help="Required when depth input is .raw.")
    parser.add_argument(
        "--depth-dtype",
        choices=["uint16", "uint8", "float32"],
        default="uint16",
        help="Depth raw dtype.",
    )
    parser.add_argument(
        "--depth-endian",
        choices=["little", "big"],
        default="little",
        help="Depth raw endianness for multi-byte dtypes.",
    )
    return parser.parse_args()


def resolve_input_views(args: argparse.Namespace) -> list[tuple[str, Path, Path, Path]]:
    has_front = any(value is not None for value in (args.front_rgb, args.front_depth, args.front_ply))
    has_back = any(value is not None for value in (args.back_rgb, args.back_depth, args.back_ply))
    has_single = any(value is not None for value in (args.rgb, args.depth, args.ply))

    if has_front or has_back:
        if not (args.front_rgb and args.front_depth and args.front_ply):
            raise RuntimeError("--front-rgb, --front-depth, and --front-ply are required together.")
        if not (args.back_rgb and args.back_depth and args.back_ply):
            raise RuntimeError("--back-rgb, --back-depth, and --back-ply are required together.")
        return [
            ("front", args.front_rgb, args.front_depth, args.front_ply),
            ("back", args.back_rgb, args.back_depth, args.back_ply),
        ]

    if has_single:
        if not (args.rgb and args.depth and args.ply):
            raise RuntimeError("--rgb, --depth, and --ply are required together for single-view mode.")
        return [("single", args.rgb, args.depth, args.ply)]

    raise RuntimeError(
        "Provide either front/back inputs "
        "(--front-rgb/--front-depth/--front-ply and --back-rgb/--back-depth/--back-ply) "
        "or single-view inputs (--rgb/--depth/--ply)."
    )


def validate_input_view_paths(input_views: list[tuple[str, Path, Path, Path]]) -> None:
    for view_name, rgb_path, depth_path, ply_path in input_views:
        if not rgb_path.exists():
            raise FileNotFoundError(f"[{view_name}] RGB input not found: {rgb_path}")
        if not depth_path.exists():
            raise FileNotFoundError(f"[{view_name}] Depth input not found: {depth_path}")
        if not ply_path.exists():
            raise FileNotFoundError(f"[{view_name}] Point cloud input not found: {ply_path}")


def compute_waist_center_uv(
    *,
    left_shoulder_uv: np.ndarray,
    right_shoulder_uv: np.ndarray,
    left_hip_uv: np.ndarray,
    right_hip_uv: np.ndarray,
    waist_up_ratio: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    shoulder_mid = (left_shoulder_uv + right_shoulder_uv) * 0.5
    hip_mid = (left_hip_uv + right_hip_uv) * 0.5
    waist_center = hip_mid + (shoulder_mid - hip_mid) * float(waist_up_ratio)
    return waist_center.astype(np.float64), shoulder_mid.astype(np.float64), hip_mid.astype(np.float64)


def filter_waist_slice_points(
    points_xyz: np.ndarray,
    *,
    waist_center_xyz: np.ndarray,
    half_height_m: float,
    radius_m: float,
    visible_keep_ratio: float,
) -> tuple[np.ndarray, dict[str, int | float]]:
    if not (0 < visible_keep_ratio <= 1):
        raise RuntimeError("--visible-keep-ratio must be in (0, 1].")
    points = np.asarray(points_xyz, dtype=np.float64)
    valid = np.isfinite(points).all(axis=1) & (points[:, 2] > 0)
    valid_points = points[valid]
    if len(valid_points) == 0:
        raise RuntimeError("Point cloud has no valid positive-Z points.")

    y_dist = np.abs(valid_points[:, 1] - float(waist_center_xyz[1]))
    xz_dist = np.linalg.norm(valid_points[:, [0, 2]] - waist_center_xyz[[0, 2]], axis=1)
    slice_keep = (y_dist <= float(half_height_m)) & (xz_dist <= float(radius_m))
    slice_points = np.asarray(valid_points[slice_keep], dtype=np.float64)
    if len(slice_points) < 30:
        raise RuntimeError("Too few waist points after slice filtering. Increase --waist-half-height-m or --waist-radius-m.")

    z_threshold = float(np.quantile(slice_points[:, 2], float(visible_keep_ratio)))
    visible_points = np.asarray(slice_points[slice_points[:, 2] <= z_threshold], dtype=np.float64)
    if len(visible_points) < 10:
        raise RuntimeError("Too few visible waist points after depth filtering. Increase --visible-keep-ratio.")

    return visible_points, {
        "input_count": int(len(points_xyz)),
        "finite_positive_count": int(len(valid_points)),
        "slice_count": int(len(slice_points)),
        "output_count": int(len(visible_points)),
        "waist_half_height_m": float(half_height_m),
        "waist_radius_m": float(radius_m),
        "visible_keep_ratio": float(visible_keep_ratio),
        "visible_depth_threshold_m": float(z_threshold),
    }


def robust_filter_waist_points(
    points_xyz: np.ndarray,
    *,
    k: int,
    std_ratio: float,
) -> tuple[np.ndarray, dict[str, int | float | str]]:
    points = np.asarray(points_xyz, dtype=np.float64)
    n = len(points)
    if n < 60:
        return points, {
            "status": "skip_small_input",
            "input_count": int(n),
            "output_count": int(n),
            "removed_count": 0,
        }

    k_eff = int(max(4, min(k, n - 1)))
    xz = points[:, [0, 2]]
    tree = cKDTree(xz)
    dists, _ = tree.query(xz, k=k_eff + 1)
    knn_mean = dists[:, 1:].mean(axis=1)
    median_dist = float(np.median(knn_mean))
    std_dist = float(np.std(knn_mean))
    threshold = median_dist + float(std_ratio) * std_dist
    keep = knn_mean <= threshold
    filtered = np.asarray(points[keep], dtype=np.float64)

    min_allowed = max(50, int(n * 0.45))
    if len(filtered) < min_allowed:
        return points, {
            "status": "reverted_low_support",
            "input_count": int(n),
            "output_count": int(n),
            "removed_count": 0,
            "candidate_output_count": int(len(filtered)),
            "k": int(k_eff),
            "median_knn_dist": median_dist,
            "std_knn_dist": std_dist,
            "threshold": float(threshold),
        }

    return filtered, {
        "status": "ok",
        "input_count": int(n),
        "output_count": int(len(filtered)),
        "removed_count": int(n - len(filtered)),
        "k": int(k_eff),
        "median_knn_dist": median_dist,
        "std_knn_dist": std_dist,
        "threshold": float(threshold),
    }


def hull_path_between(hull: np.ndarray, start_idx: int, end_idx: int) -> tuple[np.ndarray, np.ndarray]:
    n = len(hull)
    path_a = [hull[start_idx]]
    idx = int(start_idx)
    while idx != int(end_idx):
        idx = (idx + 1) % n
        path_a.append(hull[idx])

    path_b = [hull[start_idx]]
    idx = int(start_idx)
    while idx != int(end_idx):
        idx = (idx - 1 + n) % n
        path_b.append(hull[idx])

    return np.asarray(path_a, dtype=np.float64), np.asarray(path_b, dtype=np.float64)


def extract_visible_waist_arc_xz(points_xyz: np.ndarray) -> tuple[np.ndarray, dict[str, int | float | str]]:
    xz = np.asarray(points_xyz[:, [0, 2]], dtype=np.float64)
    xz = xz[np.isfinite(xz).all(axis=1)]
    if len(xz) < 3:
        return np.empty((0, 2), dtype=np.float64), {
            "status": "skip_small_input",
            "method": "none",
            "point_count": int(len(xz)),
            "arc_point_count": 0,
        }

    xz_unique = np.unique(np.round(xz, 6), axis=0).astype(np.float64)
    if len(xz_unique) < 3:
        order = np.argsort(xz_unique[:, 0])
        arc = np.asarray(xz_unique[order], dtype=np.float64)
        return arc, {
            "status": "fallback_few_unique_points",
            "method": "sort_by_x",
            "point_count": int(len(xz)),
            "unique_point_count": int(len(xz_unique)),
            "arc_point_count": int(len(arc)),
        }

    hull = cv2.convexHull(xz_unique.astype(np.float32)).reshape(-1, 2).astype(np.float64)
    left_idx = int(np.argmin(hull[:, 0]))
    right_idx = int(np.argmax(hull[:, 0]))
    if left_idx == right_idx:
        raise RuntimeError("Failed to find waist left/right hull endpoints.")

    path_a, path_b = hull_path_between(hull, left_idx, right_idx)
    mean_z_a = float(path_a[:, 1].mean())
    mean_z_b = float(path_b[:, 1].mean())
    arc = path_a if mean_z_a <= mean_z_b else path_b
    selected_path = "path_a" if mean_z_a <= mean_z_b else "path_b"
    if len(arc) >= 2 and arc[0, 0] > arc[-1, 0]:
        arc = arc[::-1]

    return np.asarray(arc, dtype=np.float64), {
        "status": "ok",
        "method": "convex_hull_visible_path",
        "visible_side_rule": "lower_mean_z_is_closer_to_camera",
        "selected_path": selected_path,
        "point_count": int(len(xz)),
        "unique_point_count": int(len(xz_unique)),
        "hull_point_count": int(len(hull)),
        "arc_point_count": int(len(arc)),
        "mean_z_path_a_m": mean_z_a,
        "mean_z_path_b_m": mean_z_b,
    }


def map_points_to_canvas(
    points_xz: np.ndarray,
    *,
    x_min: float,
    x_max: float,
    z_min: float,
    z_max: float,
    width: int,
    height: int,
    pad: int,
) -> np.ndarray:
    x_span = max(float(x_max - x_min), 1e-9)
    z_span = max(float(z_max - z_min), 1e-9)
    scale = min((width - 2 * pad) / x_span, (height - 2 * pad) / z_span)
    u = pad + (points_xz[:, 0] - x_min) * scale
    v = height - pad - (points_xz[:, 1] - z_min) * scale
    return np.round(np.column_stack([u, v])).astype(np.int32)


def save_waist_center_overlay(
    image_bgr: np.ndarray,
    *,
    left_shoulder_uv: np.ndarray,
    right_shoulder_uv: np.ndarray,
    left_hip_uv: np.ndarray,
    right_hip_uv: np.ndarray,
    shoulder_mid_uv: np.ndarray,
    hip_mid_uv: np.ndarray,
    waist_center_uv: np.ndarray,
    output_path: Path,
) -> None:
    vis = image_bgr.copy()
    points = [
        (left_shoulder_uv, (34, 197, 94), "L shoulder"),
        (right_shoulder_uv, (34, 197, 94), "R shoulder"),
        (left_hip_uv, (59, 130, 246), "L hip"),
        (right_hip_uv, (59, 130, 246), "R hip"),
        (shoulder_mid_uv, (21, 128, 61), "M_shoulder"),
        (hip_mid_uv, (29, 78, 216), "M_hip"),
        (waist_center_uv, (0, 0, 255), "waist_center"),
    ]
    cv2.line(vis, tuple(np.round(shoulder_mid_uv).astype(int)), tuple(np.round(hip_mid_uv).astype(int)), (120, 120, 120), 2)
    for uv, color, label in points:
        xy = (int(round(float(uv[0]))), int(round(float(uv[1]))))
        cv2.circle(vis, xy, 7, color, -1, lineType=cv2.LINE_AA)
        cv2.putText(vis, label, (xy[0] + 8, xy[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1, cv2.LINE_AA)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), vis)


def save_waist_slice_xz_plot(
    waist_points_xyz: np.ndarray,
    filtered_points_xyz: np.ndarray,
    waist_center_xyz: np.ndarray,
    visible_arc_xz: np.ndarray,
    *,
    output_path: Path,
) -> None:
    width, height, pad = 1000, 800, 50
    canvas = np.full((height, width, 3), 255, dtype=np.uint8)
    raw_xz = np.asarray(waist_points_xyz[:, [0, 2]], dtype=np.float64) if len(waist_points_xyz) else np.empty((0, 2))
    filtered_xz = (
        np.asarray(filtered_points_xyz[:, [0, 2]], dtype=np.float64) if len(filtered_points_xyz) else np.empty((0, 2))
    )
    center_xz = np.asarray(waist_center_xyz[[0, 2]], dtype=np.float64).reshape(1, 2)
    stacks = [center_xz]
    if len(raw_xz):
        stacks.append(raw_xz)
    if len(filtered_xz):
        stacks.append(filtered_xz)
    if len(visible_arc_xz):
        stacks.append(visible_arc_xz)
    all_xz = np.vstack(stacks)
    x_min, z_min = all_xz.min(axis=0)
    x_max, z_max = all_xz.max(axis=0)

    if len(raw_xz):
        raw_uv = map_points_to_canvas(raw_xz, x_min=x_min, x_max=x_max, z_min=z_min, z_max=z_max, width=width, height=height, pad=pad)
        for point in raw_uv:
            cv2.circle(canvas, tuple(point), 1, (205, 205, 205), -1)
    if len(filtered_xz):
        filtered_uv = map_points_to_canvas(filtered_xz, x_min=x_min, x_max=x_max, z_min=z_min, z_max=z_max, width=width, height=height, pad=pad)
        for point in filtered_uv:
            cv2.circle(canvas, tuple(point), 1, (140, 140, 140), -1)
    if len(visible_arc_xz) >= 2:
        arc_uv = map_points_to_canvas(visible_arc_xz, x_min=x_min, x_max=x_max, z_min=z_min, z_max=z_max, width=width, height=height, pad=pad)
        cv2.polylines(canvas, [arc_uv.reshape(-1, 1, 2)], False, (0, 0, 255), 2, lineType=cv2.LINE_AA)

    center_uv = map_points_to_canvas(center_xz, x_min=x_min, x_max=x_max, z_min=z_min, z_max=z_max, width=width, height=height, pad=pad)[0]
    cv2.drawMarker(canvas, tuple(center_uv), (0, 128, 255), cv2.MARKER_CROSS, 22, 3, line_type=cv2.LINE_AA)
    cv2.circle(canvas, tuple(center_uv), 7, (0, 128, 255), 2, lineType=cv2.LINE_AA)
    cv2.putText(canvas, "waist_center", (int(center_uv[0]) + 10, int(center_uv[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 128, 255), 2)
    cv2.putText(canvas, "Waist X/Z slice: gray=points, red=visible arc, orange=center", (24, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (35, 35, 35), 2)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), canvas)


def measure_waist_view(
    *,
    args: argparse.Namespace,
    view_name: str,
    image_bgr: np.ndarray,
    points_xyz: np.ndarray,
    rgb_shape: tuple[int, int],
    cloud_shape: tuple[int, int],
    joint_map: dict[str, dict[str, object]],
    output_dir: Path,
) -> dict[str, object]:
    left_shoulder = pick_joint(joint_map, "left_shoulder", min_confidence=0.30)
    right_shoulder = pick_joint(joint_map, "right_shoulder", min_confidence=0.30)
    left_hip = pick_joint(joint_map, "left_hip", min_confidence=0.30)
    right_hip = pick_joint(joint_map, "right_hip", min_confidence=0.30)
    left_shoulder_uv = joint_to_uv(left_shoulder)
    right_shoulder_uv = joint_to_uv(right_shoulder)
    left_hip_uv = joint_to_uv(left_hip)
    right_hip_uv = joint_to_uv(right_hip)
    waist_center_uv, shoulder_mid_uv, hip_mid_uv = compute_waist_center_uv(
        left_shoulder_uv=left_shoulder_uv,
        right_shoulder_uv=right_shoulder_uv,
        left_hip_uv=left_hip_uv,
        right_hip_uv=right_hip_uv,
        waist_up_ratio=float(args.waist_up_ratio),
    )

    waist_center_xyz, waist_cloud_uv, waist_sampling_meta = sample_joint_xyz_from_point_cloud(
        points_xyz=points_xyz,
        joint_uv_rgb=waist_center_uv,
        rgb_shape=rgb_shape,
        cloud_shape=cloud_shape,
        window_size=int(args.depth_window),
    )
    waist_points_xyz, waist_slice_meta = filter_waist_slice_points(
        points_xyz,
        waist_center_xyz=waist_center_xyz,
        half_height_m=float(args.waist_half_height_m),
        radius_m=float(args.waist_radius_m),
        visible_keep_ratio=float(args.visible_keep_ratio),
    )
    if args.disable_robust_waist_filter:
        filtered_points_xyz = waist_points_xyz
        robust_filter_meta = {
            "status": "disabled",
            "input_count": int(len(waist_points_xyz)),
            "output_count": int(len(waist_points_xyz)),
            "removed_count": 0,
        }
    else:
        filtered_points_xyz, robust_filter_meta = robust_filter_waist_points(
            waist_points_xyz,
            k=int(args.waist_filter_k),
            std_ratio=float(args.waist_filter_std_ratio),
        )
    visible_arc_xz, visible_arc_meta = extract_visible_waist_arc_xz(filtered_points_xyz)
    visible_arc_length_m = compute_polyline_length(visible_arc_xz)

    prefix = f"{view_name}_"
    overlay_path = output_dir / f"{prefix}waist_center_overlay.png"
    waist_points_path = output_dir / f"{prefix}waist_points.xyz.npy"
    waist_filtered_points_path = output_dir / f"{prefix}waist_points.filtered.xyz.npy"
    visible_arc_path = output_dir / f"{prefix}waist_visible_arc.xz.npy"
    waist_slice_plot_path = output_dir / f"{prefix}waist_slice_xz.png"
    save_waist_center_overlay(
        image_bgr,
        left_shoulder_uv=left_shoulder_uv,
        right_shoulder_uv=right_shoulder_uv,
        left_hip_uv=left_hip_uv,
        right_hip_uv=right_hip_uv,
        shoulder_mid_uv=shoulder_mid_uv,
        hip_mid_uv=hip_mid_uv,
        waist_center_uv=waist_center_uv,
        output_path=overlay_path,
    )
    np.save(waist_points_path, waist_points_xyz.astype(np.float32))
    np.save(waist_filtered_points_path, filtered_points_xyz.astype(np.float32))
    np.save(visible_arc_path, visible_arc_xz.astype(np.float32))
    save_waist_slice_xz_plot(
        waist_points_xyz,
        filtered_points_xyz,
        waist_center_xyz,
        visible_arc_xz,
        output_path=waist_slice_plot_path,
    )

    return {
        "view_name": view_name,
        "center_method": {
            "description": "waist_center = M_hip + (M_shoulder - M_hip) * waist_up_ratio",
            "waist_up_ratio": float(args.waist_up_ratio),
            "left_shoulder_uv": np.round(left_shoulder_uv, 3).tolist(),
            "right_shoulder_uv": np.round(right_shoulder_uv, 3).tolist(),
            "left_hip_uv": np.round(left_hip_uv, 3).tolist(),
            "right_hip_uv": np.round(right_hip_uv, 3).tolist(),
            "shoulder_mid_uv": np.round(shoulder_mid_uv, 3).tolist(),
            "hip_mid_uv": np.round(hip_mid_uv, 3).tolist(),
            "waist_center_uv": np.round(waist_center_uv, 3).tolist(),
            "waist_center_uv_on_point_cloud": [int(waist_cloud_uv[0]), int(waist_cloud_uv[1])],
            "waist_center_xyz_meter": np.round(waist_center_xyz, 6).tolist(),
            "waist_center_sampling": waist_sampling_meta,
        },
        "waist_slice": {
            "filter_meta": waist_slice_meta,
            "robust_filter_meta": robust_filter_meta,
            "visible_arc_meta": visible_arc_meta,
            "visible_arc_length_m": float(round(visible_arc_length_m, 6)),
        },
        "artifacts": {
            "waist_center_overlay_png": str(overlay_path.resolve()),
            "waist_points_npy": str(waist_points_path.resolve()),
            "waist_filtered_points_npy": str(waist_filtered_points_path.resolve()),
            "waist_visible_arc_xz_npy": str(visible_arc_path.resolve()),
            "waist_slice_xz_png": str(waist_slice_plot_path.resolve()),
        },
    }


def process_waist_view(
    *,
    args: argparse.Namespace,
    view_name: str,
    rgb_path: Path,
    depth_path: Path,
    ply_path: Path,
    pose_model: YOLO,
    output_dir: Path,
) -> dict[str, object]:
    image_bgr = load_rgb_image(
        rgb_path,
        width=int(args.rgb_width),
        height=int(args.rgb_height),
        rgb_format=str(args.rgb_format),
    )
    if image_bgr is None:
        raise RuntimeError(f"[{view_name}] Failed to read RGB image: {rgb_path}")
    depth_map_raw = load_depth_map(
        depth_path,
        width=int(args.depth_width),
        height=int(args.depth_height),
        depth_dtype=str(args.depth_dtype),
        endian=str(args.depth_endian),
    )
    rgb_h, rgb_w = image_bgr.shape[:2]
    depth_h, depth_w = depth_map_raw.shape[:2]
    points_xyz, _colors = load_point_cloud(ply_path)

    joints, detection_meta = detect_pose_joints(image_bgr, pose_model)
    joint_map = joints_list_to_map(joints)
    waist_result = measure_waist_view(
        args=args,
        view_name=view_name,
        image_bgr=image_bgr,
        points_xyz=points_xyz,
        rgb_shape=(rgb_h, rgb_w),
        cloud_shape=(depth_h, depth_w),
        joint_map=joint_map,
        output_dir=output_dir,
    )
    return {
        "view_name": view_name,
        "inputs": {
            "rgb": str(rgb_path.resolve()),
            "depth": str(depth_path.resolve()),
            "point_cloud_ply": str(ply_path.resolve()),
        },
        "detection": detection_meta,
        "waist": waist_result,
        "joints": joints,
    }


def main() -> None:
    args = parse_args()
    input_views = resolve_input_views(args)
    validate_input_view_paths(input_views)
    if not (0 < args.waist_up_ratio < 1):
        raise RuntimeError("--waist-up-ratio must be in (0, 1).")
    if args.waist_half_height_m <= 0:
        raise RuntimeError("--waist-half-height-m must be > 0.")
    if args.waist_radius_m <= 0:
        raise RuntimeError("--waist-radius-m must be > 0.")
    if not (0 < args.visible_keep_ratio <= 1):
        raise RuntimeError("--visible-keep-ratio must be in (0, 1].")
    if args.waist_filter_k < 1:
        raise RuntimeError("--waist-filter-k must be >= 1.")
    if args.waist_filter_std_ratio <= 0:
        raise RuntimeError("--waist-filter-std-ratio must be > 0.")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    pose_model = YOLO(str(args.pose_model))
    view_results = {
        view_name: process_waist_view(
            args=args,
            view_name=view_name,
            rgb_path=rgb_path,
            depth_path=depth_path,
            ply_path=ply_path,
            pose_model=pose_model,
            output_dir=args.output_dir,
        )
        for view_name, rgb_path, depth_path, ply_path in input_views
    }
    waist_total_length_m = float(
        sum(float(view_result["waist"]["waist_slice"]["visible_arc_length_m"]) for view_result in view_results.values())
    )
    measurement_summary = {
        f"{view_name}_waist_visible_arc_length_m": float(
            round(float(view_result["waist"]["waist_slice"]["visible_arc_length_m"]), 6)
        )
        for view_name, view_result in view_results.items()
    }
    measurement_summary["waist_total_visible_arc_length_m"] = float(round(waist_total_length_m, 6))

    summary_path = args.output_dir / "pose_joints.json"
    output = {
        "inputs": {
            view_name: {
                "rgb": str(rgb_path.resolve()),
                "depth": str(depth_path.resolve()),
                "point_cloud_ply": str(ply_path.resolve()),
            }
            for view_name, rgb_path, depth_path, ply_path in input_views
        },
        "run": {
            "pose_model": str(args.pose_model),
            "input_mode": "front_back" if len(input_views) == 2 else "single_view",
        },
        "measurement_params": {
            "waist_up_ratio": float(args.waist_up_ratio),
            "waist_half_height_m": float(args.waist_half_height_m),
            "waist_radius_m": float(args.waist_radius_m),
            "visible_keep_ratio": float(args.visible_keep_ratio),
            "disable_robust_waist_filter": bool(args.disable_robust_waist_filter),
            "waist_filter_k": int(args.waist_filter_k),
            "waist_filter_std_ratio": float(args.waist_filter_std_ratio),
            "depth_window": int(args.depth_window),
            "depth_scale_to_meter": float(args.depth_scale),
        },
        "views": view_results,
        "waist_total_lengths": {
            "waist_total_visible_arc_length_m": float(round(waist_total_length_m, 6)),
        },
        "measurement_summary": measurement_summary,
        "artifacts": {
            "joints_json": str(summary_path.resolve()),
        },
    }
    summary_path.write_text(json.dumps(output, ensure_ascii=True, indent=2), encoding="utf-8")

    for view_name, view_result in view_results.items():
        detection_meta = view_result["detection"]
        waist = view_result["waist"]
        center = waist["center_method"]
        waist_slice = waist["waist_slice"]
        print(f"[{view_name}] Selected person index: {detection_meta['selected_person_index']}")
        print(f"[{view_name}] Waist center uv: {center['waist_center_uv']}")
        print(f"[{view_name}] Waist center xyz (m): {center['waist_center_xyz_meter']}")
        print(f"[{view_name}] Waist slice points: {int(waist_slice['filter_meta']['output_count'])}")
        print(f"[{view_name}] Filtered waist points: {int(waist_slice['robust_filter_meta']['output_count'])}")
        print(f"[{view_name}] Visible waist arc length (m): {float(waist_slice['visible_arc_length_m']):.6f}")
    print(f"waist_total_visible_arc_length_m: {waist_total_length_m:.6f}")
    print(f"Saved joints: {summary_path.resolve()}")


if __name__ == "__main__":
    main()
