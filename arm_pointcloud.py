from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib")

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from ultralytics import YOLO

from pointcloud import load_point_cloud


COCO_KEYPOINT_NAMES = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]

DEFAULT_POSE_MODEL_PATH = Path("yolo26n-pose.pt")
DEFAULT_OUTPUT_DIR = Path("result") / "arm_pointcloud_depth_gate"


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for pose detection, PLY joint sampling, and depth-gated point display."""
    parser = argparse.ArgumentParser(
        description=(
            "Use YOLO Pose to locate shoulder/elbow keypoints, sample their XYZ directly "
            "from the organized PLY, then measure left/right arm visible arc lengths."
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
    parser.add_argument("--arm-side", choices=["left", "right", "both"], default="both", help="Target arm side.")
    parser.add_argument("--fx", type=float, default=None, help="Optional camera intrinsics fx for metadata.")
    parser.add_argument("--fy", type=float, default=None, help="Optional camera intrinsics fy for metadata.")
    parser.add_argument("--cx", type=float, default=None, help="Optional camera intrinsics cx for metadata.")
    parser.add_argument("--cy", type=float, default=None, help="Optional camera intrinsics cy for metadata.")
    parser.add_argument("--depth-scale", type=float, default=0.001, help="Depth unit scale to meters.")
    parser.add_argument("--depth-window", type=int, default=5, help="Odd median window for joint depth sampling.")
    parser.add_argument(
        "--depth-margin-m",
        type=float,
        default=0.08,
        help="Depth margin behind the farther shoulder/elbow joint, in meters.",
    )
    parser.add_argument(
        "--arm-center-ratio",
        type=float,
        default=1.0 / 3.0,
        help="Center ratio along shoulder->elbow used for the perpendicular arm slice.",
    )
    parser.add_argument(
        "--slice-half-thickness-m",
        type=float,
        default=0.01,
        help="Half thickness of the perpendicular arm slice along the shoulder->elbow axis, in meters.",
    )
    parser.add_argument(
        "--slice-radius-m",
        type=float,
        default=0.18,
        help="Maximum distance from the shoulder->elbow axis kept in the perpendicular slice, in meters.",
    )
    parser.add_argument(
        "--deviation-iqr-scale",
        type=float,
        default=1.8,
        help="Robust IQR multiplier used to filter deviating points in the arm slice U/V plane.",
    )
    parser.add_argument(
        "--deviation-min-keep-ratio",
        type=float,
        default=0.35,
        help="Minimum point ratio required after deviation filtering before falling back to the original slice.",
    )
    parser.add_argument(
        "--deviation-min-points",
        type=int,
        default=20,
        help="Minimum point count required after deviation filtering before falling back to the original slice.",
    )
    parser.add_argument(
        "--max-display-points",
        type=int,
        default=120000,
        help="Maximum points to draw in the filtered point-cloud preview. 0 means no limit.",
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


def load_rgb_image(path: Path, *, width: int, height: int, rgb_format: str) -> np.ndarray | None:
    """Load an RGB image or raw frame and return it in OpenCV BGR format."""
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


def load_depth_map(
    path: Path,
    *,
    width: int,
    height: int,
    depth_dtype: str,
    endian: str,
) -> np.ndarray:
    """Load a depth frame from npy/raw/image formats and return the raw depth array."""
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


def prepare_depth_uv(rgb_uv: np.ndarray, rgb_shape: tuple[int, int], depth_shape: tuple[int, int]) -> np.ndarray:
    """Map a keypoint coordinate from RGB image space into depth image space."""
    rgb_h, rgb_w = rgb_shape
    depth_h, depth_w = depth_shape
    if (rgb_h, rgb_w) == (depth_h, depth_w):
        return rgb_uv.astype(np.float64)
    return np.array(
        [float(rgb_uv[0]) * depth_w / rgb_w, float(rgb_uv[1]) * depth_h / rgb_h],
        dtype=np.float64,
    )


def sample_depth_median(
    depth_map_raw: np.ndarray,
    uv: np.ndarray,
    window_size: int,
) -> tuple[float, tuple[int, int]]:
    """Sample a robust median depth around a keypoint pixel."""
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


def sample_joint_xyz_from_point_cloud(
    *,
    points_xyz: np.ndarray,
    joint_uv_rgb: np.ndarray,
    rgb_shape: tuple[int, int],
    cloud_shape: tuple[int, int],
    window_size: int,
) -> tuple[np.ndarray, tuple[int, int], dict[str, int | str]]:
    """Sample one joint's XYZ directly from the organized PLY point cloud."""
    if window_size < 1 or window_size % 2 == 0:
        raise RuntimeError("--depth-window must be a positive odd integer.")

    cloud_h, cloud_w = cloud_shape
    expected_count = cloud_h * cloud_w
    if len(points_xyz) != expected_count:
        raise RuntimeError(
            f"Point cloud must be organized as {cloud_w}x{cloud_h}; "
            f"got {len(points_xyz)} points, expected {expected_count}."
        )

    joint_uv_cloud = prepare_depth_uv(joint_uv_rgb, rgb_shape, cloud_shape)
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


def filter_point_cloud_by_max_depth(
    points_xyz: np.ndarray,
    *,
    depth_high_m: float,
) -> tuple[np.ndarray, dict[str, int | float]]:
    """Keep valid point-cloud points up to the background depth cutoff."""
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


def build_arm_local_frame(
    *,
    shoulder_xyz: np.ndarray,
    elbow_xyz: np.ndarray,
    center_ratio: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """Build a local arm frame whose W axis follows shoulder->elbow."""
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


def map_points_to_arm_frame(
    points_xyz: np.ndarray,
    *,
    origin_xyz: np.ndarray,
    u_axis: np.ndarray,
    v_axis: np.ndarray,
    w_axis: np.ndarray,
) -> np.ndarray:
    """Map world/camera-space XYZ points into the local arm frame."""
    if len(points_xyz) == 0:
        return np.empty((0, 3), dtype=np.float64)
    rel = np.asarray(points_xyz, dtype=np.float64) - np.asarray(origin_xyz, dtype=np.float64)
    return np.column_stack([rel @ u_axis, rel @ v_axis, rel @ w_axis]).astype(np.float64)


def extract_perpendicular_arm_slice(
    points_xyz: np.ndarray,
    *,
    origin_xyz: np.ndarray,
    u_axis: np.ndarray,
    v_axis: np.ndarray,
    w_axis: np.ndarray,
    slice_half_thickness_m: float,
    slice_radius_m: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, int | float]]:
    """Cut a thin slice perpendicular to the arm axis and return world/local coordinates."""
    points = np.asarray(points_xyz, dtype=np.float64)
    valid = np.isfinite(points).all(axis=1) & (points[:, 2] > 0)
    valid_points = points[valid]
    local = map_points_to_arm_frame(
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


def compute_polyline_length(polyline_xy: np.ndarray) -> float:
    """Compute the length of an open 2D polyline."""
    points = np.asarray(polyline_xy, dtype=np.float64)
    if len(points) < 2:
        return 0.0
    diffs = np.diff(points, axis=0)
    return float(np.linalg.norm(diffs, axis=1).sum())


def filter_deviating_arm_slice_points(
    slice_points_local: np.ndarray,
    *,
    iqr_scale: float,
    min_keep_ratio: float,
    min_keep_points: int,
) -> tuple[np.ndarray, dict[str, int | float | str]]:
    """Remove points that deviate strongly from the main U/V slice cluster."""
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


def hull_path_between(hull_uv: np.ndarray, start_idx: int, end_idx: int) -> tuple[np.ndarray, np.ndarray]:
    """Return both clockwise paths between two hull vertices."""
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


def extract_visible_arm_arc_uv(slice_points_local: np.ndarray) -> tuple[np.ndarray, dict[str, int | float | str]]:
    """Extract the visible arm-slice arc from local U/V points with a convex hull path."""
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

    path_a, path_b = hull_path_between(hull, low_v_idx, high_v_idx)
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


def downsample_for_display(points_xyz: np.ndarray, max_points: int) -> np.ndarray:
    """Deterministically downsample points so the debug preview remains readable and lightweight."""
    if max_points <= 0 or len(points_xyz) <= max_points:
        return points_xyz
    rng = np.random.default_rng(42)
    indices = rng.choice(len(points_xyz), size=int(max_points), replace=False)
    return np.asarray(points_xyz[indices], dtype=np.float64)


def set_axes_equal_from_points(ax: plt.Axes, points: np.ndarray, *, min_radius: float = 0.02) -> None:
    """Set equal visual scale for a 3D axes from an Nx3 point set."""
    if len(points) == 0:
        radius = float(min_radius)
        ax.set_xlim(-radius, radius)
        ax.set_ylim(-radius, radius)
        ax.set_zlim(-radius, radius)
        return

    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    center = (mins + maxs) * 0.5
    radius = float(max(np.max(maxs - mins) * 0.55, min_radius))
    ax.set_xlim(float(center[0] - radius), float(center[0] + radius))
    ax.set_ylim(float(center[1] - radius), float(center[1] + radius))
    ax.set_zlim(float(center[2] - radius), float(center[2] + radius))


def save_depth_filtered_point_cloud_plot(
    points_xyz: np.ndarray,
    *,
    shoulder_xyz: np.ndarray,
    elbow_xyz: np.ndarray,
    output_path: Path,
    max_display_points: int,
) -> None:
    """Save a 3D scatter preview of the depth-gated point cloud plus shoulder/elbow markers."""
    display_points = downsample_for_display(points_xyz, max_display_points)
    fig = plt.figure(figsize=(9, 7), dpi=150)
    ax = fig.add_subplot(111, projection="3d")

    if len(display_points) > 0:
        ax.scatter(
            display_points[:, 0],
            display_points[:, 1],
            display_points[:, 2],
            s=0.4,
            c=display_points[:, 2],
            cmap="viridis",
            alpha=0.65,
            linewidths=0,
        )

    ax.scatter([shoulder_xyz[0]], [shoulder_xyz[1]], [shoulder_xyz[2]], s=45, c="#22c55e", label="shoulder")
    ax.scatter([elbow_xyz[0]], [elbow_xyz[1]], [elbow_xyz[2]], s=45, c="#e879f9", label="elbow")
    ax.plot(
        [shoulder_xyz[0], elbow_xyz[0]],
        [shoulder_xyz[1], elbow_xyz[1]],
        [shoulder_xyz[2], elbow_xyz[2]],
        c="#2563eb",
        linewidth=2.0,
        label="shoulder-elbow axis",
    )

    if len(display_points) > 0:
        mins = np.minimum(display_points.min(axis=0), np.minimum(shoulder_xyz, elbow_xyz))
        maxs = np.maximum(display_points.max(axis=0), np.maximum(shoulder_xyz, elbow_xyz))
    else:
        mins = np.minimum(shoulder_xyz, elbow_xyz)
        maxs = np.maximum(shoulder_xyz, elbow_xyz)
    center = (mins + maxs) * 0.5
    radius = float(max(np.max(maxs - mins) * 0.55, 0.05))
    ax.set_xlim(float(center[0] - radius), float(center[0] + radius))
    ax.set_ylim(float(center[1] - radius), float(center[1] + radius))
    ax.set_zlim(float(center[2] - radius), float(center[2] + radius))
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z depth (m)")
    ax.set_title(f"Depth-gated point cloud ({len(points_xyz)} points)")
    ax.legend(loc="best")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def save_arm_slice_local_3d_plot(
    slice_points_local: np.ndarray,
    *,
    output_path: Path,
    max_display_points: int,
) -> None:
    """Save the perpendicular arm slice in local U/V/W coordinates."""
    display_points = downsample_for_display(slice_points_local, max_display_points)
    fig = plt.figure(figsize=(8, 7), dpi=150)
    ax = fig.add_subplot(111, projection="3d")

    if len(display_points) > 0:
        ax.scatter(
            display_points[:, 0],
            display_points[:, 1],
            display_points[:, 2],
            s=2.0,
            c=display_points[:, 2],
            cmap="coolwarm",
            alpha=0.8,
            linewidths=0,
        )
    ax.scatter([0.0], [0.0], [0.0], s=45, c="#ef4444", label="slice center")
    ax.plot([0.0, 0.0], [0.0, 0.0], [-0.05, 0.05], c="#2563eb", linewidth=2.0, label="arm axis W")
    set_axes_equal_from_points(ax, np.vstack([display_points, np.array([[0.0, 0.0, 0.0]])]), min_radius=0.04)
    ax.set_xlabel("U radial (m)")
    ax.set_ylabel("V radial (m)")
    ax.set_zlabel("W arm axis (m)")
    ax.set_title(f"Arm slice in local frame ({len(slice_points_local)} points)")
    ax.legend(loc="best")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def save_arm_slice_2d_projection_plot(
    slice_points_local: np.ndarray,
    *,
    output_path: Path,
    max_display_points: int,
) -> None:
    """Save the perpendicular arm slice compressed onto the local U/V plane."""
    display_points = downsample_for_display(slice_points_local, max_display_points)
    fig, ax = plt.subplots(figsize=(7, 7), dpi=150)
    if len(display_points) > 0:
        ax.scatter(
            display_points[:, 0],
            display_points[:, 1],
            s=3.0,
            c=np.abs(display_points[:, 2]),
            cmap="magma",
            alpha=0.8,
            linewidths=0,
        )
    ax.scatter([0.0], [0.0], s=45, c="#ef4444", label="slice center")
    ax.axhline(0.0, color="#9ca3af", linewidth=0.8)
    ax.axvline(0.0, color="#9ca3af", linewidth=0.8)

    if len(display_points) > 0:
        mins = np.minimum(display_points[:, :2].min(axis=0), np.array([0.0, 0.0]))
        maxs = np.maximum(display_points[:, :2].max(axis=0), np.array([0.0, 0.0]))
        center = (mins + maxs) * 0.5
        radius = float(max(np.max(maxs - mins) * 0.55, 0.02))
    else:
        center = np.array([0.0, 0.0], dtype=np.float64)
        radius = 0.02
    ax.set_xlim(float(center[0] - radius), float(center[0] + radius))
    ax.set_ylim(float(center[1] - radius), float(center[1] + radius))
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("U radial (m)")
    ax.set_ylabel("V radial (m)")
    ax.set_title(f"Arm slice compressed to U/V plane ({len(slice_points_local)} points)")
    ax.legend(loc="best")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def save_arm_slice_arc_2d_plot(
    slice_points_local: np.ndarray,
    filtered_points_local: np.ndarray,
    arc_uv: np.ndarray,
    *,
    output_path: Path,
    max_display_points: int,
) -> None:
    """Save a U/V projection showing raw points, filtered points, and the measured visible arc."""
    raw_display = downsample_for_display(slice_points_local, max_display_points)
    filtered_display = downsample_for_display(filtered_points_local, max_display_points)
    fig, ax = plt.subplots(figsize=(7, 7), dpi=150)

    if len(raw_display) > 0:
        ax.scatter(
            raw_display[:, 0],
            raw_display[:, 1],
            s=2.0,
            c="#cbd5e1",
            alpha=0.35,
            linewidths=0,
            label="raw slice",
        )
    if len(filtered_display) > 0:
        ax.scatter(
            filtered_display[:, 0],
            filtered_display[:, 1],
            s=4.0,
            c=np.abs(filtered_display[:, 2]),
            cmap="magma",
            alpha=0.85,
            linewidths=0,
            label="filtered slice",
        )
    if len(arc_uv) >= 2:
        ax.plot(arc_uv[:, 0], arc_uv[:, 1], c="#16a34a", linewidth=2.0, label="visible arc")
        ax.scatter([arc_uv[0, 0], arc_uv[-1, 0]], [arc_uv[0, 1], arc_uv[-1, 1]], s=28, c="#16a34a")

    ax.scatter([0.0], [0.0], s=45, c="#ef4444", label="slice center")
    ax.axhline(0.0, color="#9ca3af", linewidth=0.8)
    ax.axvline(0.0, color="#9ca3af", linewidth=0.8)

    extents = [np.array([[0.0, 0.0]], dtype=np.float64)]
    if len(raw_display) > 0:
        extents.append(raw_display[:, :2])
    if len(filtered_display) > 0:
        extents.append(filtered_display[:, :2])
    if len(arc_uv) > 0:
        extents.append(np.asarray(arc_uv, dtype=np.float64))
    all_uv = np.vstack(extents)
    mins = all_uv.min(axis=0)
    maxs = all_uv.max(axis=0)
    center = (mins + maxs) * 0.5
    radius = float(max(np.max(maxs - mins) * 0.55, 0.02))
    ax.set_xlim(float(center[0] - radius), float(center[0] + radius))
    ax.set_ylim(float(center[1] - radius), float(center[1] + radius))
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("U radial / camera-depth direction (m)")
    ax.set_ylabel("V radial (m)")
    ax.set_title(f"Filtered arm slice visible arc ({len(filtered_points_local)} points)")
    ax.legend(loc="best")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def choose_largest_detection_index(boxes_xyxy: np.ndarray) -> int:
    """Choose the largest detected person box as the measurement target."""
    if len(boxes_xyxy) == 0:
        raise RuntimeError("No person detection was found.")
    wh = np.maximum(boxes_xyxy[:, 2:4] - boxes_xyxy[:, 0:2], 0.0)
    areas = wh[:, 0] * wh[:, 1]
    return int(np.argmax(areas))


def detect_pose_joints(image_bgr: np.ndarray, pose_model: YOLO) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Run YOLO Pose and return COCO keypoints for the largest detected person."""
    results = pose_model.predict(image_bgr, verbose=False)
    if not results:
        raise RuntimeError("YOLO Pose did not return any result.")

    result = results[0]
    if result.boxes is None or len(result.boxes) == 0 or result.keypoints is None:
        raise RuntimeError("YOLO Pose did not detect a person with keypoints.")

    boxes = result.boxes.xyxy.detach().cpu().numpy()
    person_idx = choose_largest_detection_index(boxes)
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


def joints_list_to_map(joints: list[dict[str, object]]) -> dict[str, dict[str, object]]:
    """Index a joint list by COCO keypoint name for convenient lookup."""
    return {str(item["name"]): item for item in joints}


def pick_joint(
    joint_map: dict[str, dict[str, object]],
    joint_name: str,
    *,
    min_confidence: float,
) -> dict[str, object]:
    """Fetch one named joint and require a minimum pose confidence."""
    joint = joint_map.get(joint_name)
    if joint is None:
        raise RuntimeError(f"Joint not found: {joint_name}")
    confidence = float(joint.get("confidence", 0.0))
    if confidence < min_confidence:
        raise RuntimeError(f"Joint confidence too low for {joint_name}: {confidence:.4f}")
    return joint


def joint_to_uv(joint: dict[str, object]) -> np.ndarray:
    """Convert a joint dictionary into a 2D image coordinate array."""
    return np.array([float(joint["x"]), float(joint["y"])], dtype=np.float64)


def save_pose_overlay(
    image_bgr: np.ndarray,
    *,
    shoulder_uv: np.ndarray,
    elbow_uv: np.ndarray,
    shoulder_name: str,
    elbow_name: str,
    output_path: Path,
) -> None:
    """Save a debug image with shoulder, elbow, and their connecting line drawn."""
    vis = image_bgr.copy()
    shoulder_xy = (int(round(float(shoulder_uv[0]))), int(round(float(shoulder_uv[1]))))
    elbow_xy = (int(round(float(elbow_uv[0]))), int(round(float(elbow_uv[1]))))

    cv2.line(vis, shoulder_xy, elbow_xy, (255, 0, 0), 2, cv2.LINE_AA)
    cv2.circle(vis, shoulder_xy, 7, (0, 255, 0), -1)
    cv2.circle(vis, elbow_xy, 7, (255, 0, 255), -1)
    cv2.putText(
        vis,
        shoulder_name,
        (shoulder_xy[0] + 8, shoulder_xy[1] - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 0),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        vis,
        elbow_name,
        (elbow_xy[0] + 8, elbow_xy[1] - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 0, 255),
        1,
        cv2.LINE_AA,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), vis)


def measure_arm_side(
    *,
    args: argparse.Namespace,
    arm_side: str,
    image_bgr: np.ndarray,
    points_xyz: np.ndarray,
    rgb_shape: tuple[int, int],
    cloud_shape: tuple[int, int],
    joint_map: dict[str, dict[str, object]],
    output_dir: Path,
    artifact_prefix: str | None = None,
) -> dict[str, object]:
    """Measure one arm side and save side-prefixed artifacts."""
    shoulder_name = f"{arm_side}_shoulder"
    elbow_name = f"{arm_side}_elbow"
    shoulder = pick_joint(joint_map, shoulder_name, min_confidence=0.30)
    elbow = pick_joint(joint_map, elbow_name, min_confidence=0.30)
    shoulder_uv = joint_to_uv(shoulder)
    elbow_uv = joint_to_uv(elbow)

    shoulder_xyz, shoulder_cloud_uv, shoulder_ply_meta = sample_joint_xyz_from_point_cloud(
        points_xyz=points_xyz,
        joint_uv_rgb=shoulder_uv,
        rgb_shape=rgb_shape,
        cloud_shape=cloud_shape,
        window_size=int(args.depth_window),
    )
    elbow_xyz, elbow_cloud_uv, elbow_ply_meta = sample_joint_xyz_from_point_cloud(
        points_xyz=points_xyz,
        joint_uv_rgb=elbow_uv,
        rgb_shape=rgb_shape,
        cloud_shape=cloud_shape,
        window_size=int(args.depth_window),
    )

    depth_high_m = max(float(shoulder_xyz[2]), float(elbow_xyz[2])) + float(args.depth_margin_m)
    depth_filtered_points, depth_filter_meta = filter_point_cloud_by_max_depth(
        points_xyz,
        depth_high_m=depth_high_m,
    )
    arm_center_xyz, arm_u_axis, arm_v_axis, arm_w_axis, arm_length_m = build_arm_local_frame(
        shoulder_xyz=shoulder_xyz,
        elbow_xyz=elbow_xyz,
        center_ratio=float(args.arm_center_ratio),
    )
    arm_slice_points_xyz, arm_slice_points_local, arm_slice_meta = extract_perpendicular_arm_slice(
        depth_filtered_points,
        origin_xyz=arm_center_xyz,
        u_axis=arm_u_axis,
        v_axis=arm_v_axis,
        w_axis=arm_w_axis,
        slice_half_thickness_m=float(args.slice_half_thickness_m),
        slice_radius_m=float(args.slice_radius_m),
    )
    filtered_arm_slice_points_local, deviation_filter_meta = filter_deviating_arm_slice_points(
        arm_slice_points_local,
        iqr_scale=float(args.deviation_iqr_scale),
        min_keep_ratio=float(args.deviation_min_keep_ratio),
        min_keep_points=int(args.deviation_min_points),
    )
    visible_arc_uv, visible_arc_meta = extract_visible_arm_arc_uv(filtered_arm_slice_points_local)
    visible_arc_length_m = compute_polyline_length(visible_arc_uv)

    prefix = artifact_prefix if artifact_prefix is not None else f"{arm_side}_"
    overlay_path = output_dir / f"{prefix}pose_shoulder_elbow_overlay.png"
    depth_filtered_points_path = output_dir / f"{prefix}depth_filtered_points.xyz.npy"
    depth_filtered_plot_path = output_dir / f"{prefix}depth_filtered_point_cloud.png"
    arm_slice_points_path = output_dir / f"{prefix}arm_slice_points.xyz.npy"
    arm_slice_local_points_path = output_dir / f"{prefix}arm_slice_points.local_uvw.npy"
    filtered_arm_slice_local_points_path = output_dir / f"{prefix}arm_slice_points.filtered.local_uvw.npy"
    visible_arc_uv_path = output_dir / f"{prefix}arm_slice_visible_arc.uv.npy"
    arm_slice_local_3d_plot_path = output_dir / f"{prefix}arm_slice_local_3d.png"
    arm_slice_2d_plot_path = output_dir / f"{prefix}arm_slice_uv_2d.png"
    arm_slice_arc_2d_plot_path = output_dir / f"{prefix}arm_slice_visible_arc_2d.png"

    save_pose_overlay(
        image_bgr,
        shoulder_uv=shoulder_uv,
        elbow_uv=elbow_uv,
        shoulder_name=shoulder_name,
        elbow_name=elbow_name,
        output_path=overlay_path,
    )
    np.save(depth_filtered_points_path, depth_filtered_points.astype(np.float32))
    np.save(arm_slice_points_path, arm_slice_points_xyz.astype(np.float32))
    np.save(arm_slice_local_points_path, arm_slice_points_local.astype(np.float32))
    np.save(filtered_arm_slice_local_points_path, filtered_arm_slice_points_local.astype(np.float32))
    np.save(visible_arc_uv_path, visible_arc_uv.astype(np.float32))
    save_depth_filtered_point_cloud_plot(
        depth_filtered_points,
        shoulder_xyz=shoulder_xyz,
        elbow_xyz=elbow_xyz,
        output_path=depth_filtered_plot_path,
        max_display_points=int(args.max_display_points),
    )
    save_arm_slice_local_3d_plot(
        arm_slice_points_local,
        output_path=arm_slice_local_3d_plot_path,
        max_display_points=int(args.max_display_points),
    )
    save_arm_slice_2d_projection_plot(
        arm_slice_points_local,
        output_path=arm_slice_2d_plot_path,
        max_display_points=int(args.max_display_points),
    )
    save_arm_slice_arc_2d_plot(
        arm_slice_points_local,
        filtered_arm_slice_points_local,
        visible_arc_uv,
        output_path=arm_slice_arc_2d_plot_path,
        max_display_points=int(args.max_display_points),
    )

    return {
        "arm_side": arm_side,
        "target_arm": {
            "arm_side": arm_side,
            "shoulder_joint_name": shoulder_name,
            "elbow_joint_name": elbow_name,
            "shoulder_uv": np.round(shoulder_uv, 3).tolist(),
            "elbow_uv": np.round(elbow_uv, 3).tolist(),
            "shoulder_confidence": float(shoulder["confidence"]),
            "elbow_confidence": float(elbow["confidence"]),
        },
        "point_cloud_3d": {
            "source": "organized_ply",
            "shoulder_uv_on_point_cloud": [int(shoulder_cloud_uv[0]), int(shoulder_cloud_uv[1])],
            "elbow_uv_on_point_cloud": [int(elbow_cloud_uv[0]), int(elbow_cloud_uv[1])],
            "shoulder_xyz_meter": np.round(shoulder_xyz, 6).tolist(),
            "elbow_xyz_meter": np.round(elbow_xyz, 6).tolist(),
            "shoulder_sampling": shoulder_ply_meta,
            "elbow_sampling": elbow_ply_meta,
        },
        "depth_gate": {
            "depth_margin_m": float(args.depth_margin_m),
            "depth_filter_max_m": float(depth_high_m),
            "filter_meta": depth_filter_meta,
        },
        "arm_slice": {
            "source_points": "depth_filtered_point_cloud",
            "center_ratio": float(args.arm_center_ratio),
            "arm_length_m": float(arm_length_m),
            "center_xyz_meter": np.round(arm_center_xyz, 6).tolist(),
            "u_axis": np.round(arm_u_axis, 6).tolist(),
            "v_axis": np.round(arm_v_axis, 6).tolist(),
            "w_axis_shoulder_to_elbow": np.round(arm_w_axis, 6).tolist(),
            "filter_meta": arm_slice_meta,
            "deviation_filter_meta": deviation_filter_meta,
            "visible_arc_meta": visible_arc_meta,
            "visible_arc_length_m": float(round(visible_arc_length_m, 6)),
        },
        "artifacts": {
            "pose_overlay_png": str(overlay_path.resolve()),
            "depth_filtered_points_npy": str(depth_filtered_points_path.resolve()),
            "depth_filtered_point_cloud_png": str(depth_filtered_plot_path.resolve()),
            "arm_slice_points_npy": str(arm_slice_points_path.resolve()),
            "arm_slice_local_points_npy": str(arm_slice_local_points_path.resolve()),
            "arm_slice_filtered_local_points_npy": str(filtered_arm_slice_local_points_path.resolve()),
            "arm_slice_visible_arc_uv_npy": str(visible_arc_uv_path.resolve()),
            "arm_slice_local_3d_png": str(arm_slice_local_3d_plot_path.resolve()),
            "arm_slice_uv_2d_png": str(arm_slice_2d_plot_path.resolve()),
            "arm_slice_visible_arc_2d_png": str(arm_slice_arc_2d_plot_path.resolve()),
        },
    }


def resolve_input_views(args: argparse.Namespace) -> list[tuple[str, Path, Path, Path]]:
    """Resolve CLI inputs into one or two named views."""
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
    """Check that all view input paths exist before running the heavier model work."""
    for view_name, rgb_path, depth_path, ply_path in input_views:
        if not rgb_path.exists():
            raise FileNotFoundError(f"[{view_name}] RGB input not found: {rgb_path}")
        if not depth_path.exists():
            raise FileNotFoundError(f"[{view_name}] Depth input not found: {depth_path}")
        if not ply_path.exists():
            raise FileNotFoundError(f"[{view_name}] Point cloud input not found: {ply_path}")


def process_arm_view(
    *,
    args: argparse.Namespace,
    view_name: str,
    rgb_path: Path,
    depth_path: Path,
    ply_path: Path,
    pose_model: YOLO,
    requested_sides: list[str],
    output_dir: Path,
) -> dict[str, object]:
    """Load one view, detect pose once, then measure requested arm sides."""
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
    arm_results = {
        side: measure_arm_side(
            args=args,
            arm_side=side,
            image_bgr=image_bgr,
            points_xyz=points_xyz,
            rgb_shape=(rgb_h, rgb_w),
            cloud_shape=(depth_h, depth_w),
            joint_map=joint_map,
            output_dir=output_dir,
            artifact_prefix=f"{view_name}_{side}_",
        )
        for side in requested_sides
    }

    return {
        "view_name": view_name,
        "inputs": {
            "rgb": str(rgb_path.resolve()),
            "depth": str(depth_path.resolve()),
            "point_cloud_ply": str(ply_path.resolve()),
        },
        "detection": detection_meta,
        "arms": arm_results,
        "joints": joints,
    }


def main() -> None:
    """Run pose detection, PLY joint sampling, and depth-gated point-cloud display."""
    args = parse_args()
    input_views = resolve_input_views(args)
    validate_input_view_paths(input_views)
    if args.slice_half_thickness_m <= 0:
        raise RuntimeError("--slice-half-thickness-m must be > 0.")
    if args.slice_radius_m <= 0:
        raise RuntimeError("--slice-radius-m must be > 0.")
    if args.deviation_iqr_scale <= 0:
        raise RuntimeError("--deviation-iqr-scale must be > 0.")
    if not (0 < args.deviation_min_keep_ratio <= 1):
        raise RuntimeError("--deviation-min-keep-ratio must be in (0, 1].")
    if args.deviation_min_points < 1:
        raise RuntimeError("--deviation-min-points must be >= 1.")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    cx = float(args.cx) if args.cx is not None else None
    cy = float(args.cy) if args.cy is not None else None
    pose_model = YOLO(str(args.pose_model))
    requested_sides = ["left", "right"] if str(args.arm_side) == "both" else [str(args.arm_side)]
    joints_path = args.output_dir / "pose_joints.json"

    view_results = {
        view_name: process_arm_view(
            args=args,
            view_name=view_name,
            rgb_path=rgb_path,
            depth_path=depth_path,
            ply_path=ply_path,
            pose_model=pose_model,
            requested_sides=requested_sides,
            output_dir=args.output_dir,
        )
        for view_name, rgb_path, depth_path, ply_path in input_views
    }

    arm_total_lengths: dict[str, float] = {}
    measurement_summary: dict[str, float] = {}
    for side in requested_sides:
        side_total = 0.0
        for view_name, view_result in view_results.items():
            length_m = float(view_result["arms"][side]["arm_slice"]["visible_arc_length_m"])
            measurement_summary[f"{view_name}_{side}_arm_visible_arc_length_m"] = float(round(length_m, 6))
            side_total += length_m
        arm_total_lengths[f"{side}_arm_total_visible_arc_length_m"] = float(round(side_total, 6))
        measurement_summary[f"{side}_arm_total_visible_arc_length_m"] = float(round(side_total, 6))

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
        "camera_intrinsics": {
            "fx": None if args.fx is None else float(args.fx),
            "fy": None if args.fy is None else float(args.fy),
            "cx": None if cx is None else float(cx),
            "cy": None if cy is None else float(cy),
            "depth_scale_to_meter": float(args.depth_scale),
            "note": "Intrinsics are metadata only; shoulder/elbow XYZ are sampled directly from the PLY.",
        },
        "requested_arm_side": str(args.arm_side),
        "measured_arm_sides": requested_sides,
        "views": view_results,
        "arm_total_lengths": arm_total_lengths,
        "measurement_summary": measurement_summary,
        "artifacts": {
            "joints_json": str(joints_path.resolve()),
        },
    }
    joints_path.write_text(json.dumps(output, ensure_ascii=True, indent=2), encoding="utf-8")

    for view_name, view_result in view_results.items():
        detection_meta = view_result["detection"]
        print(f"[{view_name}] Selected person index: {detection_meta['selected_person_index']}")
        for side, result in view_result["arms"].items():
            target = result["target_arm"]
            depth_gate = result["depth_gate"]
            arm_slice = result["arm_slice"]
            slice_filter_meta = arm_slice["filter_meta"]
            deviation_meta = arm_slice["deviation_filter_meta"]
            print(
                f"[{view_name}/{side}] {target['shoulder_joint_name']}: "
                f"uv={target['shoulder_uv']}, conf={float(target['shoulder_confidence']):.4f}"
            )
            print(
                f"[{view_name}/{side}] {target['elbow_joint_name']}: "
                f"uv={target['elbow_uv']}, conf={float(target['elbow_confidence']):.4f}"
            )
            print(f"[{view_name}/{side}] Depth-filtered points: {int(depth_gate['filter_meta']['output_count'])}")
            print(f"[{view_name}/{side}] Arm slice points: {int(slice_filter_meta['output_count'])}")
            print(f"[{view_name}/{side}] Filtered arm slice points: {int(deviation_meta['output_count'])}")
            print(f"[{view_name}/{side}] Visible arm arc length (m): {float(arm_slice['visible_arc_length_m']):.6f}")

    for key, value in arm_total_lengths.items():
        print(f"{key}: {value:.6f}")
    print(f"Saved joints: {joints_path.resolve()}")


if __name__ == "__main__":
    main()
