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

from .arm_pointcloud import (
    DEFAULT_POSE_MODEL_PATH,
    build_arm_local_frame,
    compute_polyline_length,
    detect_pose_joints,
    downsample_for_display,
    extract_perpendicular_arm_slice,
    extract_visible_arm_arc_uv,
    filter_deviating_arm_slice_points,
    filter_point_cloud_by_max_depth,
    joint_to_uv,
    joints_list_to_map,
    load_depth_map,
    load_rgb_image,
    pick_joint,
    sample_joint_xyz_from_point_cloud,
    set_axes_equal_from_points,
)
from .pointcloud import load_point_cloud


DEFAULT_OUTPUT_DIR = Path("result") / "leg_pointcloud"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Measure left/right thigh visible arc lengths from front/back RGB+Depth+PLY inputs. "
            "The thigh center is estimated on the hip->knee segment."
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
    parser.add_argument("--leg-side", choices=["left", "right", "both"], default="both", help="Target thigh side.")
    parser.add_argument("--hip-min-confidence", type=float, default=0.20, help="Minimum pose confidence for hip joints.")
    parser.add_argument("--knee-min-confidence", type=float, default=0.03, help="Minimum pose confidence for knee joints.")
    parser.add_argument("--fx", type=float, default=None, help="Optional camera intrinsics fx for metadata.")
    parser.add_argument("--fy", type=float, default=None, help="Optional camera intrinsics fy for metadata.")
    parser.add_argument("--cx", type=float, default=None, help="Optional camera intrinsics cx for metadata.")
    parser.add_argument("--cy", type=float, default=None, help="Optional camera intrinsics cy for metadata.")
    parser.add_argument("--depth-scale", type=float, default=0.001, help="Depth unit scale to meters.")
    parser.add_argument("--depth-window", type=int, default=5, help="Odd median window for joint point-cloud sampling.")
    parser.add_argument(
        "--depth-margin-m",
        type=float,
        default=0.08,
        help="Depth margin behind the farther hip/knee joint, in meters.",
    )
    parser.add_argument(
        "--thigh-center-ratio",
        type=float,
        default=1.0 / 3.0,
        help="Center ratio on hip->knee segment. 0=hip, 1=knee.",
    )
    parser.add_argument(
        "--slice-half-thickness-m",
        type=float,
        default=0.012,
        help="Half thickness of the perpendicular thigh slice along the hip->knee axis, in meters.",
    )
    parser.add_argument(
        "--slice-radius-m",
        type=float,
        default=0.24,
        help="Maximum distance from the hip->knee axis kept in the perpendicular slice, in meters.",
    )
    parser.add_argument(
        "--deviation-iqr-scale",
        type=float,
        default=1.8,
        help="Robust IQR multiplier used to filter deviating points in the thigh slice U/V plane.",
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
        default=25,
        help="Minimum point count required after deviation filtering before falling back to the original slice.",
    )
    parser.add_argument(
        "--max-display-points",
        type=int,
        default=120000,
        help="Maximum points to draw in debug previews. 0 means no limit.",
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


def compute_thigh_center_uv(hip_uv: np.ndarray, knee_uv: np.ndarray, ratio: float) -> np.ndarray:
    ratio_clamped = float(np.clip(ratio, 0.0, 1.0))
    return (np.asarray(hip_uv, dtype=np.float64) + (np.asarray(knee_uv, dtype=np.float64) - hip_uv) * ratio_clamped).astype(
        np.float64
    )


def clip_uv_to_shape(uv: np.ndarray, shape_hw: tuple[int, int]) -> np.ndarray:
    h, w = shape_hw
    clipped = np.asarray(uv, dtype=np.float64).copy()
    clipped[0] = float(np.clip(clipped[0], 0, max(w - 1, 0)))
    clipped[1] = float(np.clip(clipped[1], 0, max(h - 1, 0)))
    return clipped


def sample_limb_endpoint_xyz(
    *,
    points_xyz: np.ndarray,
    endpoint_uv_rgb: np.ndarray,
    fallback_uv_rgb: np.ndarray,
    rgb_shape: tuple[int, int],
    cloud_shape: tuple[int, int],
    window_size: int,
    endpoint_name: str,
) -> tuple[np.ndarray, tuple[int, int], dict[str, int | float | str | list[float]]]:
    endpoint = clip_uv_to_shape(endpoint_uv_rgb, rgb_shape)
    fallback = clip_uv_to_shape(fallback_uv_rgb, rgb_shape)
    ratios = [0.0, 0.03, 0.06, 0.10, 0.15, 0.20, 0.28, 0.36]
    failures: list[str] = []
    for ratio in ratios:
        candidate_uv = endpoint + (fallback - endpoint) * float(ratio)
        candidate_uv = clip_uv_to_shape(candidate_uv, rgb_shape)
        try:
            xyz, cloud_uv, meta = sample_joint_xyz_from_point_cloud(
                points_xyz=points_xyz,
                joint_uv_rgb=candidate_uv,
                rgb_shape=rgb_shape,
                cloud_shape=cloud_shape,
                window_size=window_size,
            )
            out_meta: dict[str, int | float | str | list[float]] = dict(meta)
            out_meta.update(
                {
                    "endpoint_name": endpoint_name,
                    "requested_uv": np.round(endpoint_uv_rgb, 3).tolist(),
                    "sample_uv": np.round(candidate_uv, 3).tolist(),
                    "fallback_toward_uv": np.round(fallback_uv_rgb, 3).tolist(),
                    "fallback_ratio_toward_other_joint": float(ratio),
                    "fallback_attempts": int(len(failures) + 1),
                }
            )
            return xyz, cloud_uv, out_meta
        except RuntimeError as exc:
            failures.append(str(exc))

    raise RuntimeError(f"No valid point-cloud XYZ for {endpoint_name}; attempts failed: {' | '.join(failures[-3:])}")


def save_pose_overlay(
    image_bgr: np.ndarray,
    *,
    hip_uv: np.ndarray,
    knee_uv: np.ndarray,
    thigh_center_uv: np.ndarray,
    hip_name: str,
    knee_name: str,
    output_path: Path,
) -> None:
    vis = image_bgr.copy()
    hip_xy = (int(round(float(hip_uv[0]))), int(round(float(hip_uv[1]))))
    knee_xy = (int(round(float(knee_uv[0]))), int(round(float(knee_uv[1]))))
    center_xy = (int(round(float(thigh_center_uv[0]))), int(round(float(thigh_center_uv[1]))))
    cv2.line(vis, hip_xy, knee_xy, (255, 0, 0), 2, cv2.LINE_AA)
    cv2.circle(vis, hip_xy, 7, (34, 197, 94), -1, lineType=cv2.LINE_AA)
    cv2.circle(vis, knee_xy, 7, (232, 121, 249), -1, lineType=cv2.LINE_AA)
    cv2.drawMarker(vis, center_xy, (0, 0, 255), cv2.MARKER_CROSS, 24, 3, line_type=cv2.LINE_AA)
    cv2.circle(vis, center_xy, 8, (0, 0, 255), 2, lineType=cv2.LINE_AA)
    labels = [(hip_xy, hip_name, (34, 197, 94)), (knee_xy, knee_name, (232, 121, 249)), (center_xy, "thigh_center", (0, 0, 255))]
    for xy, label, color in labels:
        cv2.putText(vis, label, (xy[0] + 8, xy[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 1, cv2.LINE_AA)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), vis)


def save_depth_filtered_point_cloud_plot(
    points_xyz: np.ndarray,
    *,
    hip_xyz: np.ndarray,
    knee_xyz: np.ndarray,
    thigh_center_xyz: np.ndarray,
    output_path: Path,
    max_display_points: int,
) -> None:
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
    ax.scatter([hip_xyz[0]], [hip_xyz[1]], [hip_xyz[2]], s=45, c="#22c55e", label="hip")
    ax.scatter([knee_xyz[0]], [knee_xyz[1]], [knee_xyz[2]], s=45, c="#e879f9", label="knee")
    ax.scatter([thigh_center_xyz[0]], [thigh_center_xyz[1]], [thigh_center_xyz[2]], s=55, c="#ef4444", label="thigh center")
    ax.plot([hip_xyz[0], knee_xyz[0]], [hip_xyz[1], knee_xyz[1]], [hip_xyz[2], knee_xyz[2]], c="#2563eb", linewidth=2.0)
    if len(display_points) > 0:
        all_points = np.vstack([display_points, hip_xyz, knee_xyz, thigh_center_xyz])
    else:
        all_points = np.vstack([hip_xyz, knee_xyz, thigh_center_xyz])
    set_axes_equal_from_points(ax, all_points, min_radius=0.05)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z depth (m)")
    ax.set_title(f"Depth-gated point cloud ({len(points_xyz)} points)")
    ax.legend(loc="best")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def save_thigh_slice_local_3d_plot(slice_points_local: np.ndarray, *, output_path: Path, max_display_points: int) -> None:
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
    ax.plot([0.0, 0.0], [0.0, 0.0], [-0.06, 0.06], c="#2563eb", linewidth=2.0, label="hip-knee axis W")
    set_axes_equal_from_points(ax, np.vstack([display_points, np.array([[0.0, 0.0, 0.0]])]), min_radius=0.05)
    ax.set_xlabel("U radial (m)")
    ax.set_ylabel("V radial (m)")
    ax.set_zlabel("W leg axis (m)")
    ax.set_title(f"Thigh slice in local frame ({len(slice_points_local)} points)")
    ax.legend(loc="best")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def save_thigh_slice_arc_2d_plot(
    slice_points_local: np.ndarray,
    filtered_points_local: np.ndarray,
    arc_uv: np.ndarray,
    *,
    output_path: Path,
    max_display_points: int,
) -> None:
    raw_display = downsample_for_display(slice_points_local, max_display_points)
    filtered_display = downsample_for_display(filtered_points_local, max_display_points)
    fig, ax = plt.subplots(figsize=(7, 7), dpi=150)
    if len(raw_display) > 0:
        ax.scatter(raw_display[:, 0], raw_display[:, 1], s=2.0, c="#cbd5e1", alpha=0.35, linewidths=0, label="raw slice")
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
    radius = float(max(np.max(maxs - mins) * 0.55, 0.025))
    ax.set_xlim(float(center[0] - radius), float(center[0] + radius))
    ax.set_ylim(float(center[1] - radius), float(center[1] + radius))
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("U radial / camera-depth direction (m)")
    ax.set_ylabel("V radial (m)")
    ax.set_title(f"Filtered thigh slice visible arc ({len(filtered_points_local)} points)")
    ax.legend(loc="best")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def measure_leg_side(
    *,
    args: argparse.Namespace,
    leg_side: str,
    image_bgr: np.ndarray,
    points_xyz: np.ndarray,
    rgb_shape: tuple[int, int],
    cloud_shape: tuple[int, int],
    joint_map: dict[str, dict[str, object]],
    output_dir: Path,
    artifact_prefix: str | None = None,
) -> dict[str, object]:
    hip_name = f"{leg_side}_hip"
    knee_name = f"{leg_side}_knee"
    hip = pick_joint(joint_map, hip_name, min_confidence=float(args.hip_min_confidence))
    knee = pick_joint(joint_map, knee_name, min_confidence=float(args.knee_min_confidence))
    hip_uv = joint_to_uv(hip)
    knee_uv = joint_to_uv(knee)
    thigh_center_uv = compute_thigh_center_uv(hip_uv, knee_uv, float(args.thigh_center_ratio))
    hip_sample_uv = clip_uv_to_shape(hip_uv, rgb_shape)
    knee_sample_uv = clip_uv_to_shape(knee_uv, rgb_shape)

    hip_xyz, hip_cloud_uv, hip_ply_meta = sample_limb_endpoint_xyz(
        points_xyz=points_xyz,
        endpoint_uv_rgb=hip_sample_uv,
        fallback_uv_rgb=knee_sample_uv,
        rgb_shape=rgb_shape,
        cloud_shape=cloud_shape,
        window_size=int(args.depth_window),
        endpoint_name=hip_name,
    )
    knee_xyz, knee_cloud_uv, knee_ply_meta = sample_limb_endpoint_xyz(
        points_xyz=points_xyz,
        endpoint_uv_rgb=knee_sample_uv,
        fallback_uv_rgb=hip_sample_uv,
        rgb_shape=rgb_shape,
        cloud_shape=cloud_shape,
        window_size=int(args.depth_window),
        endpoint_name=knee_name,
    )

    depth_high_m = max(float(hip_xyz[2]), float(knee_xyz[2])) + float(args.depth_margin_m)
    depth_filtered_points, depth_filter_meta = filter_point_cloud_by_max_depth(points_xyz, depth_high_m=depth_high_m)
    thigh_center_xyz, thigh_u_axis, thigh_v_axis, thigh_w_axis, leg_length_m = build_arm_local_frame(
        shoulder_xyz=hip_xyz,
        elbow_xyz=knee_xyz,
        center_ratio=float(args.thigh_center_ratio),
    )
    thigh_slice_points_xyz, thigh_slice_points_local, thigh_slice_meta = extract_perpendicular_arm_slice(
        depth_filtered_points,
        origin_xyz=thigh_center_xyz,
        u_axis=thigh_u_axis,
        v_axis=thigh_v_axis,
        w_axis=thigh_w_axis,
        slice_half_thickness_m=float(args.slice_half_thickness_m),
        slice_radius_m=float(args.slice_radius_m),
    )
    filtered_thigh_slice_points_local, deviation_filter_meta = filter_deviating_arm_slice_points(
        thigh_slice_points_local,
        iqr_scale=float(args.deviation_iqr_scale),
        min_keep_ratio=float(args.deviation_min_keep_ratio),
        min_keep_points=int(args.deviation_min_points),
    )
    visible_arc_uv, visible_arc_meta = extract_visible_arm_arc_uv(filtered_thigh_slice_points_local)
    visible_arc_length_m = compute_polyline_length(visible_arc_uv)

    prefix = artifact_prefix if artifact_prefix is not None else f"{leg_side}_"
    overlay_path = output_dir / f"{prefix}pose_hip_knee_overlay.png"
    depth_filtered_points_path = output_dir / f"{prefix}depth_filtered_points.xyz.npy"
    depth_filtered_plot_path = output_dir / f"{prefix}depth_filtered_point_cloud.png"
    thigh_slice_points_path = output_dir / f"{prefix}thigh_slice_points.xyz.npy"
    thigh_slice_local_points_path = output_dir / f"{prefix}thigh_slice_points.local_uvw.npy"
    filtered_thigh_slice_local_points_path = output_dir / f"{prefix}thigh_slice_points.filtered.local_uvw.npy"
    visible_arc_uv_path = output_dir / f"{prefix}thigh_slice_visible_arc.uv.npy"
    thigh_slice_local_3d_plot_path = output_dir / f"{prefix}thigh_slice_local_3d.png"
    thigh_slice_arc_2d_plot_path = output_dir / f"{prefix}thigh_slice_visible_arc_2d.png"

    save_pose_overlay(
        image_bgr,
        hip_uv=hip_uv,
        knee_uv=knee_uv,
        thigh_center_uv=thigh_center_uv,
        hip_name=hip_name,
        knee_name=knee_name,
        output_path=overlay_path,
    )
    np.save(depth_filtered_points_path, depth_filtered_points.astype(np.float32))
    np.save(thigh_slice_points_path, thigh_slice_points_xyz.astype(np.float32))
    np.save(thigh_slice_local_points_path, thigh_slice_points_local.astype(np.float32))
    np.save(filtered_thigh_slice_local_points_path, filtered_thigh_slice_points_local.astype(np.float32))
    np.save(visible_arc_uv_path, visible_arc_uv.astype(np.float32))
    save_depth_filtered_point_cloud_plot(
        depth_filtered_points,
        hip_xyz=hip_xyz,
        knee_xyz=knee_xyz,
        thigh_center_xyz=thigh_center_xyz,
        output_path=depth_filtered_plot_path,
        max_display_points=int(args.max_display_points),
    )
    save_thigh_slice_local_3d_plot(
        thigh_slice_points_local,
        output_path=thigh_slice_local_3d_plot_path,
        max_display_points=int(args.max_display_points),
    )
    save_thigh_slice_arc_2d_plot(
        thigh_slice_points_local,
        filtered_thigh_slice_points_local,
        visible_arc_uv,
        output_path=thigh_slice_arc_2d_plot_path,
        max_display_points=int(args.max_display_points),
    )

    return {
        "leg_side": leg_side,
        "target_leg": {
            "leg_side": leg_side,
            "hip_joint_name": hip_name,
            "knee_joint_name": knee_name,
            "hip_uv": np.round(hip_uv, 3).tolist(),
            "knee_uv": np.round(knee_uv, 3).tolist(),
            "hip_sample_uv": np.round(hip_sample_uv, 3).tolist(),
            "knee_sample_uv": np.round(knee_sample_uv, 3).tolist(),
            "thigh_center_uv": np.round(thigh_center_uv, 3).tolist(),
            "hip_confidence": float(hip["confidence"]),
            "knee_confidence": float(knee["confidence"]),
        },
        "point_cloud_3d": {
            "source": "organized_ply",
            "hip_uv_on_point_cloud": [int(hip_cloud_uv[0]), int(hip_cloud_uv[1])],
            "knee_uv_on_point_cloud": [int(knee_cloud_uv[0]), int(knee_cloud_uv[1])],
            "hip_xyz_meter": np.round(hip_xyz, 6).tolist(),
            "knee_xyz_meter": np.round(knee_xyz, 6).tolist(),
            "thigh_center_xyz_meter": np.round(thigh_center_xyz, 6).tolist(),
            "hip_sampling": hip_ply_meta,
            "knee_sampling": knee_ply_meta,
        },
        "depth_gate": {
            "depth_margin_m": float(args.depth_margin_m),
            "depth_filter_max_m": float(depth_high_m),
            "filter_meta": depth_filter_meta,
        },
        "thigh_slice": {
            "source_points": "depth_filtered_point_cloud",
            "center_ratio": float(args.thigh_center_ratio),
            "hip_knee_length_m": float(leg_length_m),
            "center_xyz_meter": np.round(thigh_center_xyz, 6).tolist(),
            "u_axis": np.round(thigh_u_axis, 6).tolist(),
            "v_axis": np.round(thigh_v_axis, 6).tolist(),
            "w_axis_hip_to_knee": np.round(thigh_w_axis, 6).tolist(),
            "filter_meta": thigh_slice_meta,
            "deviation_filter_meta": deviation_filter_meta,
            "visible_arc_meta": visible_arc_meta,
            "visible_arc_length_m": float(round(visible_arc_length_m, 6)),
        },
        "artifacts": {
            "pose_overlay_png": str(overlay_path.resolve()),
            "depth_filtered_points_npy": str(depth_filtered_points_path.resolve()),
            "depth_filtered_point_cloud_png": str(depth_filtered_plot_path.resolve()),
            "thigh_slice_points_npy": str(thigh_slice_points_path.resolve()),
            "thigh_slice_local_points_npy": str(thigh_slice_local_points_path.resolve()),
            "thigh_slice_filtered_local_points_npy": str(filtered_thigh_slice_local_points_path.resolve()),
            "thigh_slice_visible_arc_uv_npy": str(visible_arc_uv_path.resolve()),
            "thigh_slice_local_3d_png": str(thigh_slice_local_3d_plot_path.resolve()),
            "thigh_slice_visible_arc_2d_png": str(thigh_slice_arc_2d_plot_path.resolve()),
        },
    }


def process_leg_view(
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
    leg_results = {
        side: measure_leg_side(
            args=args,
            leg_side=side,
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
        "legs": leg_results,
        "joints": joints,
    }


def main() -> None:
    args = parse_args()
    input_views = resolve_input_views(args)
    validate_input_view_paths(input_views)
    if not (0 <= args.thigh_center_ratio <= 1):
        raise RuntimeError("--thigh-center-ratio must be in [0, 1].")
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
    requested_sides = ["left", "right"] if str(args.leg_side) == "both" else [str(args.leg_side)]
    summary_path = args.output_dir / "pose_joints.json"
    view_results = {
        view_name: process_leg_view(
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

    leg_total_lengths: dict[str, float] = {}
    measurement_summary: dict[str, float] = {}
    for side in requested_sides:
        side_total = 0.0
        for view_name, view_result in view_results.items():
            length_m = float(view_result["legs"][side]["thigh_slice"]["visible_arc_length_m"])
            measurement_summary[f"{view_name}_{side}_thigh_visible_arc_length_m"] = float(round(length_m, 6))
            side_total += length_m
        leg_total_lengths[f"{side}_thigh_total_visible_arc_length_m"] = float(round(side_total, 6))
        measurement_summary[f"{side}_thigh_total_visible_arc_length_m"] = float(round(side_total, 6))

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
            "note": "Intrinsics are metadata only; hip/knee XYZ are sampled directly from the PLY.",
        },
        "measurement_params": {
            "leg_side": str(args.leg_side),
            "thigh_center_ratio": float(args.thigh_center_ratio),
            "depth_margin_m": float(args.depth_margin_m),
            "slice_half_thickness_m": float(args.slice_half_thickness_m),
            "slice_radius_m": float(args.slice_radius_m),
            "deviation_iqr_scale": float(args.deviation_iqr_scale),
            "deviation_min_keep_ratio": float(args.deviation_min_keep_ratio),
            "deviation_min_points": int(args.deviation_min_points),
            "hip_min_confidence": float(args.hip_min_confidence),
            "knee_min_confidence": float(args.knee_min_confidence),
        },
        "requested_leg_side": str(args.leg_side),
        "measured_leg_sides": requested_sides,
        "views": view_results,
        "leg_total_lengths": leg_total_lengths,
        "measurement_summary": measurement_summary,
        "artifacts": {
            "joints_json": str(summary_path.resolve()),
        },
    }
    summary_path.write_text(json.dumps(output, ensure_ascii=True, indent=2), encoding="utf-8")

    for view_name, view_result in view_results.items():
        detection_meta = view_result["detection"]
        print(f"[{view_name}] Selected person index: {detection_meta['selected_person_index']}")
        for side, result in view_result["legs"].items():
            target = result["target_leg"]
            depth_gate = result["depth_gate"]
            thigh_slice = result["thigh_slice"]
            slice_filter_meta = thigh_slice["filter_meta"]
            deviation_meta = thigh_slice["deviation_filter_meta"]
            print(
                f"[{view_name}/{side}] {target['hip_joint_name']}: "
                f"uv={target['hip_uv']}, conf={float(target['hip_confidence']):.4f}"
            )
            print(
                f"[{view_name}/{side}] {target['knee_joint_name']}: "
                f"uv={target['knee_uv']}, conf={float(target['knee_confidence']):.4f}"
            )
            print(f"[{view_name}/{side}] Thigh center uv: {target['thigh_center_uv']}")
            print(f"[{view_name}/{side}] Depth-filtered points: {int(depth_gate['filter_meta']['output_count'])}")
            print(f"[{view_name}/{side}] Thigh slice points: {int(slice_filter_meta['output_count'])}")
            print(f"[{view_name}/{side}] Filtered thigh slice points: {int(deviation_meta['output_count'])}")
            print(f"[{view_name}/{side}] Visible thigh arc length (m): {float(thigh_slice['visible_arc_length_m']):.6f}")
    for key, value in leg_total_lengths.items():
        print(f"{key}: {value:.6f}")
    print(f"Saved joints: {summary_path.resolve()}")


if __name__ == "__main__":
    main()
