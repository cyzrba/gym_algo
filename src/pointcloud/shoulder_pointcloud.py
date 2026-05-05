from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str((Path.cwd() / ".mplconfig").resolve()))

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from ultralytics import YOLO

from .arm_pointcloud import (
    DEFAULT_POSE_MODEL_PATH,
    detect_pose_joints,
    joint_to_uv,
    joints_list_to_map,
    load_depth_map,
    load_rgb_image,
    pick_joint,
    sample_joint_xyz_from_point_cloud,
)
from .pointcloud import load_point_cloud


DEFAULT_OUTPUT_DIR = Path("result") / "shoulder_pointcloud"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Measure shoulder width from RGB+Depth+PLY input. The distance is the 3D line "
            "between left_shoulder and right_shoulder keypoints sampled from the organized PLY."
        )
    )
    parser.add_argument("--rgb", type=Path, default=None, help="Single-view RGB image path.")
    parser.add_argument("--depth", type=Path, default=None, help="Single-view depth image path.")
    parser.add_argument("--ply", type=Path, default=None, help="Single-view point cloud .ply path.")
    parser.add_argument("--front-rgb", type=Path, default=None, help="Front-view RGB image path.")
    parser.add_argument("--front-depth", type=Path, default=None, help="Front-view depth image path.")
    parser.add_argument("--front-ply", type=Path, default=None, help="Front-view point cloud .ply path.")
    parser.add_argument("--back-rgb", type=Path, default=None, help="Back-view RGB image path.")
    parser.add_argument("--back-depth", type=Path, default=None, help="Back-view depth image path.")
    parser.add_argument("--back-ply", type=Path, default=None, help="Back-view point cloud .ply path.")
    parser.add_argument("--pose-model", type=Path, default=DEFAULT_POSE_MODEL_PATH, help="YOLO pose model path.")
    parser.add_argument("--shoulder-min-confidence", type=float, default=0.20, help="Minimum pose confidence.")
    parser.add_argument("--depth-window", type=int, default=5, help="Odd median window for point-cloud sampling.")
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


def save_shoulder_overlay(
    image_bgr: np.ndarray,
    *,
    left_uv: np.ndarray,
    right_uv: np.ndarray,
    output_path: Path,
) -> None:
    vis = image_bgr.copy()
    left_xy = (int(round(float(left_uv[0]))), int(round(float(left_uv[1]))))
    right_xy = (int(round(float(right_uv[0]))), int(round(float(right_uv[1]))))
    mid_xy = (int(round((left_xy[0] + right_xy[0]) * 0.5)), int(round((left_xy[1] + right_xy[1]) * 0.5)))

    cv2.line(vis, left_xy, right_xy, (255, 0, 0), 2, cv2.LINE_AA)
    cv2.circle(vis, left_xy, 7, (34, 197, 94), -1, lineType=cv2.LINE_AA)
    cv2.circle(vis, right_xy, 7, (232, 121, 249), -1, lineType=cv2.LINE_AA)
    cv2.drawMarker(vis, mid_xy, (0, 0, 255), cv2.MARKER_CROSS, 22, 2, line_type=cv2.LINE_AA)
    cv2.putText(vis, "left_shoulder", (left_xy[0] + 8, left_xy[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (34, 197, 94), 1, cv2.LINE_AA)
    cv2.putText(vis, "right_shoulder", (right_xy[0] + 8, right_xy[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (232, 121, 249), 1, cv2.LINE_AA)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), vis)


def save_shoulder_3d_plot(
    *,
    left_xyz: np.ndarray,
    right_xyz: np.ndarray,
    output_path: Path,
) -> None:
    points = np.vstack([left_xyz, right_xyz]).astype(np.float64)
    fig = plt.figure(figsize=(8, 7), dpi=150)
    ax = fig.add_subplot(111, projection="3d")

    ax.plot(points[:, 0], points[:, 1], points[:, 2], c="#2563eb", linewidth=2.5, label="shoulder width")
    ax.scatter([left_xyz[0]], [left_xyz[1]], [left_xyz[2]], s=60, c="#22c55e", label="left_shoulder")
    ax.scatter([right_xyz[0]], [right_xyz[1]], [right_xyz[2]], s=60, c="#e879f9", label="right_shoulder")

    center = points.mean(axis=0)
    radius = float(max(np.max(np.ptp(points, axis=0)) * 0.65, 0.08))
    ax.set_xlim(float(center[0] - radius), float(center[0] + radius))
    ax.set_ylim(float(center[1] - radius), float(center[1] + radius))
    ax.set_zlim(float(center[2] - radius), float(center[2] + radius))
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z depth (m)")
    ax.set_title("3D shoulder width")
    ax.legend(loc="best")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def process_shoulder_view(
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
    left_shoulder = pick_joint(joint_map, "left_shoulder", min_confidence=float(args.shoulder_min_confidence))
    right_shoulder = pick_joint(joint_map, "right_shoulder", min_confidence=float(args.shoulder_min_confidence))
    left_uv = joint_to_uv(left_shoulder)
    right_uv = joint_to_uv(right_shoulder)

    left_xyz, left_cloud_uv, left_sampling = sample_joint_xyz_from_point_cloud(
        points_xyz=points_xyz,
        joint_uv_rgb=left_uv,
        rgb_shape=(rgb_h, rgb_w),
        cloud_shape=(depth_h, depth_w),
        window_size=int(args.depth_window),
    )
    right_xyz, right_cloud_uv, right_sampling = sample_joint_xyz_from_point_cloud(
        points_xyz=points_xyz,
        joint_uv_rgb=right_uv,
        rgb_shape=(rgb_h, rgb_w),
        cloud_shape=(depth_h, depth_w),
        window_size=int(args.depth_window),
    )

    shoulder_width_m = float(np.linalg.norm(left_xyz - right_xyz))
    prefix = f"{view_name}_"
    overlay_path = output_dir / f"{prefix}shoulder_width_overlay.png"
    plot_3d_path = output_dir / f"{prefix}shoulder_width_3d.png"
    save_shoulder_overlay(image_bgr, left_uv=left_uv, right_uv=right_uv, output_path=overlay_path)
    save_shoulder_3d_plot(left_xyz=left_xyz, right_xyz=right_xyz, output_path=plot_3d_path)

    return {
        "view_name": view_name,
        "inputs": {
            "rgb": str(rgb_path.resolve()),
            "depth": str(depth_path.resolve()),
            "point_cloud_ply": str(ply_path.resolve()),
        },
        "detection": detection_meta,
        "shoulders": {
            "left_shoulder": {
                "uv": np.round(left_uv, 3).tolist(),
                "uv_on_point_cloud": [int(left_cloud_uv[0]), int(left_cloud_uv[1])],
                "xyz_meter": np.round(left_xyz, 6).tolist(),
                "confidence": float(left_shoulder["confidence"]),
                "sampling": left_sampling,
            },
            "right_shoulder": {
                "uv": np.round(right_uv, 3).tolist(),
                "uv_on_point_cloud": [int(right_cloud_uv[0]), int(right_cloud_uv[1])],
                "xyz_meter": np.round(right_xyz, 6).tolist(),
                "confidence": float(right_shoulder["confidence"]),
                "sampling": right_sampling,
            },
        },
        "shoulder_width_m": float(round(shoulder_width_m, 6)),
        "artifacts": {
            "shoulder_overlay_png": str(overlay_path.resolve()),
            "shoulder_width_3d_png": str(plot_3d_path.resolve()),
        },
        "joints": joints,
    }


def main() -> None:
    args = parse_args()
    input_views = resolve_input_views(args)
    validate_input_view_paths(input_views)
    if args.depth_window < 1 or args.depth_window % 2 == 0:
        raise RuntimeError("--depth-window must be a positive odd integer.")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    pose_model = YOLO(str(args.pose_model))
    view_results = {
        view_name: process_shoulder_view(
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

    widths = [float(view_result["shoulder_width_m"]) for view_result in view_results.values()]
    measurement_summary = {
        f"{view_name}_shoulder_width_m": float(view_result["shoulder_width_m"])
        for view_name, view_result in view_results.items()
    }
    measurement_summary["shoulder_width_mean_m"] = float(round(float(np.mean(widths)), 6))

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
            "shoulder_min_confidence": float(args.shoulder_min_confidence),
            "depth_window": int(args.depth_window),
        },
        "views": view_results,
        "shoulder_widths": measurement_summary,
        "measurement_summary": measurement_summary,
        "artifacts": {
            "joints_json": str(summary_path.resolve()),
        },
    }
    summary_path.write_text(json.dumps(output, ensure_ascii=True, indent=2), encoding="utf-8")

    for view_name, view_result in view_results.items():
        detection_meta = view_result["detection"]
        left = view_result["shoulders"]["left_shoulder"]
        right = view_result["shoulders"]["right_shoulder"]
        print(f"[{view_name}] Selected person index: {detection_meta['selected_person_index']}")
        print(f"[{view_name}] left_shoulder: uv={left['uv']}, conf={float(left['confidence']):.4f}")
        print(f"[{view_name}] right_shoulder: uv={right['uv']}, conf={float(right['confidence']):.4f}")
        print(f"[{view_name}] Shoulder width (m): {float(view_result['shoulder_width_m']):.6f}")
    print(f"shoulder_width_mean_m: {measurement_summary['shoulder_width_mean_m']:.6f}")
    print(f"Saved joints: {summary_path.resolve()}")


if __name__ == "__main__":
    main()
