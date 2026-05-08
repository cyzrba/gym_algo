from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Config:
    front_rgb: Path | None = None
    front_depth: Path | None = None
    front_ply: Path | None = None
    back_rgb: Path | None = None
    back_depth: Path | None = None
    back_ply: Path | None = None
    output_dir: Path = Path("result")

    # Common defaults
    rgb_width: int = 640
    rgb_height: int = 480
    rgb_format: str = "rgb8"
    depth_width: int = 640
    depth_height: int = 480
    depth_dtype: str = "uint16"
    depth_endian: str = "little"
    depth_window: int = 5
    depth_scale: float = 0.001

    # Arm
    arm_side: str = "both"
    arm_center_ratio: float = 1.0 / 3.0
    depth_margin_m: float = 0.08
    slice_half_thickness_m: float = 0.01
    slice_radius_m: float = 0.18
    deviation_iqr_scale: float = 1.8
    deviation_min_keep_ratio: float = 0.35
    deviation_min_points: int = 20

    # Leg
    leg_side: str = "both"
    hip_min_confidence: float = 0.20
    knee_min_confidence: float = 0.03
    thigh_center_ratio: float = 2.0 / 3.0
    leg_depth_margin_m: float = 0.08
    leg_slice_half_thickness_m: float = 0.012
    leg_slice_radius_m: float = 0.24
    leg_deviation_iqr_scale: float = 1.8
    leg_deviation_min_keep_ratio: float = 0.35
    leg_deviation_min_points: int = 20
    hip_midline_margin_m: float = 0.02

    # Waist
    waist_up_ratio: float = 0.25
    waist_depth_margin_m: float = 0.15
    waist_half_height_m: float = 0.04
    waist_radius_m: float = 0.35
    visible_keep_ratio: float = 0.65
    disable_robust_waist_filter: bool = False
    waist_filter_k: int = 14
    waist_filter_std_ratio: float = 2.2

    # Shoulder
    shoulder_min_confidence: float = 0.20

    # Optional metadata
    fx: float | None = None
    fy: float | None = None
    cx: float | None = None
    cy: float | None = None
