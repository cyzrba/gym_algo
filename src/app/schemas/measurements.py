from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MeasurementParams:
    arm_side: str
    leg_side: str
    rgb_width: int
    rgb_height: int
    rgb_format: str
    depth_width: int
    depth_height: int
    depth_dtype: str
    depth_endian: str
    depth_scale: float
    depth_window: int
