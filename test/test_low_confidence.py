"""Tests for low-confidence graceful degradation.

Verifies that LowConfidenceError from pick_joint is caught at each
measurement module and results in None fields rather than exceptions.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.algo.base import LowConfidenceError, MeasurementBase
from src.algo.arm import ArmMeasurement
from src.algo.leg import LegMeasurement
from src.algo.waist import WaistMeasurement
from src.algo.shoulder import ShoulderMeasurement
from src.algo.body import BodyMeasurement
from src.algo.config import Config
from src.app.schemas.measurements import (
    ArmMeasurement as ArmSchema,
    LegMeasurement as LegSchema,
    WaistMeasurement as WaistSchema,
    ShoulderMeasurement as ShoulderSchema,
    MeasurementResult,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_joint(name: str, confidence: float = 0.90) -> dict[str, object]:
    return {"name": name, "x": 100.0, "y": 100.0, "confidence": confidence, "visible": confidence >= 0.20}


def _make_joint_map(overrides: dict[str, float] | None = None) -> dict[str, dict[str, object]]:
    """Build a full COCO joint map; ``overrides`` sets per-joint confidence."""
    from src.app.utils.constants import COCO_KEYPOINT_NAMES
    conf = overrides or {}
    return {name: _make_joint(name, conf.get(name, 0.90)) for name in COCO_KEYPOINT_NAMES}


# ---------------------------------------------------------------------------
# 1) LowConfidenceError is a RuntimeError subclass
# ---------------------------------------------------------------------------

class TestLowConfidenceError:
    def test_is_runtime_error(self):
        err = LowConfidenceError("left_hip", 0.05, 0.20)
        assert isinstance(err, RuntimeError)

    def test_attributes(self):
        err = LowConfidenceError("left_hip", 0.05, 0.20)
        assert err.joint_name == "left_hip"
        assert err.confidence == 0.05
        assert err.min_confidence == 0.20

    def test_message(self):
        err = LowConfidenceError("left_hip", 0.05, 0.20)
        assert "left_hip" in str(err)
        assert "0.0500" in str(err)


# ---------------------------------------------------------------------------
# 2) pick_joint raises LowConfidenceError (not generic RuntimeError)
# ---------------------------------------------------------------------------

class TestPickJoint:
    def test_raises_low_confidence_error(self):
        jmap = _make_joint_map({"left_hip": 0.01})
        with pytest.raises(LowConfidenceError) as exc_info:
            MeasurementBase.pick_joint(jmap, "left_hip", min_confidence=0.20)
        assert exc_info.value.joint_name == "left_hip"

    def test_passes_when_confidence_ok(self):
        jmap = _make_joint_map({"left_hip": 0.50})
        joint = MeasurementBase.pick_joint(jmap, "left_hip", min_confidence=0.20)
        assert joint["name"] == "left_hip"

    def test_missing_joint_still_runtime_error(self):
        jmap = _make_joint_map()
        with pytest.raises(RuntimeError, match="Joint not found"):
            MeasurementBase.pick_joint(jmap, "nonexistent_joint", min_confidence=0.10)


# ---------------------------------------------------------------------------
# 3) Per-part degradation: each module returns None on low confidence
# ---------------------------------------------------------------------------

class _FakeMeasurement:
    """Mixin providing a minimal config and a no-op pose_model."""

    def _make_instance(self, cls):
        cfg = Config()
        obj = object.__new__(cls)
        obj.config = cfg
        obj.pose_model = None
        return obj


class TestArmDegradation(_FakeMeasurement):
    def test_single_side_low_confidence_returns_none(self):
        inst = self._make_instance(ArmMeasurement)
        jmap = _make_joint_map({"left_shoulder": 0.01})
        result = inst._measure_arm_side(
            arm_side="left",
            points_xyz=np.zeros((100, 3)),
            rgb_shape=(480, 640),
            cloud_shape=(480, 640),
            joint_map=jmap,
        )
        assert result is None

    def test_aggregate_left_low_right_low(self):
        """Both sides low confidence → both fields None, no exception."""
        inst = self._make_instance(ArmMeasurement)
        inst.config.arm_side = "both"
        low_both = _make_joint_map({"left_shoulder": 0.01, "right_shoulder": 0.01})
        view = {
            "points_xyz": np.zeros((480 * 640, 3)),
            "rgb_shape": (480, 640),
            "cloud_shape": (480, 640),
            "joint_map": low_both,
        }
        result = inst.aggregate([view])
        assert isinstance(result, ArmSchema)
        assert result.left_arc is None
        assert result.right_arc is None

    def test_aggregate_one_side_low(self):
        """Only left low confidence → left None; right may also be None due to
        zero point cloud, but no exception is raised."""
        inst = self._make_instance(ArmMeasurement)
        inst.config.arm_side = "left"
        low_left = _make_joint_map({"left_shoulder": 0.01})
        view = {
            "points_xyz": np.zeros((480 * 640, 3)),
            "rgb_shape": (480, 640),
            "cloud_shape": (480, 640),
            "joint_map": low_left,
        }
        result = inst.aggregate([view])
        assert isinstance(result, ArmSchema)
        assert result.left_arc is None


class TestLegDegradation(_FakeMeasurement):
    def test_single_side_low_confidence_returns_none(self):
        inst = self._make_instance(LegMeasurement)
        jmap = _make_joint_map({"left_hip": 0.01})
        result = inst._measure_leg_side(
            leg_side="left",
            points_xyz=np.zeros((100, 3)),
            rgb_shape=(480, 640),
            cloud_shape=(480, 640),
            joint_map=jmap,
        )
        assert result is None


class TestWaistDegradation(_FakeMeasurement):
    def test_low_confidence_returns_none(self):
        inst = self._make_instance(WaistMeasurement)
        jmap = _make_joint_map({"left_shoulder": 0.01})
        result = inst._measure_waist(
            points_xyz=np.zeros((100, 3)),
            rgb_shape=(480, 640),
            cloud_shape=(480, 640),
            joint_map=jmap,
        )
        assert result is None

    def test_aggregate_returns_none_waist_arc(self):
        inst = self._make_instance(WaistMeasurement)
        jmap = _make_joint_map({"left_shoulder": 0.01})
        view = {
            "points_xyz": np.zeros((100, 3)),
            "rgb_shape": (480, 640),
            "cloud_shape": (480, 640),
            "joint_map": jmap,
        }
        result = inst.aggregate([view])
        assert isinstance(result, WaistSchema)
        assert result.waist_arc is None


class TestShoulderDegradation(_FakeMeasurement):
    def test_low_confidence_returns_none(self):
        inst = self._make_instance(ShoulderMeasurement)
        jmap = _make_joint_map({"left_shoulder": 0.01})
        result = inst._measure_shoulder_width(
            points_xyz=np.zeros((100, 3)),
            rgb_shape=(480, 640),
            cloud_shape=(480, 640),
            joint_map=jmap,
        )
        assert result is None

    def test_aggregate_returns_none_width(self):
        inst = self._make_instance(ShoulderMeasurement)
        jmap = _make_joint_map({"left_shoulder": 0.01})
        view = {
            "points_xyz": np.zeros((100, 3)),
            "rgb_shape": (480, 640),
            "cloud_shape": (480, 640),
            "joint_map": jmap,
        }
        result = inst.aggregate([view])
        assert isinstance(result, ShoulderSchema)
        assert result.shoulder_width is None


# ---------------------------------------------------------------------------
# 4) Non-confidence errors still propagate
# ---------------------------------------------------------------------------

class TestNonConfidenceErrorsPropagated:
    def test_missing_joint_still_raises(self):
        """RuntimeError for missing joint (not low confidence) must not be swallowed."""
        jmap = {}  # empty → Joint not found
        inst = object.__new__(ArmMeasurement)
        inst.config = Config()
        inst.pose_model = None
        with pytest.raises(RuntimeError, match="Joint not found"):
            inst._measure_arm_side_inner(
                arm_side="left",
                points_xyz=np.zeros((100, 3)),
                rgb_shape=(480, 640),
                cloud_shape=(480, 640),
                joint_map=jmap,
            )


# ---------------------------------------------------------------------------
# 5) MeasurementResult can hold partial None
# ---------------------------------------------------------------------------

class TestMeasurementResultPartial:
    def test_all_none(self):
        r = MeasurementResult(arm=None, leg=None, waist=None, shoulder=None)
        d = r.model_dump()
        assert d["arm"] is None
        assert d["leg"] is None
        assert d["waist"] is None
        assert d["shoulder"] is None

    def test_mixed(self):
        r = MeasurementResult(
            arm=ArmSchema(left_arc=0.25, right_arc=None),
            leg=None,
            waist=WaistSchema(waist_arc=0.80),
            shoulder=None,
        )
        d = r.model_dump()
        assert d["arm"]["left_arc"] == 0.25
        assert d["arm"]["right_arc"] is None
        assert d["leg"] is None
        assert d["waist"]["waist_arc"] == 0.80
        assert d["shoulder"] is None
