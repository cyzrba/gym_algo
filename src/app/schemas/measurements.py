from sqlmodel import SQLModel


class ArmMeasurement(SQLModel):
    left_arc: float | None = None
    right_arc: float | None = None


class LegMeasurement(SQLModel):
    left_arc: float | None = None
    right_arc: float | None = None


class WaistMeasurement(SQLModel):
    waist_arc: float | None = None


class ShoulderMeasurement(SQLModel):
    shoulder_width: float | None = None


class MeasurementResult(SQLModel):
    arm: ArmMeasurement | None = None
    leg: LegMeasurement | None = None
    waist: WaistMeasurement | None = None
    shoulder: ShoulderMeasurement | None = None
