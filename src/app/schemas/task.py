from __future__ import annotations

from enum import Enum

from sqlmodel import SQLModel
from .measurements import MeasurementResult


class TaskStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    SUCCESS = "success"
    FAIL = "fail"
    CANCEL = "cancel"


TASK_STATUS_TRANSITIONS: dict[TaskStatus, set[TaskStatus]] = {
    TaskStatus.PENDING: {TaskStatus.PROCESSING, TaskStatus.CANCEL},
    TaskStatus.PROCESSING: {TaskStatus.SUCCESS, TaskStatus.FAIL, TaskStatus.CANCEL},
    TaskStatus.SUCCESS: set(),
    TaskStatus.FAIL: set(),
    TaskStatus.CANCEL: set(),
}


class TaskCreate(SQLModel):
    pass


class TaskUpdate(SQLModel):
    measurements: MeasurementResult


class TaskStatusUpdate(SQLModel):
    status: str


class TaskResponse(SQLModel):
    task_id: str
    status: str
    measurements: MeasurementResult | None = None

