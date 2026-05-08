from __future__ import annotations

from sqlalchemy import Column, JSON
from sqlmodel import SQLModel, Field

from src.app.schemas.task import TaskStatus


class Task(SQLModel, table=True):
    __tablename__ = "tasks"
    __table_args__ = {"extend_existing": True}

    task_id: str = Field(primary_key=True, max_length=32)
    status: str = Field(default=TaskStatus.PENDING.value, max_length=16)
    measurements: dict | None = Field(default=None, sa_column=Column(JSON, nullable=True))
