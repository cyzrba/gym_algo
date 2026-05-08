from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession

from src.app.model.task import Task
from src.app.schemas.task import TASK_STATUS_TRANSITIONS, TaskStatus


async def create_task(session: AsyncSession, task_id: str) -> Task:
    task = Task(task_id=task_id, status=TaskStatus.PENDING.value)
    session.add(task)
    await session.commit()
    await session.refresh(task)
    return task


async def get_task(session: AsyncSession, task_id: str) -> Task | None:
    return await session.get(Task, task_id)


async def update_task(session: AsyncSession, task_id: str, measurements: dict[str, object]) -> Task | None:
    task = await session.get(Task, task_id)
    if task is None:
        return None
    task.measurements = measurements
    session.add(task)
    await session.commit()
    await session.refresh(task)
    return task


async def transition_task_status(session: AsyncSession, task_id: str, new_status: str) -> Task:
    task = await session.get(Task, task_id)
    if task is None:
        raise ValueError(f"Task not found: {task_id}")

    current = TaskStatus(task.status)
    target = TaskStatus(new_status)

    if current == target:
        return task

    allowed = TASK_STATUS_TRANSITIONS.get(current, set())
    if target not in allowed:
        raise ValueError(f"Invalid transition: {current.value} → {target.value}")

    task.status = target.value
    session.add(task)
    await session.commit()
    await session.refresh(task)
    return task


async def save_measurement_result(session: AsyncSession, task_id: str, result: object) -> Task:
    task = await session.get(Task, task_id)
    if task is None:
        raise ValueError(f"Task not found: {task_id}")

    current = TaskStatus(task.status)
    if current != TaskStatus.PROCESSING:
        raise ValueError(f"Cannot save result: task is {current.value}, expected processing")

    task.measurements = result.model_dump() if hasattr(result, "model_dump") else result
    task.status = TaskStatus.SUCCESS.value
    session.add(task)
    await session.commit()
    await session.refresh(task)
    return task


async def delete_task(session: AsyncSession, task_id: str) -> bool:
    task = await session.get(Task, task_id)
    if task is None:
        return False
    await session.delete(task)
    await session.commit()
    return True
