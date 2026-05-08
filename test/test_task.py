from __future__ import annotations

import sys
from pathlib import Path

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlmodel import SQLModel

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from app.model.task import Task
from app.schemas.task import TaskStatus
from app.services.task import create_task, delete_task, get_task, save_measurement_result, transition_task_status, update_task

TEST_DB_URL = "sqlite+aiosqlite://"


@pytest_asyncio.fixture
async def session():
    engine = create_async_engine(TEST_DB_URL, echo=False)
    test_session = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)

    async with test_session() as s:
        yield s

    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.drop_all)
    await engine.dispose()


@pytest.mark.asyncio
async def test_connection(session: AsyncSession):
    result = await session.execute(Task.__table__.select())
    assert result.fetchall() == []


@pytest.mark.asyncio
async def test_create_task(session: AsyncSession):
    task = await create_task(session, "abc123")

    assert task.task_id == "abc123"
    assert task.measurements is None
    assert task.status == TaskStatus.PENDING.value


@pytest.mark.asyncio
async def test_get_task(session: AsyncSession):
    await create_task(session, "abc123")

    task = await get_task(session, "abc123")
    assert task is not None
    assert task.task_id == "abc123"
    assert task.measurements is None


@pytest.mark.asyncio
async def test_get_task_not_found(session: AsyncSession):
    task = await get_task(session, "nonexistent")
    assert task is None


@pytest.mark.asyncio
async def test_update_task(session: AsyncSession):
    await create_task(session, "abc123")

    updated = await update_task(session, "abc123", {"arm": {"left_arc": 0.30}})
    assert updated is not None
    assert updated.measurements == {"arm": {"left_arc": 0.30}}

    task = await get_task(session, "abc123")
    assert task.measurements == {"arm": {"left_arc": 0.30}}


@pytest.mark.asyncio
async def test_update_task_not_found(session: AsyncSession):
    result = await update_task(session, "nonexistent", {"arm": {}})
    assert result is None


@pytest.mark.asyncio
async def test_delete_task(session: AsyncSession):
    await create_task(session, "abc123")

    ok = await delete_task(session, "abc123")
    assert ok is True

    task = await get_task(session, "abc123")
    assert task is None


@pytest.mark.asyncio
async def test_delete_task_not_found(session: AsyncSession):
    ok = await delete_task(session, "nonexistent")
    assert ok is False


@pytest.mark.asyncio
async def test_transition_pending_to_processing(session: AsyncSession):
    await create_task(session, "t1")
    task = await transition_task_status(session, "t1", TaskStatus.PROCESSING.value)
    assert task.status == TaskStatus.PROCESSING.value


@pytest.mark.asyncio
async def test_transition_processing_to_success(session: AsyncSession):
    await create_task(session, "t2")
    await transition_task_status(session, "t2", TaskStatus.PROCESSING.value)
    task = await transition_task_status(session, "t2", TaskStatus.SUCCESS.value)
    assert task.status == TaskStatus.SUCCESS.value


@pytest.mark.asyncio
async def test_transition_processing_to_fail(session: AsyncSession):
    await create_task(session, "t3")
    await transition_task_status(session, "t3", TaskStatus.PROCESSING.value)
    task = await transition_task_status(session, "t3", TaskStatus.FAIL.value)
    assert task.status == TaskStatus.FAIL.value


@pytest.mark.asyncio
async def test_transition_processing_to_cancel(session: AsyncSession):
    await create_task(session, "t4")
    await transition_task_status(session, "t4", TaskStatus.PROCESSING.value)
    task = await transition_task_status(session, "t4", TaskStatus.CANCEL.value)
    assert task.status == TaskStatus.CANCEL.value


@pytest.mark.asyncio
async def test_transition_pending_to_cancel(session: AsyncSession):
    await create_task(session, "t5")
    task = await transition_task_status(session, "t5", TaskStatus.CANCEL.value)
    assert task.status == TaskStatus.CANCEL.value


@pytest.mark.asyncio
async def test_transition_invalid_pending_to_success(session: AsyncSession):
    await create_task(session, "t6")
    with pytest.raises(ValueError, match="Invalid transition"):
        await transition_task_status(session, "t6", TaskStatus.SUCCESS.value)


@pytest.mark.asyncio
async def test_transition_terminal_state(session: AsyncSession):
    await create_task(session, "t7")
    await transition_task_status(session, "t7", TaskStatus.CANCEL.value)
    with pytest.raises(ValueError, match="Invalid transition"):
        await transition_task_status(session, "t7", TaskStatus.PROCESSING.value)


@pytest.mark.asyncio
async def test_transition_task_not_found(session: AsyncSession):
    with pytest.raises(ValueError, match="Task not found"):
        await transition_task_status(session, "nonexistent", TaskStatus.PROCESSING.value)


@pytest.mark.asyncio
async def test_save_measurement_result(session: AsyncSession):
    await create_task(session, "t8")
    await transition_task_status(session, "t8", TaskStatus.PROCESSING.value)
    data = {"arm": {"left_arc": 0.25, "right_arc": 0.26}}
    task = await save_measurement_result(session, "t8", data)
    assert task.status == TaskStatus.SUCCESS.value
    assert task.measurements == data


@pytest.mark.asyncio
async def test_save_measurement_result_not_processing(session: AsyncSession):
    await create_task(session, "t9")
    with pytest.raises(ValueError, match="Cannot save result"):
        await save_measurement_result(session, "t9", {"arm": {}})
