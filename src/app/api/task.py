from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from src.app.schemas.task import TaskResponse, TaskStatusUpdate, TaskUpdate
from src.app.services.task import create_task, delete_task, get_task, transition_task_status, update_task
from src.app.utils.database import get_session

router = APIRouter(prefix="/api/v1/tasks", tags=["tasks"])


@router.post("", status_code=201, response_model=TaskResponse)
async def create(session: AsyncSession = Depends(get_session)):
    try:
        task_id = uuid.uuid4().hex
        task = await create_task(session, task_id)
        return TaskResponse(task_id=task.task_id, status=task.status, measurements=task.measurements)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{task_id}", response_model=TaskResponse)
async def read(task_id: str, session: AsyncSession = Depends(get_session)):
    try:
        task = await get_task(session, task_id)
        if task is None:
            raise HTTPException(status_code=404, detail=f"Task not found: {task_id}")
        return TaskResponse(task_id=task.task_id, status=task.status, measurements=task.measurements)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/{task_id}", response_model=TaskResponse)
async def update(task_id: str, body: TaskUpdate, session: AsyncSession = Depends(get_session)):
    try:
        task = await update_task(session, task_id, body.measurements)
        if task is None:
            raise HTTPException(status_code=404, detail=f"Task not found: {task_id}")
        return TaskResponse(task_id=task.task_id, status=task.status, measurements=task.measurements)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.patch("/{task_id}/status", response_model=TaskResponse)
async def transition_status(task_id: str, body: TaskStatusUpdate, session: AsyncSession = Depends(get_session)):
    try:
        task = await transition_task_status(session, task_id, body.status)
        return TaskResponse(task_id=task.task_id, status=task.status, measurements=task.measurements)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{task_id}", status_code=204)
async def delete(task_id: str, session: AsyncSession = Depends(get_session)):
    try:
        ok = await delete_task(session, task_id)
        if not ok:
            raise HTTPException(status_code=404, detail=f"Task not found: {task_id}")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
