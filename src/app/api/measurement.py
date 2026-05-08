from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from sqlalchemy.ext.asyncio import AsyncSession

from src.app.schemas.task import TaskResponse
from src.app.services.measurement import process_measurement
from src.app.services.task import get_task, save_measurement_result, transition_task_status
from src.app.schemas.task import TaskStatus
from src.app.utils.database import get_session

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/measurements", tags=["measurements"])


@router.post("", response_model=TaskResponse)
async def create_measurement(
    task_id: str = Form(...),
    archive: UploadFile = File(...),
    session: AsyncSession = Depends(get_session),
) -> TaskResponse:
    try:
        task = await get_task(session, task_id)
        if task is None:
            raise HTTPException(status_code=404, detail=f"Task not found: {task_id}")

        await transition_task_status(session, task_id, TaskStatus.PROCESSING.value)

        result = await process_measurement(archive)

        task = await save_measurement_result(session, task_id, result)
        return TaskResponse(task_id=task.task_id, status=task.status, measurements=task.measurements)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Measurement processing failed")
        try:
            await transition_task_status(session, task_id, TaskStatus.FAIL.value)
        except Exception:
            logger.exception("Failed to transition task to FAIL status")
        raise HTTPException(status_code=500, detail="Internal server error")
