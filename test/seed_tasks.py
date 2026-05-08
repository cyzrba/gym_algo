from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import app.model  # noqa: F401
from app.utils.database import async_session, init_db
from app.model.task import Task
from sqlmodel import select

SAMPLES = [
    {
        "task_id": "a1b2c3d4e5f60001",
        "measurements": {
            "arm": {"left_arc": 0.253, "right_arc": 0.261},
            "leg": {"left_arc": 0.452, "right_arc": 0.448},
            "waist": {"waist_arc": 0.784},
            "shoulder": {"shoulder_width": 0.392},
        },
    },
    {
        "task_id": "a1b2c3d4e5f60002",
        "measurements": {
            "arm": {"left_arc": 0.280, "right_arc": 0.275},
            "leg": {"left_arc": 0.501, "right_arc": 0.498},
            "waist": {"waist_arc": 0.812},
            "shoulder": {"shoulder_width": 0.410},
        },
    },
    {
        "task_id": "a1b2c3d4e5f60003",
        "measurements": {
            "arm": {"left_arc": 0.221, "right_arc": 0.218},
            "leg": {"left_arc": 0.410, "right_arc": 0.415},
            "waist": {"waist_arc": 0.701},
            "shoulder": {"shoulder_width": 0.365},
        },
    },
]


async def main() -> None:
    await init_db()
    async with async_session() as session:
        for sample in SAMPLES:
            task = Task(task_id=sample["task_id"], measurements=sample["measurements"])
            session.add(task)
        await session.commit()

    async with async_session() as session:
        result = await session.execute(select(Task))
        for t in result.scalars():
            print(f"{t.task_id}  {t.measurements}")


if __name__ == "__main__":
    asyncio.run(main())
