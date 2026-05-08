from __future__ import annotations

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"
DB_PATH = DATA_DIR / "app.db"
SQLITE_URL = f"sqlite+aiosqlite:///{DB_PATH}"
