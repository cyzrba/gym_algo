from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal


@dataclass(frozen=True)
class SavedInputs:
    input_mode: Literal["front_back", "single"]
    paths: dict[str, Path]
