from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from .atomic import atomic_write_yaml


def read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}


def write_yaml(path: Path, payload: dict[str, Any]) -> None:
    atomic_write_yaml(path, payload)
