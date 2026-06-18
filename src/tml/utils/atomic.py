from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import yaml


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def atomic_write_yaml(path: Path, payload: dict[str, Any]) -> None:
    text = yaml.safe_dump(payload, sort_keys=False, allow_unicode=False)
    atomic_write_text(path, text)
