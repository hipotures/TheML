from __future__ import annotations

from importlib import resources
from pathlib import Path

from .connect import connect


def migrate(db_path: Path) -> None:
    schema = resources.files("tml.db").joinpath("schema.sql").read_text(encoding="utf-8")
    with connect(db_path) as conn:
        conn.executescript(schema)
