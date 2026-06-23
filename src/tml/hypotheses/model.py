from __future__ import annotations

from pathlib import Path

from tml.utils.yaml_io import read_yaml
from .revisions import latest_revision_record, migrate_hypothesis_dir


def hypothesis_dirs(project_dir: Path) -> list[Path]:
    return sorted(path for path in (project_dir / "hypotheses").glob("*") if path.is_dir())


def enabled_hypotheses(project_dir: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for path in hypothesis_dirs(project_dir):
        migrate_hypothesis_dir(project_dir, path)
        payload = latest_revision_record(path).payload if list(path.glob("??-hypothesis.yaml")) else {}
        if payload and payload.get("enabled", True):
            records.append(payload)
    return records
