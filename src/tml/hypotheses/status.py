from __future__ import annotations

from pathlib import Path

from .model import hypothesis_dirs


def filesystem_counts(project_dir: Path) -> dict[str, int]:
    hdirs = hypothesis_dirs(project_dir)
    materialized = 0
    complete = 0
    incomplete = 0
    for hdir in hdirs:
        if list((hdir / "materializations").glob("*.py")):
            materialized += 1
    for node_dir in (project_dir / "runs").glob("*/artifacts/*"):
        if (node_dir / "node.done.yaml").exists():
            complete += 1
        elif (node_dir / "node.start.yaml").exists():
            incomplete += 1
    return {
        "hypotheses": len(hdirs),
        "materialized": materialized,
        "evaluated": complete,
        "incomplete": incomplete,
    }
