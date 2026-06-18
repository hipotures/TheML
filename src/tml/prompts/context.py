from __future__ import annotations

from pathlib import Path
from typing import Any

from tml.core.config import load_project_config


def project_prompt_context(project_dir: Path, **extra: Any) -> dict[str, Any]:
    project = load_project_config(project_dir)
    task_file = project_dir / str(project.get("task_file", "task.md"))
    task_text = task_file.read_text(encoding="utf-8") if task_file.exists() else ""
    return {
        "project": project,
        "task_text": task_text,
        "data_dir": project_dir / str(project.get("data_dir", "data")),
        **extra,
    }
