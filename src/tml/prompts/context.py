from __future__ import annotations

from pathlib import Path
from typing import Any

from tml.core.config import load_project_config
from tml.hypotheses.model import enabled_hypotheses


HYPOTHESIS_MEMORY_FIELDS = (
    "hypothesis_id",
    "title",
    "summary",
    "group_name",
    "family",
    "expected_signal",
    "score",
    "status",
    "risk",
)


def project_prompt_context(project_dir: Path, **extra: Any) -> dict[str, Any]:
    project = load_project_config(project_dir)
    task_file = project_dir / str(project.get("task_file", "task.md"))
    task_text = task_file.read_text(encoding="utf-8") if task_file.exists() else ""
    data_overview_file = project_dir / "docs" / "data-overview.md"
    data_overview = data_overview_file.read_text(encoding="utf-8") if data_overview_file.exists() else ""
    return {
        "project_dir": str(project_dir),
        "project": project,
        "task_text": task_text,
        "data_overview": data_overview,
        "prior_root_group_results": _existing_hypothesis_memory(project_dir),
        "existing_hypotheses": _existing_hypothesis_memory(project_dir),
        "hypothesis_count": len(enabled_hypotheses(project_dir)),
        "data_dir": str(project.get("data_dir", "data")),
        **extra,
    }


def _existing_hypothesis_memory(project_dir: Path, *, limit: int = 12) -> list[dict[str, object]]:
    memory: list[dict[str, object]] = []
    for hypothesis in enabled_hypotheses(project_dir)[-limit:]:
        entry = {key: hypothesis[key] for key in HYPOTHESIS_MEMORY_FIELDS if hypothesis.get(key)}
        if entry.get("summary"):
            entry["prompt_summary"] = _truncate_text(str(entry["summary"]), 250)
        if entry:
            memory.append(entry)
    return memory


def _truncate_text(value: str, limit: int) -> str:
    text = " ".join(value.split())
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 1)].rstrip() + "…"
