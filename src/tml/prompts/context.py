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
    external_description = _external_description(project_dir, project)
    materialization_data_overview = _materialization_data_overview(
        data_overview,
        target_column=project.get("target", {}).get("target_column"),
    )
    if external_description:
        materialization_data_overview = "\n\n".join(
            part for part in (materialization_data_overview, external_description) if part
        )
    return {
        "project_dir": str(project_dir),
        "project": project,
        "task_text": task_text,
        "data_overview": data_overview,
        "external_description": external_description,
        "materialization_data_overview": materialization_data_overview,
        "prior_root_group_results": _existing_hypothesis_memory(project_dir),
        "existing_hypotheses": _existing_hypothesis_memory(project_dir),
        "hypothesis_count": len(enabled_hypotheses(project_dir)),
        "data_dir": str(project.get("data_dir", "data")),
        **extra,
    }


def _existing_hypothesis_memory(project_dir: Path, *, limit: int = 100) -> list[dict[str, object]]:
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


def _external_description(project_dir: Path, project: dict[str, Any]) -> str:
    external = project.get("external") if isinstance(project.get("external"), dict) else {}
    description = external.get("description") or external.get("description_file")
    if not description:
        return ""
    path = Path(str(description))
    if not path.is_absolute():
        path = project_dir / path
    if not path.exists() or not path.is_file():
        return ""
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return ""
    file_name = _display_path(project_dir, external.get("file") or external.get("path") or external.get("aux") or path.name)
    return f"# External Data Description for {file_name}\n\n{text}"


def _display_path(project_dir: Path, value: object) -> str:
    path = Path(str(value))
    if path.is_absolute():
        try:
            return path.relative_to(project_dir).as_posix()
        except ValueError:
            return path.name
    return path.as_posix()


def _materialization_data_overview(data_overview: str, *, target_column: object | None) -> str:
    target_prefix = f"{target_column} " if target_column else None
    kept: list[str] = []
    skipping_file = False
    for line in data_overview.splitlines():
        if line.startswith("-> "):
            skipping_file = "sample_submission.csv" in line
        if skipping_file:
            continue
        if target_prefix and line.startswith(target_prefix):
            continue
        kept.append(line)
    return "\n".join(kept).strip()
