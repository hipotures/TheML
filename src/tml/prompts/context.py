from __future__ import annotations

from pathlib import Path
from typing import Any

from tml.branches.algorithms import epsilon_delta, load_branch_algorithm, parse_score_epsilon
from tml.core.config import active_branch_algorithm_id, load_project_config
from tml.db.state import root_hypothesis_rows
from tml.hypotheses.model import enabled_hypotheses


HYPOTHESIS_MEMORY_FIELDS = (
    "hypothesis_id",
    "title",
    "summary",
    "group_name",
    "family",
    "expected_signal",
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
    hypotheses = enabled_hypotheses(project_dir)
    scores = _root_scores_by_hypothesis_id(project_dir, hypotheses)
    baseline = scores.get("000000")
    epsilon = _score_epsilon(project_dir, baseline)
    for hypothesis in hypotheses[-limit:]:
        entry = {key: hypothesis[key] for key in HYPOTHESIS_MEMORY_FIELDS if hypothesis.get(key)}
        if entry.get("summary"):
            entry["prompt_summary"] = _truncate_text(str(entry["summary"]), 250)
        score_result = _score_result_text(
            scores.get(str(hypothesis.get("hypothesis_id") or "")),
            baseline=baseline,
            epsilon=epsilon,
            is_baseline=str(hypothesis.get("hypothesis_id") or "") == "000000",
        )
        if score_result:
            entry["score_result"] = score_result
        if entry:
            memory.append(entry)
    return memory


def _root_scores_by_hypothesis_id(project_dir: Path, hypotheses: list[dict[str, object]]) -> dict[str, float]:
    scores: dict[str, float] = {}
    for hypothesis in hypotheses:
        hypothesis_id = str(hypothesis.get("hypothesis_id") or "")
        score = _optional_float(hypothesis.get("score"))
        if hypothesis_id and score is not None:
            scores[hypothesis_id] = score
    try:
        for row in root_hypothesis_rows(project_dir):
            hypothesis_id = str(row.get("hypothesis_id") or "")
            score = _optional_float(row.get("best_score"))
            if hypothesis_id and score is not None:
                scores[hypothesis_id] = score
    except Exception:
        pass
    return scores


def _score_epsilon(project_dir: Path, baseline: float | None) -> float:
    if baseline is None:
        return 0.0
    try:
        config = load_project_config(project_dir)
        return epsilon_delta(baseline, load_branch_algorithm(project_dir, active_branch_algorithm_id(config)).epsilon)
    except Exception:
        return epsilon_delta(baseline, parse_score_epsilon(None))


def _score_result_text(
    score: float | None,
    *,
    baseline: float | None,
    epsilon: float,
    is_baseline: bool,
) -> str:
    if score is None or baseline is None or is_baseline:
        return ""
    delta = score - baseline
    if abs(delta) < epsilon or delta == 0:
        return "score near baseline"
    if delta > 0:
        return "score above baseline"
    return "score below baseline"


def _optional_float(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return float(value)
    return None


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
