from __future__ import annotations

import json
import csv
import gzip
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

from tml.ai import AiRequest, client_for_model
from tml.core.kaggle_pages import fetch_competition_pages
from tml.prompts.renderer import render_template
from tml.utils.atomic import atomic_write_json, atomic_write_text


SKLEARN_TO_AUTOGLUON_METRIC = {
    "sklearn.metrics.accuracy_score": "accuracy",
    "sklearn.metrics.balanced_accuracy_score": "balanced_accuracy",
    "sklearn.metrics.roc_auc_score": "roc_auc",
    "sklearn.metrics.log_loss": "log_loss",
    "sklearn.metrics.f1_score": "f1",
    "sklearn.metrics.precision_score": "precision",
    "sklearn.metrics.recall_score": "recall",
    "sklearn.metrics.mean_absolute_error": "mean_absolute_error",
    "sklearn.metrics.mean_squared_error": "mean_squared_error",
    "sklearn.metrics.root_mean_squared_error": "root_mean_squared_error",
    "sklearn.metrics.r2_score": "r2",
}

AUTOGLUON_METRICS = set(SKLEARN_TO_AUTOGLUON_METRIC.values()) | {
    "average_precision",
    "f1_macro",
    "f1_micro",
    "f1_weighted",
    "mcc",
    "mean_absolute_percentage_error",
    "median_absolute_error",
    "pac_score",
    "pearsonr",
    "quadratic_kappa",
    "rmse",
    "root_mean_squared_log_error",
    "spearmanr",
}


def detect_project_metadata(
    project_dir: Path,
    *,
    slug: str,
    model: str,
    sample_submission_header: list[str],
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any] | None:
    try:
        pages = fetch_competition_pages(slug, progress=progress)
    except Exception:
        return None
    if not pages:
        return None
    rendered = render_project_metadata_prompt(
        project_dir,
        slug=slug,
        pages=pages,
        sample_submission_header=sample_submission_header,
    )
    out_dir = project_dir / "logs" / "project-metadata"
    out_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_text(out_dir / "request.md", rendered["rendered"])
    atomic_write_json(
        out_dir / "request.json",
        {
            "kind": "project-metadata",
            "model": model,
            "provider": model,
            "messages": [{"role": "user", "content": rendered["rendered"]}],
            "template_id": rendered["template_id"],
            "template_path": rendered["template_path"],
            "template_hash": rendered["template_hash"],
            "rendered_prompt_hash": rendered["rendered_hash"],
            "project_dir": ".",
            "created_at": datetime.now().isoformat(timespec="seconds"),
        },
    )
    response = client_for_model(model).call(
        AiRequest(role="metadata", model=model, prompt=rendered["rendered"])
    )
    atomic_write_text(out_dir / "response.md", response.text)
    atomic_write_json(out_dir / "response.json", {"text": response.text, **response.metadata})
    payload = _parse_json_object(response.text)
    if not payload:
        return None
    payload["pages"] = pages
    return normalize_project_metadata(payload)


def render_project_metadata_prompt(
    project_dir: Path,
    *,
    slug: str,
    pages: dict[str, str] | None = None,
    sample_submission_header: list[str] | None = None,
    progress: Callable[[str], None] | None = None,
) -> dict[str, str]:
    resolved_pages = pages if pages is not None else fetch_competition_pages(slug, progress=progress)
    return render_template(
        project_dir,
        "project.metadata",
        {
            "slug": slug,
            "pages": resolved_pages,
            "sample_submission_header": sample_submission_header or _sample_submission_header(project_dir),
        },
    )


def normalize_project_metadata(payload: dict[str, Any]) -> dict[str, Any]:
    target = payload.get("target") if isinstance(payload.get("target"), dict) else {}
    normalized_target = {
        "id_column": _optional_string(target.get("id_column")),
        "target_column": _optional_string(target.get("target_column")),
        "problem_type": _optional_string(target.get("problem_type")),
        "submission_kind": _optional_string(target.get("submission_kind")),
        **normalize_metric(target),
    }
    return {
        "goal": _optional_string(payload.get("goal")),
        "evaluation": _optional_string(payload.get("evaluation")),
        "data_description": _optional_string(payload.get("data_description")),
        "target": normalized_target,
    }


def normalize_metric(target: dict[str, Any]) -> dict[str, Any]:
    raw_metric = _optional_string(target.get("metric"))
    source = _optional_string(target.get("metric_source"))
    description = _optional_string(target.get("metric_description"))
    maximize = target.get("maximize")
    maximize = maximize if isinstance(maximize, bool) else True
    sklearn_metric = _optional_string(target.get("sklearn_metric"))

    if raw_metric and raw_metric.startswith("sklearn.metrics."):
        sklearn_metric = raw_metric
        mapped = SKLEARN_TO_AUTOGLUON_METRIC.get(raw_metric)
        if mapped:
            return {
                "metric": mapped,
                "metric_source": "autogluon",
                "sklearn_metric": raw_metric,
                "metric_description": description,
                "maximize": maximize,
            }
        return {
            "metric": "custom",
            "metric_source": "custom",
            "sklearn_metric": raw_metric,
            "metric_description": description or f"Unmapped sklearn metric: {raw_metric}",
            "maximize": maximize,
        }

    if raw_metric == "custom" or source == "custom":
        return {
            "metric": "custom",
            "metric_source": "custom",
            "sklearn_metric": sklearn_metric,
            "metric_description": description,
            "maximize": maximize,
        }

    if raw_metric in AUTOGLUON_METRICS:
        return {
            "metric": raw_metric,
            "metric_source": "autogluon",
            "sklearn_metric": sklearn_metric,
            "metric_description": description,
            "maximize": maximize,
        }

    return {
        "metric": raw_metric,
        "metric_source": source,
        "sklearn_metric": sklearn_metric,
        "metric_description": description,
        "maximize": maximize,
    }


def metadata_task_markdown(metadata: dict[str, Any], slug: str) -> str:
    parts = [f"# {slug}", ""]
    for title, key in (
        ("Goal", "goal"),
        ("Evaluation", "evaluation"),
        ("Data description", "data_description"),
    ):
        value = metadata.get(key)
        if value:
            parts.extend([f"## {title}", str(value).strip(), ""])
    return "\n".join(parts).rstrip() + "\n"


def _parse_json_object(text: str) -> dict[str, Any] | None:
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def _sample_submission_header(project_dir: Path) -> list[str]:
    config_path = project_dir / "project.yaml"
    data_dir = project_dir / "data"
    if config_path.exists():
        from tml.utils.yaml_io import read_yaml

        config = read_yaml(config_path)
        data_dir = project_dir / str(config.get("data_dir") or "data")
    sample = _data_file(data_dir, "sample_submission.csv")
    if sample.exists():
        opener = gzip.open if sample.suffix == ".gz" else Path.open
        with opener(sample, mode="rt", newline="", encoding="utf-8") as handle:
            return next(csv.reader(handle), [])
    if config_path.exists():
        from tml.utils.yaml_io import read_yaml

        target = read_yaml(config_path).get("target", {})
        if isinstance(target, dict):
            columns = [target.get("id_column"), target.get("target_column")]
            return [str(column) for column in columns if column]
    return []


def _data_file(data_dir: Path, name: str) -> Path:
    plain = data_dir / name
    if plain.exists():
        return plain
    return plain.with_name(plain.name + ".gz")


def _optional_string(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None
