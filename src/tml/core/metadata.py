from __future__ import annotations

import json
import csv
import gzip
from collections.abc import Callable
from pathlib import Path
from typing import Any

from tml.ai import ModelInvocation, run_model_invocation
from tml.core.config import repo_root_for_project
from tml.core.errors import TmlError
from tml.core.kaggle_pages import fetch_competition_pages
from tml.prompts.renderer import render_template


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


def _progress(progress: Callable[[str], None] | None, message: str) -> None:
    if progress is not None:
        progress(message)


def detect_project_metadata(
    project_dir: Path,
    *,
    slug: str,
    model: str,
    sample_submission_header: list[str],
    progress: Callable[[str], None] | None = None,
    providers: dict[str, object] | None = None,
    role_options: dict[str, object] | None = None,
) -> dict[str, Any] | None:
    try:
        pages = fetch_competition_pages(slug, progress=progress)
    except Exception:
        return None
    if not pages:
        return None
    _progress(progress, "Rendering project metadata prompt...")
    rendered = render_project_metadata_prompt(
        project_dir,
        slug=slug,
        pages=pages,
        sample_submission_header=sample_submission_header,
        progress=progress,
    )
    out_dir = project_dir / "logs" / "project-metadata"
    out_dir.mkdir(parents=True, exist_ok=True)
    _progress(progress, f"Extracting project metadata with model {model}...")
    try:
        response = run_model_invocation(
            ModelInvocation(
                role="metadata",
                model=model,
                prompt=rendered["rendered"],
                template_id=rendered["template_id"],
                template_path=rendered["template_path"],
                template_hash=rendered["template_hash"],
                rendered_prompt_hash=rendered["rendered_hash"],
                cwd=repo_root_for_project(project_dir),
                sandbox="read_only",
                metadata={"kind": "project-metadata"},
                progress=progress,
            ),
            artifact_dir=out_dir,
            providers=providers,
            role_options=role_options,
        )
    except Exception as exc:
        raise TmlError(
            f"Project metadata extraction failed with models.metadata={model!r}. "
            f"Set models.metadata in tml.yaml to mock or provider:model[:effort], "
            f"for example codex:gpt-5.4:low. Cause: {exc}"
        ) from exc
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
    _progress(progress, "EDA...")
    return render_template(
        project_dir,
        "project.metadata",
        {
        "slug": slug,
        "pages": resolved_pages,
        "sample_submission_header": sample_submission_header or _sample_submission_header(project_dir),
        "data_overview": _data_overview(project_dir),
        "sklearn_to_autogluon_metrics": dict(sorted(SKLEARN_TO_AUTOGLUON_METRIC.items())),
        "autogluon_metrics": sorted(AUTOGLUON_METRICS),
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
    raw_metric = _optional_string(target.get("autogluon_metric"))
    legacy_metric = _optional_string(target.get("metric"))
    description = _optional_string(target.get("metric_description"))
    maximize = target.get("maximize")
    maximize = maximize if isinstance(maximize, bool) else True
    sklearn_metric = _optional_string(target.get("sklearn_metric"))

    if legacy_metric and legacy_metric.startswith("sklearn.metrics.") and not sklearn_metric:
        sklearn_metric = legacy_metric
    elif legacy_metric and not raw_metric:
        raw_metric = legacy_metric

    if sklearn_metric:
        mapped = SKLEARN_TO_AUTOGLUON_METRIC.get(sklearn_metric)
        if mapped:
            return {
                "autogluon_metric": mapped,
                "sklearn_metric": sklearn_metric,
                "metric_description": description,
                "maximize": maximize,
            }
        if raw_metric and raw_metric in AUTOGLUON_METRICS:
            return {
                "autogluon_metric": raw_metric,
                "sklearn_metric": sklearn_metric,
                "metric_description": description,
                "maximize": maximize,
            }
        return {
            "autogluon_metric": "custom",
            "sklearn_metric": sklearn_metric,
            "metric_description": description or f"Unmapped sklearn metric: {sklearn_metric}",
            "maximize": maximize,
        }

    if raw_metric == "custom":
        return {
            "autogluon_metric": "custom",
            "sklearn_metric": sklearn_metric,
            "metric_description": description,
            "maximize": maximize,
        }

    if raw_metric in AUTOGLUON_METRICS:
        return {
            "autogluon_metric": raw_metric,
            "sklearn_metric": sklearn_metric,
            "metric_description": description,
            "maximize": maximize,
        }

    return {
        "autogluon_metric": raw_metric,
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


def _data_overview(project_dir: Path) -> str:
    data_dir = _project_data_dir(project_dir)
    files = sorted(
        path
        for path in data_dir.iterdir()
        if path.is_file() and (path.suffix == ".csv" or path.name.endswith(".csv.gz"))
    ) if data_dir.exists() else []
    if not files:
        return "No local CSV data files found."
    sections = [_summarize_csv_file(path) for path in files]
    return "\n\n".join(section for section in sections if section).strip()


def _project_data_dir(project_dir: Path) -> Path:
    config_path = project_dir / "project.yaml"
    if not config_path.exists():
        return project_dir / "data"
    from tml.utils.yaml_io import read_yaml

    config = read_yaml(config_path)
    return project_dir / str(config.get("data_dir") or "data")


def _summarize_csv_file(path: Path) -> str:
    try:
        opener = gzip.open if path.name.endswith(".gz") else Path.open
        with opener(path, mode="rt", newline="", encoding="utf-8") as handle:
            reader = csv.reader(handle)
            header = next(reader, [])
            columns = [_ColumnSummary(name) for name in header]
            row_count = 0
            for row in reader:
                row_count += 1
                for index, column in enumerate(columns):
                    column.add(row[index] if index < len(row) else "")
    except Exception:
        return f"-> {_display_data_name(path)} could not be summarized."

    lines = [f"-> {_display_data_name(path)} has {row_count} rows and {len(header)} columns."]
    if columns:
        lines.append("Here is some information about the columns:")
        lines.extend(column.render() for column in columns)
    return "\n".join(lines)


class _ColumnSummary:
    _MAX_UNIQUE_VALUES = 10
    _NAN_VALUES = {"", "na", "n/a", "nan", "none", "null"}

    def __init__(self, name: str) -> None:
        self.name = name
        self.nan_count = 0
        self.non_null_count = 0
        self.int_possible = True
        self.float_possible = True
        self.min_value: float | None = None
        self.max_value: float | None = None
        self.unique_values: list[str] = []
        self._unique_seen: set[str] = set()
        self._unique_overflow = False

    def add(self, raw_value: str) -> None:
        value = raw_value.strip()
        if value.lower() in self._NAN_VALUES:
            self.nan_count += 1
            return

        self.non_null_count += 1
        if not self._unique_overflow and value not in self._unique_seen:
            self._unique_seen.add(value)
            self.unique_values.append(value)
            if len(self.unique_values) > self._MAX_UNIQUE_VALUES:
                self.unique_values.clear()
                self._unique_seen.clear()
                self._unique_overflow = True

        if self.int_possible:
            try:
                int(value)
            except ValueError:
                self.int_possible = False

        if self.float_possible:
            try:
                numeric = float(value)
            except ValueError:
                self.float_possible = False
            else:
                self.min_value = numeric if self.min_value is None else min(self.min_value, numeric)
                self.max_value = numeric if self.max_value is None else max(self.max_value, numeric)

    def render(self) -> str:
        dtype = self._dtype()
        prefix = f"{self.name} ({dtype})"
        if self.non_null_count == 0:
            return f"{prefix} has only nan values: {self.nan_count} nan values"
        if self.unique_values and not self._unique_overflow:
            return (
                f"{prefix} has {len(self.unique_values)} unique values: "
                f"{_format_unique_values(self.unique_values, dtype)}, {self.nan_count} nan values"
            )
        if dtype in {"int64", "float64"} and self.min_value is not None and self.max_value is not None:
            return (
                f"{prefix} has range: {_format_number(self.min_value)} - {_format_number(self.max_value)}, "
                f"{self.nan_count} nan values"
            )
        return f"{prefix} has {self.nan_count} nan values"

    def _dtype(self) -> str:
        if self.non_null_count == 0:
            return "object"
        if self.int_possible:
            return "int64"
        if self.float_possible:
            return "float64"
        return "object"


def _display_data_name(path: Path) -> str:
    if path.name.endswith(".csv.gz"):
        return path.name[:-3]
    return path.name


def _format_unique_values(values: list[str], dtype: str) -> str:
    if dtype == "int64":
        return "[" + ", ".join(str(int(value)) for value in values) + "]"
    if dtype == "float64":
        return "[" + ", ".join(_format_number(float(value)) for value in values) + "]"
    return "[" + ", ".join(repr(value) for value in values) + "]"


def _format_number(value: float) -> str:
    return f"{value:.2f}"


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
