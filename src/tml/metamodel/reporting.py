from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
from typing import Any

from tml.metamodel.training import MetaRunResult, TargetTrainingResult


@dataclass(frozen=True)
class ReportResult:
    report_md: Path
    report_json: Path


def write_report(run_result: MetaRunResult) -> ReportResult:
    report_json = run_result.output_dir / "report.json"
    report_md = run_result.output_dir / "report.md"
    payload = _report_payload(run_result)
    report_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_md.write_text(_markdown_report(payload), encoding="utf-8")
    return ReportResult(report_md=report_md, report_json=report_json)


def _report_payload(run_result: MetaRunResult) -> dict[str, Any]:
    target_payloads = [_target_payload(result) for result in run_result.target_results]
    cv = next((item for item in target_payloads if item["target"] == "cv_score"), None)
    public = next((item for item in target_payloads if item["target"] == "public_score"), None)
    gap = next((item for item in target_payloads if item["target"] == "public_gap"), None)
    return {
        "schema_version": 1,
        "kind": "tml_meta_model_report",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "output_dir": str(run_result.output_dir),
        "dataset": {
            "record_count": run_result.dataset_result.record_count,
            "cv_score_count": run_result.dataset_result.cv_score_count,
            "public_score_count": run_result.dataset_result.public_score_count,
            "missing_fields": run_result.dataset_result.missing_fields,
            "dataset_csv": str(run_result.dataset_result.dataset_csv),
            "dataset_parquet": str(run_result.dataset_result.dataset_parquet) if run_result.dataset_result.dataset_parquet else None,
            "dataset_fingerprint": run_result.dataset_result.dataset_fingerprint,
        },
        "features": {
            "feature_count": run_result.feature_result.feature_count,
            "feature_csv": str(run_result.feature_result.feature_csv),
            "feature_spec_json": str(run_result.feature_result.feature_spec_json),
            "leakage_policy": run_result.feature_result.leakage_policy,
        },
        "targets": target_payloads,
        "answers": _answers(cv, public, gap),
    }


def _target_payload(result: TargetTrainingResult) -> dict[str, Any]:
    metrics = result.metrics
    aggregate = (
        metrics.get("random_splits", {}).get("aggregate", {})
        if isinstance(metrics.get("random_splits"), dict)
        else {}
    )
    group_metrics = (
        metrics.get("group_split", {}).get("metrics", {})
        if isinstance(metrics.get("group_split"), dict)
        else None
    )
    return {
        "target": result.target,
        "status": result.status,
        "n_examples": result.n_examples,
        "low_confidence": result.low_confidence,
        "output_dir": str(result.output_dir),
        "model_dir": str(result.model_dir) if result.model_dir else None,
        "validation_predictions_csv": str(result.validation_predictions_csv) if result.validation_predictions_csv else None,
        "metrics_json": str(result.metrics_json),
        "leaderboard_csv": str(result.leaderboard_csv) if result.leaderboard_csv else None,
        "feature_importance_csv": str(result.feature_importance_csv) if result.feature_importance_csv else None,
        "random_split_aggregate": aggregate,
        "group_split_metrics": group_metrics,
        "uncertainty": metrics.get("uncertainty"),
        "similarity_diagnostic": metrics.get("similarity_diagnostic"),
    }


def _answers(cv: dict[str, Any] | None, public: dict[str, Any] | None, gap: dict[str, Any] | None) -> dict[str, str]:
    cv_mae = _metric(cv, "mae_mean")
    cv_spearman = _metric(cv, "spearman_mean")
    cv_precision = _metric(cv, "top10_precision_mean")
    cv_recall = _metric(cv, "top10_recall_mean")
    group_mae = _group_metric(cv, "mae")
    public_mae = _metric(public, "mae_mean")
    public_spearman = _metric(public, "spearman_mean")
    cold_mae = _metric(cv, "unseen_group_mae_mean")
    known_mae = _metric(cv, "known_group_mae_mean")

    practical = "nie"
    if cv_mae is not None and cv_spearman is not None and cv_mae <= 0.00025 and cv_spearman >= 0.4:
        practical = "ostroznie jako advisory, nie jako automatyczne pruning"
    if cv_mae is not None and cv_mae > 0.0005:
        practical = "nie"

    return {
        "can_predict_cv": (
            "CV ma sygnal predykcyjny."
            if cv_mae is not None and cv_spearman is not None and cv_spearman > 0.25
            else "CV nie ma wystarczajaco mocnego sygnalu w obecnej walidacji."
        ),
        "real_prediction_error": _fmt_metric("MAE", cv_mae, fallback="Brak pelnych metryk CV."),
        "ranking_quality": (
            f"Spearman={cv_spearman:.4f}, top10 precision={_fmt(cv_precision)}, top10 recall={_fmt(cv_recall)}."
            if cv_spearman is not None
            else "Brak wiarygodnej metryki rankingowej."
        ),
        "public_score_signal": (
            f"Public score jest low-confidence: n={public.get('n_examples')}, MAE={_fmt(public_mae)}, Spearman={_fmt(public_spearman)}."
            if public and public.get("status") == "complete"
            else "Public score ma za malo danych albo model zostal pominiety."
        ),
        "public_gap_signal": (
            f"Public-CV gap model: n={gap.get('n_examples')}, MAE={_fmt(_metric(gap, 'mae_mean'))}."
            if gap and gap.get("status") == "complete"
            else "Public-CV gap nie ma wystarczajacej proby."
        ),
        "cold_start": (
            f"Unseen-group MAE={cold_mae:.6f} vs known-group MAE={known_mae:.6f}."
            if cold_mae is not None and known_mae is not None
            else "Cold-start/unseen-group segment nie jest wystarczajaco pokryty metrykami."
        ),
        "group_split_caution": (
            f"Group split MAE={group_mae:.6f}; porownaj z random split przed decyzja."
            if group_mae is not None
            else "Group split nie byl dostepny lub nie wystarczyl do oceny ostroznosciowej."
        ),
        "future_integration": (
            f"Wartosc praktyczna: {practical}. Ten modul powinien pozostac raportem/advisory do czasu mocniejszej walidacji."
        ),
    }


def _markdown_report(payload: dict[str, Any]) -> str:
    lines = [
        "# AutoGluon Experiment Meta-Model Report",
        "",
        f"Created: {payload['created_at']}",
        f"Output: `{payload['output_dir']}`",
        "",
        "## Dataset",
        "",
        f"- Records: {payload['dataset']['record_count']}",
        f"- CV labels: {payload['dataset']['cv_score_count']}",
        f"- Public labels: {payload['dataset']['public_score_count']}",
        f"- Dataset CSV: `{payload['dataset']['dataset_csv']}`",
        f"- Feature CSV: `{payload['features']['feature_csv']}`",
        f"- Feature count: {payload['features']['feature_count']}",
        "",
        "Leakage policy: " + payload["features"]["leakage_policy"],
        "",
        "## Targets",
        "",
    ]
    for target in payload["targets"]:
        lines.extend(_target_markdown(target))
    lines.extend(
        [
            "## Required Questions",
            "",
        ]
    )
    labels = [
        ("Czy `cv_score` da sie przewidywac?", "can_predict_cv"),
        ("Jaki jest realny blad predykcji?", "real_prediction_error"),
        ("Czy model dobrze szereguje kandydatow?", "ranking_quality"),
        ("Czy public score daje sygnal?", "public_score_signal"),
        ("Czy model radzi sobie z nowymi grupami?", "cold_start"),
        ("Czy walidacja moze byc zawyzona przez podobne node'y?", "group_split_caution"),
        ("Czy warto integrowac advisory/pruning?", "future_integration"),
    ]
    for label, key in labels:
        lines.append(f"- {label} {payload['answers'][key]}")
    lines.append("")
    lines.append("This report is diagnostic only. No workflow pruning or node selection logic was changed.")
    lines.append("")
    return "\n".join(lines)


def _target_markdown(target: dict[str, Any]) -> list[str]:
    lines = [
        f"### {target['target']}",
        "",
        f"- Status: {target['status']}",
        f"- Examples: {target['n_examples']}",
        f"- Low confidence: {str(target['low_confidence']).lower()}",
    ]
    aggregate = target.get("random_split_aggregate") or {}
    for key in ("mae_mean", "rmse_mean", "median_absolute_error_mean", "r2_mean", "spearman_mean", "pearson_mean", "top10_precision_mean", "top10_recall_mean"):
        if key in aggregate:
            lines.append(f"- {key}: {_fmt(aggregate.get(key))}")
    group = target.get("group_split_metrics")
    if isinstance(group, dict) and group:
        lines.append(f"- group_split_mae: {_fmt(group.get('mae'))}")
        lines.append(f"- group_split_spearman: {_fmt(group.get('spearman'))}")
    similarity = target.get("similarity_diagnostic")
    if isinstance(similarity, dict) and similarity.get("available"):
        lines.append(f"- high_similarity_row_rate: {_fmt(similarity.get('high_similarity_row_rate'))}")
        lines.append(f"- p90_nearest_group_jaccard: {_fmt(similarity.get('p90_nearest_group_jaccard'))}")
    if target.get("feature_importance_csv"):
        lines.append(f"- Feature importance: `{target['feature_importance_csv']}`")
    lines.append("")
    return lines


def _metric(target: dict[str, Any] | None, key: str) -> float | None:
    if not target:
        return None
    aggregate = target.get("random_split_aggregate")
    if not isinstance(aggregate, dict):
        return None
    value = aggregate.get(key)
    return float(value) if isinstance(value, int | float) else None


def _group_metric(target: dict[str, Any] | None, key: str) -> float | None:
    if not target:
        return None
    group = target.get("group_split_metrics")
    if not isinstance(group, dict):
        return None
    value = group.get(key)
    return float(value) if isinstance(value, int | float) else None


def _fmt_metric(label: str, value: float | None, *, fallback: str) -> str:
    return fallback if value is None else f"{label}={value:.6f} score units."


def _fmt(value: object) -> str:
    return "n/a" if not isinstance(value, int | float) else f"{float(value):.6f}"
