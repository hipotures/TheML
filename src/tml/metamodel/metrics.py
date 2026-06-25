from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any


@dataclass(frozen=True)
class EvaluationSummary:
    metrics: dict[str, float | int | None]
    advisory: list[dict[str, float | int | None]]
    cold_start_metrics: dict[str, float | int | None]


def evaluate_predictions(y_true, y_pred, *, has_unseen_group=None) -> EvaluationSummary:
    np = _require_numpy()
    true = np.asarray(y_true, dtype=float)
    pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(true) & np.isfinite(pred)
    true = true[mask]
    pred = pred[mask]
    metrics = _core_metrics(true, pred)
    metrics.update(_top_decile_metrics(true, pred))
    cold_metrics: dict[str, float | int | None] = {}
    if has_unseen_group is not None:
        cold = np.asarray(has_unseen_group)[mask].astype(bool)
        if cold.size and bool(cold.any()):
            cold_metrics = _prefix_metrics(_core_metrics(true[cold], pred[cold]), "unseen_group")
        if cold.size and bool((~cold).any()):
            cold_metrics.update(_prefix_metrics(_core_metrics(true[~cold], pred[~cold]), "known_group"))
    return EvaluationSummary(metrics=metrics, advisory=advisory_simulation(true, pred), cold_start_metrics=cold_metrics)


def aggregate_metric_dicts(items: list[dict[str, Any]]) -> dict[str, float | int | None]:
    np = _require_numpy()
    keys = sorted({key for item in items for key in item})
    result: dict[str, float | int | None] = {}
    for key in keys:
        values = [float(item[key]) for item in items if isinstance(item.get(key), int | float) and math.isfinite(float(item[key]))]
        if not values:
            result[key] = None
            continue
        result[f"{key}_mean"] = float(np.mean(values))
        result[f"{key}_std"] = float(np.std(values))
        result[f"{key}_min"] = float(np.min(values))
        result[f"{key}_max"] = float(np.max(values))
    return result


def advisory_simulation(y_true, y_pred) -> list[dict[str, float | int | None]]:
    np = _require_numpy()
    true = np.asarray(y_true, dtype=float)
    pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(true) & np.isfinite(pred)
    true = true[mask]
    pred = pred[mask]
    if true.size == 0:
        return []
    true_top_threshold = float(np.quantile(true, 0.9))
    best_all = float(np.max(true))
    rows: list[dict[str, float | int | None]] = []
    for quantile in (0.1, 0.25, 0.5):
        threshold = float(np.quantile(pred, quantile))
        skipped = pred < threshold
        saved = int(skipped.sum())
        if saved == 0:
            false_skips = 0
            false_skip_rate = 0.0
        else:
            false_skips = int(((true >= true_top_threshold) & skipped).sum())
            false_skip_rate = false_skips / saved
        kept_true = true[~skipped]
        best_kept = float(np.max(kept_true)) if kept_true.size else None
        rows.append(
            {
                "threshold_quantile": quantile,
                "prediction_threshold": threshold,
                "candidate_skip_rate": saved / int(true.size),
                "false_skip_rate": false_skip_rate,
                "missed_top_candidate_count": false_skips,
                "best_score_loss": None if best_kept is None else best_all - best_kept,
                "saved_training_count": saved,
            }
        )
    return rows


def similarity_diagnostic(records, predictions) -> dict[str, Any]:
    np = _require_numpy()
    by_node = {record.node_id: {component.logical_key for component in record.components} for record in records}
    rows: list[float] = []
    for item in predictions:
        node_id = str(item.get("node_id") or "")
        train_ids = item.get("train_node_ids")
        if not isinstance(train_ids, list):
            continue
        current = by_node.get(node_id, set())
        if not current:
            continue
        best = 0.0
        for train_id in train_ids:
            other = by_node.get(str(train_id), set())
            if not other:
                continue
            union = current | other
            score = len(current & other) / len(union) if union else 0.0
            if score > best:
                best = score
        rows.append(best)
    if not rows:
        return {
            "available": False,
            "message": "No split train-node metadata available for similarity diagnostics.",
        }
    values = np.asarray(rows, dtype=float)
    return {
        "available": True,
        "validation_rows": int(values.size),
        "mean_nearest_group_jaccard": float(np.mean(values)),
        "p90_nearest_group_jaccard": float(np.quantile(values, 0.9)),
        "max_nearest_group_jaccard": float(np.max(values)),
        "high_similarity_row_rate": float(np.mean(values >= 0.8)),
        "caution": (
            "High nearest-neighbor group similarity can inflate random split metrics; "
            "compare group-split metrics before treating advisory/pruning as reliable."
        ),
    }


def _core_metrics(true, pred) -> dict[str, float | int | None]:
    np = _require_numpy()
    if true.size == 0:
        return {"n": 0, "mae": None, "rmse": None, "median_absolute_error": None, "r2": None, "spearman": None, "pearson": None}
    error = pred - true
    abs_error = np.abs(error)
    ss_res = float(np.sum(error**2))
    ss_tot = float(np.sum((true - np.mean(true)) ** 2))
    return {
        "n": int(true.size),
        "mae": float(np.mean(abs_error)),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "median_absolute_error": float(np.median(abs_error)),
        "r2": None if ss_tot == 0 else 1.0 - ss_res / ss_tot,
        "spearman": _corr(_rank(true), _rank(pred)),
        "pearson": _corr(true, pred),
    }


def _top_decile_metrics(true, pred) -> dict[str, float | int | None]:
    np = _require_numpy()
    if true.size == 0:
        return {
            "top10_true_mae": None,
            "top10_precision": None,
            "top10_recall": None,
            "top10_true_count": 0,
            "top10_pred_count": 0,
        }
    true_threshold = float(np.quantile(true, 0.9))
    pred_threshold = float(np.quantile(pred, 0.9))
    true_top = true >= true_threshold
    pred_top = pred >= pred_threshold
    intersection = true_top & pred_top
    return {
        "top10_true_mae": float(np.mean(np.abs(pred[true_top] - true[true_top]))) if bool(true_top.any()) else None,
        "top10_precision": float(intersection.sum() / pred_top.sum()) if bool(pred_top.any()) else None,
        "top10_recall": float(intersection.sum() / true_top.sum()) if bool(true_top.any()) else None,
        "top10_true_count": int(true_top.sum()),
        "top10_pred_count": int(pred_top.sum()),
    }


def _prefix_metrics(metrics: dict[str, float | int | None], prefix: str) -> dict[str, float | int | None]:
    return {f"{prefix}_{key}": value for key, value in metrics.items()}


def _corr(left, right) -> float | None:
    np = _require_numpy()
    if len(left) < 2 or len(right) < 2:
        return None
    if float(np.std(left)) == 0.0 or float(np.std(right)) == 0.0:
        return None
    value = float(np.corrcoef(left, right)[0, 1])
    return value if math.isfinite(value) else None


def _rank(values):
    np = _require_numpy()
    order = np.argsort(values)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(len(values), dtype=float)
    return ranks


def _require_numpy():
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("Meta-model evaluation requires numpy. Install project requirements with uv pip.") from exc
    return np
