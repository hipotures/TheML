from __future__ import annotations

import math
from typing import Any


def evaluation_selection_fields(
    *,
    metric: float | None,
    run_stats: dict[str, Any] | None,
    profile: dict[str, object] | None,
) -> tuple[int | None, float | None]:
    feature_count = _feature_count(run_stats)
    selection = profile.get("selection_score") if isinstance(profile, dict) else None
    if not isinstance(selection, dict):
        return feature_count, None

    kind = str(selection.get("kind") or "").strip()
    if kind != "feature_count_penalty":
        raise ValueError(f"Unsupported selection_score.kind: {kind!r}")
    if metric is None or feature_count is None:
        return feature_count, None

    penalty = _finite_float(selection.get("penalty_per_feature"), "selection_score.penalty_per_feature")
    return feature_count, float(metric) - penalty * feature_count


def _feature_count(run_stats: dict[str, Any] | None) -> int | None:
    if not isinstance(run_stats, dict):
        return None
    raw = run_stats.get("feature_count")
    if isinstance(raw, bool):
        return None
    if isinstance(raw, int):
        return raw
    if isinstance(raw, float) and raw.is_integer():
        return int(raw)
    return None


def _finite_float(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"Invalid {label}: {value!r}")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"Invalid {label}: {value!r}")
    return parsed
