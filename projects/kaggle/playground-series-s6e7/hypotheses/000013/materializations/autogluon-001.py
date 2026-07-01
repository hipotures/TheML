import numpy as np
import pandas as pd


ACTIVITY_NUMERIC_COLUMNS = ("step_count", "exercise_duration", "calorie_expenditure")
ACTIVITY_LEVEL_MAP = {
    "sedentary": 0.0,
    "moderate": 0.5,
    "active": 1.0,
}


def _reference_mask(raw):
    if "id" not in raw.columns:
        return pd.Series(True, index=raw.index)

    ids = pd.to_numeric(raw["id"], errors="coerce")
    if ids.notna().any() and ids.min(skipna=True) <= 0 and ids.max(skipna=True) > 690087:
        mask = ids <= 690087
        if mask.any():
            return mask

    return pd.Series(True, index=raw.index)


def _safe_median(values, fallback):
    clean = pd.to_numeric(values, errors="coerce").dropna()
    if clean.empty:
        return fallback
    return float(clean.median())


def _safe_quantile(values, q, fallback):
    clean = pd.to_numeric(values, errors="coerce").dropna()
    if clean.empty:
        return fallback
    return float(clean.quantile(q))


def _robust_quantile_score(values, median, q1, q3):
    numeric = pd.to_numeric(values, errors="coerce").astype(float)
    iqr = max(float(q3 - q1), 1e-6)
    filled = numeric.fillna(median)
    return (0.5 + (filled - median) / (2.0 * iqr)).clip(0.0, 1.0)


def _tertile_level(values, low_cut, high_cut):
    numeric = pd.to_numeric(values, errors="coerce").astype(float)
    level = pd.Series(1, index=values.index, dtype="int8")
    level[numeric <= low_cut] = 0
    level[numeric >= high_cut] = 2
    return level


def add_activity_modality_balance(raw, deps, aux):
    ref_mask = _reference_mask(raw)
    ref = raw.loc[ref_mask]

    step_median = _safe_median(ref.get("step_count"), 7000.0)
    exercise_median = _safe_median(ref.get("exercise_duration"), 30.0)
    calorie_median = _safe_median(ref.get("calorie_expenditure"), 2200.0)

    step_q1 = _safe_quantile(ref.get("step_count"), 0.25, step_median)
    step_q3 = _safe_quantile(ref.get("step_count"), 0.75, step_median + 1.0)
    exercise_q1 = _safe_quantile(ref.get("exercise_duration"), 0.25, exercise_median)
    exercise_q3 = _safe_quantile(ref.get("exercise_duration"), 0.75, exercise_median + 1.0)
    calorie_q1 = _safe_quantile(ref.get("calorie_expenditure"), 0.25, calorie_median)
    calorie_q3 = _safe_quantile(ref.get("calorie_expenditure"), 0.75, calorie_median + 1.0)

    step_t1 = _safe_quantile(ref.get("step_count"), 1.0 / 3.0, step_median)
    step_t2 = _safe_quantile(ref.get("step_count"), 2.0 / 3.0, step_median)
    exercise_t1 = _safe_quantile(ref.get("exercise_duration"), 1.0 / 3.0, exercise_median)
    exercise_t2 = _safe_quantile(ref.get("exercise_duration"), 2.0 / 3.0, exercise_median)

    steps = pd.to_numeric(raw.get("step_count"), errors="coerce").astype(float).fillna(step_median)
    exercise = pd.to_numeric(raw.get("exercise_duration"), errors="coerce").astype(float).fillna(exercise_median)

    step_score = _robust_quantile_score(raw.get("step_count"), step_median, step_q1, step_q3)
    exercise_score = _robust_quantile_score(raw.get("exercise_duration"), exercise_median, exercise_q1, exercise_q3)
    calorie_score = _robust_quantile_score(raw.get("calorie_expenditure"), calorie_median, calorie_q1, calorie_q3)

    activity_volume = ((step_score + exercise_score) / 2.0).clip(0.0, 1.0)
    modality_delta = step_score - exercise_score

    step_units = steps / 1000.0
    denominator = exercise + step_units + 1.0
    structured_share = exercise / denominator
    ambulatory_share = step_units / denominator

    calorie_intensity_residual = calorie_score - activity_volume

    if "physical_activity_level" in raw.columns:
        activity_level = (
            raw["physical_activity_level"]
            .astype("string")
            .str.lower()
            .map(ACTIVITY_LEVEL_MAP)
            .astype(float)
            .fillna(0.5)
        )
    else:
        activity_level = pd.Series(0.5, index=raw.index, dtype=float)

    self_report_gap = activity_level - activity_volume

    step_level = _tertile_level(steps, step_t1, step_t2)
    exercise_level = _tertile_level(exercise, exercise_t1, exercise_t2)

    modality_bucket = pd.Series("mixed_mid", index=raw.index, dtype=object)
    modality_bucket[(step_level == 0) & (exercise_level == 0)] = "low_both"
    modality_bucket[(step_level == 2) & (exercise_level == 2)] = "high_both"
    modality_bucket[(step_level >= 1) & (exercise_level == 0)] = "walk_dominant"
    modality_bucket[(step_level == 0) & (exercise_level >= 1)] = "exercise_dominant"
    modality_bucket[(step_level == 2) & (exercise_level == 1)] = "walk_dominant"
    modality_bucket[(step_level == 1) & (exercise_level == 2)] = "exercise_dominant"

    return pd.DataFrame(
        {
            "step_quantile_score": step_score.astype(float),
            "exercise_quantile_score": exercise_score.astype(float),
            "calorie_quantile_score": calorie_score.astype(float),
            "activity_volume": activity_volume.astype(float),
            "modality_delta": modality_delta.astype(float),
            "structured_share": structured_share.astype(float),
            "ambulatory_share": ambulatory_share.astype(float),
            "calorie_intensity_residual": calorie_intensity_residual.astype(float),
            "self_report_gap": self_report_gap.astype(float),
            "modality_bucket": modality_bucket,
        },
        index=raw.index,
    )


FEATURE_GROUPS = [
    {
        "name": "activity_modality_balance",
        "fn": add_activity_modality_balance,
        "depends_on": [],
        "description": "Captures the balance between structured exercise, daily ambulation, calorie intensity, and stated activity level.",
    }
]