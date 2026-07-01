import numpy as np
import pandas as pd


STRESS_LEVEL_MAP = {"low": 0, "medium": 1, "high": 2}
SLEEP_QUALITY_MAP = {"poor": 0, "average": 1, "good": 2}
PHYSICAL_ACTIVITY_MAP = {"sedentary": 0, "moderate": 1, "active": 2}


def _ordinal_category_score(series, mapping, index):
    if series is None:
        return pd.Series(1, index=index, dtype="float64")
    normalized = series.astype("string").str.strip().str.lower()
    return normalized.map(mapping).fillna(1).astype("float64")


def _numeric_series(raw, column, low, high):
    values = pd.to_numeric(raw[column], errors="coerce") if column in raw else pd.Series(np.nan, index=raw.index)
    return values.clip(lower=low, upper=high)


def add_stress_recovery_triad_profiles(raw, deps, aux):
    index = raw.index

    stress_score = _ordinal_category_score(raw["stress_level"] if "stress_level" in raw else None, STRESS_LEVEL_MAP, index)
    sleep_quality_score = _ordinal_category_score(
        raw["sleep_quality"] if "sleep_quality" in raw else None,
        SLEEP_QUALITY_MAP,
        index,
    )
    activity_score = _ordinal_category_score(
        raw["physical_activity_level"] if "physical_activity_level" in raw else None,
        PHYSICAL_ACTIVITY_MAP,
        index,
    )

    sleep_duration = _numeric_series(raw, "sleep_duration", 3.0, 10.0)
    sleep_duration_score = pd.Series(1.0, index=index)
    sleep_duration_score = sleep_duration_score.mask(sleep_duration < 6.0, 0.0)
    sleep_duration_score = sleep_duration_score.mask((sleep_duration >= 6.0) & (sleep_duration <= 8.5), 2.0)
    sleep_duration_score = sleep_duration_score.mask(sleep_duration > 8.5, 1.0)

    step_count = _numeric_series(raw, "step_count", 1002.0, 14999.0)
    step_score = pd.Series(1.0, index=index)
    step_score = step_score.mask(step_count < 5000.0, 0.0)
    step_score = step_score.mask((step_count >= 5000.0) & (step_count < 10000.0), 1.0)
    step_score = step_score.mask(step_count >= 10000.0, 2.0)

    exercise_duration = _numeric_series(raw, "exercise_duration", 0.0, 99.8)
    exercise_score = pd.Series(1.0, index=index)
    exercise_score = exercise_score.mask(exercise_duration < 20.0, 0.0)
    exercise_score = exercise_score.mask((exercise_duration >= 20.0) & (exercise_duration < 60.0), 1.0)
    exercise_score = exercise_score.mask(exercise_duration >= 60.0, 2.0)

    movement_score = np.rint((step_score + exercise_score + activity_score) / 3.0).astype("int8")
    rest_score = np.rint((sleep_duration_score + sleep_quality_score) / 2.0).astype("int8")
    stress_score_int = stress_score.astype("int8")

    features = pd.DataFrame(index=index)
    features["triad_code"] = (9 * stress_score_int + 3 * rest_score + movement_score).astype("int8")
    features["recovery_margin"] = (rest_score + movement_score - stress_score_int).astype("int8")
    features["stressed_buffered"] = (
        (stress_score_int == 2) & (rest_score >= 1) & (movement_score >= 1)
    ).astype("int8")
    features["exhausted_inactive"] = (
        (stress_score_int >= 1) & (rest_score == 0) & (movement_score == 0)
    ).astype("int8")

    return features


FEATURE_GROUPS = [
    {
        "name": "stress_recovery_triad_profiles",
        "fn": add_stress_recovery_triad_profiles,
        "depends_on": [],
        "description": "Encodes stress, rest, and movement into recovery-balance profile features.",
    }
]