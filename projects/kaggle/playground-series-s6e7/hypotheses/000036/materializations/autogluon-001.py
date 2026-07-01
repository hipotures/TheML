import numpy as np
import pandas as pd


ACTIVITY_MET_MAP = {
    "sedentary": 2.0,
    "moderate": 4.0,
    "active": 8.0,
}

ACTIVITY_DOSE_BINS = (-np.inf, 300.0, 600.0, 1200.0, 2400.0, np.inf)
ACTIVITY_DOSE_LABELS = (
    "minimal",
    "below_guideline",
    "meets_guideline",
    "high_benefit",
    "extreme",
)


def _numeric_series(raw, column_name):
    if column_name in raw.columns:
        return pd.to_numeric(raw[column_name], errors="coerce")
    return pd.Series(np.nan, index=raw.index, dtype="float64")


def _text_series(raw, column_name):
    if column_name in raw.columns:
        return raw[column_name].astype("string").str.strip().str.lower()
    return pd.Series(pd.NA, index=raw.index, dtype="string")


def add_met_equivalent_activity_dose(raw, deps, aux):
    step_count = _numeric_series(raw, "step_count")
    exercise_duration = _numeric_series(raw, "exercise_duration")
    activity_level = _text_series(raw, "physical_activity_level")

    step_missing = step_count.isna()
    exercise_missing = exercise_duration.isna()
    intensity_unknown = ~activity_level.isin(ACTIVITY_MET_MAP.keys())

    clipped_steps = step_count.clip(lower=1000.0, upper=14999.0)
    clipped_exercise = exercise_duration.clip(lower=0.0, upper=120.0)

    intensity_met = activity_level.map(ACTIVITY_MET_MAP).astype("float64").fillna(4.0)

    exercise_daily_met_minutes = clipped_exercise.fillna(0.0) * intensity_met
    walking_minutes_daily = clipped_steps.fillna(0.0) / 100.0
    walking_daily_met_minutes = walking_minutes_daily * 3.3

    daily_met_minutes = exercise_daily_met_minutes + walking_daily_met_minutes
    weekly_met_minutes = daily_met_minutes * 7.0

    ratio_600 = weekly_met_minutes / 600.0
    ratio_1200 = weekly_met_minutes / 1200.0

    component_count = (~step_missing).astype("int8") + (~exercise_missing).astype("int8")
    both_components_missing = step_missing & exercise_missing

    dose_bin = pd.cut(
        weekly_met_minutes,
        bins=list(ACTIVITY_DOSE_BINS),
        labels=list(ACTIVITY_DOSE_LABELS),
        right=False,
    ).astype("string")
    dose_bin = dose_bin.mask(both_components_missing, "missing_activity_components")

    new_features = pd.DataFrame(index=raw.index)
    new_features["activity_intensity_met"] = intensity_met.astype("float32")
    new_features["activity_intensity_unknown"] = intensity_unknown.astype("int8")
    new_features["activity_step_component_missing"] = step_missing.astype("int8")
    new_features["activity_exercise_component_missing"] = exercise_missing.astype("int8")
    new_features["activity_component_completeness_count"] = component_count.astype("int8")
    new_features["activity_clipped_step_count"] = clipped_steps.astype("float32")
    new_features["activity_clipped_exercise_minutes"] = clipped_exercise.astype("float32")
    new_features["walking_daily_met_minutes"] = walking_daily_met_minutes.astype("float32")
    new_features["exercise_daily_met_minutes"] = exercise_daily_met_minutes.astype("float32")
    new_features["total_daily_met_minutes"] = daily_met_minutes.astype("float32")
    new_features["weekly_met_minutes"] = weekly_met_minutes.astype("float32")
    new_features["weekly_met_ratio_600"] = ratio_600.astype("float32")
    new_features["weekly_met_ratio_1200"] = ratio_1200.astype("float32")
    new_features["weekly_met_ratio_600_clipped"] = ratio_600.clip(lower=0.0, upper=6.0).astype("float32")
    new_features["weekly_met_ratio_1200_clipped"] = ratio_1200.clip(lower=0.0, upper=4.0).astype("float32")
    new_features["log1p_weekly_met_ratio_600"] = np.log1p(ratio_600.clip(lower=0.0)).astype("float32")
    new_features["meets_600_met_guideline"] = (weekly_met_minutes >= 600.0).astype("int8")
    new_features["meets_1200_met_guideline"] = (weekly_met_minutes >= 1200.0).astype("int8")
    new_features["activity_dose_bin"] = dose_bin

    return new_features


FEATURE_GROUPS = [
    {
        "name": "met_equivalent_activity_dose",
        "fn": add_met_equivalent_activity_dose,
        "depends_on": [],
        "description": "Standardizes steps, exercise duration, and reported activity level into guideline-relative weekly MET-minute activity dose features.",
    }
]