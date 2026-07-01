import numpy as np
import pandas as pd


_STEP_SUPPORT_BINS = (-np.inf, 5000.0, 7500.0, 10000.0, 12500.0, np.inf)
_STEP_SUPPORT_VALUES = (-2.0, -1.0, 0.0, 1.0, 2.0)

_EXERCISE_SUPPORT_BINS = (-np.inf, 0.0, 15.0, 30.0, np.inf)
_EXERCISE_SUPPORT_VALUES = (-1.0, 0.0, 1.0, 2.0)

_ACTIVITY_LEVEL_MAP = {
    "sedentary": -1.0,
    "moderate": 0.0,
    "active": 1.0,
}


def _numeric_series(raw, column_name):
    if column_name in raw.columns:
        return pd.to_numeric(raw[column_name], errors="coerce")
    return pd.Series(np.nan, index=raw.index, dtype="float64")


def _string_series(raw, column_name):
    if column_name in raw.columns:
        return raw[column_name].astype("string").str.lower()
    return pd.Series(pd.NA, index=raw.index, dtype="string")


def add_weight_status_activity_buffer(raw, deps, aux):
    bmi = _numeric_series(raw, "bmi")
    step_count = _numeric_series(raw, "step_count")
    exercise_duration = _numeric_series(raw, "exercise_duration")
    activity_level = _string_series(raw, "physical_activity_level")

    out = pd.DataFrame(index=raw.index)

    bmi_zone_ordinal = pd.Series(0, index=raw.index, dtype="int8")
    bmi_zone_ordinal = bmi_zone_ordinal.mask(bmi < 18.5, 1)
    bmi_zone_ordinal = bmi_zone_ordinal.mask((bmi >= 18.5) & (bmi < 25.0), 2)
    bmi_zone_ordinal = bmi_zone_ordinal.mask((bmi >= 25.0) & (bmi < 30.0), 3)
    bmi_zone_ordinal = bmi_zone_ordinal.mask(bmi >= 30.0, 4)

    bmi_zone_label = pd.Series("unknown", index=raw.index, dtype="object")
    bmi_zone_label = bmi_zone_label.mask(bmi_zone_ordinal == 1, "underweight")
    bmi_zone_label = bmi_zone_label.mask(bmi_zone_ordinal == 2, "healthy")
    bmi_zone_label = bmi_zone_label.mask(bmi_zone_ordinal == 3, "overweight")
    bmi_zone_label = bmi_zone_label.mask(bmi_zone_ordinal == 4, "obesity")

    below_healthy = ((18.5 - bmi) / 5.0).clip(lower=0.0, upper=2.0)
    above_healthy = ((bmi - 25.0) / 5.0).clip(lower=0.0, upper=2.0)
    body_mass_severity = pd.concat([below_healthy, above_healthy], axis=1).max(axis=1)
    body_mass_severity = body_mass_severity.where(bmi.notna(), np.nan)

    step_support = pd.Series(np.nan, index=raw.index, dtype="float64")
    for lower, upper, value in zip(_STEP_SUPPORT_BINS[:-1], _STEP_SUPPORT_BINS[1:], _STEP_SUPPORT_VALUES):
        step_support = step_support.mask((step_count >= lower) & (step_count < upper), value)

    exercise_support = pd.Series(np.nan, index=raw.index, dtype="float64")
    exercise_known = exercise_duration.notna()
    exercise_support = exercise_support.mask(exercise_known & (exercise_duration <= 0.0), -1.0)
    exercise_support = exercise_support.mask(exercise_known & (exercise_duration > 0.0) & (exercise_duration < 15.0), 0.0)
    exercise_support = exercise_support.mask(exercise_known & (exercise_duration >= 15.0) & (exercise_duration < 30.0), 1.0)
    exercise_support = exercise_support.mask(exercise_known & (exercise_duration >= 30.0), 2.0)

    level_support = activity_level.map(_ACTIVITY_LEVEL_MAP).astype("float64")

    support_components = pd.concat(
        [step_support, exercise_support, level_support],
        axis=1,
    )
    activity_support_score = support_components.mean(axis=1, skipna=True)
    activity_known_count = support_components.notna().sum(axis=1).astype("int8")

    activity_support_state = pd.Series("unknown", index=raw.index, dtype="object")
    activity_support_state = activity_support_state.mask(activity_known_count.gt(0) & (activity_support_score < -0.5), "low")
    activity_support_state = activity_support_state.mask(
        activity_known_count.gt(0) & (activity_support_score >= -0.5) & (activity_support_score <= 0.5),
        "neutral",
    )
    activity_support_state = activity_support_state.mask(activity_known_count.gt(0) & (activity_support_score > 0.5), "high")

    sedentary_side = (-activity_support_score).clip(lower=0.0, upper=2.0).where(activity_known_count.gt(0), np.nan)
    active_side = activity_support_score.clip(lower=0.0, upper=2.0).where(activity_known_count.gt(0), np.nan)
    healthy_bmi_flag = (bmi_zone_ordinal == 2).astype("int8")
    nonhealthy_bmi_flag = bmi.notna() & (bmi_zone_ordinal != 2)

    out["bmi_zone_ordinal"] = bmi_zone_ordinal
    out["bmi_nonhealthy_flag"] = nonhealthy_bmi_flag.astype("int8")
    out["bmi_nonhealthy_severity"] = body_mass_severity.astype("float32")
    out["activity_support_score"] = activity_support_score.astype("float32")
    out["activity_support_state"] = activity_support_state
    out["activity_support_observed_count"] = activity_known_count
    out["amplified_weight_risk"] = (body_mass_severity * sedentary_side).astype("float32")
    out["buffered_weight_risk"] = (body_mass_severity * active_side).astype("float32")
    out["healthy_weight_inactivity"] = (healthy_bmi_flag * sedentary_side).astype("float32")
    out["bmi_activity_cross"] = (bmi_zone_label.astype("string") + "__" + activity_support_state.astype("string")).astype("object")

    return out


FEATURE_GROUPS = [
    {
        "name": "weight_status_activity_buffer",
        "fn": add_weight_status_activity_buffer,
        "depends_on": [],
        "description": "Contextual BMI status features that distinguish activity-buffered weight risk from activity-amplified weight risk.",
    }
]