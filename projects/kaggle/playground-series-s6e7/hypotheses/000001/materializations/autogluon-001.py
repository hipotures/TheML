import numpy as np
import pandas as pd


GUIDELINE_COMPONENT_COLUMNS = (
    "bmi_points",
    "sleep_duration_points",
    "heart_rate_points",
    "step_count_points",
    "exercise_duration_points",
    "water_intake_points",
    "diet_type_points",
    "stress_level_points",
    "sleep_quality_points",
    "physical_activity_level_points",
    "smoking_alcohol_points",
)


def _numeric(raw, column):
    if column not in raw.columns:
        return pd.Series(np.nan, index=raw.index, dtype="float64")
    return pd.to_numeric(raw[column], errors="coerce")


def _lower_text(raw, column):
    if column not in raw.columns:
        return pd.Series(pd.NA, index=raw.index, dtype="object")
    return raw[column].astype("string").str.strip().str.lower()


def add_guideline_deviation_burden(raw, deps, aux):
    out = pd.DataFrame(index=raw.index)

    bmi = _numeric(raw, "bmi")
    bmi_points = pd.Series(np.nan, index=raw.index, dtype="float64")
    bmi_points = bmi_points.mask(bmi < 18.5, 1.0)
    bmi_points = bmi_points.mask((bmi >= 18.5) & (bmi < 25.0), 0.0)
    bmi_points = bmi_points.mask((bmi >= 25.0) & (bmi < 30.0), 1.0)
    bmi_points = bmi_points.mask(bmi >= 30.0, 2.0)
    out["bmi_points"] = bmi_points

    sleep_duration = _numeric(raw, "sleep_duration")
    sleep_points = pd.Series(np.nan, index=raw.index, dtype="float64")
    sleep_points = sleep_points.mask(sleep_duration < 6.0, 2.0)
    sleep_points = sleep_points.mask((sleep_duration >= 6.0) & (sleep_duration < 7.0), 1.0)
    sleep_points = sleep_points.mask((sleep_duration >= 7.0) & (sleep_duration <= 9.0), 0.0)
    sleep_points = sleep_points.mask(sleep_duration > 9.0, 1.0)
    out["sleep_duration_points"] = sleep_points

    heart_rate = _numeric(raw, "heart_rate")
    heart_points = pd.Series(np.nan, index=raw.index, dtype="float64")
    heart_points = heart_points.mask((heart_rate >= 60.0) & (heart_rate <= 100.0), 0.0)
    heart_points = heart_points.mask((heart_rate >= 50.0) & (heart_rate < 60.0), 1.0)
    heart_points = heart_points.mask(heart_rate > 100.0, 2.0)
    out["heart_rate_points"] = heart_points

    step_count = _numeric(raw, "step_count")
    step_points = pd.Series(np.nan, index=raw.index, dtype="float64")
    step_points = step_points.mask(step_count < 4000.0, 2.0)
    step_points = step_points.mask((step_count >= 4000.0) & (step_count < 7000.0), 1.0)
    step_points = step_points.mask(step_count >= 7000.0, 0.0)
    out["step_count_points"] = step_points

    exercise_duration = _numeric(raw, "exercise_duration")
    exercise_points = pd.Series(np.nan, index=raw.index, dtype="float64")
    exercise_points = exercise_points.mask(exercise_duration == 0.0, 2.0)
    exercise_points = exercise_points.mask((exercise_duration > 0.0) & (exercise_duration < 30.0), 1.0)
    exercise_points = exercise_points.mask(exercise_duration >= 30.0, 0.0)
    out["exercise_duration_points"] = exercise_points

    water_intake = _numeric(raw, "water_intake")
    gender = _lower_text(raw, "gender")
    water_target = pd.Series(3.2, index=raw.index, dtype="float64")
    water_target = water_target.mask(gender == "male", 3.7)
    water_target = water_target.mask(gender == "female", 2.7)
    water_ratio = water_intake / water_target
    water_points = pd.Series(np.nan, index=raw.index, dtype="float64")
    water_points = water_points.mask(water_ratio >= 1.0, 0.0)
    water_points = water_points.mask((water_ratio >= 0.7) & (water_ratio < 1.0), 1.0)
    water_points = water_points.mask(water_ratio < 0.7, 2.0)
    out["water_intake_points"] = water_points
    out["water_intake_target_liters"] = water_target.where(water_intake.notna())
    out["water_intake_target_ratio"] = water_ratio

    diet_type = _lower_text(raw, "diet_type")
    out["diet_type_points"] = diet_type.map({"balanced": 0.0, "veg": 1.0, "non-veg": 1.0}).astype("float64")

    stress_level = _lower_text(raw, "stress_level")
    out["stress_level_points"] = stress_level.map({"low": 0.0, "medium": 1.0, "high": 2.0}).astype("float64")

    sleep_quality = _lower_text(raw, "sleep_quality")
    out["sleep_quality_points"] = sleep_quality.map({"good": 0.0, "average": 1.0, "poor": 2.0}).astype("float64")

    activity_level = _lower_text(raw, "physical_activity_level")
    out["physical_activity_level_points"] = activity_level.map(
        {"active": 0.0, "moderate": 1.0, "sedentary": 2.0}
    ).astype("float64")

    smoking_alcohol = _lower_text(raw, "smoking_alcohol")
    out["smoking_alcohol_points"] = smoking_alcohol.map(
        {"no": 0.0, "occasional": 1.0, "yes": 2.0}
    ).astype("float64")

    components = out.loc[:, list(GUIDELINE_COMPONENT_COLUMNS)]
    available_count = components.notna().sum(axis=1)

    out["burden_sum"] = components.sum(axis=1, min_count=1)
    out["burden_mean"] = components.mean(axis=1)
    out["severe_deviation_count"] = components.eq(2.0).sum(axis=1).where(available_count > 0)
    out["healthy_alignment_count"] = components.eq(0.0).sum(axis=1).where(available_count > 0)
    out["burden_available_count"] = available_count.astype("float64")

    return out


FEATURE_GROUPS = [
    {
        "name": "guideline_deviation_burden",
        "fn": add_guideline_deviation_burden,
        "depends_on": [],
        "description": "Deterministic public-guideline deviation scores and aggregate lifestyle-health burden features.",
    }
]