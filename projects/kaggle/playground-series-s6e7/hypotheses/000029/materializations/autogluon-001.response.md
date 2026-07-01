import numpy as np
import pandas as pd


SLEEP_QUALITY_MAP = {
    "poor": 0.0,
    "average": 0.5,
    "good": 1.0,
}

PHYSICAL_ACTIVITY_MAP = {
    "sedentary": 0.0,
    "moderate": 0.5,
    "active": 1.0,
}

DIET_TYPE_MAP = {
    "non-veg": 0.35,
    "veg": 0.7,
    "balanced": 1.0,
}

STRESS_LEVEL_MAP = {
    "low": 0.0,
    "medium": 0.5,
    "high": 1.0,
}

SMOKING_ALCOHOL_MAP = {
    "no": 0.0,
    "occasional": 0.5,
    "yes": 1.0,
}


def _numeric_column(raw, name):
    if name not in raw.columns:
        return pd.Series(np.nan, index=raw.index, dtype="float64")
    return pd.to_numeric(raw[name], errors="coerce")


def _mapped_column(raw, name, mapping):
    if name not in raw.columns:
        return pd.Series(np.nan, index=raw.index, dtype="float64")
    values = raw[name].astype("string").str.strip().str.lower()
    return values.map(mapping).astype("float64")


def _row_mean_with_nan(columns, index):
    frame = pd.concat(columns, axis=1)
    result = frame.mean(axis=1, skipna=True)
    return result.reindex(index)


def add_risk_protection_behavior_gap(raw, deps, aux):
    sleep_duration = _numeric_column(raw, "sleep_duration")
    heart_rate = _numeric_column(raw, "heart_rate")
    bmi = _numeric_column(raw, "bmi")
    step_count = _numeric_column(raw, "step_count")
    exercise_duration = _numeric_column(raw, "exercise_duration")
    water_intake = _numeric_column(raw, "water_intake")

    sleep_amount = (1.0 - (sleep_duration - 7.5).abs() / 3.5).clip(0.0, 1.0)
    sleep_quality = _mapped_column(raw, "sleep_quality", SLEEP_QUALITY_MAP)

    movement_steps = ((step_count - 4000.0) / 6000.0).clip(0.0, 1.0)
    movement_exercise = (exercise_duration / 45.0).clip(0.0, 1.0)
    movement_activity = _mapped_column(raw, "physical_activity_level", PHYSICAL_ACTIVITY_MAP)
    movement = _row_mean_with_nan(
        [movement_steps, movement_exercise, movement_activity],
        raw.index,
    )

    self_care_water = ((water_intake - 0.5) / 2.0).clip(0.0, 1.0)
    self_care_diet = _mapped_column(raw, "diet_type", DIET_TYPE_MAP)
    self_care = _row_mean_with_nan([self_care_water, self_care_diet], raw.index)

    stress = _mapped_column(raw, "stress_level", STRESS_LEVEL_MAP)
    smoking_alcohol = _mapped_column(raw, "smoking_alcohol", SMOKING_ALCOHOL_MAP)

    heart_load = ((heart_rate - 75.0) / 28.0).clip(0.0, 1.0)
    bmi_distance = pd.concat([18.5 - bmi, bmi - 25.0], axis=1).max(axis=1)
    bmi_load = (bmi_distance / 8.0).clip(0.0, 1.0)
    cardio_body_load = _row_mean_with_nan([heart_load, bmi_load], raw.index)

    sleep_deficit = ((7.0 - sleep_duration) / 4.0).clip(0.0, 1.0)

    protective_score = _row_mean_with_nan(
        [sleep_amount, sleep_quality, movement, self_care],
        raw.index,
    ).fillna(0.5)

    harm_score = _row_mean_with_nan(
        [stress, smoking_alcohol, cardio_body_load, sleep_deficit],
        raw.index,
    ).fillna(0.5)

    protection_minus_harm = protective_score - harm_score
    buffered_risk = pd.concat([protective_score, harm_score], axis=1).min(axis=1)
    unsupported_exposure = harm_score * (1.0 - protective_score)

    high_protection = protective_score >= 0.6
    high_harm = harm_score > 0.35
    quadrant = pd.Series(0, index=raw.index, dtype="int8")
    quadrant.loc[high_protection & ~high_harm] = 3
    quadrant.loc[high_protection & high_harm] = 2
    quadrant.loc[~high_protection & ~high_harm] = 1
    quadrant.loc[~high_protection & high_harm] = 0

    return pd.DataFrame(
        {
            "protective_score": protective_score.astype("float32"),
            "harm_score": harm_score.astype("float32"),
            "protection_minus_harm": protection_minus_harm.astype("float32"),
            "buffered_risk": buffered_risk.astype("float32"),
            "unsupported_exposure": unsupported_exposure.astype("float32"),
            "risk_protection_quadrant": quadrant,
        },
        index=raw.index,
    )


FEATURE_GROUPS = [
    {
        "name": "risk_protection_behavior_gap",
        "fn": add_risk_protection_behavior_gap,
        "depends_on": [],
        "description": "Contrasts protective routines against damaging exposures to summarize buffered and unsupported health risk profiles.",
    }
]