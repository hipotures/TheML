import numpy as np
import pandas as pd


ACTIVITY_SUPPORT_MAP = {
    "sedentary": 0.0,
    "moderate": 0.5,
    "active": 1.0,
}


def _numeric(raw, column):
    if column in raw.columns:
        return pd.to_numeric(raw[column], errors="coerce")
    return pd.Series(np.nan, index=raw.index, dtype="float64")


def _categorical(raw, column):
    if column in raw.columns:
        return raw[column].astype("string").str.lower().str.strip()
    return pd.Series(pd.NA, index=raw.index, dtype="string")


def _observed_mean(parts, neutral=0.5):
    frame = pd.concat(parts, axis=1)
    observed = frame.notna().sum(axis=1)
    summed = frame.fillna(0.0).sum(axis=1)
    return (summed / observed.replace(0, np.nan)).fillna(neutral)


def add_sedentary_offset_balance(raw, deps, aux):
    sleep_duration = _numeric(raw, "sleep_duration")
    step_count = _numeric(raw, "step_count")
    exercise_duration = _numeric(raw, "exercise_duration")
    calorie_expenditure = _numeric(raw, "calorie_expenditure")
    activity_level = _categorical(raw, "physical_activity_level")

    waking_hours = (24.0 - sleep_duration).clip(lower=12.0, upper=21.0)
    step_density = (step_count / waking_hours.replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan)

    exercise_sufficiency = (exercise_duration / 30.0).clip(lower=0.0, upper=2.0)
    incidental_movement_sufficiency = (step_count / 7500.0).clip(lower=0.0, upper=2.0)
    activity_label_support = activity_level.map(ACTIVITY_SUPPORT_MAP).astype("float64")

    calorie_support = calorie_expenditure.rank(pct=True, method="average")
    calorie_support = calorie_support.clip(lower=0.0, upper=1.0)

    sedentary_label_pressure = 1.0 - activity_label_support
    low_step_density_pressure = 1.0 - (step_density / 625.0).clip(lower=0.0, upper=1.0)
    low_calorie_pressure = 1.0 - calorie_support
    long_waking_pressure = ((waking_hours - 15.0) / 6.0).clip(lower=0.0, upper=1.0)

    sedentary_pressure = _observed_mean(
        [
            sedentary_label_pressure,
            low_step_density_pressure,
            low_calorie_pressure,
            long_waking_pressure,
        ]
    ).clip(lower=0.0, upper=1.0)

    offset_score = _observed_mean(
        [
            (exercise_sufficiency / 2.0).clip(lower=0.0, upper=1.0),
            (incidental_movement_sufficiency / 2.0).clip(lower=0.0, upper=1.0),
            activity_label_support,
        ]
    ).clip(lower=0.0, upper=1.0)

    offset_gap = (sedentary_pressure - offset_score).clip(lower=-1.0, upper=1.0)

    high_sedentary = sedentary_pressure >= 0.5
    high_offset = offset_score >= 0.5
    offset_quadrant = pd.Series("low_sedentary_low_offset", index=raw.index, dtype="object")
    offset_quadrant.loc[high_sedentary & ~high_offset] = "high_sedentary_low_offset"
    offset_quadrant.loc[high_sedentary & high_offset] = "high_sedentary_high_offset"
    offset_quadrant.loc[~high_sedentary & high_offset] = "low_sedentary_high_offset"

    concentrated_exercise = (
        (exercise_sufficiency >= 1.0)
        & (incidental_movement_sufficiency.fillna(1.0) < 0.8)
        & (activity_label_support.fillna(0.5) <= 0.5)
    )

    new_features = pd.DataFrame(
        {
            "waking_hours_est": waking_hours.fillna(16.0).clip(lower=12.0, upper=21.0),
            "step_density_waking_hour": step_density.fillna(625.0).clip(lower=0.0, upper=1500.0),
            "exercise_sufficiency_30m": exercise_sufficiency.fillna(0.5).clip(lower=0.0, upper=2.0),
            "incidental_movement_sufficiency_7500": incidental_movement_sufficiency.fillna(0.5).clip(lower=0.0, upper=2.0),
            "activity_label_support": activity_label_support.fillna(0.5).clip(lower=0.0, upper=1.0),
            "calorie_support_rank": calorie_support.fillna(0.5).clip(lower=0.0, upper=1.0),
            "sedentary_pressure": sedentary_pressure,
            "offset_score": offset_score,
            "offset_gap": offset_gap,
            "offset_quadrant": offset_quadrant,
            "concentrated_exercise_flag": concentrated_exercise.astype("int8"),
        },
        index=raw.index,
    )

    return new_features


FEATURE_GROUPS = [
    {
        "name": "sedentary_offset_balance",
        "fn": add_sedentary_offset_balance,
        "depends_on": [],
        "description": "Measures whether exercise and incidental movement appear sufficient to offset sedentary waking-day pressure.",
    }
]