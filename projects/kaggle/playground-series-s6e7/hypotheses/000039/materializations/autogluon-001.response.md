import numpy as np
import pandas as pd


MOVEMENT_COLUMNS = (
    "step_count",
    "exercise_duration",
    "calorie_expenditure",
)

RECOVERY_COLUMNS = (
    "sleep_duration",
    "sleep_quality",
    "stress_level",
    "water_intake",
    "smoking_alcohol",
    "diet_type",
)

PHYSIOLOGY_COLUMNS = (
    "heart_rate",
    "bmi",
)

ALL_CONTEXT_COLUMNS = MOVEMENT_COLUMNS + RECOVERY_COLUMNS + PHYSIOLOGY_COLUMNS

SLEEP_QUALITY_SUPPORT = {
    "good": 1.0,
    "average": 0.5,
    "poor": 0.0,
}

STRESS_SUPPORT = {
    "low": 1.0,
    "medium": 0.5,
    "high": 0.0,
}

SMOKING_ALCOHOL_SUPPORT = {
    "no": 1.0,
    "occasional": 0.5,
    "yes": 0.0,
}

DIET_SUPPORT = {
    "balanced": 1.0,
    "veg": 0.7,
    "non-veg": 0.4,
}


def _numeric(raw, column):
    if column not in raw.columns:
        return pd.Series(np.nan, index=raw.index, dtype="float64")
    return pd.to_numeric(raw[column], errors="coerce")


def _category_support(raw, column, mapping):
    if column not in raw.columns:
        return pd.Series(np.nan, index=raw.index, dtype="float64")
    normalized = raw[column].astype("string").str.strip().str.lower()
    return normalized.map(mapping).astype("float64")


def _row_mean_with_neutral(series_list, index):
    block = pd.concat(series_list, axis=1)
    observed_count = block.notna().sum(axis=1)
    values = block.mean(axis=1)
    return values.where(observed_count > 0, 0.5).astype("float64")


def add_low_movement_recovery_context(raw, deps, aux):
    new_features = pd.DataFrame(index=raw.index)

    for column in ALL_CONTEXT_COLUMNS:
        if column in raw.columns:
            new_features[column + "_missing"] = raw[column].isna().astype("int8")
        else:
            new_features[column + "_missing"] = pd.Series(1, index=raw.index, dtype="int8")

    step_count = _numeric(raw, "step_count")
    exercise_duration = _numeric(raw, "exercise_duration")
    calorie_expenditure = _numeric(raw, "calorie_expenditure")

    step_shortfall = ((7000.0 - step_count) / 6000.0).clip(lower=0.0, upper=1.0)
    exercise_shortfall = ((30.0 - exercise_duration) / 30.0).clip(lower=0.0, upper=1.0)
    calorie_shortfall = ((1800.0 - calorie_expenditure) / 600.0).clip(lower=0.0, upper=1.0)

    movement_shortfall = _row_mean_with_neutral(
        [step_shortfall, exercise_shortfall, calorie_shortfall],
        raw.index,
    )

    sleep_duration = _numeric(raw, "sleep_duration")
    sleep_support = pd.Series(np.nan, index=raw.index, dtype="float64")
    sleep_support = sleep_support.mask(
        sleep_duration.notna(),
        np.where(
            sleep_duration < 6.0,
            ((sleep_duration - 3.0) / 3.0).clip(0.0, 1.0),
            np.where(
                sleep_duration <= 7.0,
                0.7 + ((sleep_duration - 6.0) / 1.0) * 0.3,
                np.where(
                    sleep_duration <= 9.0,
                    1.0,
                    np.where(
                        sleep_duration <= 10.0,
                        1.0 - ((sleep_duration - 9.0) / 1.0) * 0.3,
                        ((12.0 - sleep_duration) / 2.0).clip(0.0, 0.7),
                    ),
                ),
            ),
        ),
    )

    water_intake = _numeric(raw, "water_intake")
    water_support = ((water_intake - 1.5) / 1.5).clip(lower=0.0, upper=1.0)

    recovery_support = _row_mean_with_neutral(
        [
            sleep_support,
            _category_support(raw, "sleep_quality", SLEEP_QUALITY_SUPPORT),
            _category_support(raw, "stress_level", STRESS_SUPPORT),
            water_support,
            _category_support(raw, "smoking_alcohol", SMOKING_ALCOHOL_SUPPORT),
            _category_support(raw, "diet_type", DIET_SUPPORT),
        ],
        raw.index,
    )

    heart_rate = _numeric(raw, "heart_rate")
    heart_rate_burden = ((heart_rate - 75.0) / 25.0).clip(lower=0.0, upper=1.0)

    bmi = _numeric(raw, "bmi")
    low_bmi_burden = ((18.5 - bmi) / 2.5).clip(lower=0.0, upper=1.0)
    high_bmi_burden = ((bmi - 24.9) / 10.0).clip(lower=0.0, upper=1.0)
    bmi_burden = pd.concat([low_bmi_burden, high_bmi_burden], axis=1).max(axis=1)
    bmi_burden = bmi_burden.where(bmi.notna(), np.nan)

    physiology_strain = _row_mean_with_neutral(
        [heart_rate_burden, bmi_burden],
        raw.index,
    )

    observed_parts = []
    for column in ALL_CONTEXT_COLUMNS:
        if column in raw.columns:
            observed_parts.append(raw[column].notna().astype("float64"))
        else:
            observed_parts.append(pd.Series(0.0, index=raw.index, dtype="float64"))
    movement_context_observed_fraction = pd.concat(observed_parts, axis=1).mean(axis=1)

    new_features["movement_shortfall"] = movement_shortfall
    new_features["recovery_support"] = recovery_support
    new_features["physiology_strain"] = physiology_strain
    new_features["rest_day_plausibility"] = movement_shortfall * recovery_support * (1.0 - physiology_strain)
    new_features["sedentary_strain"] = movement_shortfall * physiology_strain * (1.0 - recovery_support)
    new_features["low_movement_ambiguity"] = movement_shortfall * (
        1.0 - (recovery_support - physiology_strain).abs()
    )
    new_features["movement_context_observed_fraction"] = movement_context_observed_fraction

    return new_features


FEATURE_GROUPS = [
    {
        "name": "low_movement_recovery_context",
        "fn": add_low_movement_recovery_context,
        "depends_on": [],
        "description": "Separates low observed movement that looks like recovery from low movement paired with physiological strain.",
    }
]