import numpy as np
import pandas as pd


STRESS_RISK_MAP = {"low": 0.0, "medium": 0.5, "high": 1.0}
SMOKING_ALCOHOL_RISK_MAP = {"no": 0.0, "occasional": 0.5, "yes": 1.0}
DIET_RISK_MAP = {"balanced": 0.0, "veg": 0.25, "non-veg": 0.5}
SLEEP_QUALITY_RISK_MAP = {"good": 0.0, "average": 0.5, "poor": 1.0}
ACTIVITY_LEVEL_RISK_MAP = {"active": 0.0, "moderate": 0.5, "sedentary": 1.0}


def _numeric(raw, column):
    if column not in raw.columns:
        return pd.Series(np.nan, index=raw.index, dtype="float64")
    return pd.to_numeric(raw[column], errors="coerce").astype("float64")


def _mapped_category(raw, column, mapping):
    if column not in raw.columns:
        return pd.Series(np.nan, index=raw.index, dtype="float64")
    values = raw[column].astype("string").str.strip().str.lower()
    return values.map(mapping).astype("float64")


def _clip01(series):
    return series.clip(lower=0.0, upper=1.0)


def _observed_mean(components, index):
    block = pd.concat(components, axis=1)
    return block.mean(axis=1, skipna=True).fillna(0.5).astype("float64")


def add_lifestyle_risk_cascade_alignment(raw, deps, aux):
    stress_risk = _mapped_category(raw, "stress_level", STRESS_RISK_MAP)
    smoking_alcohol_risk = _mapped_category(raw, "smoking_alcohol", SMOKING_ALCOHOL_RISK_MAP)
    diet_risk = _mapped_category(raw, "diet_type", DIET_RISK_MAP)
    water = _numeric(raw, "water_intake")
    water_deficit = _clip01((2.0 - water) / 1.5)

    exposure_load = _observed_mean(
        [stress_risk, smoking_alcohol_risk, diet_risk, water_deficit],
        raw.index,
    )

    sleep_quality_risk = _mapped_category(raw, "sleep_quality", SLEEP_QUALITY_RISK_MAP)
    sleep_duration = _numeric(raw, "sleep_duration")
    short_sleep_risk = (7.0 - sleep_duration) / 4.0
    long_sleep_risk = (sleep_duration - 9.0) / 1.0
    sleep_duration_risk = _clip01(pd.concat([short_sleep_risk, long_sleep_risk], axis=1).max(axis=1))

    recovery_impairment = _observed_mean(
        [sleep_quality_risk, sleep_duration_risk],
        raw.index,
    )

    activity_level_risk = _mapped_category(raw, "physical_activity_level", ACTIVITY_LEVEL_RISK_MAP)
    step_count = _numeric(raw, "step_count")
    exercise_duration = _numeric(raw, "exercise_duration")
    calorie_expenditure = _numeric(raw, "calorie_expenditure")

    step_deficit = _clip01((10000.0 - step_count) / 9000.0)
    exercise_deficit = _clip01((30.0 - exercise_duration) / 30.0)
    calorie_output_deficit = _clip01((1800.0 - calorie_expenditure) / 600.0)

    activity_shortfall = _observed_mean(
        [activity_level_risk, step_deficit, exercise_deficit, calorie_output_deficit],
        raw.index,
    )

    bmi = _numeric(raw, "bmi")
    heart_rate = _numeric(raw, "heart_rate")

    bmi_low_distance = (18.5 - bmi) / 2.5
    bmi_high_distance = (bmi - 25.0) / 10.0
    bmi_strain = _clip01(pd.concat([bmi_low_distance, bmi_high_distance], axis=1).max(axis=1))

    high_hr_strain = (heart_rate - 70.0) / 30.0
    low_hr_strain = 0.5 * (55.0 - heart_rate) / 10.0
    heart_rate_strain = _clip01(pd.concat([high_hr_strain, low_hr_strain], axis=1).max(axis=1))

    physiology_strain = _observed_mean(
        [bmi_strain, heart_rate_strain],
        raw.index,
    )

    gap_recovery_minus_exposure = recovery_impairment - exposure_load
    gap_activity_minus_recovery = activity_shortfall - recovery_impairment
    gap_physiology_minus_activity = physiology_strain - activity_shortfall

    cascade_slope = (
        -0.3 * exposure_load
        - 0.1 * recovery_impairment
        + 0.1 * activity_shortfall
        + 0.3 * physiology_strain
    )

    monotone_propagation_fraction = (
        (recovery_impairment >= exposure_load - 0.1).astype("float64")
        + (activity_shortfall >= recovery_impairment - 0.1).astype("float64")
        + (physiology_strain >= activity_shortfall - 0.1).astype("float64")
    ) / 3.0

    propagated_risk = (
        (
            (exposure_load + 0.05)
            * (recovery_impairment + 0.05)
            * (activity_shortfall + 0.05)
            * (physiology_strain + 0.05)
        )
        ** 0.25
        - 0.05
    ).clip(lower=0.0, upper=1.0)

    downstream_max = pd.concat(
        [recovery_impairment, activity_shortfall, physiology_strain],
        axis=1,
    ).max(axis=1)
    contained_exposure = (exposure_load - downstream_max).clip(lower=0.0)

    upstream_mean = pd.concat(
        [exposure_load, recovery_impairment, activity_shortfall],
        axis=1,
    ).mean(axis=1)
    unexplained_physiology = (physiology_strain - upstream_mean).clip(lower=0.0)

    return pd.DataFrame(
        {
            "exposure_load": exposure_load,
            "recovery_impairment": recovery_impairment,
            "activity_shortfall": activity_shortfall,
            "physiology_strain": physiology_strain,
            "cascade_slope": cascade_slope,
            "gap_recovery_minus_exposure": gap_recovery_minus_exposure,
            "gap_activity_minus_recovery": gap_activity_minus_recovery,
            "gap_physiology_minus_activity": gap_physiology_minus_activity,
            "monotone_propagation_fraction": monotone_propagation_fraction,
            "propagated_risk": propagated_risk,
            "contained_exposure": contained_exposure,
            "unexplained_physiology": unexplained_physiology,
        },
        index=raw.index,
    )


FEATURE_GROUPS = [
    {
        "name": "lifestyle_risk_cascade_alignment",
        "fn": add_lifestyle_risk_cascade_alignment,
        "depends_on": [],
        "description": "Encodes ordered lifestyle, recovery, activity, and physiological risks as a coherent cascade alignment feature group.",
    }
]