import numpy as np
import pandas as pd


BEHAVIOR_COMPONENT_COLUMNS = (
    "sleep_duration",
    "step_count",
    "exercise_duration",
    "water_intake",
    "diet_type",
    "stress_level",
    "sleep_quality",
    "physical_activity_level",
    "smoking_alcohol",
)

PHYSIOLOGY_COMPONENT_COLUMNS = (
    "bmi",
    "heart_rate",
)

DIET_LOAD_MAP = {
    "balanced": 0.0,
    "veg": 0.2,
    "non-veg": 0.45,
}

STRESS_LOAD_MAP = {
    "low": 0.0,
    "medium": 0.5,
    "high": 1.0,
}

SLEEP_QUALITY_LOAD_MAP = {
    "good": 0.0,
    "average": 0.5,
    "poor": 1.0,
}

ACTIVITY_LOAD_MAP = {
    "active": 0.0,
    "moderate": 0.4,
    "sedentary": 1.0,
}

SMOKING_ALCOHOL_LOAD_MAP = {
    "no": 0.0,
    "occasional": 0.55,
    "yes": 1.0,
}


def _numeric_series(raw, column):
    if column in raw.columns:
        return pd.to_numeric(raw[column], errors="coerce")
    return pd.Series(np.nan, index=raw.index, dtype="float64")


def _category_series(raw, column):
    if column in raw.columns:
        return raw[column].astype("string").str.lower().str.strip()
    return pd.Series(pd.NA, index=raw.index, dtype="string")


def _bounded_mean(frame, columns, default_value):
    available = frame[columns].notna().sum(axis=1).astype("float64")
    load = frame[columns].mean(axis=1, skipna=True).fillna(default_value)
    coverage = available / float(len(columns))
    return load.clip(0.0, 1.0), coverage.clip(0.0, 1.0)


def _sleep_load(values):
    load = pd.Series(0.0, index=values.index, dtype="float64")
    low_mask = values < 7.0
    high_mask = values > 9.0
    load.loc[low_mask] = ((7.0 - values.loc[low_mask]) / 1.0).clip(0.0, 1.0)
    load.loc[high_mask] = ((values.loc[high_mask] - 9.0) / 0.5).clip(0.0, 1.0)
    load.loc[values.isna()] = np.nan
    return load


def _steps_load(values):
    load = pd.Series(0.0, index=values.index, dtype="float64")
    low_mask = values < 7500.0
    high_mask = values > 12000.0
    load.loc[low_mask] = ((7500.0 - values.loc[low_mask]) / 2500.0).clip(0.0, 1.0)
    load.loc[high_mask] = ((values.loc[high_mask] - 12000.0) / 3000.0).clip(0.0, 0.25)
    load.loc[values.isna()] = np.nan
    return load


def _exercise_load(values):
    load = pd.Series(0.0, index=values.index, dtype="float64")
    low_mask = values < 30.0
    high_mask = values > 75.0
    load.loc[low_mask] = ((30.0 - values.loc[low_mask]) / 10.0).clip(0.0, 1.0)
    load.loc[high_mask] = ((values.loc[high_mask] - 75.0) / 45.0).clip(0.0, 0.2)
    load.loc[values.isna()] = np.nan
    return load


def _water_load(values):
    load = pd.Series(0.0, index=values.index, dtype="float64")
    low_mask = values < 1.5
    high_mask = values > 3.5
    load.loc[low_mask] = ((1.5 - values.loc[low_mask]) / 1.0).clip(0.0, 1.0)
    load.loc[high_mask] = ((values.loc[high_mask] - 3.5) / 1.5).clip(0.0, 0.35)
    load.loc[values.isna()] = np.nan
    return load


def _bmi_load(values):
    load = pd.Series(0.0, index=values.index, dtype="float64")
    low_mask = values < 18.5
    overweight_mask = (values >= 25.0) & (values < 30.0)
    high_mask = values >= 30.0

    load.loc[low_mask] = ((18.5 - values.loc[low_mask]) / 2.5).clip(0.0, 1.0) * 0.65
    load.loc[overweight_mask] = 0.35 + ((values.loc[overweight_mask] - 25.0) / 5.0).clip(0.0, 1.0) * 0.25
    load.loc[high_mask] = 0.75 + ((values.loc[high_mask] - 30.0) / 5.0).clip(0.0, 1.0) * 0.25
    load.loc[values.isna()] = np.nan
    return load.clip(0.0, 1.0)


def _heart_rate_load(values):
    load = pd.Series(0.0, index=values.index, dtype="float64")
    low_mask = values < 60.0
    mild_high_mask = (values > 80.0) & (values <= 90.0)
    high_mask = values > 90.0

    load.loc[low_mask] = ((60.0 - values.loc[low_mask]) / 10.0).clip(0.0, 1.0) * 0.45
    load.loc[mild_high_mask] = ((values.loc[mild_high_mask] - 80.0) / 10.0).clip(0.0, 1.0) * 0.55
    load.loc[high_mask] = 0.65 + ((values.loc[high_mask] - 90.0) / 30.0).clip(0.0, 1.0) * 0.35
    load.loc[values.isna()] = np.nan
    return load.clip(0.0, 1.0)


def add_behavior_physiology_lag_phase(raw, deps, aux):
    behavior_components = pd.DataFrame(index=raw.index)
    behavior_components["sleep_amount_load"] = _sleep_load(_numeric_series(raw, "sleep_duration"))
    behavior_components["movement_load"] = _steps_load(_numeric_series(raw, "step_count"))
    behavior_components["exercise_load"] = _exercise_load(_numeric_series(raw, "exercise_duration"))
    behavior_components["hydration_load"] = _water_load(_numeric_series(raw, "water_intake"))
    behavior_components["diet_load"] = _category_series(raw, "diet_type").map(DIET_LOAD_MAP).astype("float64")
    behavior_components["stress_load"] = _category_series(raw, "stress_level").map(STRESS_LOAD_MAP).astype("float64")
    behavior_components["sleep_quality_load"] = _category_series(raw, "sleep_quality").map(SLEEP_QUALITY_LOAD_MAP).astype("float64")
    behavior_components["stated_activity_load"] = _category_series(raw, "physical_activity_level").map(ACTIVITY_LOAD_MAP).astype("float64")
    behavior_components["substance_load"] = _category_series(raw, "smoking_alcohol").map(SMOKING_ALCOHOL_LOAD_MAP).astype("float64")

    physiology_components = pd.DataFrame(index=raw.index)
    physiology_components["bmi_strain_load"] = _bmi_load(_numeric_series(raw, "bmi"))
    physiology_components["heart_rate_strain_load"] = _heart_rate_load(_numeric_series(raw, "heart_rate"))

    behavior_load, behavior_coverage = _bounded_mean(
        behavior_components,
        tuple(behavior_components.columns),
        0.5,
    )
    physiology_load, physiology_coverage = _bounded_mean(
        physiology_components,
        tuple(physiology_components.columns),
        0.5,
    )

    lag_gap = physiology_load - behavior_load
    absolute_lag = lag_gap.abs()
    coverage_weighted_lag = lag_gap * np.sqrt(behavior_coverage * physiology_coverage)

    both_low = (behavior_load <= 0.33) & (physiology_load <= 0.33)
    both_high = (behavior_load >= 0.67) & (physiology_load >= 0.67)
    behavior_high_phys_low = (behavior_load >= 0.67) & (physiology_load <= 0.33)
    physiology_high_behavior_low = (physiology_load >= 0.67) & (behavior_load <= 0.33)

    phase_code = pd.Series("transitional_mixed", index=raw.index, dtype="object")
    phase_code.loc[both_low] = "consistent_fit_like"
    phase_code.loc[both_high] = "entrenched_risk"
    phase_code.loc[behavior_high_phys_low] = "latent_behavior_risk"
    phase_code.loc[physiology_high_behavior_low] = "embodied_legacy_risk"

    phase_ordinal = pd.Series(2, index=raw.index, dtype="int8")
    phase_ordinal.loc[both_low] = 0
    phase_ordinal.loc[behavior_high_phys_low] = 1
    phase_ordinal.loc[physiology_high_behavior_low] = 3
    phase_ordinal.loc[both_high] = 4

    new_features = pd.DataFrame(index=raw.index)
    new_features["behavior_load"] = behavior_load.astype("float32")
    new_features["physiology_load"] = physiology_load.astype("float32")
    new_features["behavior_coverage"] = behavior_coverage.astype("float32")
    new_features["physiology_coverage"] = physiology_coverage.astype("float32")
    new_features["lag_gap"] = lag_gap.astype("float32")
    new_features["absolute_lag"] = absolute_lag.astype("float32")
    new_features["coverage_weighted_lag"] = coverage_weighted_lag.astype("float32")
    new_features["phase_code"] = phase_code
    new_features["phase_ordinal"] = phase_ordinal
    new_features["behavior_high_physiology_low"] = behavior_high_phys_low.astype("int8")
    new_features["physiology_high_behavior_low"] = physiology_high_behavior_low.astype("int8")
    new_features["both_domains_low_load"] = both_low.astype("int8")
    new_features["both_domains_high_load"] = both_high.astype("int8")
    return new_features


FEATURE_GROUPS = [
    {
        "name": "behavior_physiology_lag_phase",
        "fn": add_behavior_physiology_lag_phase,
        "depends_on": [],
        "description": "Contrasts near-term behavior load with embodied physiology load to identify lagged health-risk phases.",
    }
]