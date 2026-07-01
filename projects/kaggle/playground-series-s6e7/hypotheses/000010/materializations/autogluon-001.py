import numpy as np
import pandas as pd


GENDER_BASELINE_NEED = {
    "male": 3.0,
    "female": 2.2,
    "other": 2.6,
}

SMOKING_ALCOHOL_ADDON = {
    "yes": 0.25,
    "occasional": 0.10,
    "no": 0.0,
}

HYDRATION_TIER_LABELS = [
    "unknown",
    "severe_deficit",
    "deficit",
    "adequate",
    "high",
    "excess",
]


def add_activity_adjusted_hydration_reserve(raw, deps, aux):
    index = raw.index

    gender = raw.get("gender", pd.Series(index=index, dtype="object"))
    gender_norm = gender.astype("string").str.lower()
    baseline_need = gender_norm.map(GENDER_BASELINE_NEED).fillna(2.6).astype("float64")

    smoking_alcohol = raw.get("smoking_alcohol", pd.Series(index=index, dtype="object"))
    smoking_norm = smoking_alcohol.astype("string").str.lower()
    smoking_alcohol_addon = smoking_norm.map(SMOKING_ALCOHOL_ADDON).fillna(0.0).astype("float64")

    exercise_duration = pd.to_numeric(
        raw.get("exercise_duration", pd.Series(index=index, dtype="float64")),
        errors="coerce",
    ).clip(lower=0.0)
    step_count = pd.to_numeric(
        raw.get("step_count", pd.Series(index=index, dtype="float64")),
        errors="coerce",
    ).clip(lower=0.0)
    calorie_expenditure = pd.to_numeric(
        raw.get("calorie_expenditure", pd.Series(index=index, dtype="float64")),
        errors="coerce",
    ).clip(lower=0.0)
    water_intake = pd.to_numeric(
        raw.get("water_intake", pd.Series(index=index, dtype="float64")),
        errors="coerce",
    )

    if "id" in raw.columns:
        id_values = pd.to_numeric(raw["id"], errors="coerce")
        train_mask = id_values < 690088
        train_calories = calorie_expenditure.where(train_mask)
        calorie_median = train_calories.median()
    else:
        calorie_median = calorie_expenditure.median()

    if pd.isna(calorie_median):
        calorie_median = 0.0

    exercise_addon = 0.6 * exercise_duration.fillna(0.0) / 60.0
    step_addon = 0.15 * (step_count.fillna(0.0) - 8000.0).clip(lower=0.0) / 5000.0
    calorie_addon = 0.00015 * (calorie_expenditure.fillna(0.0) - calorie_median).clip(lower=0.0)
    activity_addon = exercise_addon + step_addon + calorie_addon

    estimated_need = baseline_need + activity_addon + smoking_alcohol_addon
    safe_estimated_need = estimated_need.clip(lower=0.25)
    safe_water_intake = water_intake.clip(lower=0.0)

    hydration_gap = water_intake - estimated_need
    hydration_ratio = (water_intake / safe_estimated_need).clip(lower=0.0, upper=2.0)
    activity_water_pressure = activity_addon / safe_water_intake.fillna(0.0).clip(lower=0.25)

    hydration_tier = pd.Series("unknown", index=index, dtype="object")
    known_ratio = hydration_ratio.notna()
    hydration_tier.loc[known_ratio & (hydration_ratio < 0.55)] = "severe_deficit"
    hydration_tier.loc[known_ratio & (hydration_ratio >= 0.55) & (hydration_ratio < 0.80)] = "deficit"
    hydration_tier.loc[known_ratio & (hydration_ratio >= 0.80) & (hydration_ratio < 1.25)] = "adequate"
    hydration_tier.loc[known_ratio & (hydration_ratio >= 1.25) & (hydration_ratio < 1.60)] = "high"
    hydration_tier.loc[known_ratio & (hydration_ratio >= 1.60)] = "excess"

    new_features = pd.DataFrame(
        {
            "estimated_hydration_need_l": estimated_need.astype("float64"),
            "hydration_gap_l": hydration_gap.astype("float64"),
            "hydration_ratio": hydration_ratio.astype("float64"),
            "activity_water_pressure": activity_water_pressure.astype("float64"),
            "activity_hydration_addon_l": activity_addon.astype("float64"),
            "smoking_alcohol_hydration_addon_l": smoking_alcohol_addon.astype("float64"),
            "hydration_tier": hydration_tier,
        },
        index=index,
    )

    return new_features


FEATURE_GROUPS = [
    {
        "name": "activity_adjusted_hydration_reserve",
        "fn": add_activity_adjusted_hydration_reserve,
        "depends_on": [],
        "description": "Estimates hydration reserve by comparing reported water intake with activity-, calorie-, smoking-, and gender-adjusted fluid demand.",
    }
]