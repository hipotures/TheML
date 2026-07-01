import numpy as np
import pandas as pd


TRAIN_ID_MAX = 690087

GENDER_MAP = {
    "male": 1.0,
    "female": 0.0,
    "other": 0.5,
}

EXPOSURE_MAP = {
    "no": 0.0,
    "occasional": 0.5,
    "yes": 1.0,
}

PAL_MAP = {
    "sedentary": 0.0,
    "moderate": 2.0,
    "active": 4.0,
}

NUMERIC_COLS = (
    "bmi",
    "heart_rate",
    "step_count",
    "exercise_duration",
)


def _reference_frame(raw):
    if "id" not in raw.columns:
        return raw

    ids = pd.to_numeric(raw["id"], errors="coerce")
    mask = ids.notna() & (ids <= TRAIN_ID_MAX)
    if bool(mask.any()):
        return raw.loc[mask]
    return raw


def _numeric_fill_and_clip(raw, ref, col):
    values = pd.to_numeric(raw[col], errors="coerce") if col in raw.columns else pd.Series(np.nan, index=raw.index)
    ref_values = pd.to_numeric(ref[col], errors="coerce") if col in ref.columns else pd.Series(np.nan, index=ref.index)

    median = ref_values.median()
    if pd.isna(median):
        median = values.median()
    if pd.isna(median):
        median = 0.0

    lower = ref_values.quantile(0.01)
    upper = ref_values.quantile(0.99)
    if pd.isna(lower):
        lower = values.quantile(0.01)
    if pd.isna(upper):
        upper = values.quantile(0.99)
    if pd.isna(lower):
        lower = median
    if pd.isna(upper):
        upper = median
    if lower > upper:
        lower, upper = upper, lower

    return values.fillna(median).clip(lower=lower, upper=upper)


def _mapped_category(raw, col, mapping, default):
    if col not in raw.columns:
        return pd.Series(default, index=raw.index, dtype="float64")

    normalized = raw[col].astype("string").str.lower().str.strip()
    mapped = normalized.map(mapping)
    return mapped.astype("float64").fillna(default)


def add_nonexercise_fitness_reserve(raw, deps, aux):
    ref = _reference_frame(raw)

    bmi = _numeric_fill_and_clip(raw, ref, "bmi")
    heart_rate = _numeric_fill_and_clip(raw, ref, "heart_rate")
    step_count = _numeric_fill_and_clip(raw, ref, "step_count")
    exercise_duration = _numeric_fill_and_clip(raw, ref, "exercise_duration")

    raw_steps = pd.to_numeric(raw["step_count"], errors="coerce") if "step_count" in raw.columns else pd.Series(np.nan, index=raw.index)
    raw_exercise = pd.to_numeric(raw["exercise_duration"], errors="coerce") if "exercise_duration" in raw.columns else pd.Series(np.nan, index=raw.index)

    sex_code = _mapped_category(raw, "gender", GENDER_MAP, 0.5)
    exposure = _mapped_category(raw, "smoking_alcohol", EXPOSURE_MAP, 0.5)
    pal_base = _mapped_category(raw, "physical_activity_level", PAL_MAP, 2.0)

    objective_activity_points = (
        (step_count >= 7500.0).astype("float64")
        + (step_count >= 10000.0).astype("float64")
        + (exercise_duration >= 30.0).astype("float64")
        + (exercise_duration >= 60.0).astype("float64")
    ).clip(lower=0.0, upper=4.0)
    both_activity_missing = raw_steps.isna() & raw_exercise.isna()
    objective_activity_points = objective_activity_points.where(~both_activity_missing, 2.0)

    pa_index = ((pal_base + objective_activity_points) / 2.0).clip(lower=0.0, upper=4.0)

    ecrf_met_proxy = (
        18.07
        + 2.77 * sex_code
        - 0.17 * bmi
        - 0.03 * heart_rate
        + pa_index
        - 0.6 * exposure
    )

    ref_ecrf = ecrf_met_proxy.loc[ref.index.intersection(ecrf_met_proxy.index)]
    ecrf_mean = ref_ecrf.mean()
    ecrf_std = ref_ecrf.std()
    if pd.isna(ecrf_mean):
        ecrf_mean = ecrf_met_proxy.mean()
    if pd.isna(ecrf_mean):
        ecrf_mean = 0.0
    if pd.isna(ecrf_std) or ecrf_std == 0:
        ecrf_std = ecrf_met_proxy.std()
    if pd.isna(ecrf_std) or ecrf_std == 0:
        ecrf_std = 1.0

    lower_tertile = ref_ecrf.quantile(1.0 / 3.0)
    upper_tertile = ref_ecrf.quantile(2.0 / 3.0)
    if pd.isna(lower_tertile):
        lower_tertile = ecrf_met_proxy.quantile(1.0 / 3.0)
    if pd.isna(upper_tertile):
        upper_tertile = ecrf_met_proxy.quantile(2.0 / 3.0)
    if pd.isna(lower_tertile):
        lower_tertile = ecrf_mean
    if pd.isna(upper_tertile):
        upper_tertile = ecrf_mean
    if lower_tertile > upper_tertile:
        lower_tertile, upper_tertile = upper_tertile, lower_tertile

    ecrf_band = pd.Series("mid", index=raw.index, dtype="object")
    ecrf_band = ecrf_band.mask(ecrf_met_proxy <= lower_tertile, "low")
    ecrf_band = ecrf_band.mask(ecrf_met_proxy >= upper_tertile, "high")

    ecrf_tertile_rank = pd.Series(2.0, index=raw.index)
    ecrf_tertile_rank = ecrf_tertile_rank.mask(ecrf_met_proxy <= lower_tertile, 1.0)
    ecrf_tertile_rank = ecrf_tertile_rank.mask(ecrf_met_proxy >= upper_tertile, 3.0)

    activity_capacity_gap = objective_activity_points - ecrf_tertile_rank

    features = pd.DataFrame(index=raw.index)
    features["ecrf_met_proxy"] = ecrf_met_proxy.astype("float64")
    features["ecrf_z"] = ((ecrf_met_proxy - ecrf_mean) / ecrf_std).astype("float64")
    features["ecrf_band"] = ecrf_band
    features["objective_activity_points"] = objective_activity_points.astype("float64")
    features["pa_index"] = pa_index.astype("float64")
    features["activity_capacity_gap"] = activity_capacity_gap.astype("float64")
    features["high_demand_low_capacity"] = ((objective_activity_points >= 3.0) & (ecrf_band == "low")).astype("int8")
    features["low_demand_high_capacity"] = ((objective_activity_points <= 1.0) & (ecrf_band == "high")).astype("int8")

    return features


FEATURE_GROUPS = [
    {
        "name": "nonexercise_fitness_reserve",
        "fn": add_nonexercise_fitness_reserve,
        "depends_on": [],
        "description": "Estimates latent cardiorespiratory reserve from body context, pulse, activity, and substance exposure.",
    }
]