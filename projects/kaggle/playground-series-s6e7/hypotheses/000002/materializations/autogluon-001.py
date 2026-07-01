import numpy as np
import pandas as pd

TRAIN_ID_MAX = 690087
EPSILON = 1e-12
HR_PER_STEP_LOW_MAX = 0.0081
HR_PER_STEP_MEDIUM_MAX = 0.0147
LOWER_QUANTILE = 0.005
UPPER_QUANTILE = 0.995


def _numeric(raw, name):
    if name not in raw.columns:
        return pd.Series(np.nan, index=raw.index, dtype="float64")
    return pd.to_numeric(raw[name], errors="coerce")


def _safe_divide(numerator, denominator, scale=1.0):
    denominator = denominator.where(denominator.abs() > EPSILON)
    return scale * numerator / denominator


def _train_mask(raw):
    if "id" not in raw.columns:
        return pd.Series(True, index=raw.index)
    ids = pd.to_numeric(raw["id"], errors="coerce")
    mask = ids.le(TRAIN_ID_MAX)
    if bool(mask.any()):
        return mask.fillna(False)
    return pd.Series(True, index=raw.index)


def _robust_z_from_train(values, train_mask):
    train_values = values.loc[train_mask].dropna()
    if train_values.empty:
        return pd.Series(np.nan, index=values.index, dtype="float64")

    lower = train_values.quantile(LOWER_QUANTILE)
    upper = train_values.quantile(UPPER_QUANTILE)
    clipped = values.clip(lower=lower, upper=upper)
    train_clipped = clipped.loc[train_mask].dropna()

    median = train_clipped.median()
    mad = (train_clipped - median).abs().median()
    if not np.isfinite(mad) or mad <= EPSILON:
        scale = train_clipped.std(ddof=0)
    else:
        scale = 1.4826 * mad

    if not np.isfinite(scale) or scale <= EPSILON:
        return pd.Series(0.0, index=values.index, dtype="float64").where(values.notna())

    return (clipped - median) / scale


def add_activity_normalized_cardio_strain(raw, deps, aux):
    heart_rate = _numeric(raw, "heart_rate")
    step_count = _numeric(raw, "step_count")
    exercise_duration = _numeric(raw, "exercise_duration")
    calorie_expenditure = _numeric(raw, "calorie_expenditure")

    exercise_plus_one = exercise_duration + 1.0

    hr_per_1k_steps = _safe_divide(heart_rate, step_count, scale=1000.0)
    hr_per_exercise_min_plus1 = _safe_divide(heart_rate, exercise_plus_one)
    calories_per_1k_steps = _safe_divide(calorie_expenditure, step_count, scale=1000.0)
    calories_per_exercise_min_plus1 = _safe_divide(calorie_expenditure, exercise_plus_one)
    steps_per_exercise_min_plus1 = _safe_divide(step_count, exercise_plus_one)

    train_mask = _train_mask(raw)
    hr_step_z = _robust_z_from_train(hr_per_1k_steps, train_mask)
    calorie_step_z = _robust_z_from_train(calories_per_1k_steps, train_mask)
    cardio_calorie_step_strain_residual = hr_step_z - calorie_step_z

    hr_per_step = _safe_divide(heart_rate, step_count)
    hr_per_step_band = pd.Series("unknown", index=raw.index, dtype="object")
    valid_band = hr_per_step.notna()
    hr_per_step_band.loc[valid_band & hr_per_step.le(HR_PER_STEP_LOW_MAX)] = "low"
    hr_per_step_band.loc[
        valid_band
        & hr_per_step.gt(HR_PER_STEP_LOW_MAX)
        & hr_per_step.le(HR_PER_STEP_MEDIUM_MAX)
    ] = "medium"
    hr_per_step_band.loc[valid_band & hr_per_step.gt(HR_PER_STEP_MEDIUM_MAX)] = "high"

    return pd.DataFrame(
        {
            "hr_per_1k_steps": hr_per_1k_steps,
            "hr_per_exercise_min_plus1": hr_per_exercise_min_plus1,
            "calories_per_1k_steps": calories_per_1k_steps,
            "calories_per_exercise_min_plus1": calories_per_exercise_min_plus1,
            "steps_per_exercise_min_plus1": steps_per_exercise_min_plus1,
            "cardio_calorie_step_strain_residual": cardio_calorie_step_strain_residual,
            "hr_per_step_band": hr_per_step_band,
        },
        index=raw.index,
    )


FEATURE_GROUPS = [
    {
        "name": "activity_normalized_cardio_strain",
        "fn": add_activity_normalized_cardio_strain,
        "depends_on": [],
        "description": "Activity-normalized cardiovascular and energy strain ratios comparing heart-rate and calorie signals against movement and exercise output.",
    }
]