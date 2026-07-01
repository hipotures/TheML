import numpy as np
import pandas as pd


TRAIN_ID_MAX = 690087
SLEEP_NORMAL_MIN = 7.0
SLEEP_NORMAL_MAX = 9.0
SLEEP_SEVERE_SHORT_MAX = 5.0
SLEEP_SHORT_MAX = 7.0
BMI_UNDER_MAX = 18.5
BMI_NORMAL_MAX = 25.0
BMI_OVER_MAX = 30.0
QUALITY_PENALTY_MAP = {
    "good": 0.0,
    "average": 0.5,
    "poor": 1.0,
}


def _training_mask(raw):
    if "id" not in raw.columns:
        return pd.Series(True, index=raw.index)

    ids = pd.to_numeric(raw["id"], errors="coerce")
    mask = ids <= TRAIN_ID_MAX
    if mask.any():
        return mask.fillna(False)

    return pd.Series(True, index=raw.index)


def _train_quantile(series, train_mask, q, fallback):
    values = pd.to_numeric(series, errors="coerce")
    train_values = values.loc[train_mask].dropna()
    if train_values.empty:
        return fallback
    result = train_values.quantile(q)
    if pd.isna(result):
        return fallback
    return float(result)


def _train_median(series, train_mask, fallback):
    values = pd.to_numeric(series, errors="coerce")
    train_values = values.loc[train_mask].dropna()
    if train_values.empty:
        return fallback
    result = train_values.median()
    if pd.isna(result):
        return fallback
    return float(result)


def add_sleep_strain_echo(raw, deps, aux):
    index = raw.index
    train_mask = _training_mask(raw)

    sleep_raw = pd.to_numeric(raw.get("sleep_duration"), errors="coerce")
    heart_raw = pd.to_numeric(raw.get("heart_rate"), errors="coerce")
    bmi_raw = pd.to_numeric(raw.get("bmi"), errors="coerce")
    step_raw = pd.to_numeric(raw.get("step_count"), errors="coerce")
    exercise_raw = pd.to_numeric(raw.get("exercise_duration"), errors="coerce")

    sleep_median = _train_median(sleep_raw, train_mask, SLEEP_NORMAL_MIN)
    heart_median = _train_median(heart_raw, train_mask, 75.0)
    bmi_median = _train_median(bmi_raw, train_mask, 22.0)
    step_median = _train_median(step_raw, train_mask, 7000.0)
    exercise_median = _train_median(exercise_raw, train_mask, 30.0)

    heart_q75 = _train_quantile(heart_raw, train_mask, 0.75, heart_median)
    heart_q90 = _train_quantile(heart_raw, train_mask, 0.90, heart_q75)
    step_q25 = _train_quantile(step_raw, train_mask, 0.25, step_median)
    step_q75 = _train_quantile(step_raw, train_mask, 0.75, step_median)
    exercise_q25 = _train_quantile(exercise_raw, train_mask, 0.25, exercise_median)
    exercise_q75 = _train_quantile(exercise_raw, train_mask, 0.75, exercise_median)

    sleep = sleep_raw.fillna(sleep_median)
    heart = heart_raw.fillna(heart_median)
    bmi = bmi_raw.fillna(bmi_median)
    step = step_raw.fillna(step_median)
    exercise = exercise_raw.fillna(exercise_median)

    quality = raw.get("sleep_quality")
    if quality is None:
        quality_normalized = pd.Series("average", index=index)
    else:
        quality_normalized = quality.astype("string").str.lower().str.strip()

    quality_penalty = quality_normalized.map(QUALITY_PENALTY_MAP).fillna(0.5).astype(float)
    quality_bad_flag = (quality_penalty >= 1.0).astype(int)

    sleep_debt = (SLEEP_NORMAL_MIN - sleep).clip(lower=0.0)
    sleep_excess = (sleep - SLEEP_NORMAL_MAX).clip(lower=0.0)
    rest_deficit = sleep_debt.clip(upper=3.0) + 0.5 * sleep_excess.clip(upper=2.0) + quality_penalty

    pulse_tail = pd.Series(0.0, index=index)
    if heart_q90 > heart_q75:
        pulse_tail = pulse_tail.mask(heart >= heart_q75, 0.5)
        pulse_tail = pulse_tail.mask(heart >= heart_q90, 1.0)
    else:
        pulse_tail = pulse_tail.mask(heart > heart_q75, 1.0)

    bmi_tail = pd.Series(0.0, index=index)
    bmi_tail = bmi_tail.mask(((bmi >= 16.0) & (bmi < BMI_UNDER_MAX)) | ((bmi >= BMI_NORMAL_MAX) & (bmi < BMI_OVER_MAX)), 0.5)
    bmi_tail = bmi_tail.mask(bmi >= BMI_OVER_MAX, 1.0)

    exertion_high = ((step >= step_q75) | (exercise >= exercise_q75)).astype(float)
    exertion_low = ((step <= step_q25) & (exercise <= exercise_q25)).astype(float)

    numeric_context_missing = sleep_raw.isna() & heart_raw.isna() & bmi_raw.isna() & step_raw.isna() & exercise_raw.isna()
    pulse_tail = pulse_tail.mask(numeric_context_missing, 0.0)
    bmi_tail = bmi_tail.mask(numeric_context_missing, 0.0)
    exertion_high = exertion_high.mask(numeric_context_missing, 0.0)
    exertion_low = exertion_low.mask(numeric_context_missing, 0.0)

    sleep_band = pd.Series("normal", index=index, dtype="object")
    sleep_band = sleep_band.mask(sleep < SLEEP_SEVERE_SHORT_MAX, "severe_short")
    sleep_band = sleep_band.mask((sleep >= SLEEP_SEVERE_SHORT_MAX) & (sleep < SLEEP_SHORT_MAX), "short")
    sleep_band = sleep_band.mask(sleep > SLEEP_NORMAL_MAX, "long")
    sleep_band = sleep_band.mask(sleep_raw.isna(), "missing")

    high_physio_strain = ((pulse_tail + bmi_tail) >= 1.0).astype(int)
    normal_sleep = (sleep >= SLEEP_NORMAL_MIN) & (sleep <= SLEEP_NORMAL_MAX)
    good_quality = quality_penalty == 0.0

    restorative_alignment = (
        normal_sleep
        & good_quality
        & (bmi_tail == 0.0)
        & (pulse_tail == 0.0)
        & (~numeric_context_missing)
    ).astype(int)

    phenotype_code = (
        sleep_band.astype(str)
        + "_qbad"
        + quality_bad_flag.astype(str)
        + "_strain"
        + high_physio_strain.astype(str)
    )

    features = pd.DataFrame(index=index)
    features["rest_deficit"] = rest_deficit.astype(float)
    features["sleep_physio_echo"] = (rest_deficit * (pulse_tail + bmi_tail)).astype(float)
    features["sleep_exertion_overreach"] = (rest_deficit * pulse_tail * exertion_high).astype(float)
    features["quiet_sleep_deficit"] = (rest_deficit * (1.0 - pulse_tail) * (1.0 - bmi_tail)).astype(float)
    features["restorative_alignment"] = restorative_alignment.astype(int)
    features["sleep_strain_phenotype"] = phenotype_code.astype("string")
    features["sleep_band"] = sleep_band.astype("string")
    features["pulse_tail"] = pulse_tail.astype(float)
    features["bmi_tail"] = bmi_tail.astype(float)
    features["exertion_high"] = exertion_high.astype(float)
    features["exertion_low"] = exertion_low.astype(float)

    return features


FEATURE_GROUPS = [
    {
        "name": "sleep_strain_echo",
        "fn": add_sleep_strain_echo,
        "depends_on": [],
        "description": "Sleep quantity and perceived quality features coupled with pulse, BMI, and exertion strain.",
    }
]