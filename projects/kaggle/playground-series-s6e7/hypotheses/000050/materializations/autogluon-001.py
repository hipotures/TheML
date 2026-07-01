import numpy as np
import pandas as pd


STRESS_LEVEL_MAP = {
    "low": 0.0,
    "medium": 1.0,
    "high": 2.0,
}

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
    "non-veg": 0.25,
    "veg": 0.75,
    "balanced": 1.0,
}

SMOKING_ALCOHOL_MAP = {
    "yes": 0.0,
    "occasional": 0.5,
    "no": 1.0,
}

GENDER_LEVELS = ("female", "male", "other", "unknown")


def _numeric_series(raw, column, default=0.5):
    if column not in raw.columns:
        return pd.Series(default, index=raw.index, dtype="float64")
    return pd.to_numeric(raw[column], errors="coerce")


def _category_series(raw, column, default="unknown"):
    if column not in raw.columns:
        return pd.Series(default, index=raw.index, dtype="object")
    return raw[column].astype("object").where(raw[column].notna(), default).astype(str).str.lower()


def _mapped_score(raw, column, mapping, default=0.5):
    values = _category_series(raw, column)
    return values.map(mapping).fillna(default).astype("float64")


def _clipped_numeric(raw, column, default=0.5):
    values = _numeric_series(raw, column)
    valid = values.dropna()
    if valid.empty:
        return pd.Series(default, index=raw.index, dtype="float64")
    return values.clip(lower=valid.min(), upper=valid.max())


def _training_mask(raw):
    if "id" not in raw.columns:
        return pd.Series(True, index=raw.index)
    ids = pd.to_numeric(raw["id"], errors="coerce")
    if ids.notna().any() and ids.max(skipna=True) > 690087 and ids.min(skipna=True) <= 690087:
        return ids <= 690087
    return pd.Series(True, index=raw.index)


def _gender_step_thresholds(raw, gender, step_count):
    train_mask = _training_mask(raw)
    valid_train = train_mask & step_count.notna()
    global_threshold = step_count.loc[valid_train].quantile(0.60)
    if pd.isna(global_threshold):
        global_threshold = step_count.dropna().quantile(0.60)
    if pd.isna(global_threshold):
        global_threshold = 8000.0

    thresholds = {}
    for level in GENDER_LEVELS:
        mask = valid_train & (gender == level)
        threshold = step_count.loc[mask].quantile(0.60)
        thresholds[level] = float(global_threshold if pd.isna(threshold) else threshold)
    return thresholds, float(global_threshold)


def add_gender_stress_buffer_margin(raw, deps, aux):
    idx = raw.index

    gender = _category_series(raw, "gender")
    gender = gender.where(gender.isin(GENDER_LEVELS[:-1]), "unknown")

    stress_label = _category_series(raw, "stress_level")
    stress_score = stress_label.map(STRESS_LEVEL_MAP).fillna(1.0).astype("float64")
    stress_norm = stress_score / 2.0

    sleep_duration = _clipped_numeric(raw, "sleep_duration")
    sleep_duration_score = pd.Series(
        np.where(sleep_duration.isna(), 0.5, np.where((sleep_duration >= 7.0) & (sleep_duration <= 9.0), 1.0, 0.0)),
        index=idx,
        dtype="float64",
    )
    sleep_quality_score = _mapped_score(raw, "sleep_quality", SLEEP_QUALITY_MAP)
    sleep_buffer = (sleep_duration_score + sleep_quality_score) / 2.0

    step_count = _clipped_numeric(raw, "step_count")
    thresholds, global_step_threshold = _gender_step_thresholds(raw, gender, step_count)
    step_threshold = gender.map(thresholds).fillna(global_step_threshold).astype("float64")
    step_score = pd.Series(
        np.where(step_count.isna(), 0.5, np.where(step_count >= step_threshold, 1.0, 0.0)),
        index=idx,
        dtype="float64",
    )

    exercise_duration = _clipped_numeric(raw, "exercise_duration")
    exercise_score = pd.Series(
        np.where(exercise_duration.isna(), 0.5, np.where(exercise_duration >= 30.0, 1.0, 0.0)),
        index=idx,
        dtype="float64",
    )
    activity_level_score = _mapped_score(raw, "physical_activity_level", PHYSICAL_ACTIVITY_MAP)
    activity_buffer = (step_score + exercise_score + activity_level_score) / 3.0

    diet_buffer = _mapped_score(raw, "diet_type", DIET_TYPE_MAP)
    substance_buffer = _mapped_score(raw, "smoking_alcohol", SMOKING_ALCOHOL_MAP)

    bmi = _clipped_numeric(raw, "bmi")
    bmi_buffer = pd.Series(
        np.select(
            [
                bmi.isna(),
                (bmi >= 18.5) & (bmi <= 24.9),
                ((bmi >= 16.0) & (bmi < 18.5)) | ((bmi > 24.9) & (bmi < 30.0)),
                bmi >= 30.0,
            ],
            [0.5, 1.0, 0.5, 0.0],
            default=0.5,
        ),
        index=idx,
        dtype="float64",
    )

    components = pd.DataFrame(
        {
            "sleep_buffer": sleep_buffer,
            "activity_buffer": activity_buffer,
            "diet_buffer": diet_buffer,
            "substance_buffer": substance_buffer,
            "bmi_buffer": bmi_buffer,
        },
        index=idx,
    )

    buffer_mean = components.mean(axis=1)
    buffer_count = (components >= 0.75).sum(axis=1).astype("int16")
    stress_buffer_gap = buffer_mean - stress_norm
    stress_unbuffered_load = stress_norm * (1.0 - buffer_mean)

    buffer_band = pd.Series(
        np.select(
            [buffer_mean < 0.4, buffer_mean < 0.75],
            ["low", "mixed"],
            default="high",
        ),
        index=idx,
        dtype="object",
    )
    stress_cell = stress_label.where(stress_label.isin(("low", "medium", "high")), "unknown")
    gender_stress_buffer_cell = gender + "_" + stress_cell + "_" + buffer_band

    out = pd.DataFrame(index=idx)
    out["sleep_buffer_score"] = sleep_buffer.astype("float32")
    out["activity_buffer_score"] = activity_buffer.astype("float32")
    out["diet_buffer_score"] = diet_buffer.astype("float32")
    out["substance_buffer_score"] = substance_buffer.astype("float32")
    out["bmi_buffer_score"] = bmi_buffer.astype("float32")
    out["healthy_buffer_count"] = buffer_count
    out["healthy_buffer_mean"] = buffer_mean.astype("float32")
    out["stress_score"] = stress_score.astype("float32")
    out["stress_buffer_gap"] = stress_buffer_gap.astype("float32")
    out["stress_unbuffered_load"] = stress_unbuffered_load.astype("float32")

    for level in GENDER_LEVELS:
        mask = (gender == level).astype("float32")
        out[level + "_stress_buffer_gap"] = (stress_buffer_gap * mask).astype("float32")
        out[level + "_stress_unbuffered_load"] = (stress_unbuffered_load * mask).astype("float32")

    out["gender_stress_buffer_cell"] = gender_stress_buffer_cell.astype("object")
    return out


FEATURE_GROUPS = [
    {
        "name": "gender_stress_buffer_margin",
        "fn": add_gender_stress_buffer_margin,
        "depends_on": [],
        "description": "Scores gender-contextual healthy lifestyle buffers against perceived stress to capture buffered versus unbuffered stress load.",
    }
]