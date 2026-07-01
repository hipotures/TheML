import numpy as np
import pandas as pd


CALORIE_TRAIN_ID_MAX = 690087

CATEGORICAL_STAGE_MAPS = {
    "stress_level": {
        "low": -1.0,
        "medium": 0.0,
        "high": 1.0,
    },
    "sleep_quality": {
        "good": -1.0,
        "average": 0.0,
        "poor": 1.0,
    },
    "physical_activity_level": {
        "active": -1.0,
        "moderate": 0.0,
        "sedentary": 1.0,
    },
    "smoking_alcohol": {
        "no": -1.0,
        "occasional": 0.0,
        "yes": 1.0,
    },
    "diet_type": {
        "balanced": -1.0,
        "veg": 0.0,
        "non-veg": 1.0,
    },
}


def _numeric_series(raw, column):
    if column not in raw.columns:
        return pd.Series(np.nan, index=raw.index, dtype="float64")
    return pd.to_numeric(raw[column], errors="coerce")


def _stage_sleep(raw):
    x = _numeric_series(raw, "sleep_duration")
    out = pd.Series(np.nan, index=raw.index, dtype="float64")
    out[(x >= 7.0) & (x <= 9.0)] = -1.0
    out[((x >= 6.0) & (x < 7.0)) | ((x > 9.0) & (x <= 10.0))] = 0.0
    out[(x < 6.0) | (x > 10.0)] = 1.0
    return out


def _stage_bmi(raw):
    x = _numeric_series(raw, "bmi")
    out = pd.Series(np.nan, index=raw.index, dtype="float64")
    out[(x >= 18.5) & (x < 25.0)] = -1.0
    out[((x >= 16.0) & (x < 18.5)) | ((x >= 25.0) & (x < 30.0))] = 0.0
    out[(x < 16.0) | (x >= 30.0)] = 1.0
    return out


def _stage_heart_rate(raw):
    x = _numeric_series(raw, "heart_rate")
    out = pd.Series(np.nan, index=raw.index, dtype="float64")
    out[(x >= 50.0) & (x <= 72.0)] = -1.0
    out[(x < 50.0) | ((x > 72.0) & (x <= 88.0))] = 0.0
    out[x > 88.0] = 1.0
    return out


def _stage_exercise(raw):
    x = _numeric_series(raw, "exercise_duration")
    out = pd.Series(np.nan, index=raw.index, dtype="float64")
    out[x >= 30.0] = -1.0
    out[(x >= 10.0) & (x < 30.0)] = 0.0
    out[x < 10.0] = 1.0
    return out


def _stage_steps(raw):
    x = _numeric_series(raw, "step_count")
    out = pd.Series(np.nan, index=raw.index, dtype="float64")
    out[x >= 8000.0] = -1.0
    out[(x >= 4000.0) & (x < 8000.0)] = 0.0
    out[x < 4000.0] = 1.0
    return out


def _stage_water(raw):
    x = _numeric_series(raw, "water_intake")
    out = pd.Series(np.nan, index=raw.index, dtype="float64")
    out[(x >= 2.0) & (x <= 3.5)] = -1.0
    out[((x >= 1.0) & (x < 2.0)) | ((x > 3.5) & (x <= 4.5))] = 0.0
    out[(x < 1.0) | (x > 4.5)] = 1.0
    return out


def _stage_calories(raw):
    x = _numeric_series(raw, "calorie_expenditure")
    train_mask = pd.Series(True, index=raw.index)

    if "id" in raw.columns:
        ids = pd.to_numeric(raw["id"], errors="coerce")
        train_mask = ids <= CALORIE_TRAIN_ID_MAX

    fit_values = x[train_mask & x.notna()]
    if fit_values.empty:
        fit_values = x[x.notna()]

    out = pd.Series(np.nan, index=raw.index, dtype="float64")
    if fit_values.empty:
        return out

    q20 = fit_values.quantile(0.20)
    q40 = fit_values.quantile(0.40)
    q85 = fit_values.quantile(0.85)
    q95 = fit_values.quantile(0.95)

    out[(x >= q40) & (x <= q85)] = -1.0
    out[((x >= q20) & (x < q40)) | ((x > q85) & (x <= q95))] = 0.0
    out[(x < q20) | (x > q95)] = 1.0
    return out


def _stage_categorical(raw, column):
    if column not in raw.columns:
        return pd.Series(np.nan, index=raw.index, dtype="float64")

    normalized = raw[column].astype("string").str.strip().str.lower()
    return normalized.map(CATEGORICAL_STAGE_MAPS[column]).astype("float64")


def add_at_risk_corridor_occupancy(raw, deps, aux):
    stages = pd.DataFrame(index=raw.index)
    stages["sleep_stage"] = _stage_sleep(raw)
    stages["bmi_stage"] = _stage_bmi(raw)
    stages["heart_rate_stage"] = _stage_heart_rate(raw)
    stages["exercise_stage"] = _stage_exercise(raw)
    stages["step_stage"] = _stage_steps(raw)
    stages["water_stage"] = _stage_water(raw)
    stages["calorie_stage"] = _stage_calories(raw)
    stages["stress_stage"] = _stage_categorical(raw, "stress_level")
    stages["sleep_quality_stage"] = _stage_categorical(raw, "sleep_quality")
    stages["activity_stage"] = _stage_categorical(raw, "physical_activity_level")
    stages["smoking_alcohol_stage"] = _stage_categorical(raw, "smoking_alcohol")
    stages["diet_stage"] = _stage_categorical(raw, "diet_type")

    known_stage_count = stages.notna().sum(axis=1).astype("float64")
    fit_stage_count = stages.eq(-1.0).sum(axis=1).astype("float64")
    corridor_stage_count = stages.eq(0.0).sum(axis=1).astype("float64")
    unhealthy_stage_count = stages.eq(1.0).sum(axis=1).astype("float64")

    denom = known_stage_count.replace(0.0, np.nan)
    fit_share = (fit_stage_count / denom).fillna(0.0)
    corridor_share = (corridor_stage_count / denom).fillna(0.0)
    unhealthy_share = (unhealthy_stage_count / denom).fillna(0.0)

    mean_stage = stages.mean(axis=1).fillna(0.0)
    max_stage = stages.max(axis=1).fillna(0.0)
    min_stage = stages.min(axis=1).fillna(0.0)

    shares = pd.DataFrame(
        {
            "fit": fit_share,
            "corridor": corridor_share,
            "unhealthy": unhealthy_share,
        },
        index=raw.index,
    )
    positive_shares = shares.where(shares > 0.0, 1.0)
    stage_entropy = -(shares.where(shares > 0.0, 0.0) * np.log(positive_shares)).sum(axis=1)

    features = stages.copy()
    features["fit_stage_count"] = fit_stage_count
    features["corridor_stage_count"] = corridor_stage_count
    features["unhealthy_stage_count"] = unhealthy_stage_count
    features["known_stage_count"] = known_stage_count
    features["fit_share"] = fit_share
    features["corridor_share"] = corridor_share
    features["unhealthy_share"] = unhealthy_share
    features["signed_stage_margin"] = fit_share - unhealthy_share
    features["corridor_dominance"] = corridor_share - pd.concat([fit_share, unhealthy_share], axis=1).max(axis=1)
    features["extreme_polarization"] = pd.concat([fit_share, unhealthy_share], axis=1).min(axis=1)
    features["mean_stage"] = mean_stage
    features["max_stage"] = max_stage
    features["min_stage"] = min_stage
    features["stage_entropy"] = stage_entropy.fillna(0.0)

    return features


FEATURE_GROUPS = [
    {
        "name": "at_risk_corridor_occupancy",
        "fn": add_at_risk_corridor_occupancy,
        "depends_on": [],
        "description": "Stages observable health signals into fit-like, at-risk-corridor, and unhealthy-like evidence and summarizes corridor occupancy.",
    }
]