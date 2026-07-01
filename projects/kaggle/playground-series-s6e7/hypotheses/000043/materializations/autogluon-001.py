import numpy as np
import pandas as pd


TRAIN_ID_CUTOFF = 690088

SLEEP_QUALITY_MAP = {"poor": -1.0, "average": 0.0, "good": 1.0}
STRESS_LEVEL_MAP = {"high": -1.0, "medium": 0.0, "low": 1.0}
PHYSICAL_ACTIVITY_MAP = {"sedentary": -1.0, "moderate": 0.0, "active": 1.0}
SMOKING_ALCOHOL_MAP = {"yes": -1.0, "occasional": 0.0, "no": 1.0}
DIET_TYPE_MAP = {"balanced": 1.0, "veg": 0.25, "non-veg": -0.25}


def _numeric_series(raw, column):
    if column not in raw.columns:
        return pd.Series(0.0, index=raw.index, dtype="float64")
    return pd.to_numeric(raw[column], errors="coerce")


def _mapped_series(raw, column, mapping):
    if column not in raw.columns:
        return pd.Series(0.0, index=raw.index, dtype="float64")
    values = raw[column].map(mapping)
    return values.fillna(0.0).astype("float64")


def _triangular_score(values, low, best_low, best_high, high):
    score = pd.Series(0.0, index=values.index, dtype="float64")
    valid = values.notna()

    below = valid & (values < best_low)
    above = valid & (values > best_high)
    middle = valid & (values >= best_low) & (values <= best_high)

    score.loc[middle] = 1.0
    score.loc[below] = -1.0 + 2.0 * ((values.loc[below].clip(lower=low, upper=best_low) - low) / (best_low - low))
    score.loc[above] = 1.0 - 2.0 * ((values.loc[above].clip(lower=best_high, upper=high) - best_high) / (high - best_high))
    return score.clip(-1.0, 1.0).fillna(0.0).astype("float64")


def _scaled_clip_score(values, low, high):
    valid = values.notna()
    score = pd.Series(0.0, index=values.index, dtype="float64")
    score.loc[valid] = -1.0 + 2.0 * ((values.loc[valid].clip(lower=low, upper=high) - low) / (high - low))
    return score.clip(-1.0, 1.0).fillna(0.0).astype("float64")


def _train_mask(raw):
    if "id" not in raw.columns:
        return pd.Series(True, index=raw.index)
    ids = pd.to_numeric(raw["id"], errors="coerce")
    mask = ids < TRAIN_ID_CUTOFF
    if mask.any():
        return mask.fillna(False)
    return pd.Series(True, index=raw.index)


def _percentile_rank(values, reference):
    reference = np.sort(reference[np.isfinite(reference)])
    if reference.size == 0:
        return np.full(values.shape, 0.5, dtype="float64")
    left = np.searchsorted(reference, values, side="left")
    right = np.searchsorted(reference, values, side="right")
    return ((left + right) * 0.5 / reference.size).astype("float64")


def add_shared_lifestyle_covariance_factor(raw, deps, aux):
    sleep_quality = _mapped_series(raw, "sleep_quality", SLEEP_QUALITY_MAP)
    sleep_duration = _triangular_score(_numeric_series(raw, "sleep_duration"), 3.0, 7.0, 9.0, 10.0)
    stress_level = _mapped_series(raw, "stress_level", STRESS_LEVEL_MAP)
    physical_activity = _mapped_series(raw, "physical_activity_level", PHYSICAL_ACTIVITY_MAP)

    step_score = _scaled_clip_score(_numeric_series(raw, "step_count"), 3000.0, 10000.0)
    exercise_score = _scaled_clip_score(_numeric_series(raw, "exercise_duration"), 0.0, 60.0)
    movement = (step_score + exercise_score) * 0.5

    smoking_alcohol = _mapped_series(raw, "smoking_alcohol", SMOKING_ALCOHOL_MAP)
    diet_type = _mapped_series(raw, "diet_type", DIET_TYPE_MAP)
    water_intake = _triangular_score(_numeric_series(raw, "water_intake"), 0.5, 2.0, 3.5, 4.72)

    base = pd.DataFrame(
        {
            "sleep_quality_base": sleep_quality,
            "sleep_duration_base": sleep_duration,
            "stress_base": stress_level,
            "activity_level_base": physical_activity,
            "movement_base": movement,
            "smoking_alcohol_base": smoking_alcohol,
            "diet_base": diet_type,
            "water_base": water_intake,
        },
        index=raw.index,
    )

    train = _train_mask(raw)
    train_base = base.loc[train]
    if train_base.empty:
        train_base = base

    medians = train_base.median(axis=0)
    q75 = train_base.quantile(0.75, axis=0)
    q25 = train_base.quantile(0.25, axis=0)
    iqr = (q75 - q25).replace(0.0, 1.0).fillna(1.0)

    standardized = ((base - medians) / iqr).replace([np.inf, -np.inf], 0.0).fillna(0.0)
    train_standardized = standardized.loc[train]
    if train_standardized.shape[0] < 2:
        train_standardized = standardized

    cov = np.cov(train_standardized.to_numpy(dtype="float64"), rowvar=False)
    if cov.ndim == 0:
        direction = np.ones(standardized.shape[1], dtype="float64")
    else:
        cov = np.nan_to_num(cov, nan=0.0, posinf=0.0, neginf=0.0)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        direction = eigenvectors[:, int(np.argmax(eigenvalues))].astype("float64")

    norm = np.linalg.norm(direction)
    if not np.isfinite(norm) or norm == 0.0:
        direction = np.ones(standardized.shape[1], dtype="float64") / np.sqrt(float(standardized.shape[1]))
    else:
        direction = direction / norm

    protective_anchor = np.ones(standardized.shape[1], dtype="float64")
    if float(np.dot(direction, protective_anchor)) < 0.0:
        direction = -direction

    matrix = standardized.to_numpy(dtype="float64")
    factor = matrix @ direction
    reconstruction = np.outer(factor, direction)
    residual = matrix - reconstruction
    residual_rms = np.sqrt(np.mean(residual * residual, axis=1))

    train_factor = factor[train.to_numpy(dtype=bool)]
    factor_percentile = _percentile_rank(factor, train_factor)
    factor_sign = np.sign(factor)
    base_signs = np.sign(base.to_numpy(dtype="float64"))
    opposing_count = ((base_signs != 0.0) & (factor_sign[:, None] != 0.0) & (base_signs != factor_sign[:, None])).sum(axis=1)

    return pd.DataFrame(
        {
            "common_factor_score": factor.astype("float64"),
            "common_factor_train_percentile": factor_percentile.astype("float64"),
            "common_factor_abs_magnitude": np.abs(factor).astype("float64"),
            "one_factor_residual_rms": residual_rms.astype("float64"),
            "opposing_base_score_count": opposing_count.astype("int16"),
        },
        index=raw.index,
    )


FEATURE_GROUPS = [
    {
        "name": "shared_lifestyle_covariance_factor",
        "fn": add_shared_lifestyle_covariance_factor,
        "depends_on": [],
        "description": "Unsupervised lifestyle covariance factor capturing coherent protective or unfavorable routine health behavior patterns.",
    }
]