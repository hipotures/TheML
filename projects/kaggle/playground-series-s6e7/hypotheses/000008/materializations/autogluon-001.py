import numpy as np
import pandas as pd


NUMERIC_FEATURES = (
    "sleep_duration",
    "heart_rate",
    "bmi",
    "calorie_expenditure",
    "step_count",
    "exercise_duration",
    "water_intake",
)

GENDER_COLUMN = "gender"
MISSING_GROUP = "__missing__"
MIN_GROUP_SIZE = 1000
TRAIN_ID_MAX = 690087


def _as_train_mask(raw):
    if "id" not in raw.columns:
        return pd.Series(True, index=raw.index)

    ids = pd.to_numeric(raw["id"], errors="coerce")
    mask = ids.le(TRAIN_ID_MAX)
    if mask.any():
        return mask.fillna(False)

    return pd.Series(True, index=raw.index)


def _normalise_group(values):
    return values.astype("object").where(values.notna(), MISSING_GROUP).astype(str)


def _midrank_percentile(fit_values, transform_values):
    observed = pd.to_numeric(fit_values, errors="coerce").dropna().to_numpy(dtype=float)
    result = np.full(len(transform_values), np.nan, dtype=float)

    if observed.size == 0:
        return result

    observed.sort()
    x = pd.to_numeric(transform_values, errors="coerce").to_numpy(dtype=float)
    valid = ~np.isnan(x)

    if not valid.any():
        return result

    valid_x = x[valid]
    left = np.searchsorted(observed, valid_x, side="left")
    right = np.searchsorted(observed, valid_x, side="right")
    pct = (left + 0.5 * (right - left)) / float(observed.size)
    pct = np.where(valid_x < observed[0], 0.001, pct)
    pct = np.where(valid_x > observed[-1], 0.999, pct)
    result[valid] = np.clip(pct, 0.001, 0.999)
    return result


def add_gender_relative_numeric_percentiles(raw, deps, aux):
    train_mask = _as_train_mask(raw)
    train = raw.loc[train_mask]
    raw_gender = _normalise_group(raw[GENDER_COLUMN]) if GENDER_COLUMN in raw.columns else pd.Series(MISSING_GROUP, index=raw.index)
    train_gender = raw_gender.loc[train.index]

    features = pd.DataFrame(index=raw.index)
    tail_columns = []

    for col in NUMERIC_FEATURES:
        if col not in raw.columns:
            continue

        global_pct = _midrank_percentile(train[col], raw[col])
        gender_pct = global_pct.copy()

        for gender_value in train_gender.unique():
            fit_mask = train_gender.eq(gender_value)
            fit_values = train.loc[fit_mask, col].dropna()

            if len(fit_values) < MIN_GROUP_SIZE:
                continue

            row_mask = raw_gender.eq(gender_value).to_numpy()
            gender_pct[row_mask] = _midrank_percentile(fit_values, raw.loc[row_mask, col])

        source_missing = pd.to_numeric(raw[col], errors="coerce").isna().to_numpy()
        gender_pct[source_missing] = np.nan
        global_pct[source_missing] = np.nan

        diff = gender_pct - global_pct
        tail = 2.0 * np.abs(gender_pct - 0.5)

        features[col + "_gender_percentile"] = gender_pct
        features[col + "_gender_minus_global_percentile"] = diff
        features[col + "_gender_tail_extremeness"] = tail
        tail_columns.append(col + "_gender_tail_extremeness")

    if tail_columns:
        tail_frame = features[tail_columns]
        features["gender_tail_extremeness_mean"] = tail_frame.mean(axis=1)
        features["gender_tail_extremeness_max"] = tail_frame.max(axis=1)
        features["gender_tail_extremeness_ge_0_8_count"] = tail_frame.ge(0.8).sum(axis=1).astype("float64")
    else:
        features["gender_tail_extremeness_mean"] = np.nan
        features["gender_tail_extremeness_max"] = np.nan
        features["gender_tail_extremeness_ge_0_8_count"] = 0.0

    return features


FEATURE_GROUPS = [
    {
        "name": "gender_relative_numeric_percentiles",
        "fn": add_gender_relative_numeric_percentiles,
        "depends_on": [],
        "description": "Empirical global and gender-relative percentile features for numeric health measurements.",
    }
]