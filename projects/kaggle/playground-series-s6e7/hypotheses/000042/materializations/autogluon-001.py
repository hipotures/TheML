import numpy as np
import pandas as pd


TRAIN_ID_CUTOFF = 690088

NUMERIC_TAIL_SPECS = (
    ("sleep_duration", "low_adverse"),
    ("heart_rate", "high_adverse"),
    ("bmi", "bmi_distance_adverse"),
    ("calorie_expenditure", "low_adverse"),
    ("step_count", "low_adverse"),
    ("exercise_duration", "low_adverse"),
    ("water_intake", "low_adverse"),
)

ORDINAL_ADVERSE_MAPS = {
    "stress_level": {"low": 0.0, "medium": 0.5, "high": 1.0},
    "sleep_quality": {"good": 0.0, "average": 0.5, "poor": 1.0},
    "physical_activity_level": {"active": 0.0, "moderate": 0.5, "sedentary": 1.0},
    "smoking_alcohol": {"no": 0.0, "occasional": 0.5, "yes": 1.0},
    "diet_type": {"balanced": 0.0, "veg": 0.35, "non-veg": 0.75},
}


def _empty_aligned_frame(index):
    return pd.DataFrame(index=index)


def _training_mask(raw):
    if "id" not in raw.columns:
        return pd.Series(True, index=raw.index)
    ids = pd.to_numeric(raw["id"], errors="coerce")
    mask = ids.ge(0) & ids.lt(TRAIN_ID_CUTOFF)
    if int(mask.sum()) < 100:
        return pd.Series(True, index=raw.index)
    return mask


def _quantiles(values, mask):
    train_values = values.loc[mask]
    train_values = train_values[np.isfinite(train_values)]
    if train_values.empty:
        train_values = values[np.isfinite(values)]
    if train_values.empty:
        return 0.25, 0.5, 0.75
    return tuple(train_values.quantile([0.25, 0.5, 0.75]).astype(float).tolist())


def _scale_low_adverse(values, q25, q50, q75):
    spread_low = max(q50 - q25, 1e-6)
    spread_high = max(q75 - q50, 1e-6)
    adverse = ((q50 - values) / spread_low).clip(lower=0.0, upper=1.0)
    protective = ((values - q50) / spread_high).clip(lower=0.0, upper=1.0)
    return adverse, protective


def _scale_high_adverse(values, q25, q50, q75):
    spread_low = max(q50 - q25, 1e-6)
    spread_high = max(q75 - q50, 1e-6)
    adverse = ((values - q50) / spread_high).clip(lower=0.0, upper=1.0)
    protective = ((q50 - values) / spread_low).clip(lower=0.0, upper=1.0)
    return adverse, protective


def _scale_bmi_distance(values, mask):
    distance = (values - 22.0).abs()
    train_distance = distance.loc[mask]
    train_distance = train_distance[np.isfinite(train_distance)]
    if train_distance.empty:
        train_distance = distance[np.isfinite(distance)]
    if train_distance.empty:
        q25, q50, q75 = 1.0, 3.0, 6.0
    else:
        q25, q50, q75 = tuple(train_distance.quantile([0.25, 0.5, 0.75]).astype(float).tolist())
    spread_low = max(q50 - q25, 1e-6)
    spread_high = max(q75 - q50, 1e-6)
    adverse = ((distance - q50) / spread_high).clip(lower=0.0, upper=1.0)
    protective = ((q50 - distance) / spread_low).clip(lower=0.0, upper=1.0)
    return adverse, protective


def add_empirical_tail_stress_constellation(raw, deps, aux):
    index = raw.index
    train_mask = _training_mask(raw)
    adverse_parts = []
    protective_parts = []
    observed_parts = []

    for column, mode in NUMERIC_TAIL_SPECS:
        if column not in raw.columns:
            continue
        values = pd.to_numeric(raw[column], errors="coerce")
        observed = values.notna().astype(float)

        if mode == "bmi_distance_adverse":
            adverse, protective = _scale_bmi_distance(values, train_mask)
        else:
            q25, q50, q75 = _quantiles(values, train_mask)
            if mode == "high_adverse":
                adverse, protective = _scale_high_adverse(values, q25, q50, q75)
            else:
                adverse, protective = _scale_low_adverse(values, q25, q50, q75)

        adverse_parts.append(adverse.where(values.notna(), np.nan).rename(column))
        protective_parts.append(protective.where(values.notna(), np.nan).rename(column))
        observed_parts.append(observed.rename(column))

    for column, mapping in ORDINAL_ADVERSE_MAPS.items():
        if column not in raw.columns:
            continue
        normalized = raw[column].astype("string").str.lower()
        adverse = normalized.map(mapping).astype(float)
        protective = 1.0 - adverse
        observed = adverse.notna().astype(float)

        adverse_parts.append(adverse.rename(column))
        protective_parts.append(protective.where(adverse.notna(), np.nan).rename(column))
        observed_parts.append(observed.rename(column))

    if not adverse_parts:
        return _empty_aligned_frame(index)

    adverse_scores = pd.concat(adverse_parts, axis=1).reindex(index)
    protective_scores = pd.concat(protective_parts, axis=1).reindex(index)
    observed = pd.concat(observed_parts, axis=1).reindex(index).fillna(0.0)

    observed_count = observed.sum(axis=1)
    denom = observed_count.replace(0.0, np.nan)

    adverse_75 = adverse_scores.ge(0.75)
    adverse_90 = adverse_scores.ge(0.90)
    protective_75 = protective_scores.ge(0.75)
    protective_90 = protective_scores.ge(0.90)

    adverse_count_75 = adverse_75.sum(axis=1).astype(float)
    adverse_count_90 = adverse_90.sum(axis=1).astype(float)
    protective_count_75 = protective_75.sum(axis=1).astype(float)
    protective_count_90 = protective_90.sum(axis=1).astype(float)

    adverse_intensity_75 = adverse_scores.sub(0.75).clip(lower=0.0).sum(axis=1)
    adverse_intensity_90 = adverse_scores.sub(0.90).clip(lower=0.0).sum(axis=1)
    protective_intensity_75 = protective_scores.sub(0.75).clip(lower=0.0).sum(axis=1)
    protective_intensity_90 = protective_scores.sub(0.90).clip(lower=0.0).sum(axis=1)

    strongest_adverse = adverse_scores.max(axis=1, skipna=True).fillna(0.0)
    strongest_protective = protective_scores.max(axis=1, skipna=True).fillna(0.0)

    adverse_rate_75 = adverse_count_75.div(denom).fillna(0.0)
    adverse_rate_90 = adverse_count_90.div(denom).fillna(0.0)
    protective_rate_75 = protective_count_75.div(denom).fillna(0.0)
    protective_rate_90 = protective_count_90.div(denom).fillna(0.0)

    adverse_mean = adverse_scores.sum(axis=1, min_count=1).div(denom).fillna(0.0)
    protective_mean = protective_scores.sum(axis=1, min_count=1).div(denom).fillna(0.0)

    mixed_flag_75 = (adverse_count_75.gt(0) & protective_count_75.gt(0)).astype(np.int8)
    mixed_flag_90 = (adverse_count_90.gt(0) & protective_count_90.gt(0)).astype(np.int8)
    mixed_intensity_75 = np.minimum(adverse_intensity_75, protective_intensity_75)
    mixed_intensity_90 = np.minimum(adverse_intensity_90, protective_intensity_90)

    features = pd.DataFrame(index=index)
    features["observed_tail_signal_count"] = observed_count.astype(float)
    features["observed_tail_signal_rate"] = observed_count.div(float(len(adverse_parts))).fillna(0.0)

    features["adverse_tail_count_75"] = adverse_count_75
    features["adverse_tail_count_90"] = adverse_count_90
    features["protective_tail_count_75"] = protective_count_75
    features["protective_tail_count_90"] = protective_count_90

    features["adverse_tail_rate_75"] = adverse_rate_75
    features["adverse_tail_rate_90"] = adverse_rate_90
    features["protective_tail_rate_75"] = protective_rate_75
    features["protective_tail_rate_90"] = protective_rate_90

    features["adverse_tail_intensity_75"] = adverse_intensity_75
    features["adverse_tail_intensity_90"] = adverse_intensity_90
    features["protective_tail_intensity_75"] = protective_intensity_75
    features["protective_tail_intensity_90"] = protective_intensity_90

    features["strongest_adverse_tail"] = strongest_adverse
    features["strongest_protective_tail"] = strongest_protective
    features["tail_count_balance_75"] = adverse_count_75 - protective_count_75
    features["tail_count_balance_90"] = adverse_count_90 - protective_count_90
    features["tail_rate_balance_75"] = adverse_rate_75 - protective_rate_75
    features["tail_rate_balance_90"] = adverse_rate_90 - protective_rate_90
    features["tail_mean_balance"] = adverse_mean - protective_mean
    features["tail_strongest_balance"] = strongest_adverse - strongest_protective

    features["mixed_extreme_flag_75"] = mixed_flag_75
    features["mixed_extreme_flag_90"] = mixed_flag_90
    features["mixed_extreme_intensity_75"] = mixed_intensity_75
    features["mixed_extreme_intensity_90"] = mixed_intensity_90

    return features


FEATURE_GROUPS = [
    {
        "name": "empirical_tail_stress_constellation",
        "fn": add_empirical_tail_stress_constellation,
        "depends_on": [],
        "description": "Train-anchored empirical co-tail summaries for adverse, protective, and mixed student health stress constellations.",
    }
]