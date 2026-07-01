import numpy as np
import pandas as pd


TRAIN_ID_MAX = 690087
MISSING_TOKEN = "__missing__"
SMOOTHING_ALPHA = 1.0
RARE_FREQUENCY_THRESHOLD = 0.001
RARE_COUNT_THRESHOLD = 50.0

FEATURE_COLUMNS = (
    "sleep_duration",
    "heart_rate",
    "bmi",
    "calorie_expenditure",
    "step_count",
    "exercise_duration",
    "water_intake",
    "diet_type",
    "stress_level",
    "sleep_quality",
    "physical_activity_level",
    "smoking_alcohol",
    "gender",
)

NUMERIC_BIN_WIDTHS = {
    "sleep_duration": 0.5,
    "heart_rate": 5.0,
    "bmi": 1.0,
    "calorie_expenditure": 100.0,
    "step_count": 1000.0,
    "exercise_duration": 10.0,
    "water_intake": 0.25,
}


def _training_view(raw):
    if "id" not in raw.columns:
        return raw

    ids = pd.to_numeric(raw["id"], errors="coerce")
    train_mask = ids.le(TRAIN_ID_MAX)
    if bool(train_mask.any()):
        return raw.loc[train_mask]

    return raw


def _available_feature_columns(raw):
    return [column for column in FEATURE_COLUMNS if column in raw.columns]


def _exact_tokens(series):
    tokens = series.astype("object")
    return tokens.where(series.notna(), MISSING_TOKEN)


def _coarse_numeric_tokens(series, width):
    numeric_values = pd.to_numeric(series, errors="coerce").to_numpy(dtype="float64", copy=False)
    valid_mask = np.isfinite(numeric_values)

    tokens = np.empty(len(series), dtype="object")
    tokens[~valid_mask] = MISSING_TOKEN
    tokens[valid_mask] = np.floor((numeric_values[valid_mask] / width) + 1e-12).astype("int64")
    return pd.Series(tokens, index=series.index)


def _fit_token_stats(tokens):
    counts = tokens.value_counts(dropna=False)
    observed_count = float(len(tokens))
    vocabulary_size = float(len(counts))
    denominator = observed_count + (SMOOTHING_ALPHA * vocabulary_size)

    if denominator <= 0.0:
        return {}, {}, {}, 1.0

    frequencies = (counts.astype("float64") + SMOOTHING_ALPHA) / denominator
    percentiles = counts.rank(method="average", pct=True).astype("float64")
    floor_frequency = SMOOTHING_ALPHA / denominator

    return counts.to_dict(), frequencies.to_dict(), percentiles.to_dict(), floor_frequency


def _new_accumulator(row_count):
    return {
        "count": 0,
        "log_sum": np.zeros(row_count, dtype="float64"),
        "log_sq_sum": np.zeros(row_count, dtype="float64"),
        "log_min": np.full(row_count, np.inf, dtype="float64"),
        "log_max": np.full(row_count, -np.inf, dtype="float64"),
        "pct_sum": np.zeros(row_count, dtype="float64"),
        "pct_sq_sum": np.zeros(row_count, dtype="float64"),
        "pct_min": np.full(row_count, np.inf, dtype="float64"),
        "pct_max": np.full(row_count, -np.inf, dtype="float64"),
        "rare_count": np.zeros(row_count, dtype="float64"),
        "unseen_count": np.zeros(row_count, dtype="float64"),
    }


def _update_accumulator(accumulator, fit_tokens, row_tokens):
    counts, frequencies, percentiles, floor_frequency = _fit_token_stats(fit_tokens)

    mapped_counts = row_tokens.map(counts)
    unseen = mapped_counts.isna().to_numpy(dtype="bool")
    count_values = mapped_counts.fillna(0.0).to_numpy(dtype="float64")

    frequency_values = row_tokens.map(frequencies).fillna(floor_frequency).to_numpy(dtype="float64")
    frequency_values = np.maximum(frequency_values, np.finfo("float64").tiny)
    log_frequency = np.log(frequency_values)

    percentile_values = row_tokens.map(percentiles).fillna(0.0).to_numpy(dtype="float64")

    accumulator["count"] += 1
    accumulator["log_sum"] += log_frequency
    accumulator["log_sq_sum"] += log_frequency * log_frequency
    accumulator["log_min"] = np.minimum(accumulator["log_min"], log_frequency)
    accumulator["log_max"] = np.maximum(accumulator["log_max"], log_frequency)

    accumulator["pct_sum"] += percentile_values
    accumulator["pct_sq_sum"] += percentile_values * percentile_values
    accumulator["pct_min"] = np.minimum(accumulator["pct_min"], percentile_values)
    accumulator["pct_max"] = np.maximum(accumulator["pct_max"], percentile_values)

    rare = (frequency_values < RARE_FREQUENCY_THRESHOLD) | (count_values < RARE_COUNT_THRESHOLD)
    accumulator["rare_count"] += rare.astype("float64")
    accumulator["unseen_count"] += unseen.astype("float64")


def _finalize_accumulator(prefix, accumulator):
    feature_count = accumulator["count"]
    if feature_count == 0:
        return {}

    scale = 1.0 / float(feature_count)

    log_mean = accumulator["log_sum"] * scale
    log_variance = np.maximum((accumulator["log_sq_sum"] * scale) - (log_mean * log_mean), 0.0)

    pct_mean = accumulator["pct_sum"] * scale
    pct_variance = np.maximum((accumulator["pct_sq_sum"] * scale) - (pct_mean * pct_mean), 0.0)

    return {
        prefix + "_log_frequency_mean": log_mean,
        prefix + "_log_frequency_min": accumulator["log_min"],
        prefix + "_log_frequency_max": accumulator["log_max"],
        prefix + "_log_frequency_std": np.sqrt(log_variance),
        prefix + "_negative_log_frequency_sum": -accumulator["log_sum"],
        prefix + "_frequency_percentile_mean": pct_mean,
        prefix + "_frequency_percentile_min": accumulator["pct_min"],
        prefix + "_frequency_percentile_max": accumulator["pct_max"],
        prefix + "_frequency_percentile_std": np.sqrt(pct_variance),
        prefix + "_rare_token_count": accumulator["rare_count"],
        prefix + "_unseen_token_count": accumulator["unseen_count"],
    }


def add_source_value_popularity_signature(raw, deps, aux):
    fit_raw = _training_view(raw)
    feature_columns = _available_feature_columns(raw)

    exact_accumulator = _new_accumulator(len(raw))
    coarse_accumulator = _new_accumulator(len(raw))

    for column in feature_columns:
        fit_exact_tokens = _exact_tokens(fit_raw[column])
        row_exact_tokens = _exact_tokens(raw[column])
        _update_accumulator(exact_accumulator, fit_exact_tokens, row_exact_tokens)

        if column in NUMERIC_BIN_WIDTHS:
            width = NUMERIC_BIN_WIDTHS[column]
            fit_coarse_tokens = _coarse_numeric_tokens(fit_raw[column], width)
            row_coarse_tokens = _coarse_numeric_tokens(raw[column], width)
        else:
            fit_coarse_tokens = fit_exact_tokens
            row_coarse_tokens = row_exact_tokens

        _update_accumulator(coarse_accumulator, fit_coarse_tokens, row_coarse_tokens)

    features = {}
    features.update(_finalize_accumulator("exact", exact_accumulator))
    features.update(_finalize_accumulator("coarse", coarse_accumulator))

    return pd.DataFrame(features, index=raw.index)


FEATURE_GROUPS = [
    {
        "name": "source_value_popularity_signature",
        "fn": add_source_value_popularity_signature,
        "depends_on": [],
        "description": "Summarizes exact and coarse source-value popularity patterns fitted on training covariates.",
    }
]