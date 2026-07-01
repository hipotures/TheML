import numpy as np
import pandas as pd


NUMERIC_COLUMNS = (
    "sleep_duration",
    "heart_rate",
    "bmi",
    "calorie_expenditure",
    "step_count",
    "exercise_duration",
    "water_intake",
)

CATEGORICAL_COLUMNS = (
    "diet_type",
    "stress_level",
    "sleep_quality",
    "physical_activity_level",
    "smoking_alcohol",
    "gender",
)

PAIR_SPECS = (
    ("sleep_quality", "sleep_duration"),
    ("stress_level", "heart_rate"),
    ("physical_activity_level", "step_count"),
    ("physical_activity_level", "exercise_duration"),
    ("diet_type", "bmi"),
    ("smoking_alcohol", "heart_rate"),
    ("physical_activity_level", "calorie_expenditure"),
)

MISSING_TOKEN = "__missing__"
UNSEEN_TOKEN = "__unseen__"


def _numeric_series(frame, column, index):
    if column in frame.columns:
        return pd.to_numeric(frame[column], errors="coerce")
    return pd.Series(np.nan, index=index, dtype="float64")


def _categorical_series(frame, column, index):
    if column in frame.columns:
        return frame[column].astype("object").where(frame[column].notna(), MISSING_TOKEN).astype(str)
    return pd.Series(MISSING_TOKEN, index=index, dtype="object")


def _reference_frame(raw, aux):
    if aux is not None and len(aux) > 0:
        available = [col for col in NUMERIC_COLUMNS + CATEGORICAL_COLUMNS if col in aux.columns]
        if available:
            return aux
    return raw


def _safe_log_frequency_map(source_values):
    counts = source_values.value_counts(dropna=False)
    total = float(len(source_values))
    cardinality = float(max(len(counts), 1))
    denom = total + cardinality + 1.0
    mapping = {key: np.log((float(value) + 1.0) / denom) for key, value in counts.items()}
    unseen_log = np.log(1.0 / denom)
    return mapping, unseen_log


def _quintile_codes(values, source_values):
    result = pd.Series(-1, index=values.index, dtype="int16")
    source_non_missing = source_values.dropna().astype("float64")
    valid = values.notna()

    if len(source_non_missing) == 0:
        result.loc[valid] = 2
        return result

    quantiles = source_non_missing.quantile([0.2, 0.4, 0.6, 0.8]).to_numpy(dtype="float64")
    quantiles = np.maximum.accumulate(quantiles)
    result.loc[valid] = np.searchsorted(quantiles, values.loc[valid].to_numpy(dtype="float64"), side="right").astype("int16")
    return result


def add_external_source_support_signature(raw, deps, aux):
    ref = _reference_frame(raw, aux)
    out = pd.DataFrame(index=raw.index)

    percentile_cols = []
    tail_cols = []
    low_005_cols = []
    low_010_cols = []
    quintile_by_numeric = {}

    for column in NUMERIC_COLUMNS:
        values = _numeric_series(raw, column, raw.index)
        source_values = _numeric_series(ref, column, ref.index).dropna().astype("float64")
        missing = values.isna()

        pct = pd.Series(0.5, index=raw.index, dtype="float64")
        if len(source_values) > 0:
            sorted_source = np.sort(source_values.to_numpy(dtype="float64"))
            valid_values = values.loc[~missing].to_numpy(dtype="float64")
            ranks = np.searchsorted(sorted_source, valid_values, side="right") / float(len(sorted_source))
            pct.loc[~missing] = np.clip(ranks, 0.001, 0.999)
        tail = np.minimum(pct, 1.0 - pct)
        tail.loc[missing] = 0.0

        quintile = _quintile_codes(values, _numeric_series(ref, column, ref.index))
        quintile_by_numeric[column] = quintile

        out[column + "_source_pct"] = pct
        out[column + "_source_tail_support"] = tail
        out[column + "_source_missing"] = missing.astype("int8")
        out[column + "_source_quintile"] = quintile.astype("category")

        percentile_cols.append(column + "_source_pct")
        tail_cols.append(column + "_source_tail_support")
        low_005_cols.append(column + "_source_tail_lt_005")
        low_010_cols.append(column + "_source_tail_lt_010")
        out[column + "_source_tail_lt_005"] = (tail < 0.05).astype("int8")
        out[column + "_source_tail_lt_010"] = (tail < 0.10).astype("int8")

    out["numeric_source_pct_mean"] = out[percentile_cols].mean(axis=1)
    out["numeric_source_tail_mean"] = out[tail_cols].mean(axis=1)
    out["numeric_source_tail_min"] = out[tail_cols].min(axis=1)
    out["numeric_source_tail_lt_005_count"] = out[low_005_cols].sum(axis=1).astype("int16")
    out["numeric_source_tail_lt_010_count"] = out[low_010_cols].sum(axis=1).astype("int16")

    categorical_support_cols = []
    raw_cat_values = {}

    for column in CATEGORICAL_COLUMNS:
        raw_values = _categorical_series(raw, column, raw.index)
        source_values = _categorical_series(ref, column, ref.index)
        mapping, unseen_log = _safe_log_frequency_map(source_values)

        raw_cat_values[column] = raw_values
        out[column + "_source_log_freq"] = raw_values.map(mapping).fillna(unseen_log).astype("float64")
        out[column + "_source_unseen"] = (~raw_values.isin(mapping.keys())).astype("int8")
        categorical_support_cols.append(column + "_source_log_freq")

    raw_tuple = raw_cat_values[CATEGORICAL_COLUMNS[0]].copy()
    source_tuple = _categorical_series(ref, CATEGORICAL_COLUMNS[0], ref.index)
    for column in CATEGORICAL_COLUMNS[1:]:
        raw_tuple = raw_tuple + "|" + raw_cat_values[column]
        source_tuple = source_tuple + "|" + _categorical_series(ref, column, ref.index)

    tuple_mapping, tuple_unseen_log = _safe_log_frequency_map(source_tuple)
    out["categorical_tuple_source_log_freq"] = raw_tuple.map(tuple_mapping).fillna(tuple_unseen_log).astype("float64")
    out["categorical_source_log_freq_mean"] = out[categorical_support_cols].mean(axis=1)
    out["categorical_source_log_freq_min"] = out[categorical_support_cols].min(axis=1)

    pair_support_cols = []
    for cat_col, num_col in PAIR_SPECS:
        raw_cat = raw_cat_values.get(cat_col, _categorical_series(raw, cat_col, raw.index))
        source_cat = _categorical_series(ref, cat_col, ref.index)
        raw_quintile = quintile_by_numeric[num_col].astype(str)
        source_quintile = _quintile_codes(_numeric_series(ref, num_col, ref.index), _numeric_series(ref, num_col, ref.index)).astype(str)

        raw_pair = raw_cat + "|" + raw_quintile
        source_pair = source_cat + "|" + source_quintile
        pair_mapping, pair_unseen_log = _safe_log_frequency_map(source_pair)

        feature_name = cat_col + "_x_" + num_col + "_quintile_source_log_freq"
        out[feature_name] = raw_pair.map(pair_mapping).fillna(pair_unseen_log).astype("float64")
        pair_support_cols.append(feature_name)

    out["pair_source_log_freq_mean"] = out[pair_support_cols].mean(axis=1)
    out["pair_source_log_freq_min"] = out[pair_support_cols].min(axis=1)

    return out


FEATURE_GROUPS = [
    {
        "name": "external_source_support_signature",
        "fn": add_external_source_support_signature,
        "depends_on": [],
        "description": "External-source distribution support features from shared lifestyle, physiology, and categorical profile coverage.",
    }
]