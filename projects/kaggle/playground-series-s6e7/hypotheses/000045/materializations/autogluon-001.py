import numpy as np
import pandas as pd


CLASS_LABELS = ("at-risk", "unhealthy", "fit")
SUPPORT_COLUMN = "__support__"
MISSING_NUMERIC_CODE = -2147483648
MISSING_COARSE_CODE = -1
MISSING_TEXT_TOKEN = "__missing__"
SMOOTHING_ALPHA = 12.0
EXACT_MIN_SUPPORT = 2.0
COARSE_MIN_SUPPORT = 5.0

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

SIGNATURES = (
    (
        "all_profile",
        (
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
        ),
    ),
    (
        "objective_numeric",
        (
            "sleep_duration",
            "heart_rate",
            "bmi",
            "calorie_expenditure",
            "step_count",
            "exercise_duration",
            "water_intake",
        ),
    ),
    (
        "self_report_categorical",
        (
            "diet_type",
            "stress_level",
            "sleep_quality",
            "physical_activity_level",
            "smoking_alcohol",
            "gender",
        ),
    ),
    (
        "recovery_activity",
        (
            "sleep_duration",
            "sleep_quality",
            "stress_level",
            "heart_rate",
            "step_count",
            "exercise_duration",
            "physical_activity_level",
            "water_intake",
        ),
    ),
)

EXACT_NUMERIC_SCALES = {
    "sleep_duration": 10.0,
    "heart_rate": 10.0,
    "bmi": 100.0,
    "calorie_expenditure": 1.0,
    "step_count": 1.0,
    "exercise_duration": 10.0,
    "water_intake": 100.0,
}

COARSE_BINS = {
    "sleep_duration": (4.5, 6.0, 7.0, 8.0, 9.0),
    "heart_rate": (55.0, 65.0, 75.0, 85.0, 95.0, 105.0),
    "bmi": (18.5, 25.0, 30.0),
    "calorie_expenditure": (1600.0, 2000.0, 2400.0, 2800.0, 3200.0),
    "step_count": (2500.0, 5000.0, 7500.0, 10000.0, 12500.0),
    "exercise_duration": (10.0, 30.0, 60.0, 90.0),
    "water_intake": (1.5, 2.0, 2.5, 3.0, 3.5, 4.0),
}


def _numeric_exact_codes(frame, column):
    codes = np.full(len(frame.index), MISSING_NUMERIC_CODE, dtype=np.int64)
    if column not in frame.columns:
        return codes

    values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float, copy=False)
    mask = np.isfinite(values)
    if mask.any():
        scale = EXACT_NUMERIC_SCALES.get(column, 100.0)
        codes[mask] = np.rint(values[mask] * scale).astype(np.int64)
    return codes


def _numeric_coarse_codes(frame, column):
    codes = np.full(len(frame.index), MISSING_COARSE_CODE, dtype=np.int16)
    if column not in frame.columns:
        return codes

    values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float, copy=False)
    mask = np.isfinite(values)
    if mask.any():
        edges = np.asarray(COARSE_BINS.get(column, ()), dtype=float)
        codes[mask] = np.searchsorted(edges, values[mask], side="right").astype(np.int16)
    return codes


def _categorical_tokens(frame, column):
    if column not in frame.columns:
        return pd.Series(MISSING_TEXT_TOKEN, index=frame.index, dtype="string")

    tokens = frame[column].astype("string").str.strip().str.lower()
    tokens = tokens.fillna(MISSING_TEXT_TOKEN)
    return tokens.mask(tokens == "", MISSING_TEXT_TOKEN)


def _profile_key(frame, columns, resolution):
    pieces = {}
    for column in columns:
        if column in NUMERIC_COLUMNS:
            if resolution == "coarse":
                pieces[column] = _numeric_coarse_codes(frame, column)
            else:
                pieces[column] = _numeric_exact_codes(frame, column)
        else:
            pieces[column] = _categorical_tokens(frame, column)

    if not pieces:
        return pd.Series(np.zeros(len(frame.index), dtype=np.uint64), index=frame.index)

    tokens = pd.DataFrame(pieces, index=frame.index)
    return pd.util.hash_pandas_object(tokens, index=False, categorize=True).astype("uint64")


def _normalized_labels(aux):
    if aux is None or not isinstance(aux, pd.DataFrame) or "health_condition" not in aux.columns:
        return pd.Series(pd.NA, index=pd.RangeIndex(0), dtype="string")

    labels = aux["health_condition"].astype("string").str.strip().str.lower()
    return labels


def _global_label_counts(labels):
    counts = np.zeros(len(CLASS_LABELS), dtype=np.float64)
    if labels.empty:
        return counts, 0.0

    for i, label in enumerate(CLASS_LABELS):
        counts[i] = float((labels == label).sum())

    return counts, float(counts.sum())


def _empty_count_table():
    return pd.DataFrame(columns=list(CLASS_LABELS) + [SUPPORT_COLUMN])


def _count_table(keys, labels):
    if labels.empty:
        return _empty_count_table()

    valid = labels.isin(CLASS_LABELS).to_numpy()
    if not valid.any():
        return _empty_count_table()

    work = pd.DataFrame(
        {
            "key": keys.to_numpy()[valid],
            "label": labels.to_numpy()[valid],
        }
    )
    counts = work.groupby(["key", "label"]).size().unstack(fill_value=0)

    for label in CLASS_LABELS:
        if label not in counts.columns:
            counts[label] = 0

    counts = counts.loc[:, list(CLASS_LABELS)].astype(np.float32)
    counts[SUPPORT_COLUMN] = counts.sum(axis=1).astype(np.float32)
    return counts


def _lookup_counts(keys, table, index):
    columns = list(CLASS_LABELS) + [SUPPORT_COLUMN]
    if table.empty:
        return pd.DataFrame(np.zeros((len(index), len(columns)), dtype=np.float32), index=index, columns=columns)

    looked_up = table.reindex(keys.to_numpy()).fillna(0.0)
    looked_up.index = index
    return looked_up.loc[:, columns].astype(np.float32, copy=False)


def _raw_key_log_support(keys):
    counts = keys.value_counts(sort=False)
    support = keys.map(counts).to_numpy(dtype=np.float32, copy=False)
    return np.log1p(support).astype(np.float32)


def _selected_counts(exact_counts, coarse_counts, global_counts, global_support):
    class_columns = list(CLASS_LABELS)
    n_rows = len(exact_counts.index)

    selected = np.tile(global_counts, (n_rows, 1)).astype(np.float64)
    support = np.full(n_rows, global_support, dtype=np.float64)
    level = np.zeros(n_rows, dtype=np.int8)

    exact_support = exact_counts[SUPPORT_COLUMN].to_numpy(dtype=np.float64, copy=False)
    coarse_support = coarse_counts[SUPPORT_COLUMN].to_numpy(dtype=np.float64, copy=False)

    use_exact = exact_support >= EXACT_MIN_SUPPORT
    use_coarse = (~use_exact) & (coarse_support >= COARSE_MIN_SUPPORT)

    if use_coarse.any():
        selected[use_coarse] = coarse_counts.loc[:, class_columns].to_numpy(dtype=np.float64, copy=False)[use_coarse]
        support[use_coarse] = coarse_support[use_coarse]
        level[use_coarse] = 1

    if use_exact.any():
        selected[use_exact] = exact_counts.loc[:, class_columns].to_numpy(dtype=np.float64, copy=False)[use_exact]
        support[use_exact] = exact_support[use_exact]
        level[use_exact] = 2

    return selected, support, level


def _append_label_features(features, prefix, counts, support, level, global_counts, global_support):
    n_classes = len(CLASS_LABELS)
    if global_support > 0.0:
        prior = global_counts / global_support
    else:
        prior = np.full(n_classes, 1.0 / n_classes, dtype=np.float64)

    probs = (counts + (SMOOTHING_ALPHA * prior)) / (support[:, None] + SMOOTHING_ALPHA)
    probs = np.clip(probs, 1.0e-12, 1.0)

    sorted_probs = np.sort(probs, axis=1)
    dominant = np.argmax(probs, axis=1).astype(np.int16)
    if global_support <= 0.0:
        dominant[:] = -1

    entropy = -np.sum(probs * np.log(probs), axis=1) / np.log(float(n_classes))

    for i, label in enumerate(CLASS_LABELS):
        safe_label = label.replace("-", "_")
        features[prefix + "_prop_" + safe_label] = probs[:, i].astype(np.float32)

    features[prefix + "_dominant_label_code"] = dominant
    features[prefix + "_dominant_purity"] = sorted_probs[:, -1].astype(np.float32)
    features[prefix + "_top2_margin"] = (sorted_probs[:, -1] - sorted_probs[:, -2]).astype(np.float32)
    features[prefix + "_class_entropy"] = entropy.astype(np.float32)
    features[prefix + "_label_log_support"] = np.log1p(support).astype(np.float32)
    features[prefix + "_label_support_level"] = level.astype(np.int8)


def add_profile_label_consistency_signature(raw, deps, aux):
    features = pd.DataFrame(index=raw.index)

    aux_frame = aux if isinstance(aux, pd.DataFrame) else pd.DataFrame()
    labels = _normalized_labels(aux_frame)
    global_counts, global_support = _global_label_counts(labels)

    for signature_name, columns in SIGNATURES:
        raw_exact_key = _profile_key(raw, columns, "exact")
        raw_coarse_key = _profile_key(raw, columns, "coarse")

        features[signature_name + "_raw_exact_log_support"] = _raw_key_log_support(raw_exact_key)
        features[signature_name + "_raw_coarse_log_support"] = _raw_key_log_support(raw_coarse_key)

        if global_support > 0.0:
            aux_exact_key = _profile_key(aux_frame, columns, "exact")
            aux_coarse_key = _profile_key(aux_frame, columns, "coarse")
            exact_table = _count_table(aux_exact_key, labels)
            coarse_table = _count_table(aux_coarse_key, labels)
        else:
            exact_table = _empty_count_table()
            coarse_table = _empty_count_table()

        exact_counts = _lookup_counts(raw_exact_key, exact_table, raw.index)
        coarse_counts = _lookup_counts(raw_coarse_key, coarse_table, raw.index)
        counts, support, level = _selected_counts(exact_counts, coarse_counts, global_counts, global_support)

        _append_label_features(
            features,
            signature_name,
            counts,
            support,
            level,
            global_counts,
            global_support,
        )

    return features


FEATURE_GROUPS = [
    {
        "name": "profile_label_consistency_signature",
        "fn": add_profile_label_consistency_signature,
        "depends_on": [],
        "description": "Auxiliary profile label-consistency and ambiguity signatures from canonical student health profiles.",
    }
]