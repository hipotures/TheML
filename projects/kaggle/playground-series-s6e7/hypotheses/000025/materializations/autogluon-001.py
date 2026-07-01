import numpy as np
import pandas as pd

MISSING_TOKEN = "missing"
CLASS_ORDER = ("unhealthy", "at-risk", "fit")
CLASS_SUFFIXES = {
    "unhealthy": "unhealthy",
    "at-risk": "at_risk",
    "fit": "fit",
}
DEFAULT_CLASS_COUNTS = {
    "unhealthy": 3816.0,
    "at-risk": 43718.0,
    "fit": 2466.0,
}
DEFAULT_CALORIE_EDGES = (1650.0, 2050.0, 2450.0, 2900.0)
CALORIE_QUANTILES = (0.2, 0.4, 0.6, 0.8)
SMOOTHING_STRENGTH = 30.0
MIN_FULL_SUPPORT = 20.0
MIN_FALLBACK_SUPPORT = 40.0
EPSILON = 1e-12

KEY_SPECS = (
    (
        "recovery_state",
        ("sleep_bin", "stress_level_bin", "sleep_quality_bin"),
        ("sleep_bin", "stress_level_bin"),
    ),
    (
        "activity_state",
        ("steps_bin", "exercise_bin", "physical_activity_bin"),
        ("steps_bin", "physical_activity_bin"),
    ),
    (
        "cardiometabolic_state",
        ("bmi_bin", "heart_rate_bin", "calorie_bin"),
        ("bmi_bin", "heart_rate_bin"),
    ),
    (
        "self_care_state",
        ("water_bin", "diet_type_bin", "smoking_alcohol_bin"),
        ("water_bin", "diet_type_bin"),
    ),
    (
        "whole_health_state",
        (
            "sleep_bin",
            "stress_level_bin",
            "physical_activity_bin",
            "bmi_bin",
            "smoking_alcohol_bin",
        ),
        ("sleep_bin", "stress_level_bin", "bmi_bin"),
    ),
)


def _numeric_series(frame, column):
    if column in frame.columns:
        return pd.to_numeric(frame[column], errors="coerce")
    return pd.Series(np.nan, index=frame.index, dtype="float64")


def _text_state(frame, column, prefix):
    if column not in frame.columns:
        return pd.Series(prefix + "_" + MISSING_TOKEN, index=frame.index, dtype="object")

    text = frame[column].astype("string").str.strip().str.lower()
    text = text.fillna(MISSING_TOKEN)
    text = text.mask(text.eq(""), MISSING_TOKEN)
    text = text.str.replace(r"\s+", "_", regex=True)
    text = text.str.replace("-", "_", regex=False)
    return text.radd(prefix + "_").astype("object")


def _sleep_bin(frame):
    values = _numeric_series(frame, "sleep_duration")
    out = pd.Series("sleep_missing", index=frame.index, dtype="object")
    valid = values.notna()
    out.loc[valid & (values < 6.0)] = "sleep_short"
    out.loc[valid & (values >= 6.0) & (values <= 8.0)] = "sleep_adequate"
    out.loc[valid & (values > 8.0)] = "sleep_long"
    return out


def _heart_rate_bin(frame):
    values = _numeric_series(frame, "heart_rate")
    out = pd.Series("heart_missing", index=frame.index, dtype="object")
    valid = values.notna()
    out.loc[valid & (values < 60.0)] = "heart_low"
    out.loc[valid & (values >= 60.0) & (values <= 85.0)] = "heart_normal"
    out.loc[valid & (values > 85.0) & (values <= 100.0)] = "heart_elevated"
    out.loc[valid & (values > 100.0)] = "heart_high"
    return out


def _bmi_bin(frame):
    values = _numeric_series(frame, "bmi")
    out = pd.Series("bmi_missing", index=frame.index, dtype="object")
    valid = values.notna()
    out.loc[valid & (values < 18.5)] = "bmi_under"
    out.loc[valid & (values >= 18.5) & (values < 25.0)] = "bmi_normal"
    out.loc[valid & (values >= 25.0) & (values < 30.0)] = "bmi_over"
    out.loc[valid & (values >= 30.0)] = "bmi_obese"
    return out


def _steps_bin(frame):
    values = _numeric_series(frame, "step_count")
    out = pd.Series("steps_missing", index=frame.index, dtype="object")
    valid = values.notna()
    out.loc[valid & (values < 3000.0)] = "steps_very_low"
    out.loc[valid & (values >= 3000.0) & (values < 7000.0)] = "steps_low"
    out.loc[valid & (values >= 7000.0) & (values < 10000.0)] = "steps_moderate"
    out.loc[valid & (values >= 10000.0) & (values < 12500.0)] = "steps_high"
    out.loc[valid & (values >= 12500.0)] = "steps_very_high"
    return out


def _exercise_bin(frame):
    values = _numeric_series(frame, "exercise_duration")
    out = pd.Series("exercise_missing", index=frame.index, dtype="object")
    valid = values.notna()
    out.loc[valid & (values <= 0.0)] = "exercise_none"
    out.loc[valid & (values > 0.0) & (values < 20.0)] = "exercise_light"
    out.loc[valid & (values >= 20.0) & (values < 45.0)] = "exercise_moderate"
    out.loc[valid & (values >= 45.0) & (values < 75.0)] = "exercise_high"
    out.loc[valid & (values >= 75.0)] = "exercise_extreme"
    return out


def _water_bin(frame):
    values = _numeric_series(frame, "water_intake")
    out = pd.Series("water_missing", index=frame.index, dtype="object")
    valid = values.notna()
    out.loc[valid & (values < 1.5)] = "water_low"
    out.loc[valid & (values >= 1.5) & (values < 2.5)] = "water_moderate"
    out.loc[valid & (values >= 2.5) & (values < 3.5)] = "water_high"
    out.loc[valid & (values >= 3.5)] = "water_very_high"
    return out


def _calorie_edges(frame):
    values = _numeric_series(frame, "calorie_expenditure").dropna()
    if values.shape[0] < 10:
        return DEFAULT_CALORIE_EDGES

    quantiles = values.quantile(CALORIE_QUANTILES)
    edges = []
    previous = None
    for value in quantiles.tolist():
        value = float(value)
        if not np.isfinite(value):
            return DEFAULT_CALORIE_EDGES
        if previous is not None and value <= previous:
            return DEFAULT_CALORIE_EDGES
        edges.append(value)
        previous = value

    if len(edges) != 4:
        return DEFAULT_CALORIE_EDGES
    return tuple(edges)


def _calorie_bin(frame, edges):
    values = _numeric_series(frame, "calorie_expenditure")
    edge_1, edge_2, edge_3, edge_4 = edges
    out = pd.Series("calorie_missing", index=frame.index, dtype="object")
    valid = values.notna()
    out.loc[valid & (values <= edge_1)] = "calorie_q1"
    out.loc[valid & (values > edge_1) & (values <= edge_2)] = "calorie_q2"
    out.loc[valid & (values > edge_2) & (values <= edge_3)] = "calorie_q3"
    out.loc[valid & (values > edge_3) & (values <= edge_4)] = "calorie_q4"
    out.loc[valid & (values > edge_4)] = "calorie_q5"
    return out


def _build_state_frame(frame, calorie_edges):
    state = pd.DataFrame(index=frame.index)
    state["sleep_bin"] = _sleep_bin(frame)
    state["heart_rate_bin"] = _heart_rate_bin(frame)
    state["bmi_bin"] = _bmi_bin(frame)
    state["calorie_bin"] = _calorie_bin(frame, calorie_edges)
    state["steps_bin"] = _steps_bin(frame)
    state["exercise_bin"] = _exercise_bin(frame)
    state["water_bin"] = _water_bin(frame)
    state["diet_type_bin"] = _text_state(frame, "diet_type", "diet")
    state["stress_level_bin"] = _text_state(frame, "stress_level", "stress")
    state["sleep_quality_bin"] = _text_state(frame, "sleep_quality", "sleep_quality")
    state["physical_activity_bin"] = _text_state(
        frame, "physical_activity_level", "activity"
    )
    state["smoking_alcohol_bin"] = _text_state(frame, "smoking_alcohol", "smoke_alcohol")
    return state


def _make_key(state, columns):
    key = state[columns[0]].astype("string")
    for column in columns[1:]:
        key = key.str.cat(state[column].astype("string"), sep="|")
    return key.astype("object")


def _label_series(aux):
    if not isinstance(aux, pd.DataFrame) or "health_condition" not in aux.columns:
        return pd.Series(dtype="object")

    labels = aux["health_condition"].astype("string").str.strip().str.lower()
    labels = labels.str.replace("_", "-", regex=False)
    labels = labels.str.replace(" ", "-", regex=False)
    return labels.astype("object")


def _default_priors():
    total = 0.0
    for class_label in CLASS_ORDER:
        total += float(DEFAULT_CLASS_COUNTS.get(class_label, 0.0))

    if total <= 0.0:
        uniform = 1.0 / float(len(CLASS_ORDER))
        return {class_label: uniform for class_label in CLASS_ORDER}

    return {
        class_label: float(DEFAULT_CLASS_COUNTS.get(class_label, 0.0)) / total
        for class_label in CLASS_ORDER
    }


def _priors_from_labels(labels):
    valid = labels[labels.isin(CLASS_ORDER)]
    if valid.shape[0] == 0:
        return _default_priors()

    counts = valid.value_counts()
    total = float(valid.shape[0] + len(CLASS_ORDER))
    return {
        class_label: float(counts.get(class_label, 0.0) + 1.0) / total
        for class_label in CLASS_ORDER
    }


def _empty_counts():
    return pd.DataFrame(columns=CLASS_ORDER, dtype="float64")


def _counts_by_key(keys, labels):
    if keys.shape[0] == 0 or labels.shape[0] == 0 or keys.shape[0] != labels.shape[0]:
        return _empty_counts()

    valid = labels.isin(CLASS_ORDER) & keys.notna()
    if int(valid.sum()) == 0:
        return _empty_counts()

    counts = pd.crosstab(keys.loc[valid], labels.loc[valid])
    counts = counts.reindex(columns=CLASS_ORDER, fill_value=0.0)
    return counts.astype("float64")


def _probability_tables(counts, priors):
    support = counts.sum(axis=1).astype("float64")
    probabilities = pd.DataFrame(index=counts.index)

    denominator = support + SMOOTHING_STRENGTH
    for class_label in CLASS_ORDER:
        probabilities[class_label] = (
            counts[class_label].astype("float64")
            + SMOOTHING_STRENGTH * float(priors[class_label])
        ) / denominator

    return support, probabilities


def _mapped_float(keys, values, default_value):
    if values.shape[0] == 0:
        return np.full(keys.shape[0], float(default_value), dtype="float64")

    mapped = keys.map(values)
    return mapped.fillna(float(default_value)).to_numpy(dtype="float64", copy=False)


def _score_state(raw_state, aux_state, labels, priors, name, full_columns, fallback_columns):
    raw_full_key = _make_key(raw_state, full_columns)
    raw_fallback_key = _make_key(raw_state, fallback_columns)
    aux_full_key = _make_key(aux_state, full_columns)
    aux_fallback_key = _make_key(aux_state, fallback_columns)

    full_counts = _counts_by_key(aux_full_key, labels)
    fallback_counts = _counts_by_key(aux_fallback_key, labels)
    full_support_table, full_probability_table = _probability_tables(full_counts, priors)
    fallback_support_table, fallback_probability_table = _probability_tables(
        fallback_counts, priors
    )

    full_support = _mapped_float(raw_full_key, full_support_table, 0.0)
    fallback_support = _mapped_float(raw_fallback_key, fallback_support_table, 0.0)
    use_full = full_support >= MIN_FULL_SUPPORT
    use_fallback = (~use_full) & (fallback_support >= MIN_FALLBACK_SUPPORT)

    selected_support = np.where(use_full, full_support, np.where(use_fallback, fallback_support, 0.0))
    source_level = np.where(use_full, 2, np.where(use_fallback, 1, 0)).astype("int8")

    class_probabilities = {}
    for class_label in CLASS_ORDER:
        full_probability = _mapped_float(
            raw_full_key, full_probability_table[class_label], priors[class_label]
        )
        fallback_probability = _mapped_float(
            raw_fallback_key, fallback_probability_table[class_label], priors[class_label]
        )
        class_probabilities[class_label] = np.where(
            use_full,
            full_probability,
            np.where(use_fallback, fallback_probability, float(priors[class_label])),
        )

    probability_matrix = np.column_stack(
        [class_probabilities[class_label] for class_label in CLASS_ORDER]
    )
    clipped = np.clip(probability_matrix, EPSILON, 1.0)
    entropy = -np.sum(probability_matrix * np.log(clipped), axis=1) / np.log(
        float(len(CLASS_ORDER))
    )
    sorted_probabilities = np.sort(probability_matrix, axis=1)

    features = pd.DataFrame(index=raw_state.index)
    features[name + "_state_key"] = raw_full_key
    features[name + "_backoff_key"] = raw_fallback_key
    features[name + "_source_level"] = source_level
    features[name + "_support_log1p"] = np.log1p(selected_support)
    features[name + "_support_weight"] = selected_support / (
        selected_support + SMOOTHING_STRENGTH
    )

    for class_label in CLASS_ORDER:
        suffix = CLASS_SUFFIXES[class_label]
        features[name + "_" + suffix + "_prob"] = class_probabilities[class_label]

    features[name + "_fit_minus_unhealthy"] = (
        class_probabilities["fit"] - class_probabilities["unhealthy"]
    )
    features[name + "_fit_minus_at_risk"] = (
        class_probabilities["fit"] - class_probabilities["at-risk"]
    )
    features[name + "_unhealthy_minus_at_risk"] = (
        class_probabilities["unhealthy"] - class_probabilities["at-risk"]
    )
    features[name + "_dominant_margin"] = (
        sorted_probabilities[:, -1] - sorted_probabilities[:, -2]
    )
    features[name + "_top_prob"] = np.max(probability_matrix, axis=1)
    features[name + "_top_class_code"] = np.argmax(probability_matrix, axis=1).astype(
        "int8"
    )
    features[name + "_entropy"] = entropy
    return features


def add_coarse_health_state_class_affinity(raw, deps, aux):
    _ = deps

    calorie_edges = _calorie_edges(raw)
    raw_state = _build_state_frame(raw, calorie_edges)

    if isinstance(aux, pd.DataFrame):
        aux_frame = aux
    else:
        aux_frame = pd.DataFrame()

    aux_state = _build_state_frame(aux_frame, calorie_edges)
    labels = _label_series(aux_frame)
    if labels.shape[0] != aux_state.shape[0]:
        labels = pd.Series(dtype="object")
        aux_state = aux_state.iloc[0:0]

    priors = _priors_from_labels(labels)
    out = pd.DataFrame(index=raw.index)

    for name, full_columns, fallback_columns in KEY_SPECS:
        block = _score_state(
            raw_state,
            aux_state,
            labels,
            priors,
            name,
            full_columns,
            fallback_columns,
        )
        for column in block.columns:
            out[column] = block[column]

    return out


FEATURE_GROUPS = [
    {
        "name": "coarse_health_state_class_affinity",
        "fn": add_coarse_health_state_class_affinity,
        "depends_on": [],
        "description": "Smoothed external health-condition affinity features for coarse lifestyle, recovery, and physiology state composites.",
    }
]