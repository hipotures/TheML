import numpy as np
import pandas as pd

CLASS_LABELS = ("at-risk", "fit", "unhealthy")
CLASS_FEATURE_SUFFIXES = {
    "at-risk": "at_risk",
    "fit": "fit",
    "unhealthy": "unhealthy",
}

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

N_QUANTILE_BINS = 8
ONE_WAY_SMOOTHING = 24.0
CROSS_SMOOTHING = 48.0
PROBABILITY_FLOOR = 0.000000000001


def _column_as_series(frame, column, index):
    if column in frame.columns:
        return pd.Series(frame[column].to_numpy(copy=False), index=index)
    return pd.Series(np.nan, index=index)


def _numeric_series(frame, column, index):
    return pd.to_numeric(_column_as_series(frame, column, index), errors="coerce")


def _normalised_labels(aux):
    labels = pd.Series(aux["health_condition"].to_numpy(copy=False), index=aux.index)
    labels = labels.astype("string").str.strip().str.lower()
    labels = labels.str.replace("_", "-", regex=False)
    labels = labels.str.replace(r"\s+", "-", regex=True)
    return labels


def _quantile_edges(frame, column):
    values = pd.to_numeric(_column_as_series(frame, column, frame.index), errors="coerce")
    arr = values.to_numpy(dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return None
    quantiles = np.linspace(0.0, 1.0, N_QUANTILE_BINS + 1)
    edges = np.nanquantile(arr, quantiles)
    edges = edges[np.isfinite(edges)]
    return np.unique(edges)


def _build_quantile_edges(frame):
    edges = {}
    for column in NUMERIC_COLUMNS:
        edges[column] = _quantile_edges(frame, column)
    return edges


def _numeric_quantile_tokens(frame, column, edges, index):
    values = _numeric_series(frame, column, index)
    out = pd.Series(column + ":missing", index=index, dtype="object")
    valid = values.notna()
    if not bool(valid.any()):
        return out

    if edges is None or len(edges) < 3:
        out.loc[valid] = column + ":observed"
        return out

    inner_edges = np.asarray(edges[1:-1], dtype=float)
    bin_ids = np.searchsorted(inner_edges, values.loc[valid].to_numpy(dtype=float), side="right")
    out.loc[valid] = np.char.add(column + ":q", bin_ids.astype(str))
    return out


def _categorical_tokens(frame, column, index):
    values = _column_as_series(frame, column, index)
    cleaned = values.astype("string").str.strip().str.lower()
    cleaned = cleaned.str.replace(r"\s+", "_", regex=True)
    missing = cleaned.isna() | cleaned.eq("")
    tokens = cleaned.mask(missing, "missing").astype("object")
    return column + ":" + tokens


def _sleep_zone_tokens(frame, index):
    values = _numeric_series(frame, "sleep_duration", index)
    out = pd.Series("sleep_zone:missing", index=index, dtype="object")
    valid = values.notna()
    out.loc[valid & values.lt(7.0)] = "sleep_zone:short"
    out.loc[valid & values.ge(7.0) & values.le(9.0)] = "sleep_zone:recommended"
    out.loc[valid & values.gt(9.0)] = "sleep_zone:long"
    return out


def _bmi_zone_tokens(frame, index):
    values = _numeric_series(frame, "bmi", index)
    out = pd.Series("bmi_zone:missing", index=index, dtype="object")
    valid = values.notna()
    out.loc[valid & values.lt(18.5)] = "bmi_zone:underweight"
    out.loc[valid & values.ge(18.5) & values.lt(25.0)] = "bmi_zone:healthy"
    out.loc[valid & values.ge(25.0) & values.lt(30.0)] = "bmi_zone:overweight"
    out.loc[valid & values.ge(30.0)] = "bmi_zone:obese"
    return out


def _heart_rate_zone_tokens(frame, index):
    values = _numeric_series(frame, "heart_rate", index)
    out = pd.Series("heart_rate_zone:missing", index=index, dtype="object")
    valid = values.notna()
    out.loc[valid & values.lt(60.0)] = "heart_rate_zone:low"
    out.loc[valid & values.ge(60.0) & values.le(100.0)] = "heart_rate_zone:typical"
    out.loc[valid & values.gt(100.0)] = "heart_rate_zone:high"
    return out


def _hydration_zone_tokens(frame, index):
    values = _numeric_series(frame, "water_intake", index)
    out = pd.Series("hydration_zone:missing", index=index, dtype="object")
    valid = values.notna()
    out.loc[valid & values.lt(2.0)] = "hydration_zone:low"
    out.loc[valid & values.ge(2.0) & values.le(3.7)] = "hydration_zone:typical"
    out.loc[valid & values.gt(3.7)] = "hydration_zone:high"
    return out


def _activity_num_zone_tokens(frame, index):
    steps = _numeric_series(frame, "step_count", index)
    exercise = _numeric_series(frame, "exercise_duration", index)
    out = pd.Series("activity_num_zone:missing", index=index, dtype="object")

    known = steps.notna() | exercise.notna()
    active = steps.ge(10000.0) | exercise.ge(60.0)
    moderate = (steps.ge(5000.0) | exercise.ge(30.0)) & ~active
    low = known & ~active & ~moderate

    out.loc[low] = "activity_num_zone:low"
    out.loc[moderate] = "activity_num_zone:moderate"
    out.loc[active] = "activity_num_zone:active"
    return out


def _cross_tokens(prefix, index, *parts):
    text = parts[0].astype("object").astype(str)
    missing = parts[0].astype("object").astype(str).str.endswith(":missing")
    for part in parts[1:]:
        part_text = part.astype("object").astype(str)
        text = text + "|" + part_text
        missing = missing | part_text.str.endswith(":missing")

    out = prefix + ":" + text
    out.loc[missing] = prefix + ":missing"
    return pd.Series(out.to_numpy(copy=False), index=index, dtype="object")


def _build_state_frame(frame, quantile_edges):
    index = frame.index
    states = pd.DataFrame(index=index)

    for column in NUMERIC_COLUMNS:
        states["q_" + column] = _numeric_quantile_tokens(
            frame, column, quantile_edges.get(column), index
        )

    states["sleep_zone"] = _sleep_zone_tokens(frame, index)
    states["bmi_zone"] = _bmi_zone_tokens(frame, index)
    states["heart_rate_zone"] = _heart_rate_zone_tokens(frame, index)
    states["hydration_zone"] = _hydration_zone_tokens(frame, index)
    states["activity_num_zone"] = _activity_num_zone_tokens(frame, index)

    for column in CATEGORICAL_COLUMNS:
        states["cat_" + column] = _categorical_tokens(frame, column, index)

    states["cross_bmi_heart_rate"] = _cross_tokens(
        "bmi_heart_rate",
        index,
        states["bmi_zone"],
        states["heart_rate_zone"],
    )
    states["cross_sleep_quality_stress"] = _cross_tokens(
        "sleep_quality_stress",
        index,
        states["sleep_zone"],
        states["cat_sleep_quality"],
        states["cat_stress_level"],
    )
    states["cross_activity_reported"] = _cross_tokens(
        "activity_reported",
        index,
        states["activity_num_zone"],
        states["cat_physical_activity_level"],
    )
    states["cross_smoking_diet"] = _cross_tokens(
        "smoking_diet",
        index,
        states["cat_smoking_alcohol"],
        states["cat_diet_type"],
    )
    states["cross_hydration_activity"] = _cross_tokens(
        "hydration_activity",
        index,
        states["hydration_zone"],
        states["activity_num_zone"],
    )

    return states


def _state_column_groups():
    one_way = []
    for column in NUMERIC_COLUMNS:
        one_way.append("q_" + column)

    one_way.extend(
        [
            "sleep_zone",
            "bmi_zone",
            "heart_rate_zone",
            "hydration_zone",
            "activity_num_zone",
        ]
    )

    for column in CATEGORICAL_COLUMNS:
        one_way.append("cat_" + column)

    crossed = [
        "cross_bmi_heart_rate",
        "cross_sleep_quality_stress",
        "cross_activity_reported",
        "cross_smoking_diet",
        "cross_hydration_activity",
    ]
    return one_way, crossed


def _class_priors(labels):
    counts = labels.value_counts().reindex(CLASS_LABELS, fill_value=0).to_numpy(dtype=float)
    return (counts + 1.0) / (counts.sum() + float(len(CLASS_LABELS)))


def _probability_table(state_values, labels, priors, alpha):
    counts = pd.crosstab(state_values.to_numpy(copy=False), labels.to_numpy(copy=False))
    counts = counts.reindex(columns=CLASS_LABELS, fill_value=0)

    support = counts.sum(axis=1).astype(float)
    count_values = counts.to_numpy(dtype=float)
    support_values = support.to_numpy(dtype=float)

    probs = (count_values + alpha * priors.reshape(1, -1)) / (
        support_values.reshape(-1, 1) + alpha
    )
    prob_table = pd.DataFrame(probs, index=counts.index, columns=CLASS_LABELS)
    return prob_table, support


def _accumulate_table(raw_tokens, prob_table, support_table, prob_sum, support_sum, weight_sum, weight):
    token_values = raw_tokens.to_numpy(copy=False)
    mapped = prob_table.reindex(token_values)
    values = mapped.to_numpy(dtype=float)
    matched = np.isfinite(values).all(axis=1)

    if not bool(matched.any()):
        return

    prob_sum[matched] += values[matched] * weight
    support_values = support_table.reindex(token_values).to_numpy(dtype=float)
    support_sum[matched] += support_values[matched] * weight
    weight_sum[matched] += weight


def _format_output(index, probs, support):
    probs = np.clip(probs, PROBABILITY_FLOOR, 1.0)
    probs = probs / probs.sum(axis=1, keepdims=True)
    log_probs = np.log(probs)

    sorted_probs = np.sort(probs, axis=1)
    margin = sorted_probs[:, -1] - sorted_probs[:, -2]
    entropy = -(probs * log_probs).sum(axis=1)

    out = pd.DataFrame(index=index)
    for pos, label in enumerate(CLASS_LABELS):
        out["logp_" + CLASS_FEATURE_SUFFIXES[label]] = log_probs[:, pos]
    out["class_probability_margin"] = margin
    out["class_probability_entropy"] = entropy
    out["matched_external_support"] = support
    return out


def _neutral_output(index):
    n_rows = len(index)
    probs = np.full(
        (n_rows, len(CLASS_LABELS)),
        1.0 / float(len(CLASS_LABELS)),
        dtype=float,
    )
    support = np.zeros(n_rows, dtype=float)
    return _format_output(index, probs, support)


def add_external_source_state_likelihood(raw, deps, aux):
    if aux is None or len(aux) == 0 or "health_condition" not in aux.columns:
        return _neutral_output(raw.index)

    labels = _normalised_labels(aux)
    valid_label = labels.isin(CLASS_LABELS).fillna(False)
    if not bool(valid_label.any()):
        return _neutral_output(raw.index)

    aux_valid = aux.loc[valid_label]
    labels_valid = labels.loc[valid_label].astype("object")

    priors = _class_priors(labels_valid)
    quantile_edges = _build_quantile_edges(aux_valid)

    aux_states = _build_state_frame(aux_valid, quantile_edges)
    raw_states = _build_state_frame(raw, quantile_edges)
    one_way_columns, crossed_columns = _state_column_groups()

    n_rows = len(raw.index)
    prob_sum = np.zeros((n_rows, len(CLASS_LABELS)), dtype=float)
    support_sum = np.zeros(n_rows, dtype=float)
    weight_sum = np.zeros(n_rows, dtype=float)

    for column in one_way_columns:
        prob_table, support_table = _probability_table(
            aux_states[column], labels_valid, priors, ONE_WAY_SMOOTHING
        )
        _accumulate_table(
            raw_states[column],
            prob_table,
            support_table,
            prob_sum,
            support_sum,
            weight_sum,
            1.0,
        )

    for column in crossed_columns:
        prob_table, support_table = _probability_table(
            aux_states[column], labels_valid, priors, CROSS_SMOOTHING
        )
        _accumulate_table(
            raw_states[column],
            prob_table,
            support_table,
            prob_sum,
            support_sum,
            weight_sum,
            1.25,
        )

    has_match = weight_sum > 0.0
    probs = np.empty((n_rows, len(CLASS_LABELS)), dtype=float)
    probs[has_match] = prob_sum[has_match] / weight_sum[has_match].reshape(-1, 1)
    probs[~has_match] = priors.reshape(1, -1)

    support = np.zeros(n_rows, dtype=float)
    support[has_match] = support_sum[has_match] / weight_sum[has_match]

    return _format_output(raw.index, probs, support)


FEATURE_GROUPS = [
    {
        "name": "external_source_state_likelihood",
        "fn": add_external_source_state_likelihood,
        "depends_on": [],
        "description": "Smoothed auxiliary-source health-condition affinities from matched lifestyle state tokens.",
    }
]