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

CATEGORICAL_FEATURES = (
    "diet_type",
    "stress_level",
    "sleep_quality",
    "physical_activity_level",
    "smoking_alcohol",
    "gender",
)

CLASS_LABELS = ("unhealthy", "at-risk", "fit")

NUMERIC_WEIGHTS = {
    "sleep_duration": 1.15,
    "heart_rate": 1.00,
    "bmi": 1.10,
    "calorie_expenditure": 0.85,
    "step_count": 1.05,
    "exercise_duration": 1.05,
    "water_intake": 0.75,
}

CATEGORICAL_WEIGHTS = {
    "diet_type": 0.85,
    "stress_level": 1.15,
    "sleep_quality": 1.10,
    "physical_activity_level": 1.05,
    "smoking_alcohol": 0.95,
    "gender": 0.35,
}

K_NEAREST_PERSONAS = 25
DISTANCE_CAP = 4.0
EPSILON = 0.000001
CHUNK_SIZE = 4096


def _safe_iqr(values):
    clean = pd.to_numeric(values, errors="coerce").dropna()
    if clean.empty:
        return 1.0
    q75 = clean.quantile(0.75)
    q25 = clean.quantile(0.25)
    iqr = float(q75 - q25)
    if not np.isfinite(iqr) or iqr <= EPSILON:
        std = float(clean.std())
        if np.isfinite(std) and std > EPSILON:
            return std
        return 1.0
    return iqr


def _mode_or_missing(values):
    modes = values.dropna().mode()
    if modes.empty:
        return "__missing__"
    return str(modes.iloc[0])


def _mode_purity(values):
    clean = values.dropna()
    if clean.empty:
        return 0.0
    counts = clean.astype(str).value_counts()
    return float(counts.iloc[0] / len(clean))


def _entropy_from_probs(prob_matrix):
    clipped = np.clip(prob_matrix, EPSILON, 1.0)
    return -np.sum(clipped * np.log(clipped), axis=1) / np.log(float(prob_matrix.shape[1]))


def _empty_affinity_frame(index, prefix_values):
    n_rows = len(index)
    data = {}
    for label in CLASS_LABELS:
        data[f"persona_affinity_{label.replace('-', '_')}"] = np.full(
            n_rows, prefix_values.get(label, 1.0 / len(CLASS_LABELS)), dtype=np.float32
        )
    data["persona_affinity_margin_top2"] = np.zeros(n_rows, dtype=np.float32)
    data["persona_unhealthy_fit_log_ratio"] = np.zeros(n_rows, dtype=np.float32)
    data["persona_label_entropy"] = np.ones(n_rows, dtype=np.float32)
    data["persona_mean_categorical_purity"] = np.zeros(n_rows, dtype=np.float32)
    data["persona_mean_numeric_variability"] = np.zeros(n_rows, dtype=np.float32)
    data["persona_nearest_distance"] = np.ones(n_rows, dtype=np.float32)
    data["persona_effective_support"] = np.zeros(n_rows, dtype=np.float32)
    data["persona_observed_feature_fraction"] = np.zeros(n_rows, dtype=np.float32)
    return pd.DataFrame(data, index=index)


def add_external_student_persona_affinity(raw, deps, aux):
    del deps

    if aux is None or aux.empty or "student_id" not in aux.columns or "health_condition" not in aux.columns:
        return _empty_affinity_frame(raw.index, {})

    usable_numeric = [col for col in NUMERIC_FEATURES if col in raw.columns and col in aux.columns]
    usable_categorical = [col for col in CATEGORICAL_FEATURES if col in raw.columns and col in aux.columns]
    if not usable_numeric and not usable_categorical:
        return _empty_affinity_frame(raw.index, {})

    aux_work = aux[["student_id", "health_condition"] + usable_numeric + usable_categorical].copy()
    aux_work["student_id"] = aux_work["student_id"].astype(str)
    aux_work["health_condition"] = aux_work["health_condition"].astype(str)

    grouped = aux_work.groupby("student_id", sort=False)
    persona_ids = list(grouped.groups.keys())
    n_personas = len(persona_ids)
    if n_personas == 0:
        return _empty_affinity_frame(raw.index, {})

    persona_index = pd.Index(persona_ids, name="student_id")

    numeric_medians = pd.DataFrame(index=persona_index)
    numeric_variability = pd.DataFrame(index=persona_index)
    global_iqrs = {}
    for col in usable_numeric:
        numeric_values = pd.to_numeric(aux_work[col], errors="coerce")
        global_iqrs[col] = _safe_iqr(numeric_values)
        numeric_medians[col] = grouped[col].median().reindex(persona_index).astype(float)
        numeric_variability[col] = grouped[col].apply(_safe_iqr).reindex(persona_index).astype(float) / global_iqrs[col]

    categorical_modes = pd.DataFrame(index=persona_index)
    categorical_purities = pd.DataFrame(index=persona_index)
    for col in usable_categorical:
        categorical_modes[col] = grouped[col].apply(_mode_or_missing).reindex(persona_index).astype(str)
        categorical_purities[col] = grouped[col].apply(_mode_purity).reindex(persona_index).astype(float)

    class_counts = pd.crosstab(aux_work["student_id"], aux_work["health_condition"]).reindex(persona_index).fillna(0.0)
    for label in CLASS_LABELS:
        if label not in class_counts.columns:
            class_counts[label] = 0.0
    class_counts = class_counts[list(CLASS_LABELS)]
    class_totals = class_counts.sum(axis=1).replace(0.0, np.nan)
    class_probs = class_counts.div(class_totals, axis=0).fillna(1.0 / len(CLASS_LABELS))

    global_counts = aux_work["health_condition"].value_counts()
    global_probs = {
        label: float(global_counts.get(label, 0.0) / max(float(global_counts.sum()), 1.0))
        for label in CLASS_LABELS
    }

    if numeric_variability.empty:
        persona_numeric_variability = np.zeros(n_personas, dtype=np.float32)
    else:
        persona_numeric_variability = numeric_variability.clip(0.0, DISTANCE_CAP).mean(axis=1).fillna(0.0).to_numpy(dtype=np.float32)

    if categorical_purities.empty:
        persona_categorical_purity = np.zeros(n_personas, dtype=np.float32)
    else:
        persona_categorical_purity = categorical_purities.mean(axis=1).fillna(0.0).to_numpy(dtype=np.float32)

    class_prob_array = class_probs.to_numpy(dtype=np.float32)

    numeric_median_arrays = {
        col: numeric_medians[col].fillna(aux_work[col].median()).to_numpy(dtype=np.float32)
        for col in usable_numeric
    }
    categorical_mode_arrays = {
        col: categorical_modes[col].fillna("__missing__").astype(str).to_numpy()
        for col in usable_categorical
    }

    n_rows = len(raw)
    out = {
        f"persona_affinity_{label.replace('-', '_')}": np.zeros(n_rows, dtype=np.float32)
        for label in CLASS_LABELS
    }
    out["persona_affinity_margin_top2"] = np.zeros(n_rows, dtype=np.float32)
    out["persona_unhealthy_fit_log_ratio"] = np.zeros(n_rows, dtype=np.float32)
    out["persona_label_entropy"] = np.ones(n_rows, dtype=np.float32)
    out["persona_mean_categorical_purity"] = np.zeros(n_rows, dtype=np.float32)
    out["persona_mean_numeric_variability"] = np.zeros(n_rows, dtype=np.float32)
    out["persona_nearest_distance"] = np.ones(n_rows, dtype=np.float32)
    out["persona_effective_support"] = np.zeros(n_rows, dtype=np.float32)
    out["persona_observed_feature_fraction"] = np.zeros(n_rows, dtype=np.float32)

    raw_numeric = {
        col: pd.to_numeric(raw[col], errors="coerce").to_numpy(dtype=np.float32)
        for col in usable_numeric
    }
    raw_categorical = {
        col: raw[col].astype("object").where(raw[col].notna(), None).to_numpy()
        for col in usable_categorical
    }

    total_possible_weight = float(
        sum(NUMERIC_WEIGHTS[col] for col in usable_numeric)
        + sum(CATEGORICAL_WEIGHTS[col] for col in usable_categorical)
    )
    if total_possible_weight <= EPSILON:
        return _empty_affinity_frame(raw.index, global_probs)

    k = min(K_NEAREST_PERSONAS, n_personas)
    max_distance = 1.0

    for start in range(0, n_rows, CHUNK_SIZE):
        end = min(start + CHUNK_SIZE, n_rows)
        chunk_len = end - start

        distance_sum = np.zeros((chunk_len, n_personas), dtype=np.float32)
        observed_weight = np.zeros(chunk_len, dtype=np.float32)

        for col in usable_numeric:
            values = raw_numeric[col][start:end]
            observed = np.isfinite(values)
            if not observed.any():
                continue
            weight = float(NUMERIC_WEIGHTS[col])
            scaled = np.abs(values[observed, None] - numeric_median_arrays[col][None, :]) / float(global_iqrs[col])
            scaled = np.minimum(scaled, DISTANCE_CAP) / DISTANCE_CAP
            distance_sum[observed, :] += scaled.astype(np.float32) * weight
            observed_weight[observed] += weight

        for col in usable_categorical:
            values = raw_categorical[col][start:end]
            observed = pd.notna(values)
            if not np.any(observed):
                continue
            weight = float(CATEGORICAL_WEIGHTS[col])
            mismatches = values[observed].astype(str)[:, None] != categorical_mode_arrays[col][None, :]
            distance_sum[observed, :] += mismatches.astype(np.float32) * weight
            observed_weight[observed] += weight

        has_observed = observed_weight > EPSILON
        normalized_distance = np.full((chunk_len, n_personas), max_distance, dtype=np.float32)
        normalized_distance[has_observed, :] = distance_sum[has_observed, :] / observed_weight[has_observed, None]

        if k < n_personas:
            nearest_idx = np.argpartition(normalized_distance, kth=k - 1, axis=1)[:, :k]
            nearest_dist = np.take_along_axis(normalized_distance, nearest_idx, axis=1)
        else:
            nearest_idx = np.tile(np.arange(n_personas), (chunk_len, 1))
            nearest_dist = normalized_distance

        kernel_weights = np.exp(-4.0 * nearest_dist).astype(np.float32)
        kernel_weights[~has_observed, :] = 1.0
        weight_sums = kernel_weights.sum(axis=1)
        safe_weight_sums = np.where(weight_sums > EPSILON, weight_sums, 1.0)

        nearest_class_probs = class_prob_array[nearest_idx]
        weighted_probs = (nearest_class_probs * kernel_weights[:, :, None]).sum(axis=1) / safe_weight_sums[:, None]
        weighted_probs[~has_observed, :] = np.array([global_probs[label] for label in CLASS_LABELS], dtype=np.float32)

        sorted_probs = np.sort(weighted_probs, axis=1)
        entropy = _entropy_from_probs(weighted_probs).astype(np.float32)

        nearest_purity = persona_categorical_purity[nearest_idx]
        nearest_variability = persona_numeric_variability[nearest_idx]
        weighted_purity = (nearest_purity * kernel_weights).sum(axis=1) / safe_weight_sums
        weighted_variability = (nearest_variability * kernel_weights).sum(axis=1) / safe_weight_sums

        effective_support = (weight_sums * weight_sums) / np.maximum((kernel_weights * kernel_weights).sum(axis=1), EPSILON)

        row_slice = slice(start, end)
        for class_pos, label in enumerate(CLASS_LABELS):
            out[f"persona_affinity_{label.replace('-', '_')}"][row_slice] = weighted_probs[:, class_pos]

        out["persona_affinity_margin_top2"][row_slice] = sorted_probs[:, -1] - sorted_probs[:, -2]
        out["persona_unhealthy_fit_log_ratio"][row_slice] = np.log(
            (weighted_probs[:, 0] + EPSILON) / (weighted_probs[:, 2] + EPSILON)
        ).astype(np.float32)
        out["persona_label_entropy"][row_slice] = entropy
        out["persona_mean_categorical_purity"][row_slice] = weighted_purity.astype(np.float32)
        out["persona_mean_numeric_variability"][row_slice] = weighted_variability.astype(np.float32)
        out["persona_nearest_distance"][row_slice] = nearest_dist.min(axis=1).astype(np.float32)
        out["persona_effective_support"][row_slice] = effective_support.astype(np.float32)
        out["persona_observed_feature_fraction"][row_slice] = (observed_weight / total_possible_weight).astype(np.float32)

    return pd.DataFrame(out, index=raw.index)


FEATURE_GROUPS = [
    {
        "name": "external_student_persona_affinity",
        "fn": add_external_student_persona_affinity,
        "depends_on": [],
        "description": "Auxiliary source-only student persona affinity features based on nearest external student health behavior profiles.",
    }
]