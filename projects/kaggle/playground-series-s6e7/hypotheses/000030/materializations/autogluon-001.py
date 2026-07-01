import numpy as np
import pandas as pd

AUX_STUDENT_COLUMN = "student_id"
AUX_TIMESTAMP_COLUMN = "timestamp"
AUX_TARGET_COLUMN = "health_condition"

MIN_PHASE_BUCKET_ROWS = 200
DISTANCE_CHUNK_SIZE = 200000
NUMERIC_EPSILON = 1.0e-6
MISSING_TOKEN = "__missing__"
OTHER_TOKEN = "__other__"

COMMON_NUMERIC_COLUMNS = (
    "sleep_duration",
    "heart_rate",
    "bmi",
    "calorie_expenditure",
    "step_count",
    "exercise_duration",
    "water_intake",
)

COMMON_CATEGORICAL_COLUMNS = (
    "diet_type",
    "stress_level",
    "sleep_quality",
    "physical_activity_level",
    "smoking_alcohol",
    "gender",
)

CATEGORY_LEVELS = {
    "diet_type": ("balanced", "non-veg", "veg"),
    "stress_level": ("low", "medium", "high"),
    "sleep_quality": ("poor", "average", "good"),
    "physical_activity_level": ("sedentary", "moderate", "active"),
    "smoking_alcohol": ("no", "occasional", "yes"),
    "gender": ("female", "male", "other"),
}

SEVERITY_BY_LABEL = {
    "fit": 0,
    "at-risk": 1,
    "at_risk": 1,
    "at risk": 1,
    "unhealthy": 2,
}

SEVERITY_CLASS_NAME = {
    0: "fit",
    1: "at_risk",
    2: "unhealthy",
}

POSSIBLE_PHASE_BUCKETS = (
    "stable_fit",
    "stable_at_risk",
    "stable_unhealthy",
    "pre_worsen_fit",
    "pre_worsen_at_risk",
    "pre_recover_at_risk",
    "pre_recover_unhealthy",
    "post_worsen_at_risk",
    "post_worsen_unhealthy",
    "post_recover_fit",
    "post_recover_at_risk",
    "pre_worsen",
    "pre_recover",
    "post_worsen",
    "post_recover",
)

SUMMARY_FEATURE_COLUMNS = (
    "best_worsening_affinity",
    "best_recovery_affinity",
    "best_stable_affinity",
    "worsening_minus_recovery",
    "recovery_minus_stable",
    "severity_weighted_phase_affinity",
)


def _neutral_features(index):
    columns = []
    for bucket in POSSIBLE_PHASE_BUCKETS:
        columns.append("affinity_" + bucket)
    for column in SUMMARY_FEATURE_COLUMNS:
        columns.append(column)
    data = np.zeros((len(index), len(columns)), dtype=np.float32)
    return pd.DataFrame(data, columns=columns, index=index)


def _to_float_array(series):
    numeric = pd.to_numeric(series, errors="coerce")
    values = numeric.astype("float64").to_numpy()
    values[~np.isfinite(values)] = np.nan
    return values


def _numeric_scaling_params(aux_frame, numeric_columns):
    params = {}
    for column in numeric_columns:
        values = _to_float_array(aux_frame[column])
        values = values[np.isfinite(values)]

        if values.size == 0:
            params[column] = (0.0, 0.0, 0.0, 1.0)
            continue

        lower = float(np.nanquantile(values, 0.01))
        upper = float(np.nanquantile(values, 0.99))
        if not np.isfinite(lower):
            lower = float(np.nanmin(values))
        if not np.isfinite(upper):
            upper = float(np.nanmax(values))
        if upper < lower:
            lower, upper = upper, lower

        clipped = np.clip(values, lower, upper)
        median = float(np.nanmedian(clipped))
        q25 = float(np.nanquantile(clipped, 0.25))
        q75 = float(np.nanquantile(clipped, 0.75))
        scale = q75 - q25

        if not np.isfinite(scale) or scale < NUMERIC_EPSILON:
            spread = upper - lower
            if np.isfinite(spread) and spread >= NUMERIC_EPSILON:
                scale = spread
            else:
                scale = 1.0

        if not np.isfinite(median):
            median = 0.0

        params[column] = (lower, upper, median, float(scale))

    return params


def _normalized_category_values(frame, column, index):
    if column in frame.columns:
        values = frame[column].astype("object")
    else:
        values = pd.Series(MISSING_TOKEN, index=index, dtype="object")

    values = values.where(~pd.isna(values), MISSING_TOKEN)
    values = values.astype(str).str.strip().str.lower()

    allowed = set(CATEGORY_LEVELS.get(column, ()))
    allowed.add(MISSING_TOKEN)
    values = values.where(values.isin(allowed), OTHER_TOKEN)
    return values.to_numpy(dtype=object)


def _encoded_dimension(numeric_columns, categorical_columns):
    dimension = len(numeric_columns) * 2
    for column in categorical_columns:
        dimension += len(CATEGORY_LEVELS.get(column, ())) + 2
    return dimension


def _encode_feature_matrix(frame, index, numeric_columns, categorical_columns, params, is_project_rows):
    row_count = len(index)
    dimension = _encoded_dimension(numeric_columns, categorical_columns)
    matrix = np.zeros((row_count, dimension), dtype=np.float32)

    position = 0
    numeric_missing = []

    for column in numeric_columns:
        if column in frame.columns:
            values = _to_float_array(frame[column])
        else:
            values = np.full(row_count, np.nan, dtype=np.float64)

        missing = ~np.isfinite(values)
        lower, upper, median, scale = params[column]

        filled = np.where(missing, median, values)
        clipped = np.clip(filled, lower, upper)
        scaled = (clipped - median) / scale
        scaled[~np.isfinite(scaled)] = 0.0

        matrix[:, position] = scaled.astype(np.float32)
        position += 1
        numeric_missing.append(missing)

    for missing in numeric_missing:
        if is_project_rows:
            matrix[:, position] = missing.astype(np.float32)
        position += 1

    for column in categorical_columns:
        values = _normalized_category_values(frame, column, index)
        levels = CATEGORY_LEVELS.get(column, ()) + (MISSING_TOKEN, OTHER_TOKEN)
        for level in levels:
            matrix[:, position] = (values == level).astype(np.float32)
            position += 1

    return matrix


def _assign_transition_phases(severity_values, previous_values, next_values):
    phases = []

    for current, previous_value, next_value in zip(severity_values, previous_values, next_values):
        current_int = int(current)
        class_name = SEVERITY_CLASS_NAME.get(current_int, "unknown")

        if pd.notna(next_value) and int(next_value) > current_int:
            phase = "pre_worsen_" + class_name
        elif pd.notna(next_value) and int(next_value) < current_int:
            phase = "pre_recover_" + class_name
        elif pd.notna(previous_value) and int(previous_value) < current_int:
            phase = "post_worsen_" + class_name
        elif pd.notna(previous_value) and int(previous_value) > current_int:
            phase = "post_recover_" + class_name
        else:
            phase = "stable_" + class_name

        phases.append(phase)

    return phases


def _phase_direction(phase):
    if phase.startswith("pre_worsen"):
        return "pre_worsen"
    if phase.startswith("pre_recover"):
        return "pre_recover"
    if phase.startswith("post_worsen"):
        return "post_worsen"
    if phase.startswith("post_recover"):
        return "post_recover"
    return phase


def _collapsed_phase_buckets(phases):
    counts = pd.Series(phases).value_counts()
    collapsed = []

    for phase in phases:
        if phase.startswith("stable_"):
            collapsed.append(phase)
        elif int(counts.get(phase, 0)) >= MIN_PHASE_BUCKET_ROWS:
            collapsed.append(phase)
        else:
            collapsed.append(_phase_direction(phase))

    return collapsed


def add_external_transition_phase_affinity(raw, deps, aux):
    index = raw.index

    if aux is None or not isinstance(aux, pd.DataFrame) or aux.empty:
        return _neutral_features(index)

    required_columns = (AUX_STUDENT_COLUMN, AUX_TIMESTAMP_COLUMN, AUX_TARGET_COLUMN)
    for column in required_columns:
        if column not in aux.columns:
            return _neutral_features(index)

    numeric_columns = [
        column for column in COMMON_NUMERIC_COLUMNS
        if column in raw.columns and column in aux.columns
    ]
    categorical_columns = [
        column for column in COMMON_CATEGORICAL_COLUMNS
        if column in raw.columns and column in aux.columns
    ]

    if not numeric_columns and not categorical_columns:
        return _neutral_features(index)

    work_columns = list(required_columns) + numeric_columns + categorical_columns
    work = aux.loc[:, work_columns].copy()

    target_values = work[AUX_TARGET_COLUMN].astype("object")
    target_values = target_values.where(~pd.isna(target_values), "")
    target_values = target_values.astype(str).str.strip().str.lower()
    work["_severity"] = target_values.map(SEVERITY_BY_LABEL)
    work = work.loc[work["_severity"].notna()].copy()

    if work.empty:
        return _neutral_features(index)

    work["_severity"] = work["_severity"].astype("int8")
    work["_source_order"] = np.arange(len(work), dtype=np.int64)
    work["_timestamp_order"] = pd.to_datetime(work[AUX_TIMESTAMP_COLUMN], errors="coerce")
    work = work.sort_values(
        [AUX_STUDENT_COLUMN, "_timestamp_order", "_source_order"],
        kind="mergesort",
    )

    grouped_severity = work.groupby(AUX_STUDENT_COLUMN, sort=False)["_severity"]
    previous_severity = grouped_severity.shift(1)
    next_severity = grouped_severity.shift(-1)

    work["_phase_detail"] = _assign_transition_phases(
        work["_severity"].to_numpy(),
        previous_severity.to_numpy(),
        next_severity.to_numpy(),
    )
    work["_phase_bucket"] = _collapsed_phase_buckets(work["_phase_detail"].tolist())

    params = _numeric_scaling_params(work, numeric_columns)
    aux_matrix = _encode_feature_matrix(
        work,
        work.index,
        numeric_columns,
        categorical_columns,
        params,
        is_project_rows=False,
    )

    if aux_matrix.shape[1] == 0:
        return _neutral_features(index)

    bucket_values = work["_phase_bucket"].to_numpy(dtype=object)
    severity_values = work["_severity"].astype("float64").to_numpy()
    retained_buckets = []
    centroids = []
    bucket_severity = {}

    for bucket in sorted(work["_phase_bucket"].dropna().unique().tolist()):
        mask = bucket_values == bucket
        if not np.any(mask):
            continue

        block = aux_matrix[mask]
        centroid = block.mean(axis=0).astype(np.float32)

        if numeric_columns:
            numeric_width = len(numeric_columns)
            centroid[:numeric_width] = np.median(block[:, :numeric_width], axis=0).astype(np.float32)

        retained_buckets.append(bucket)
        centroids.append(centroid)
        bucket_severity[bucket] = float(np.mean(severity_values[mask]))

    if not centroids:
        return _neutral_features(index)

    centroid_matrix = np.vstack(centroids).astype(np.float32)
    raw_matrix = _encode_feature_matrix(
        raw,
        index,
        numeric_columns,
        categorical_columns,
        params,
        is_project_rows=True,
    )

    possible_bucket_position = {}
    for position, bucket in enumerate(POSSIBLE_PHASE_BUCKETS):
        possible_bucket_position[bucket] = position

    retained_positions = []
    for bucket in retained_buckets:
        retained_positions.append(possible_bucket_position.get(bucket))

    affinities = np.zeros((len(index), len(POSSIBLE_PHASE_BUCKETS)), dtype=np.float32)
    feature_width = float(centroid_matrix.shape[1])
    centroid_norm = np.einsum("ij,ij->i", centroid_matrix, centroid_matrix) / feature_width

    for start in range(0, len(index), DISTANCE_CHUNK_SIZE):
        end = min(start + DISTANCE_CHUNK_SIZE, len(index))
        block = raw_matrix[start:end]

        block_norm = np.einsum("ij,ij->i", block, block) / feature_width
        dot_products = block @ centroid_matrix.T
        distance_squared = block_norm[:, None] + centroid_norm[None, :] - (2.0 / feature_width) * dot_products
        np.maximum(distance_squared, 0.0, out=distance_squared)
        np.sqrt(distance_squared, out=distance_squared)

        inverse_distance = 1.0 / (1.0 + distance_squared)
        inverse_sum = inverse_distance.sum(axis=1, keepdims=True)
        soft_affinity = inverse_distance / np.where(inverse_sum > 0.0, inverse_sum, 1.0)

        for retained_position, possible_position in enumerate(retained_positions):
            if possible_position is not None:
                affinities[start:end, possible_position] = soft_affinity[:, retained_position].astype(np.float32)

    phase_columns = []
    for bucket in POSSIBLE_PHASE_BUCKETS:
        phase_columns.append("affinity_" + bucket)

    features = pd.DataFrame(affinities, columns=phase_columns, index=index)

    worsening_positions = [
        position for position, bucket in enumerate(POSSIBLE_PHASE_BUCKETS)
        if "worsen" in bucket
    ]
    recovery_positions = [
        position for position, bucket in enumerate(POSSIBLE_PHASE_BUCKETS)
        if "recover" in bucket
    ]
    stable_positions = [
        position for position, bucket in enumerate(POSSIBLE_PHASE_BUCKETS)
        if bucket.startswith("stable_")
    ]

    best_worsening = affinities[:, worsening_positions].max(axis=1)
    best_recovery = affinities[:, recovery_positions].max(axis=1)
    best_stable = affinities[:, stable_positions].max(axis=1)

    severity_weights = np.zeros(len(POSSIBLE_PHASE_BUCKETS), dtype=np.float32)
    for bucket, position in possible_bucket_position.items():
        if bucket in bucket_severity:
            severity_weights[position] = np.float32(bucket_severity[bucket])

    features["best_worsening_affinity"] = best_worsening.astype(np.float32)
    features["best_recovery_affinity"] = best_recovery.astype(np.float32)
    features["best_stable_affinity"] = best_stable.astype(np.float32)
    features["worsening_minus_recovery"] = (best_worsening - best_recovery).astype(np.float32)
    features["recovery_minus_stable"] = (best_recovery - best_stable).astype(np.float32)
    features["severity_weighted_phase_affinity"] = (affinities @ severity_weights).astype(np.float32)

    return features


FEATURE_GROUPS = [
    {
        "name": "external_transition_phase_affinity",
        "fn": add_external_transition_phase_affinity,
        "depends_on": [],
        "description": "Auxiliary transition-phase centroid affinities from external student health trajectories.",
    }
]