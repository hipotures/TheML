import numpy as np
import pandas as pd


NUMERIC_COLUMNS = [
    "sleep_duration",
    "heart_rate",
    "bmi",
    "calorie_expenditure",
    "step_count",
    "exercise_duration",
    "water_intake",
]

CATEGORICAL_COLUMNS = [
    "diet_type",
    "stress_level",
    "sleep_quality",
    "physical_activity_level",
    "smoking_alcohol",
    "gender",
]

TERTILE_NAMES = ["low", "middle", "high"]


def _empty_external_temporal_stability_affinity(index):
    return pd.DataFrame(
        {
            "low_instability_similarity": np.full(len(index), 1.0 / 3.0, dtype=np.float32),
            "middle_instability_similarity": np.full(len(index), 1.0 / 3.0, dtype=np.float32),
            "high_instability_similarity": np.full(len(index), 1.0 / 3.0, dtype=np.float32),
            "expected_instability_affinity": np.zeros(len(index), dtype=np.float32),
            "high_minus_low_instability_similarity": np.zeros(len(index), dtype=np.float32),
            "observed_profile_coverage": np.zeros(len(index), dtype=np.float32),
        },
        index=index,
    )


def _normalized_entropy(values):
    counts = values.dropna().astype(str).value_counts()
    total = float(counts.sum())
    if total <= 0.0 or len(counts) <= 1:
        return 0.0
    probs = counts.to_numpy(dtype=np.float64) / total
    entropy = -float(np.sum(probs * np.log(probs)))
    return entropy / float(np.log(len(counts)))


def _categorical_switch_rate(values):
    vals = values.dropna().astype(str).to_numpy()
    if len(vals) <= 1:
        return 0.0
    return float(np.mean(vals[1:] != vals[:-1]))


def _student_instability(student_frame, source_iqr):
    components = []

    for col in NUMERIC_COLUMNS:
        if col not in student_frame.columns:
            continue
        values = pd.to_numeric(student_frame[col], errors="coerce").dropna()
        denom = float(source_iqr.get(col, 0.0)) + 1.0e-6
        if len(values) > 0:
            q75 = float(values.quantile(0.75))
            q25 = float(values.quantile(0.25))
            components.append((q75 - q25) / denom)
            components.append(float(values.std(ddof=0)) / denom)
        if len(values) > 1:
            diffs = np.abs(np.diff(values.to_numpy(dtype=np.float64)))
            components.append(float(np.median(diffs)) / denom)
        elif len(values) > 0:
            components.append(0.0)

    for col in CATEGORICAL_COLUMNS:
        if col not in student_frame.columns:
            continue
        values = student_frame[col]
        if values.notna().any():
            components.append(_normalized_entropy(values))
            components.append(_categorical_switch_rate(values))

    if not components:
        return 0.0

    clipped = np.clip(np.asarray(components, dtype=np.float64), 0.0, 5.0)
    return float(np.mean(clipped))


def _make_centroids(aux_sorted, student_scores, source_iqr):
    scored = student_scores.copy()
    try:
        scored["tertile"] = pd.qcut(
            scored["instability_score"].rank(method="first"),
            q=3,
            labels=TERTILE_NAMES,
        ).astype(str)
    except ValueError:
        scored["tertile"] = "middle"

    aux_scored = aux_sorted.merge(
        scored[["student_id", "instability_score", "tertile"]],
        on="student_id",
        how="inner",
    )

    centroids = {}
    global_instability = float(scored["instability_score"].mean()) if len(scored) else 0.0

    for tertile in TERTILE_NAMES:
        part = aux_scored.loc[aux_scored["tertile"] == tertile]
        if part.empty:
            part = aux_scored

        numeric_centroid = {}
        for col in NUMERIC_COLUMNS:
            if col in part.columns:
                numeric_centroid[col] = float(pd.to_numeric(part[col], errors="coerce").median())

        categorical_centroid = {}
        for col in CATEGORICAL_COLUMNS:
            if col in part.columns:
                freqs = part[col].dropna().astype(str).value_counts(normalize=True)
                categorical_centroid[col] = freqs.to_dict()

        if "instability_score" in part.columns and part["instability_score"].notna().any():
            instability_mean = float(part["instability_score"].mean())
        else:
            instability_mean = global_instability

        centroids[tertile] = {
            "numeric": numeric_centroid,
            "categorical": categorical_centroid,
            "instability": instability_mean,
        }

    return centroids, global_instability


def add_external_temporal_stability_affinity(raw, deps, aux):
    if aux is None or len(aux) == 0 or "student_id" not in aux.columns:
        return _empty_external_temporal_stability_affinity(raw.index)

    shared_numeric = [col for col in NUMERIC_COLUMNS if col in raw.columns and col in aux.columns]
    shared_categorical = [col for col in CATEGORICAL_COLUMNS if col in raw.columns and col in aux.columns]
    if not shared_numeric and not shared_categorical:
        return _empty_external_temporal_stability_affinity(raw.index)

    aux_work = aux.copy(deep=False)
    if "timestamp" in aux_work.columns:
        aux_work = aux_work.assign(_parsed_timestamp=pd.to_datetime(aux_work["timestamp"], errors="coerce"))
        aux_work = aux_work.sort_values(["student_id", "_parsed_timestamp"], kind="mergesort")
    else:
        aux_work = aux_work.sort_values(["student_id"], kind="mergesort")

    source_iqr = {}
    for col in NUMERIC_COLUMNS:
        if col in aux_work.columns:
            values = pd.to_numeric(aux_work[col], errors="coerce")
            source_iqr[col] = float(values.quantile(0.75) - values.quantile(0.25))

    score_rows = []
    for student_id, student_frame in aux_work.groupby("student_id", sort=False):
        score_rows.append(
            {
                "student_id": student_id,
                "instability_score": _student_instability(student_frame, source_iqr),
            }
        )

    if not score_rows:
        return _empty_external_temporal_stability_affinity(raw.index)

    student_scores = pd.DataFrame(score_rows)
    centroids, global_instability = _make_centroids(aux_work, student_scores, source_iqr)

    n_rows = len(raw)
    distances = {}
    observed_counts = np.zeros(n_rows, dtype=np.float64)

    for tertile in TERTILE_NAMES:
        distances[tertile] = np.zeros(n_rows, dtype=np.float64)

    for col in shared_numeric:
        raw_values = pd.to_numeric(raw[col], errors="coerce").to_numpy(dtype=np.float64)
        observed = np.isfinite(raw_values)
        observed_counts += observed.astype(np.float64)
        scale = float(source_iqr.get(col, 0.0)) + 1.0e-6

        for tertile in TERTILE_NAMES:
            center = centroids[tertile]["numeric"].get(col)
            if center is None or not np.isfinite(center):
                continue
            diff = ((raw_values - center) / scale) ** 2
            distances[tertile] += np.where(observed, diff, 0.0)

    for col in shared_categorical:
        raw_values = raw[col].astype("object")
        observed = raw_values.notna().to_numpy()
        observed_counts += observed.astype(np.float64)
        raw_strings = raw_values.astype(str)

        for tertile in TERTILE_NAMES:
            freqs = centroids[tertile]["categorical"].get(col, {})
            matched_freq = raw_strings.map(freqs).fillna(0.0).to_numpy(dtype=np.float64)
            mismatch = 1.0 - matched_freq
            distances[tertile] += np.where(observed, mismatch, 0.0)

    has_observed = observed_counts > 0.0
    max_components = float(len(shared_numeric) + len(shared_categorical))
    coverage = np.where(max_components > 0.0, observed_counts / max_components, 0.0)

    similarities = {}
    for tertile in TERTILE_NAMES:
        normalized_distance = np.divide(
            distances[tertile],
            observed_counts,
            out=np.zeros(n_rows, dtype=np.float64),
            where=has_observed,
        )
        sim = np.exp(-normalized_distance)
        sim = np.where(has_observed, sim, 1.0 / 3.0)
        similarities[tertile] = sim

    sim_sum = similarities["low"] + similarities["middle"] + similarities["high"]
    safe_sim_sum = np.where(sim_sum > 0.0, sim_sum, 1.0)

    expected = np.zeros(n_rows, dtype=np.float64)
    for tertile in TERTILE_NAMES:
        expected += similarities[tertile] * float(centroids[tertile]["instability"])
    expected = expected / safe_sim_sum
    expected = np.where(has_observed, expected, global_instability)

    result = pd.DataFrame(
        {
            "low_instability_similarity": similarities["low"].astype(np.float32),
            "middle_instability_similarity": similarities["middle"].astype(np.float32),
            "high_instability_similarity": similarities["high"].astype(np.float32),
            "expected_instability_affinity": expected.astype(np.float32),
            "high_minus_low_instability_similarity": (
                similarities["high"] - similarities["low"]
            ).astype(np.float32),
            "observed_profile_coverage": coverage.astype(np.float32),
        },
        index=raw.index,
    )

    result.loc[~has_observed, "high_minus_low_instability_similarity"] = 0.0
    result.loc[~has_observed, "observed_profile_coverage"] = 0.0

    return result


FEATURE_GROUPS = [
    {
        "name": "external_temporal_stability_affinity",
        "fn": add_external_temporal_stability_affinity,
        "depends_on": [],
        "description": "Source-only temporal routine stability affinities from auxiliary longitudinal student health observations.",
    }
]