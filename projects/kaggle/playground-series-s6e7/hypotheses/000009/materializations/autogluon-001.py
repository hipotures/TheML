import itertools
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

TRAIN_ID_MAX = 690087
N_BINS = 8
EPS = 1.0e-12


def _training_mask(raw):
    if "id" in raw.columns:
        ids = pd.to_numeric(raw["id"], errors="coerce")
        mask = ids.le(TRAIN_ID_MAX)
        if mask.any():
            return mask.fillna(False).to_numpy()
    return np.ones(len(raw), dtype=bool)


def _normalize_categorical(series):
    out = series.astype("string").str.strip().str.lower()
    return out.fillna("MISSING").replace("", "MISSING").astype(object)


def _bin_numeric_from_train(series, train_mask):
    values = pd.to_numeric(series, errors="coerce")
    train_values = values.loc[train_mask & values.notna()]

    if train_values.empty:
        return pd.Series("MISSING", index=series.index, dtype=object)

    quantiles = np.linspace(0.0, 1.0, N_BINS + 1)
    edges = np.unique(np.nanquantile(train_values.to_numpy(dtype=float), quantiles))

    binned = pd.Series("MISSING", index=series.index, dtype=object)
    observed = values.notna()

    if len(edges) <= 1:
        binned.loc[observed] = "bin_0"
        return binned

    codes = np.searchsorted(edges[1:-1], values.loc[observed].to_numpy(dtype=float), side="right")
    codes = np.clip(codes, 0, len(edges) - 2)
    binned.loc[observed] = pd.Series(codes, index=values.loc[observed].index).map(lambda x: "bin_%d" % int(x))
    return binned


def _tokenize_features(raw, train_mask):
    tokens = {}

    for col in NUMERIC_COLUMNS:
        if col in raw.columns:
            tokens[col] = _bin_numeric_from_train(raw[col], train_mask)
        else:
            tokens[col] = pd.Series("MISSING", index=raw.index, dtype=object)

    for col in CATEGORICAL_COLUMNS:
        if col in raw.columns:
            normalized = _normalize_categorical(raw[col])
            seen = set(normalized.loc[train_mask].unique())
            tokens[col] = normalized.where(normalized.isin(seen), "UNSEEN")
        else:
            tokens[col] = pd.Series("MISSING", index=raw.index, dtype=object)

    return tokens


def add_behavioral_cooccurrence_rarity(raw, deps, aux):
    train_mask = _training_mask(raw)
    n_train = int(np.sum(train_mask))
    n_rows = len(raw)

    if n_rows == 0:
        return pd.DataFrame(index=raw.index)

    tokens = _tokenize_features(raw, train_mask)
    feature_names = list(NUMERIC_COLUMNS) + list(CATEGORICAL_COLUMNS)
    pairs = list(itertools.combinations(feature_names, 2))
    n_pairs = max(len(pairs), 1)

    neg_log_matrix = np.zeros((n_rows, len(pairs)), dtype=float)
    low_prob_matrix = np.zeros((n_rows, len(pairs)), dtype=bool)
    unseen_matrix = np.zeros((n_rows, len(pairs)), dtype=bool)
    abs_lift_matrix = np.zeros((n_rows, len(pairs)), dtype=float)

    for pair_idx, (left_name, right_name) in enumerate(pairs):
        left = tokens[left_name].astype(object)
        right = tokens[right_name].astype(object)
        pair_values = left.astype(str) + "\x1f" + right.astype(str)

        train_left = left.loc[train_mask]
        train_right = right.loc[train_mask]
        train_pairs = pair_values.loc[train_mask]

        pair_counts = train_pairs.value_counts(dropna=False)
        left_counts = train_left.value_counts(dropna=False)
        right_counts = train_right.value_counts(dropna=False)

        pair_cardinality = int(pair_counts.shape[0])
        left_cardinality = int(left_counts.shape[0])
        right_cardinality = int(right_counts.shape[0])

        pair_denominator = float(n_train + pair_cardinality + 1)
        left_denominator = float(n_train + left_cardinality + 1)
        right_denominator = float(n_train + right_cardinality + 1)

        pair_prob = (pair_values.map(pair_counts).fillna(0.0).to_numpy(dtype=float) + 1.0) / pair_denominator
        left_prob = (left.map(left_counts).fillna(0.0).to_numpy(dtype=float) + 1.0) / left_denominator
        right_prob = (right.map(right_counts).fillna(0.0).to_numpy(dtype=float) + 1.0) / right_denominator

        neg_log = -np.log(np.maximum(pair_prob, EPS))
        train_neg_log = neg_log[train_mask]
        cap_value = float(np.nanquantile(train_neg_log, 0.999)) if train_neg_log.size else float(np.nanmax(neg_log))
        low_prob_threshold = float(np.nanquantile(pair_prob[train_mask], 0.01)) if n_train else 0.0

        neg_log_matrix[:, pair_idx] = neg_log
        low_prob_matrix[:, pair_idx] = pair_prob < low_prob_threshold
        unseen_matrix[:, pair_idx] = pair_values.map(pair_counts).isna().to_numpy()
        abs_lift_matrix[:, pair_idx] = np.abs(np.log(np.maximum(pair_prob, EPS) / np.maximum(left_prob * right_prob, EPS)))

        if pair_idx == 0:
            capped_max = np.minimum(neg_log, cap_value)
        else:
            capped_max = np.maximum(capped_max, np.minimum(neg_log, cap_value))

    low_count = low_prob_matrix.sum(axis=1).astype(np.int16)
    unseen_count = unseen_matrix.sum(axis=1).astype(np.int16)

    return pd.DataFrame(
        {
            "mean_neg_log_pair_probability": neg_log_matrix.mean(axis=1),
            "p90_neg_log_pair_probability": np.nanquantile(neg_log_matrix, 0.90, axis=1),
            "max_neg_log_pair_probability_capped": capped_max,
            "low_probability_pair_count": low_count,
            "low_probability_pair_proportion": low_count.astype(float) / float(n_pairs),
            "unseen_pair_value_count": unseen_count,
            "mean_abs_pair_lift": abs_lift_matrix.mean(axis=1),
        },
        index=raw.index,
    )


FEATURE_GROUPS = [
    {
        "name": "behavioral_cooccurrence_rarity",
        "fn": add_behavioral_cooccurrence_rarity,
        "depends_on": [],
        "description": "Unsupervised rarity and lift scores for binned lifestyle, physiology, and self-report feature co-occurrences.",
    }
]