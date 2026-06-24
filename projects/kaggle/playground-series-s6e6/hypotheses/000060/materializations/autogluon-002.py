import numpy as np
import pandas as pd

REDSHIFT_BINS = (-0.01, 0.15, 0.40, 1.0, 1.4, 2.0, 2.8, 3.5, 4.5, 7.02)
IMAG_BINS = (-1e308, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 1e308)
CLASS_ORDER = ("GALAXY", "QSO", "STAR")
TARGET_COLUMN_CANDIDATES = ("target", "label", "y", "class", "Class", "ClassLabel", "label_encoded", "target_label", "y_true")
ALPHA = 30.0
BETA = 120.0
CLAMP_LOW = 1e-6
MARGIN_EPS = 1e-12

def _extract_aux_label_series(raw_index, aux):
    if aux is None or not isinstance(aux, pd.DataFrame) or len(aux) == 0:
        return pd.Series([np.nan] * len(raw_index), index=raw_index, dtype=object)

    label_col = None
    for col in TARGET_COLUMN_CANDIDATES:
        if col in aux.columns:
            label_col = col
            break
    if label_col is None:
        return pd.Series([np.nan] * len(raw_index), index=raw_index, dtype=object)

    aux_labels = aux[label_col]
    labels = pd.Series([np.nan] * len(raw_index), index=raw_index, dtype=object)

    if len(aux_labels) == len(raw_index) and aux.index.equals(raw_index):
        return aux_labels.astype(object).copy()

    overlap = raw_index.intersection(aux.index)
    if len(overlap) > 0:
        labels.loc[overlap] = aux.loc[overlap, label_col].astype(object).to_numpy()
        return labels

    n = min(len(aux_labels), len(raw_index))
    if n > 0:
        labels.iloc[:n] = aux_labels.iloc[:n].to_numpy()
    return labels

def _to_canonical_label(value):
    if pd.isna(value):
        return np.nan

    if isinstance(value, (int, np.integer)):
        if value == 0:
            return "GALAXY"
        if value == 1:
            return "QSO"
        if value in (2, 3):
            return "STAR"
        return np.nan

    if isinstance(value, (float, np.floating)):
        if np.isnan(value):
            return np.nan
        if float(value).is_integer():
            i = int(value)
            if i == 0:
                return "GALAXY"
            if i == 1:
                return "QSO"
            if i in (2, 3):
                return "STAR"
        return np.nan

    text = str(value).strip().upper()
    if text in ("GALAXY",):
        return "GALAXY"
    if text in ("QSO",):
        return "QSO"
    if text in ("STAR",):
        return "STAR"

    try:
        numeric = float(text)
    except (TypeError, ValueError):
        return np.nan

    if not np.isfinite(numeric):
        return np.nan
    if numeric.is_integer():
        integer = int(numeric)
        if integer == 0:
            return "GALAXY"
        if integer == 1:
            return "QSO"
        if integer in (2, 3):
            return "STAR"
    return np.nan

def _normalize_label_series(labels):
    normalized = labels.apply(_to_canonical_label)
    return normalized.where(normalized.isin(CLASS_ORDER), pd.NA).astype(object)

def _build_level_prior(train_data, level_cols, class_prior):
    grouped = train_data.groupby(level_cols + ["label"], observed=True).size().unstack(fill_value=0)
    grouped = grouped.reindex(columns=CLASS_ORDER, fill_value=0).astype(float)
    counts = grouped.sum(axis=1)
    level_prior = (grouped + ALPHA * class_prior).div(counts + ALPHA, axis=0)
    level_lambda = counts / (counts + BETA)
    return level_prior, level_lambda

def _blend_level(posterior_df, key_frame, level_prior, level_lambda):
    if level_prior is None or len(level_prior) == 0:
        return posterior_df

    keys = pd.MultiIndex.from_frame(key_frame)
    level_vals = level_prior.reindex(keys)
    if len(level_vals) == 0:
        return posterior_df

    lam = level_lambda.reindex(keys).to_numpy()
    level_arr = level_vals.to_numpy()
    posterior = posterior_df.to_numpy()

    seen = np.isfinite(level_arr).all(axis=1)
    if not np.any(seen):
        return posterior_df

    posterior_seen = posterior[seen]
    posterior[seen] = lam[seen, None] * level_arr[seen] + (1.0 - lam[seen, None]) * posterior_seen
    return pd.DataFrame(posterior, index=posterior_df.index, columns=posterior_df.columns)

def add_metadata_redshift_magnitude_class_priors(raw, deps, aux):
    index = raw.index
    n_rows = len(index)
    if n_rows == 0:
        return pd.DataFrame(index=index)

    redshift = pd.to_numeric(raw["redshift"], errors="coerce")
    i_mag = pd.to_numeric(raw["i"], errors="coerce")
    spectral_type = raw["spectral_type"].astype("string").fillna("UNKNOWN").str.strip()
    galaxy_population = raw["galaxy_population"].astype("string").fillna("UNKNOWN").str.strip()

    redshift_bin = pd.cut(redshift, bins=REDSHIFT_BINS, right=False, include_lowest=True).astype("string")
    i_bin = pd.cut(i_mag, bins=IMAG_BINS, right=False, include_lowest=True).astype("string")

    raw_labels = _extract_aux_label_series(index, aux)
    labels = _normalize_label_series(raw_labels)
    train_mask = labels.notna()

    class_prior = pd.Series(
        [1.0 / len(CLASS_ORDER)] * len(CLASS_ORDER),
        index=CLASS_ORDER,
        dtype=float,
    )

    if train_mask.any():
        y_train = labels.loc[train_mask]
        class_counts = y_train.value_counts().reindex(CLASS_ORDER, fill_value=0).astype(float)
        if class_counts.sum() > 0:
            class_prior = class_counts / class_counts.sum()

        train_meta = pd.DataFrame(
            {
                "spectral_type": spectral_type.loc[train_mask],
                "galaxy_population": galaxy_population.loc[train_mask],
                "redshift_bin": redshift_bin.loc[train_mask],
                "i_bin": i_bin.loc[train_mask],
                "label": y_train,
            },
            index=index[train_mask],
        )

        level4_prior, level4_lambda = _build_level_prior(
            train_meta,
            ["spectral_type", "galaxy_population", "redshift_bin", "i_bin"],
            class_prior,
        )
        level3_prior, level3_lambda = _build_level_prior(
            train_meta,
            ["spectral_type", "galaxy_population", "redshift_bin"],
            class_prior,
        )
        level2_prior, level2_lambda = _build_level_prior(
            train_meta,
            ["spectral_type", "galaxy_population"],
            class_prior,
        )
    else:
        level4_prior = pd.DataFrame(columns=CLASS_ORDER)
        level4_lambda = pd.Series(dtype=float)
        level3_prior = pd.DataFrame(columns=CLASS_ORDER)
        level3_lambda = pd.Series(dtype=float)
        level2_prior = pd.DataFrame(columns=CLASS_ORDER)
        level2_lambda = pd.Series(dtype=float)

    key_frame4 = pd.DataFrame(
        {
            "spectral_type": spectral_type,
            "galaxy_population": galaxy_population,
            "redshift_bin": redshift_bin,
            "i_bin": i_bin,
        },
        index=index,
    )
    key_frame3 = key_frame4[["spectral_type", "galaxy_population", "redshift_bin"]]
    key_frame2 = key_frame4[["spectral_type", "galaxy_population"]]

    posterior = pd.DataFrame(
        np.tile(class_prior.to_numpy(), (n_rows, 1)),
        columns=list(CLASS_ORDER),
        index=index,
    )

    posterior = _blend_level(posterior, key_frame4, level4_prior, level4_lambda)
    posterior = _blend_level(posterior, key_frame3, level3_prior, level3_lambda)
    posterior = _blend_level(posterior, key_frame2, level2_prior, level2_lambda)

    probs = np.array(posterior, dtype=float)
    probs = np.clip(probs, CLAMP_LOW, 1.0 - CLAMP_LOW)
    probs_sum = probs.sum(axis=1, keepdims=True)
    probs = np.divide(probs, np.where(probs_sum == 0, 1.0, probs_sum))
    probs = np.clip(probs, CLAMP_LOW, 1.0 - CLAMP_LOW)

    entropy = -np.sum(probs * np.log(probs), axis=1)
    sorted_probs = np.sort(probs, axis=1)[:, ::-1]
    max_prob = sorted_probs[:, 0]
    top2_gap = sorted_probs[:, 0] - sorted_probs[:, 1]
    margin = np.log((probs + MARGIN_EPS) / (1.0 - probs + 3.0 * MARGIN_EPS))

    features = pd.DataFrame(index=index)
    features["p_GALAXY"] = probs[:, 0]
    features["p_QSO"] = probs[:, 1]
    features["p_STAR"] = probs[:, 2]
    features["class_entropy"] = entropy
    features["max_probability"] = max_prob
    features["top2_gap"] = top2_gap
    features["logit_GALAXY"] = margin[:, 0]
    features["logit_QSO"] = margin[:, 1]
    features["logit_STAR"] = margin[:, 2]

    return features

FEATURE_GROUPS = [
    {
        "name": "metadata_redshift_magnitude_class_priors",
        "fn": add_metadata_redshift_magnitude_class_priors,
        "depends_on": [],
        "description": "Injects hierarchy-smoothed, confidence-clipped class-prior probabilities from metadata, redshift, and i-band context.",
    },
]