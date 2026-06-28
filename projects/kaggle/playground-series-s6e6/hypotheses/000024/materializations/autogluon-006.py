import numpy as np
import pandas as pd


CATEGORY_COLUMNS = ("spectral_type", "galaxy_population")
MAGNITUDE_COLUMNS = ("u", "g", "r", "i", "z")
NUMERIC_CONTEXT_COLUMNS = ("u", "g", "r", "i", "z", "redshift_clip", "redshift_log")
GROUP_FAMILIES = ("spectral_type", "galaxy_population", "catalog_pair")
MAX_QUANTILE_BINS = 12
MIN_GROUP_SIZE = 30
SMOOTHING_COUNT = 1.0
LOW_BIN_FREQ_QUANTILE = 0.10
REDSHIFT_CLIP_QUANTILE = 0.995


def _safe_float_series(raw, column, index):
    if column in raw.columns:
        return pd.to_numeric(raw[column], errors="coerce").astype(float)
    return pd.Series(np.nan, index=index, dtype=float)


def _safe_category_series(raw, column, index):
    if column in raw.columns:
        values = raw[column].astype("string").fillna("__missing__")
        return values.astype(str)
    return pd.Series("__missing__", index=index, dtype=object)


def _smoothed_frequency(values):
    values = pd.Series(values, index=values.index if hasattr(values, "index") else None)
    counts = values.value_counts(dropna=False)
    n_rows = float(len(values))
    n_keys = float(max(len(counts), 1))
    freqs = (counts + SMOOTHING_COUNT) / (n_rows + SMOOTHING_COUNT * n_keys)
    unseen = SMOOTHING_COUNT / (n_rows + SMOOTHING_COUNT * n_keys)
    mapped = values.map(freqs).fillna(unseen).astype(float)
    return mapped, np.log(np.maximum(mapped.to_numpy(dtype=float), np.finfo(float).tiny))


def _global_percentile_rank(values):
    arr = np.asarray(values, dtype=float)
    out = np.full(arr.shape[0], 0.5, dtype=float)
    valid = np.isfinite(arr)
    if valid.sum() <= 1:
        return out

    valid_values = arr[valid]
    sorter = np.sort(valid_values)
    left = np.searchsorted(sorter, valid_values, side="left")
    right = np.searchsorted(sorter, valid_values, side="right")
    avg_position = (left + right - 1.0) / 2.0
    denom = max(float(len(sorter) - 1), 1.0)
    out[valid] = np.clip(avg_position / denom, 0.0, 1.0)
    return out


def _rank_against_sorted(values, sorted_values):
    arr = np.asarray(values, dtype=float)
    out = np.full(arr.shape[0], 0.5, dtype=float)
    if len(sorted_values) <= 1:
        return out

    valid = np.isfinite(arr)
    if not valid.any():
        return out

    left = np.searchsorted(sorted_values, arr[valid], side="left")
    right = np.searchsorted(sorted_values, arr[valid], side="right")
    avg_position = (left + right - 1.0) / 2.0
    denom = max(float(len(sorted_values) - 1), 1.0)
    out[valid] = np.clip(avg_position / denom, 0.0, 1.0)
    return out


def _quantile_edges(values):
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size <= 1:
        return np.array([], dtype=float)

    quantiles = np.linspace(0.0, 1.0, MAX_QUANTILE_BINS + 1)
    edges = np.quantile(arr, quantiles)
    edges = np.unique(edges)
    if edges.size <= 1:
        return np.array([], dtype=float)
    return edges


def _quantile_bin_features(values):
    arr = np.asarray(values, dtype=float)
    edges = _quantile_edges(arr)
    n = arr.shape[0]

    if edges.size <= 1:
        bin_id = np.zeros(n, dtype=float)
        freq = np.ones(n, dtype=float)
        return bin_id, freq, np.log(freq)

    inner_edges = edges[1:-1]
    bin_id = np.searchsorted(inner_edges, arr, side="right").astype(float)
    bin_id[~np.isfinite(arr)] = -1.0

    valid_bins = bin_id[bin_id >= 0].astype(int)
    if valid_bins.size == 0:
        freq = np.ones(n, dtype=float)
        return bin_id, freq, np.log(freq)

    counts = np.bincount(valid_bins, minlength=max(int(valid_bins.max()) + 1, 1))
    freq_by_bin = counts.astype(float) / float(valid_bins.size)
    freq = np.full(n, 1.0 / float(max(len(freq_by_bin), 1)), dtype=float)
    valid_mask = bin_id >= 0
    freq[valid_mask] = freq_by_bin[bin_id[valid_mask].astype(int)]
    return bin_id, freq, np.log(np.maximum(freq, np.finfo(float).tiny))


def _conditional_rank(values, groups, global_rank):
    arr = np.asarray(values, dtype=float)
    group_values = pd.Series(groups, index=groups.index if hasattr(groups, "index") else None)
    out = np.asarray(global_rank, dtype=float).copy()
    fallback = np.ones(arr.shape[0], dtype=np.int8)

    grouped_indices = group_values.groupby(group_values, sort=False, dropna=False).indices
    for _, positions in grouped_indices.items():
        pos = np.asarray(positions, dtype=int)
        valid_pos = pos[np.isfinite(arr[pos])]
        if valid_pos.size < MIN_GROUP_SIZE:
            continue

        sorted_values = np.sort(arr[valid_pos])
        out[pos] = _rank_against_sorted(arr[pos], sorted_values)
        fallback[pos] = 0

    return out, fallback


def add_aide_catalog_rank_frequency_context(raw, deps, aux):
    index = raw.index
    n_rows = len(raw)
    features = pd.DataFrame(index=index)

    spectral = _safe_category_series(raw, "spectral_type", index)
    population = _safe_category_series(raw, "galaxy_population", index)
    catalog_pair = spectral + "__x__" + population

    category_map = {
        "spectral_type": spectral,
        "galaxy_population": population,
        "catalog_pair": catalog_pair,
    }

    for name, values in category_map.items():
        freq, log_freq = _smoothed_frequency(values)
        features[name + "_freq"] = freq.to_numpy(dtype=float)
        features[name + "_log_freq"] = log_freq

    redshift = _safe_float_series(raw, "redshift", index)
    finite_redshift = redshift[np.isfinite(redshift.to_numpy(dtype=float))]
    if len(finite_redshift) > 0:
        redshift_upper = float(finite_redshift.quantile(REDSHIFT_CLIP_QUANTILE))
        if not np.isfinite(redshift_upper):
            redshift_upper = 0.0
    else:
        redshift_upper = 0.0

    redshift_clip = redshift.clip(lower=0.0, upper=max(redshift_upper, 0.0))
    redshift_log = np.log1p(redshift_clip)

    numeric_values = {}
    for column in MAGNITUDE_COLUMNS:
        numeric_values[column] = _safe_float_series(raw, column, index)
    numeric_values["redshift_clip"] = redshift_clip
    numeric_values["redshift_log"] = pd.Series(redshift_log, index=index, dtype=float)

    global_rank_arrays = {}
    bin_freq_arrays = {}
    conditional_rank_arrays = {family: [] for family in GROUP_FAMILIES}
    fallback_arrays = {family: [] for family in GROUP_FAMILIES}

    for column in NUMERIC_CONTEXT_COLUMNS:
        values = numeric_values[column].to_numpy(dtype=float)

        if column in ("redshift_clip", "redshift_log"):
            features[column] = values

        global_rank = _global_percentile_rank(values)
        global_rank_arrays[column] = global_rank
        features[column + "_global_rank"] = global_rank

        bin_id, bin_freq, bin_log_freq = _quantile_bin_features(values)
        bin_freq_arrays[column] = bin_freq
        features[column + "_qbin_id"] = bin_id
        features[column + "_qbin_freq"] = bin_freq
        features[column + "_qbin_log_freq"] = bin_log_freq

        for family, group_values in category_map.items():
            conditional_rank, fallback = _conditional_rank(values, group_values, global_rank)
            conditional_rank_arrays[family].append(conditional_rank)
            fallback_arrays[family].append(fallback.astype(float))
            features[column + "_" + family + "_rank"] = conditional_rank
            features[column + "_" + family + "_rank_fallback"] = fallback

    rank_matrix = np.column_stack([global_rank_arrays[column] for column in NUMERIC_CONTEXT_COLUMNS])
    bin_freq_matrix = np.column_stack([bin_freq_arrays[column] for column in NUMERIC_CONTEXT_COLUMNS])

    features["rank_mean"] = np.mean(rank_matrix, axis=1)
    features["rank_std"] = np.std(rank_matrix, axis=1)
    features["rank_min"] = np.min(rank_matrix, axis=1)
    features["rank_max"] = np.max(rank_matrix, axis=1)
    features["rank_range"] = features["rank_max"] - features["rank_min"]

    for family in GROUP_FAMILIES:
        family_matrix = np.column_stack(conditional_rank_arrays[family])
        fallback_matrix = np.column_stack(fallback_arrays[family])
        available = fallback_matrix < 0.5
        diff = np.abs(family_matrix - rank_matrix)
        available_count = available.sum(axis=1)
        diff_sum = np.where(available, diff, 0.0).sum(axis=1)
        features[family + "_rank_abs_diff_mean"] = np.divide(
            diff_sum,
            np.maximum(available_count, 1),
            out=np.zeros(n_rows, dtype=float),
            where=np.maximum(available_count, 1) > 0,
        )
        features[family + "_rank_fallback_count"] = fallback_matrix.sum(axis=1)

    features["qbin_freq_mean"] = np.mean(bin_freq_matrix, axis=1)

    finite_bin_freqs = bin_freq_matrix[np.isfinite(bin_freq_matrix)]
    if finite_bin_freqs.size > 0:
        low_freq_threshold = float(np.quantile(finite_bin_freqs, LOW_BIN_FREQ_QUANTILE))
    else:
        low_freq_threshold = 0.0
    features["low_qbin_freq_count"] = (bin_freq_matrix < low_freq_threshold).sum(axis=1).astype(float)

    features.replace([np.inf, -np.inf], np.nan, inplace=True)
    features = features.fillna(0.0)
    return features


FEATURE_GROUPS = [
    {
        "name": "aide_catalog_rank_frequency_context",
        "fn": add_aide_catalog_rank_frequency_context,
        "depends_on": [],
        "description": "Unsupervised catalog tag frequency, global rank, quantile-bin, and group-conditioned rank context features.",
    }
]