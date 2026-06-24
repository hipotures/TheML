import numpy as np
import pandas as pd

_NUMERIC_BASE_FEATURES = ("u", "g", "r", "i", "z")
_RANK_FEATURES = ("u", "g", "r", "i", "z", "redshift_clip", "redshift_log")
_QUANTILES_Q12 = (0.0, 0.08333333333333333, 0.16666666666666666, 0.25, 0.3333333333333333, 0.4166666666666667, 0.5, 0.5833333333333334, 0.6666666666666666, 0.75, 0.8333333333333334, 0.9166666666666666, 1.0)
_MISSING_CATEGORY_TOKEN = "__AUTOGLUON_MISSING__"


def _to_bool_mask(values):
    if values is None:
        return None
    values = pd.Series(values)
    if pd.api.types.is_bool_dtype(values):
        return values.fillna(False).astype(bool)
    if pd.api.types.is_numeric_dtype(values):
        return values.fillna(0).astype(float).gt(0)
    if pd.api.types.is_object_dtype(values) or pd.api.types.is_string_dtype(values):
        text = values.astype(str).str.strip().str.lower()
        train_tokens = {"train", "1", "true", "t", "yes", "y", "training"}
        test_tokens = {"test", "0", "false", "f", "no", "n", "validation", "valid", "holdout"}
        is_train = text.isin(train_tokens)
        is_test = text.isin(test_tokens)
        if is_train.any() or is_test.any():
            if is_train.any() and not is_test.any():
                return is_train
            if is_test.any() and not is_train.any():
                return ~is_test
            if is_train.any() and is_test.any():
                return is_train
    return None


def _infer_train_mask(raw, aux):
    if isinstance(aux, pd.Series):
        if len(aux) == len(raw):
            mask = _to_bool_mask(aux)
            if mask is not None:
                return pd.Series(mask.to_numpy(), index=raw.index, dtype=bool)
        return pd.Series(True, index=raw.index, dtype=bool)
    if not isinstance(aux, pd.DataFrame) or aux.empty:
        return pd.Series(True, index=raw.index, dtype=bool)
    if len(aux) != len(raw):
        return pd.Series(True, index=raw.index, dtype=bool)

    for col in ("is_train", "is_train_mask", "train", "train_mask", "train_flag", "split", "dataset"):
        if col in aux.columns:
            mask = _to_bool_mask(aux[col])
            if mask is not None:
                return pd.Series(mask.to_numpy(), index=raw.index, dtype=bool)
    return pd.Series(True, index=raw.index, dtype=bool)


def _as_float_series(raw, key):
    if key in raw.columns:
        return pd.to_numeric(raw[key], errors="coerce")
    return pd.Series(np.nan, index=raw.index, dtype="float64")


def _empirical_rank_by_reference(reference, target, n_total):
    target = pd.to_numeric(target, errors="coerce")
    out = pd.Series(np.nan, index=target.index, dtype="float64")
    if n_total <= 0:
        return out

    reference = pd.to_numeric(reference, errors="coerce")
    reference = reference[np.isfinite(reference)]
    if reference.empty:
        return out

    ref_vals = reference.to_numpy(dtype=float)
    uniq_vals, counts = np.unique(ref_vals, return_counts=True)
    starts = np.concatenate(([0], np.cumsum(counts)[:-1]))
    avg_rank = (2.0 * starts + counts + 1.0) / 2.0
    pct = (avg_rank - 0.5) / float(n_total - 0.5)

    t = target.to_numpy(dtype=float)
    valid = np.isfinite(t)
    if np.any(valid):
        mapped = np.interp(t[valid], uniq_vals, pct, left=pct[0], right=pct[-1])
        out.iloc[np.where(valid)[0]] = mapped
    return out


def _group_rank_with_fallback(feature, group_vals, train_mask, global_ranks):
    feature = pd.to_numeric(feature, errors="coerce")
    group_vals = pd.Series(group_vals, index=feature.index, dtype="object")
    out = global_ranks.copy()

    train_rows = train_mask & group_vals.notna() & feature.notna()
    if not train_rows.any():
        return out

    train_groups = group_vals.loc[train_rows]
    for grp_val, grp_idx in train_groups.groupby(train_groups).groups.items():
        n_grp = len(grp_idx)
        if n_grp < 2:
            continue
        grp_rank = _empirical_rank_by_reference(feature.loc[grp_idx], feature, n_grp)
        apply_to = group_vals.eq(grp_val) & feature.notna()
        out.loc[apply_to] = grp_rank.loc[apply_to]
    return out


def _shrink_quantile_edges(edges):
    edges = np.asarray(edges, dtype=float)
    if edges.size < 2:
        return edges
    out = edges.copy()
    for i in range(1, out.size):
        if not np.isfinite(out[i]) or out[i] <= out[i - 1]:
            nxt = np.nextafter(out[i - 1], np.inf)
            if not np.isfinite(nxt) or nxt <= out[i - 1]:
                nxt = out[i - 1] + (abs(out[i - 1]) + 1.0) * 1e-12
            out[i] = nxt
    return out


def _bin_frequency_info(feature, train_mask, n_train):
    idx = feature.index
    feature = pd.to_numeric(feature, errors="coerce")
    freq_series = pd.Series(np.nan, index=idx, dtype="float64")
    low_flag = pd.Series(False, index=idx, dtype="bool")
    bin_index = pd.Series(pd.NA, index=idx, dtype="Int64")

    if n_train <= 0:
        return freq_series, low_flag, bin_index

    train_vals = feature.loc[train_mask]
    train_vals = train_vals[np.isfinite(train_vals)]
    if train_vals.empty:
        return freq_series, low_flag, bin_index

    try:
        edges = np.quantile(train_vals.to_numpy(dtype=float), _QUANTILES_Q12)
    except Exception:
        return freq_series, low_flag, bin_index

    if not np.isfinite(edges).all() or np.isclose(float(edges[0]), float(edges[-1])):
        valid = np.isfinite(feature.to_numpy(dtype=float))
        if valid.any():
            val = 1.0 / float(max(1, n_train))
            freq_series.loc[idx[valid]] = val
            bin_index.loc[idx[valid]] = 0
        return freq_series, low_flag, bin_index

    edges = _shrink_quantile_edges(edges)
    try:
        train_bins = pd.cut(train_vals, bins=edges, labels=False, include_lowest=True, duplicates="drop")
        all_bins = pd.cut(feature, bins=edges, labels=False, include_lowest=True, duplicates="drop")
    except Exception:
        return freq_series, low_flag, bin_index

    if train_bins.empty:
        return freq_series, low_flag, bin_index

    n_bins = len(train_bins.cat.categories) if pd.api.types.is_categorical_dtype(train_bins) else int(all_bins.max() + 1 if all_bins.notna().any() else 0)
    if n_bins <= 0:
        return freq_series, low_flag, bin_index

    counts = train_bins.value_counts(dropna=True).reindex(range(n_bins), fill_value=0).astype(float)
    freq_by_bin = counts / float(n_train)

    valid = all_bins.notna()
    codes = all_bins.loc[valid].astype(int)
    freq_series.loc[valid] = codes.map(freq_by_bin)
    bin_index.loc[valid] = codes.astype("Int64")

    train_valid = valid.loc[train_mask]
    train_freq = freq_series.loc[train_mask & train_valid]
    if train_freq.notna().any():
        threshold = float(np.nanquantile(train_freq.to_numpy(dtype=float), 0.10))
        if np.isfinite(threshold):
            low_flag.loc[freq_series.notna() & (freq_series < threshold)] = True

    return freq_series, low_flag, bin_index


def _smoothed_category_features(category_series, train_mask, n_train):
    cat = pd.Series(category_series, index=category_series.index, dtype="object")
    cat = cat.astype(str).where(cat.notna(), _MISSING_CATEGORY_TOKEN)
    train_counts = cat.loc[train_mask].value_counts(dropna=False)
    vocab = int(train_counts.size)
    if vocab <= 0:
        vocab = 1
    freq = cat.map(train_counts).fillna(0.0).astype("float64")
    freq = (freq + 1.0) / (float(max(1, n_train)) + float(vocab))
    return freq, np.log1p(freq)


def add_aide_catalog_rank_frequency_context(raw, deps, aux):
    idx = raw.index
    train_mask = _infer_train_mask(raw, aux).astype(bool)
    train_mask = train_mask.reindex(idx, fill_value=False)
    n_train = int(train_mask.sum())
    if n_train <= 0:
        train_mask = pd.Series(True, index=idx, dtype=bool)
        n_train = len(raw)

    spectral = pd.Series(raw["spectral_type"], index=idx) if "spectral_type" in raw.columns else pd.Series(_MISSING_CATEGORY_TOKEN, index=idx)
    population = pd.Series(raw["galaxy_population"], index=idx) if "galaxy_population" in raw.columns else pd.Series(_MISSING_CATEGORY_TOKEN, index=idx)
    spectral = spectral.astype(str).where(spectral.notna(), _MISSING_CATEGORY_TOKEN)
    population = population.astype(str).where(population.notna(), _MISSING_CATEGORY_TOKEN)
    type_pair = spectral.str.cat(population, sep="|")

    spectral_freq, spectral_logfreq = _smoothed_category_features(spectral, train_mask, n_train)
    population_freq, population_logfreq = _smoothed_category_features(population, train_mask, n_train)
    pair_freq, pair_logfreq = _smoothed_category_features(type_pair, train_mask, n_train)

    base_features = {}
    for col in _NUMERIC_BASE_FEATURES:
        base_features[col] = _as_float_series(raw, col)

    redshift = _as_float_series(raw, "redshift")
    train_redshift = redshift.loc[train_mask]
    train_redshift = train_redshift[np.isfinite(train_redshift)]
    if train_redshift.empty:
        redshift_clip_cap = 0.0
    else:
        redshift_clip_cap = float(np.nanquantile(train_redshift.to_numpy(dtype=float), 0.995))
    redshift_clip_cap = max(0.0, redshift_clip_cap)
    redshift_clip = pd.Series(np.clip(redshift.to_numpy(dtype=float), 0.0, redshift_clip_cap), index=idx, dtype="float64")
    redshift_log = pd.Series(np.log1p(redshift_clip.to_numpy(dtype=float)), index=idx, dtype="float64")
    base_features["redshift_clip"] = redshift_clip
    base_features["redshift_log"] = redshift_log

    global_rank = {}
    for name, vals in base_features.items():
        global_rank[name] = _empirical_rank_by_reference(vals.loc[train_mask], vals, n_train)

    by_type_rank = {}
    by_pop_rank = {}
    by_pair_rank = {}
    for name, vals in base_features.items():
        by_type_rank[name] = _group_rank_with_fallback(vals, spectral, train_mask, global_rank[name])
        by_pop_rank[name] = _group_rank_with_fallback(vals, population, train_mask, global_rank[name])
        by_pair_rank[name] = _group_rank_with_fallback(vals, type_pair, train_mask, global_rank[name])

    bin_freq = {}
    bin_low = {}
    bin_idx = {}
    for name, vals in base_features.items():
        bf, low, bi = _bin_frequency_info(vals, train_mask, n_train)
        bin_freq[name] = bf
        bin_low[name] = low
        bin_idx[name] = bi

    out = {
        "cat_freq_spectral_type": spectral_freq,
        "cat_logfreq_spectral_type": spectral_logfreq,
        "cat_freq_galaxy_population": population_freq,
        "cat_logfreq_galaxy_population": population_logfreq,
        "cat_freq_type_population": pair_freq,
        "cat_logfreq_type_population": pair_logfreq,
    }

    for name in _RANK_FEATURES:
        out[f"rank_global_{name}"] = global_rank[name]
        out[f"rank_by_spectral_type_{name}"] = by_type_rank[name]
        out[f"rank_by_galaxy_population_{name}"] = by_pop_rank[name]
        out[f"rank_by_type_population_{name}"] = by_pair_rank[name]
        out[f"bin_freq_{name}"] = bin_freq[name]
        out[f"bin_index_{name}"] = bin_idx[name]

    global_rank_df = pd.DataFrame(global_rank)
    by_type_df = pd.DataFrame(by_type_rank)
    by_pop_df = pd.DataFrame(by_pop_rank)
    by_pair_df = pd.DataFrame(by_pair_rank)
    low_flag_df = pd.DataFrame(bin_low)

    out["global_rank_mean"] = global_rank_df.mean(axis=1)
    out["global_rank_std"] = global_rank_df.std(axis=1, ddof=0)
    out["within_rank_mean_spectral_type"] = by_type_df.mean(axis=1)
    out["within_rank_mean_galaxy_population"] = by_pop_df.mean(axis=1)
    out["within_rank_mean_type_population"] = by_pair_df.mean(axis=1)
    out["low_freq_bin_count"] = low_flag_df.sum(axis=1).astype("int64")

    return pd.DataFrame(out, index=idx)


FEATURE_GROUPS = [
    {
        "name": "aide_catalog_rank_frequency_context",
        "fn": add_aide_catalog_rank_frequency_context,
        "depends_on": [],
        "description": "Build rank-frequency context features from numeric observables and catalog tags with catalog-conditioned ranking, quantile bins, and low-frequency-bin counters.",
    },
]