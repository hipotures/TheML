import numpy as np
import pandas as pd


_FEATURE_COLUMNS = ("u", "g", "r", "i", "z", "redshift", "spectral_type", "galaxy_population")
_CONTEXT_COLUMNS = ("spectral_type", "galaxy_population")
_INDEX_COLUMNS = ("p1", "v", "l")
_MIN_CELL_N = 500
_Z_CLIP = 8.0
_EPS = 1e-6
_TRAIN_ID_MAX = 577346


def _training_mask(raw):
    if "id" not in raw.columns:
        return pd.Series(True, index=raw.index)
    ids = pd.to_numeric(raw["id"], errors="coerce")
    mask = ids.le(_TRAIN_ID_MAX)
    if bool(mask.any()):
        return mask.fillna(False)
    return pd.Series(True, index=raw.index)


def _safe_numeric(raw, col):
    return pd.to_numeric(raw[col], errors="coerce").astype("float64")


def _mad_scale(values, center):
    abs_dev = np.abs(values - center)
    mad = np.nanmedian(abs_dev)
    q25, q75 = np.nanpercentile(values, (25.0, 75.0))
    mad_scale = 1.4826 * mad if np.isfinite(mad) else np.nan
    iqr_scale = (q75 - q25) / 1.349 if np.isfinite(q25) and np.isfinite(q75) else np.nan
    return mad_scale, iqr_scale


def _robust_center_scale(values, fallback_scale):
    arr = np.asarray(values, dtype="float64")
    arr = arr[np.isfinite(arr)]
    n = int(arr.size)
    if n == 0:
        return np.nan, fallback_scale, 0

    center = float(np.nanmedian(arr))
    mad_scale, iqr_scale = _mad_scale(arr, center)

    candidates = (mad_scale, iqr_scale, fallback_scale, _EPS)
    scale = _EPS
    for candidate in candidates:
        if candidate is not None and np.isfinite(candidate) and candidate > 0:
            scale = float(max(candidate, _EPS))
            break

    return center, scale, n


def _make_redshift_bins(redshift_fit):
    values = np.asarray(redshift_fit, dtype="float64")
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.asarray([-np.inf, np.inf], dtype="float64")

    quantiles = np.nanpercentile(values, np.arange(0.0, 101.0, 10.0))
    unique_edges = np.unique(quantiles[np.isfinite(quantiles)])
    if unique_edges.size < 2:
        return np.asarray([-np.inf, np.inf], dtype="float64")

    unique_edges[0] = -np.inf
    unique_edges[-1] = np.inf
    return unique_edges.astype("float64")


def _assign_zbin(redshift, edges):
    codes = np.searchsorted(edges[1:-1], np.asarray(redshift, dtype="float64"), side="right")
    max_code = max(0, len(edges) - 2)
    return np.clip(codes, 0, max_code).astype("int16")


def _stats_by_keys(frame, value_col, key_cols, fallback_scale):
    grouped = frame.groupby(list(key_cols), observed=True, sort=False)[value_col]
    stats = {}
    for key, values in grouped:
        if not isinstance(key, tuple):
            key = (key,)
        center, scale, n = _robust_center_scale(values.to_numpy(dtype="float64"), fallback_scale)
        if n >= _MIN_CELL_N and np.isfinite(center) and np.isfinite(scale):
            stats[key] = (center, scale)
    return stats


def _lookup_stats(keys, stats, global_center, global_scale):
    centers = np.empty(len(keys), dtype="float64")
    scales = np.empty(len(keys), dtype="float64")
    for idx, key in enumerate(keys):
        stat = stats.get(key)
        if stat is None:
            centers[idx] = global_center
            scales[idx] = global_scale
        else:
            centers[idx] = stat[0]
            scales[idx] = stat[1]
    return centers, scales


def _winsor_limits(values):
    arr = np.asarray(values, dtype="float64")
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return -np.inf, np.inf
    low, high = np.nanpercentile(arr, (0.1, 99.9))
    if not np.isfinite(low):
        low = -np.inf
    if not np.isfinite(high):
        high = np.inf
    if low > high:
        low, high = high, low
    return float(low), float(high)


def _clip_series(values, low, high):
    return pd.Series(np.asarray(values, dtype="float64"), index=values.index if isinstance(values, pd.Series) else None).clip(lower=low, upper=high)


def add_segue_stellar_atmospheric_indices(raw, deps, aux):
    idx = raw.index
    train_mask = _training_mask(raw)

    u = _safe_numeric(raw, "u")
    g = _safe_numeric(raw, "g")
    r = _safe_numeric(raw, "r")
    i_mag = _safe_numeric(raw, "i")
    z_mag = _safe_numeric(raw, "z")
    redshift = _safe_numeric(raw, "redshift")

    spectral = raw["spectral_type"].astype("string").fillna("__missing__")
    population = raw["galaxy_population"].astype("string").fillna("__missing__")

    ug = u - g
    gr = g - r
    ri = r - i_mag
    iz = i_mag - z_mag

    p1_raw = 0.91 * ug + 0.415 * gr - 1.280
    v_raw = 0.283 * ug - 0.354 * gr + 0.455 * ri + 0.766 * iz
    l_raw = -0.436 * u + 1.129 * g - 0.119 * r - 0.574 * i_mag + 0.1984

    fit = pd.DataFrame(
        {
            "redshift": redshift.loc[train_mask],
            "spectral_type": spectral.loc[train_mask],
            "galaxy_population": population.loc[train_mask],
            "p1": p1_raw.loc[train_mask],
            "v": v_raw.loc[train_mask],
            "l": l_raw.loc[train_mask],
        },
        index=raw.index[train_mask],
    )

    edges = _make_redshift_bins(fit["redshift"])
    zbin_all = _assign_zbin(redshift, edges)
    zbin_fit = _assign_zbin(fit["redshift"], edges)
    fit = fit.assign(zbin=zbin_fit)

    features = pd.DataFrame(index=idx)
    raw_indices = {"p1": p1_raw, "v": v_raw, "l": l_raw}
    contextual_z = {}

    for name in _INDEX_COLUMNS:
        fit_values = fit[name].to_numpy(dtype="float64")
        global_center, global_scale, _ = _robust_center_scale(fit_values, _EPS)
        if not np.isfinite(global_center):
            global_center = 0.0
        if not np.isfinite(global_scale) or global_scale <= 0:
            global_scale = _EPS

        spectral_stats = _stats_by_keys(fit, name, ("spectral_type",), global_scale)
        zbin_spectral_stats = _stats_by_keys(fit, name, ("zbin", "spectral_type"), global_scale)
        full_stats = _stats_by_keys(fit, name, ("zbin", "spectral_type", "galaxy_population"), global_scale)

        spectral_keys = [(s,) for s in spectral.to_numpy()]
        zbin_spectral_keys = list(zip(zbin_all, spectral.to_numpy()))
        full_keys = list(zip(zbin_all, spectral.to_numpy(), population.to_numpy()))

        spectral_center, spectral_scale = _lookup_stats(
            spectral_keys, spectral_stats, global_center, global_scale
        )
        mid_center, mid_scale = _lookup_stats(
            zbin_spectral_keys, zbin_spectral_stats, global_center, global_scale
        )

        context_center = np.empty(len(raw), dtype="float64")
        context_scale = np.empty(len(raw), dtype="float64")
        for row_pos, key in enumerate(full_keys):
            stat = full_stats.get(key)
            if stat is not None:
                context_center[row_pos] = stat[0]
                context_scale[row_pos] = stat[1]
            else:
                mid_stat = zbin_spectral_stats.get(zbin_spectral_keys[row_pos])
                if mid_stat is not None:
                    context_center[row_pos] = mid_stat[0]
                    context_scale[row_pos] = mid_stat[1]
                else:
                    spec_stat = spectral_stats.get(spectral_keys[row_pos])
                    if spec_stat is not None:
                        context_center[row_pos] = spec_stat[0]
                        context_scale[row_pos] = spec_stat[1]
                    else:
                        context_center[row_pos] = global_center
                        context_scale[row_pos] = global_scale

        values = raw_indices[name].to_numpy(dtype="float64")
        z_global = np.nan_to_num((values - global_center) / global_scale, nan=0.0, posinf=0.0, neginf=0.0)
        z_spectral = np.nan_to_num((values - spectral_center) / spectral_scale, nan=0.0, posinf=0.0, neginf=0.0)
        z_context = np.nan_to_num((values - context_center) / context_scale, nan=0.0, posinf=0.0, neginf=0.0)

        z_global = np.clip(z_global, -_Z_CLIP, _Z_CLIP)
        z_spectral = np.clip(z_spectral, -_Z_CLIP, _Z_CLIP)
        z_context = np.clip(z_context, -_Z_CLIP, _Z_CLIP)

        low, high = _winsor_limits(fit_values)
        features[name] = raw_indices[name].clip(lower=low, upper=high).replace([np.inf, -np.inf], np.nan).fillna(global_center)
        features[f"z_{name}_contextual"] = z_context
        features[f"z_{name}_global"] = z_global
        features[f"z_{name}_spectral"] = z_spectral
        features[f"abs_z_{name}_contextual"] = np.abs(z_context)
        features[f"delta_z_{name}_global_contextual"] = z_global - z_context
        contextual_z[name] = z_context

    p1 = features["p1"]
    v = features["v"]
    l = features["l"]

    extras = pd.DataFrame(index=idx)
    extras["p1_low"] = np.maximum(0.0, -0.70 - p1)
    extras["p1_high"] = np.maximum(0.0, p1 + 0.25)
    extras["p1_band_violation"] = extras["p1_low"] + extras["p1_high"]
    extras["l_low"] = np.maximum(0.0, 0.07 - l)
    extras["l_high"] = np.maximum(0.0, l - 0.135)
    extras["l_band_violation"] = extras["l_low"] + extras["l_high"]
    extras["v_abs"] = np.abs(v)

    z_p1 = contextual_z["p1"]
    z_v = contextual_z["v"]
    z_l = contextual_z["l"]
    abs_z_p1 = np.abs(z_p1)
    abs_z_v = np.abs(z_v)
    abs_z_l = np.abs(z_l)

    extras["abs_z_mean"] = (abs_z_p1 + abs_z_v + abs_z_l) / 3.0
    extras["abs_z_max"] = np.maximum.reduce([abs_z_p1, abs_z_v, abs_z_l])
    extras["locus_dist"] = np.sqrt(z_p1 * z_p1 + z_v * z_v + z_l * z_l)
    extras["signed_sum"] = z_p1 + z_v + z_l
    extras["tail_count"] = ((abs_z_p1 > 3.0).astype("int8") + (abs_z_v > 3.0).astype("int8") + (abs_z_l > 3.0).astype("int8"))
    extras["cross_vp"] = extras["v_abs"] * z_p1
    extras["cross_vl"] = extras["v_abs"] * z_l

    fit_extras = extras.loc[train_mask]
    for col in extras.columns:
        low, high = _winsor_limits(fit_extras[col])
        extras[col] = extras[col].clip(lower=low, upper=high).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    features["zbin"] = pd.Series(zbin_all, index=idx).astype("int16")
    features = pd.concat([features, extras], axis=1)
    features = features.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    return features


FEATURE_GROUPS = [
    {
        "name": "segue_stellar_atmospheric_indices",
        "fn": add_segue_stellar_atmospheric_indices,
        "depends_on": [],
        "description": "Robust SEGUE-style stellar-atmosphere color-locus residuals by redshift, spectral type, and population context.",
    }
]