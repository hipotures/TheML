import numpy as np
import pandas as pd


BANDS = ("u", "g", "r", "i", "z")
COLOR_BY_BAND = {
    "u": "ug",
    "g": "gr",
    "r": "ri",
    "i": "iz",
    "z": "gr",
}
REGIME_EDGES = (0.0, 0.05, 0.3, 0.8, 1.8, 7.1)
MIN_FIT_ROWS = 4000
MIN_CELL_ROWS = 100
TARGET_CELL_ROWS = 150
MAX_Z_BINS = 25
MAX_COLOR_BINS = 10
EPS = 1.0e-6


def _numeric_series(raw, name, default=0.0):
    if name in raw.columns:
        return pd.to_numeric(raw[name], errors="coerce").astype(float)
    return pd.Series(default, index=raw.index, dtype=float)


def _string_series(raw, name, default="missing"):
    if name in raw.columns:
        return raw[name].astype("string").fillna(default).astype(str)
    return pd.Series(default, index=raw.index, dtype=object)


def _fit_mask(raw, aux):
    mask = pd.Series(True, index=raw.index)
    if isinstance(aux, pd.DataFrame) and len(aux) == len(raw):
        for col in ("is_train", "train", "fit_mask"):
            if col in aux.columns:
                return aux[col].astype(bool).reindex(raw.index, fill_value=False)
        for col in ("split", "dataset", "fold_role"):
            if col in aux.columns:
                vals = aux[col].astype(str).str.lower()
                return vals.isin(("train", "fit", "training")).reindex(raw.index, fill_value=False)
    return mask


def _safe_quantile(values, q, fallback):
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float(fallback)
    return float(np.nanquantile(arr, q))


def _mad(values):
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return 1.0
    med = float(np.nanmedian(arr))
    val = float(np.nanmedian(np.abs(arr - med)))
    if not np.isfinite(val) or val <= EPS:
        val = float(np.nanstd(arr))
    if not np.isfinite(val) or val <= EPS:
        val = 1.0
    return val


def _robust_surface_fit(z, c, y):
    z = np.asarray(z, dtype=float)
    c = np.asarray(c, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(z) & np.isfinite(c) & np.isfinite(y)
    if int(mask.sum()) < 20:
        return None
    z = z[mask]
    c = c[mask]
    y = y[mask]
    x = np.column_stack((z, z * z, c, c * c, z * c))
    beta = np.zeros(x.shape[1], dtype=float)
    weights = np.ones(len(y), dtype=float)
    for _ in range(6):
        sw = np.sqrt(weights)
        xw = x * sw[:, None]
        yw = y * sw
        try:
            beta = np.linalg.lstsq(xw, yw, rcond=None)[0]
        except np.linalg.LinAlgError:
            return None
        resid = y - x.dot(beta)
        scale = 1.4826 * _mad(resid)
        if not np.isfinite(scale) or scale <= EPS:
            break
        cutoff = 1.345 * scale
        weights = np.minimum(1.0, cutoff / (np.abs(resid) + EPS))
    return beta


def _surface_predict(beta, z, c):
    if beta is None:
        return np.zeros(len(z), dtype=float)
    z = np.asarray(z, dtype=float)
    c = np.asarray(c, dtype=float)
    return (
        beta[0] * z
        + beta[1] * z * z
        + beta[2] * c
        + beta[3] * c * c
        + beta[4] * z * c
    )


def _rank_bins(values, fit_mask, max_bins, target_rows):
    s = pd.Series(values)
    fit_values = s.loc[fit_mask].replace([np.inf, -np.inf], np.nan).dropna()
    if fit_values.empty:
        return pd.Series(0, index=s.index, dtype=np.int16), 1
    n_bins = int(min(max_bins, max(2, len(fit_values) // target_rows)))
    if n_bins <= 1:
        return pd.Series(0, index=s.index, dtype=np.int16), 1
    ranks = fit_values.rank(method="first")
    _, edges = pd.qcut(ranks, q=n_bins, retbins=True, duplicates="drop")
    actual_bins = max(1, len(edges) - 1)
    fit_quantiles = np.nanquantile(fit_values.to_numpy(dtype=float), np.linspace(0.0, 1.0, actual_bins + 1))
    fit_quantiles = np.unique(fit_quantiles)
    if len(fit_quantiles) <= 2:
        return pd.Series(0, index=s.index, dtype=np.int16), 1
    binned = np.searchsorted(fit_quantiles[1:-1], s.to_numpy(dtype=float), side="right")
    binned = np.where(np.isfinite(s.to_numpy(dtype=float)), binned, 0)
    return pd.Series(binned.astype(np.int16), index=s.index), int(len(fit_quantiles) - 1)


def _group_baseline(values, keys, fit_mask):
    df = pd.DataFrame({"value": values, "key": keys, "fit": fit_mask})
    fit_df = df.loc[df["fit"] & np.isfinite(df["value"])]
    global_med = float(fit_df["value"].median()) if len(fit_df) else 0.0
    global_mad = _mad(fit_df["value"].to_numpy(dtype=float)) if len(fit_df) else 1.0
    if fit_df.empty:
        med = pd.Series(global_med, index=df.index, dtype=float)
        mad = pd.Series(global_mad, index=df.index, dtype=float)
        return med, mad
    med_map = fit_df.groupby("key", sort=False)["value"].median()
    mad_map = fit_df.groupby("key", sort=False)["value"].apply(_mad)
    med = df["key"].map(med_map).fillna(global_med).astype(float)
    mad = df["key"].map(mad_map).fillna(global_mad).clip(lower=EPS).astype(float)
    return med, mad


def _cell_stats(values, cell_keys, baseline_keys, fit_mask):
    df = pd.DataFrame(
        {
            "value": np.asarray(values, dtype=float),
            "cell": cell_keys.astype(str),
            "base": baseline_keys.astype(str),
            "fit": fit_mask.to_numpy(dtype=bool),
        },
        index=fit_mask.index,
    )
    fit_df = df.loc[df["fit"] & np.isfinite(df["value"])]
    base_med, base_mad = _group_baseline(df["value"], df["base"], fit_mask)
    if fit_df.empty:
        return base_med, base_mad, pd.Series(True, index=df.index, dtype=bool), pd.Series(0, index=df.index, dtype=np.int16)
    grouped = fit_df.groupby("cell", sort=False)["value"]
    counts = grouped.size()
    med_map = grouped.median()
    mad_map = grouped.apply(_mad)
    cell_counts = df["cell"].map(counts).fillna(0).astype(np.int16)
    low_support = cell_counts < MIN_CELL_ROWS
    med = df["cell"].map(med_map).astype(float)
    mad = df["cell"].map(mad_map).astype(float)
    med = med.where(~low_support, base_med).fillna(base_med).astype(float)
    mad = mad.where(~low_support, base_mad).fillna(base_mad).clip(lower=EPS).astype(float)
    return med, mad, low_support.astype(bool), cell_counts


def _make_effective_keys(stratum, spectral, regime, fit_mask):
    full = stratum.astype(str) + "|r" + regime.astype(str)
    spec = spectral.astype(str) + "|r" + regime.astype(str)
    glob = pd.Series("global|r", index=regime.index, dtype=object) + regime.astype(str)

    fit_full_counts = full.loc[fit_mask].value_counts()
    fit_spec_counts = spec.loc[fit_mask].value_counts()
    fit_glob_counts = glob.loc[fit_mask].value_counts()

    full_ok = full.map(fit_full_counts).fillna(0) >= MIN_FIT_ROWS
    spec_ok = spec.map(fit_spec_counts).fillna(0) >= MIN_FIT_ROWS
    glob_ok = glob.map(fit_glob_counts).fillna(0) >= MIN_FIT_ROWS

    effective = full.where(full_ok, spec.where(spec_ok, glob))
    fallback_level = pd.Series(0, index=regime.index, dtype=np.int8)
    fallback_level = fallback_level.where(full_ok, 1)
    fallback_level = fallback_level.where(full_ok | spec_ok, 2)
    fallback_level = fallback_level.where(full_ok | spec_ok | glob_ok, 3)
    return effective.astype(str), fallback_level.astype(np.int8)


def add_k_correction_residual_manifold(raw, deps, aux):
    index = raw.index
    fit_mask = _fit_mask(raw, aux)

    alpha = _numeric_series(raw, "alpha")
    delta = _numeric_series(raw, "delta")
    redshift = _numeric_series(raw, "redshift")
    spectral = _string_series(raw, "spectral_type")
    population = _string_series(raw, "galaxy_population")
    stratum = spectral + "|" + population

    z_low = _safe_quantile(redshift.loc[fit_mask], 0.005, 0.0)
    z_high = _safe_quantile(redshift.loc[fit_mask], 0.995, 7.1)
    z_low = min(max(z_low, REGIME_EDGES[0]), REGIME_EDGES[-1])
    z_high = max(min(z_high, REGIME_EDGES[-1]), z_low + EPS)
    zc = redshift.clip(lower=z_low, upper=z_high)

    regime_edges = np.asarray(REGIME_EDGES, dtype=float)
    regime = pd.Series(np.searchsorted(regime_edges[1:-1], zc.to_numpy(dtype=float), side="right"), index=index, dtype=np.int8)
    effective_key, fallback_level = _make_effective_keys(stratum, spectral, regime, fit_mask)

    band_values = {band: _numeric_series(raw, band) for band in BANDS}
    colors = {
        "ug": band_values["u"] - band_values["g"],
        "gr": band_values["g"] - band_values["r"],
        "ri": band_values["r"] - band_values["i"],
        "iz": band_values["i"] - band_values["z"],
    }

    features = pd.DataFrame(index=index)
    features["z_clipped_low"] = (redshift < z_low).astype(np.int8)
    features["z_clipped_high"] = (redshift > z_high).astype(np.int8)
    features["regime"] = regime.astype(np.int8)
    features["fallback_level"] = fallback_level.astype(np.int8)
    features["sky_alpha_sin"] = np.sin(np.deg2rad(alpha.to_numpy(dtype=float)))
    features["sky_alpha_cos"] = np.cos(np.deg2rad(alpha.to_numpy(dtype=float)))
    features["sky_delta_sin"] = np.sin(np.deg2rad(delta.to_numpy(dtype=float)))

    corrected = {}
    blend_width = 0.04 * (1.0 + zc.to_numpy(dtype=float))
    boundary_distance = np.full(len(raw), np.inf, dtype=float)
    for edge in REGIME_EDGES[1:-1]:
        boundary_distance = np.minimum(boundary_distance, np.abs(zc.to_numpy(dtype=float) - edge))
    features["near_regime_boundary"] = (boundary_distance <= blend_width).astype(np.int8)
    features["regime_boundary_weight"] = np.where(
        np.isfinite(boundary_distance),
        np.clip(1.0 - boundary_distance / (blend_width + EPS), 0.0, 1.0),
        0.0,
    )

    for band in BANDS:
        color_name = COLOR_BY_BAND[band]
        color = colors[color_name]
        c_low = _safe_quantile(color.loc[fit_mask], 0.005, float(color.median()))
        c_high = _safe_quantile(color.loc[fit_mask], 0.995, float(color.median()))
        if c_high <= c_low:
            c_high = c_low + EPS
        cc = color.clip(lower=c_low, upper=c_high)

        ref_key = stratum
        ref_candidate = redshift < 0.03
        ref_fit = fit_mask & ref_candidate
        if int(ref_fit.sum()) < 100:
            ref_cut = _safe_quantile(redshift.loc[fit_mask], 0.02, float(redshift.min()))
            ref_fit = fit_mask & (redshift <= ref_cut)
        ref_df = pd.DataFrame({"m": band_values[band], "key": ref_key, "fit": ref_fit})
        global_ref = float(ref_df.loc[ref_df["fit"], "m"].median()) if int(ref_df["fit"].sum()) else float(band_values[band].median())
        ref_map = ref_df.loc[ref_df["fit"]].groupby("key", sort=False)["m"].median()
        ref_median = ref_key.map(ref_map).fillna(global_ref).astype(float)

        response = band_values[band] - ref_median
        fit_df = pd.DataFrame(
            {
                "z": zc,
                "c": cc,
                "y": response,
                "key": effective_key,
                "fit": fit_mask,
            },
            index=index,
        )

        predictions = pd.Series(0.0, index=index, dtype=float)
        for key, pos in fit_df.groupby("key", sort=False).groups.items():
            pos_index = pd.Index(pos)
            train_pos = pos_index[fit_mask.reindex(pos_index).to_numpy(dtype=bool)]
            if len(train_pos) < 20:
                continue
            beta = _robust_surface_fit(
                fit_df.loc[train_pos, "z"].to_numpy(dtype=float),
                fit_df.loc[train_pos, "c"].to_numpy(dtype=float),
                fit_df.loc[train_pos, "y"].to_numpy(dtype=float),
            )
            predictions.loc[pos_index] = _surface_predict(
                beta,
                fit_df.loc[pos_index, "z"].to_numpy(dtype=float),
                fit_df.loc[pos_index, "c"].to_numpy(dtype=float),
            )

        corrected_band = band_values[band] - predictions
        corrected[band] = corrected_band
        features[f"{band}_k_delta"] = predictions.astype(float)
        features[f"{band}_rest_mag"] = corrected_band.astype(float)
        features[f"{band}_{color_name}_clip_low"] = (color < c_low).astype(np.int8)
        features[f"{band}_{color_name}_clip_high"] = (color > c_high).astype(np.int8)

    corrected_colors = {
        "d_ug": corrected["u"] - corrected["g"],
        "d_gr": corrected["g"] - corrected["r"],
        "d_ri": corrected["r"] - corrected["i"],
        "d_iz": corrected["i"] - corrected["z"],
    }
    curvatures = {
        "curv_ugr": corrected_colors["d_ug"] - corrected_colors["d_gr"],
        "curv_gri": corrected_colors["d_gr"] - corrected_colors["d_ri"],
        "curv_riz": corrected_colors["d_ri"] - corrected_colors["d_iz"],
    }

    residual_inputs = {}
    for band in BANDS:
        residual_inputs[f"{band}_rest"] = corrected[band]
    residual_inputs.update(corrected_colors)
    residual_inputs.update(curvatures)

    z_bin, z_bin_count = _rank_bins(zc, fit_mask, MAX_Z_BINS, TARGET_CELL_ROWS)
    features["z_quantile_bin"] = z_bin.astype(np.int16)
    features["z_quantile_bin_count"] = np.int16(z_bin_count)

    low_support_any = pd.Series(False, index=index, dtype=bool)
    min_support = pd.Series(np.iinfo(np.int16).max, index=index, dtype=np.int16)

    for name, values in residual_inputs.items():
        if name in ("u_rest", "d_ug", "curv_ugr"):
            color_for_bins = colors["ug"]
        elif name in ("g_rest", "d_gr", "curv_gri"):
            color_for_bins = colors["gr"]
        elif name in ("r_rest", "d_ri", "curv_riz"):
            color_for_bins = colors["ri"]
        else:
            color_for_bins = colors["iz"]

        c_bin, c_bin_count = _rank_bins(color_for_bins, fit_mask, MAX_COLOR_BINS, TARGET_CELL_ROWS)
        cell_key = effective_key + "|zb" + z_bin.astype(str) + "|cb" + c_bin.astype(str)
        base_key = effective_key
        med, mad, low_support, counts = _cell_stats(values, cell_key, base_key, fit_mask)

        resid = pd.Series(values, index=index, dtype=float) - med
        features[f"{name}_resid"] = resid.astype(float)
        features[f"{name}_scaled_resid"] = (resid / (mad + EPS)).replace([np.inf, -np.inf], 0.0).fillna(0.0).astype(float)
        features[f"{name}_low_support"] = low_support.astype(np.int8)
        features[f"{name}_cell_support"] = counts.astype(np.int16)
        features[f"{name}_color_bin_count"] = np.int16(c_bin_count)

        low_support_any = low_support_any | low_support
        min_support = pd.Series(np.minimum(min_support.to_numpy(dtype=np.int16), counts.to_numpy(dtype=np.int16)), index=index)

    features["any_low_support_cell"] = low_support_any.astype(np.int8)
    features["min_cell_support"] = min_support.replace(np.iinfo(np.int16).max, 0).astype(np.int16)
    features = features.replace([np.inf, -np.inf], np.nan).fillna(0)

    return features


FEATURE_GROUPS = [
    {
        "name": "k_correction_residual_manifold",
        "fn": add_k_correction_residual_manifold,
        "depends_on": [],
        "description": "Train-fitted redshift and color conditioned pseudo-rest-frame photometric residual manifold features.",
    }
]