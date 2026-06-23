import numpy as np
import pandas as pd

_EPS = 1e-9
_BANDS = ("u", "g", "r", "i", "z")
_COLOR_TERMS = {
    "u": ("u", "g"),
    "g": ("g", "r"),
    "r": ("r", "i"),
    "i": ("i", "z"),
    "z": ("g", "r"),
}
_REGIME_BINS = (0.5, 2.0)
_REGIME_LIMITS = ((-1.0, 0.5), (0.5, 2.0), (2.0, 12.0))
_REGIME_MIN_FIT_POINTS = 240
_REGIME_RESID_SUPPORT = 24
_Z_BINS = 6
_C_BINS = 8


def _as_float_frame(df, cols):
    if not isinstance(df, pd.DataFrame):
        return pd.DataFrame(columns=cols)
    data = {}
    for col in cols:
        if col not in df.columns:
            return pd.DataFrame(columns=cols)
        data[col] = pd.to_numeric(df[col], errors="coerce")
    return pd.DataFrame(data, index=df.index)


def _assign_regime(redshift):
    z = np.asarray(redshift, dtype=float)
    regime = np.full(z.shape, -1, dtype=np.int8)
    finite = np.isfinite(z)
    if finite.any():
        regime[finite] = np.digitize(
            z[finite],
            np.array(_REGIME_BINS, dtype=float),
            right=False,
        ).astype(np.int8)
    return regime


def _compute_colors(frame):
    colors = {}
    for band, (left, right) in _COLOR_TERMS.items():
        if left not in frame.columns or right not in frame.columns:
            return {}
        colors[band] = (frame[left] - frame[right]).to_numpy(dtype=float)
    return colors


def _poly_design(z, color):
    z = np.asarray(z, dtype=float)
    c = np.asarray(color, dtype=float)
    z2 = z * z
    return np.column_stack(
        (
            z,
            z * c,
            z * c * c,
            z2,
            z2 * c,
            z2 * c * c,
        )
    )


def _poly_predict(coeffs, z, color):
    return _poly_design(z, color) @ np.asarray(coeffs, dtype=float)


def _mad(values):
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.nan
    med = np.nanmedian(arr)
    return float(np.nanmedian(np.abs(arr - med)))


def _fit_coefficients(z, color, y, reference, min_points):
    z = np.asarray(z, dtype=float)
    color = np.asarray(color, dtype=float)
    y = np.asarray(y, dtype=float)
    valid = np.isfinite(z) & np.isfinite(color) & np.isfinite(y)

    if valid.sum() < 6:
        if reference is None:
            return np.zeros(6, dtype=float)
        return np.asarray(reference, dtype=float).copy()

    z = z[valid]
    color = color[valid]
    y = y[valid]
    n = z.size

    if n >= 80:
        lo, hi = np.nanpercentile(y, [2.0, 98.0])
    else:
        lo, hi = np.nanmin(y), np.nanmax(y)
    y = np.clip(y, lo, hi)

    X = _poly_design(z, color)
    coef = None

    try:
        from sklearn.linear_model import HuberRegressor
    except Exception:
        HuberRegressor = None

    if HuberRegressor is not None:
        try:
            model = HuberRegressor(
                fit_intercept=False,
                epsilon=1.35,
                alpha=1e-3,
                max_iter=300,
            )
            model.fit(X, y)
            coef = np.asarray(model.coef_, dtype=float)
        except Exception:
            coef = None

    if coef is None:
        try:
            ridge = 1e-4
            XtX = X.T @ X + ridge * np.eye(X.shape[1])
            Xty = X.T @ y
            coef = np.linalg.lstsq(XtX, Xty, rcond=None)[0]
        except Exception:
            if reference is None:
                coef = np.zeros(6, dtype=float)
            else:
                coef = np.asarray(reference, dtype=float).copy()

    if not np.all(np.isfinite(coef)):
        if reference is None:
            coef = np.zeros(6, dtype=float)
        else:
            coef = np.asarray(reference, dtype=float).copy()

    if reference is not None:
        blend = np.clip(n / float(min_points), 0.15, 1.0)
        coef = blend * coef + (1.0 - blend) * np.asarray(reference, dtype=float)
        span = np.maximum(np.abs(reference) * 6.0, 0.05)
        coef = np.clip(coef, np.asarray(reference) - span, np.asarray(reference) + span)

    return coef


def _safe_edges(values, bins):
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size < 2:
        return None

    edges = np.quantile(arr, np.linspace(0.0, 1.0, bins + 1))
    edges = np.unique(np.asarray(edges, dtype=float))
    if edges.size < bins + 1:
        if edges.size == 0:
            return None
        if edges.size == 1:
            return np.array([edges[0] - 1.0, edges[0] + 1.0], dtype=float)
        lo = float(edges.min())
        hi = float(edges.max())
        if lo == hi:
            return np.array([lo - 1.0, hi + 1.0], dtype=float)
        return np.linspace(lo, hi, num=bins + 1, dtype=float)

    if edges[0] == edges[-1]:
        return np.array([edges[0] - 1.0, edges[-1] + 1.0], dtype=float)

    eps = 1e-12 * (edges[-1] - edges[0])
    edges[0] -= eps
    edges[-1] += eps
    return edges


def _bin_codes(values, edges):
    values = np.asarray(values, dtype=float)
    if edges is None or len(edges) < 2:
        return np.full(values.shape, -1, dtype=np.int64)
    raw = pd.cut(values, bins=edges, include_lowest=True, labels=False)

    if hasattr(raw, "codes"):
        codes = np.asarray(raw.codes)
    elif hasattr(raw, "to_numpy"):
        codes = raw.to_numpy()
    else:
        codes = np.asarray(raw)

    codes = np.asarray(codes)
    if np.issubdtype(codes.dtype, np.floating):
        return np.where(np.isfinite(codes), codes.astype(np.int64), -1)
    return codes.astype(np.int64)


def _compute_regime_local_residuals(values, redshift, color, regime):
    n = len(values)
    resid = np.full(n, np.nan, dtype=float)
    zscore = np.full(n, np.nan, dtype=float)
    low_support = np.zeros(n, dtype=bool)

    for rid in (0, 1, 2):
        idx = np.flatnonzero(regime == rid)
        if idx.size == 0:
            continue

        idx_valid = idx[np.isfinite(redshift[idx]) & np.isfinite(color[idx]) & np.isfinite(values[idx])]
        if idx_valid.size == 0:
            continue

        v = np.asarray(values[idx_valid], dtype=float)
        z = np.asarray(redshift[idx_valid], dtype=float)
        c = np.asarray(color[idx_valid], dtype=float)

        global_med = float(np.nanmedian(v))
        global_mad = _mad(v)
        if not np.isfinite(global_mad) or global_mad <= _EPS:
            global_mad = _EPS

        if idx_valid.size <= max(_REGIME_RESID_SUPPORT, 4):
            r = v - global_med
            resid[idx_valid] = r
            zscore[idx_valid] = r / global_mad
            low_support[idx_valid] = True
            continue

        z_edges = _safe_edges(z, _Z_BINS)
        c_edges = _safe_edges(c, _C_BINS)
        z_code = _bin_codes(z, z_edges)
        c_code = _bin_codes(c, c_edges)
        valid_cells = (z_code >= 0) & (c_code >= 0)

        if not np.any(valid_cells):
            r = v - global_med
            resid[idx_valid] = r
            zscore[idx_valid] = r / global_mad
            low_support[idx_valid] = True
            continue

        n_c = len(c_edges) - 1 if c_edges is not None and len(c_edges) > 1 else 1
        combo = z_code * max(n_c, 1) + c_code

        df = pd.DataFrame({"combo": combo[valid_cells], "value": v[valid_cells]})
        stats = df.groupby("combo")["value"].agg(
            median=lambda s: float(np.nanmedian(s)),
            mad=lambda s: float(_mad(s.to_numpy())),
            support="size",
        )

        remap = stats.reindex(combo, fill_value=np.nan)
        med = remap["median"].to_numpy(dtype=float)
        mad = remap["mad"].to_numpy(dtype=float)
        supp = remap["support"].to_numpy(dtype=float)

        med = np.where(np.isfinite(med), med, global_med)
        mad = np.where(np.isfinite(mad), mad, global_mad)
        mad = np.where(mad <= _EPS, global_mad, mad)

        r = v - med
        zs = r / mad

        resid[idx_valid] = r
        zscore[idx_valid] = zs
        low_support[idx_valid] = np.where(np.isfinite(supp), supp < _REGIME_RESID_SUPPORT, True)

        invalid_cells = np.logical_not(valid_cells)
        if np.any(invalid_cells):
            inv = idx_valid[invalid_cells]
            fallback = values[inv] - global_med
            resid[inv] = fallback
            zscore[inv] = fallback / global_mad
            low_support[inv] = True

    return resid, zscore, low_support


def _build_fit_frame(raw, aux):
    base = _as_float_frame(raw, ("redshift", "u", "g", "r", "i", "z"))
    if base.empty:
        return base

    frames = [base]
    if isinstance(aux, pd.DataFrame) and not aux.empty:
        aux_frame = _as_float_frame(aux, ("redshift", "u", "g", "r", "i", "z"))
        if not aux_frame.empty:
            frames.append(aux_frame)

    fit = pd.concat(frames, ignore_index=True)

    finite = np.isfinite(fit["redshift"])
    finite &= fit["u"].between(-80.0, 40.0)
    finite &= fit["g"].between(-80.0, 40.0)
    finite &= fit["r"].between(-80.0, 40.0)
    finite &= fit["i"].between(-80.0, 40.0)
    finite &= fit["z"].between(-80.0, 40.0)
    finite &= fit["redshift"].between(-1.0, 12.0)

    return fit.loc[finite].copy()


def add_k_correction_residual_manifold(raw, deps, aux):
    raw_df = _as_float_frame(raw, ("redshift", "u", "g", "r", "i", "z"))
    if raw_df.empty:
        return pd.DataFrame(index=raw.index)

    z_raw = raw_df["redshift"].to_numpy(dtype=float)
    regime = _assign_regime(z_raw)
    colors_raw = _compute_colors(raw_df)
    if len(colors_raw) != len(_BANDS):
        return pd.DataFrame(index=raw.index)

    phot = {b: raw_df[b].to_numpy(dtype=float) for b in _BANDS}

    fit_df = _build_fit_frame(raw_df, aux)
    if fit_df.empty:
        fit_df = raw_df.copy()

    fit_z = fit_df["redshift"].to_numpy(dtype=float)
    fit_colors = _compute_colors(fit_df)
    if len(fit_colors) != len(_BANDS):
        return pd.DataFrame(index=raw.index)
    fit_regime = _assign_regime(fit_z)

    band_models = {}
    band_hulls = {}

    for band in _BANDS:
        c_fit = fit_colors.get(band, np.zeros_like(fit_z, dtype=float))
        y_fit = fit_df[band].to_numpy(dtype=float)
        global_coef = _fit_coefficients(fit_z, c_fit, y_fit, reference=None, min_points=_REGIME_MIN_FIT_POINTS)

        model_by_regime = {}
        hull_by_regime = {}

        for rid in (0, 1, 2):
            idx = np.flatnonzero(fit_regime == rid)
            lower, upper = _REGIME_LIMITS[rid]
            if idx.size >= 12:
                z_sub = fit_z[idx]
                c_sub = c_fit[idx]
                model_by_regime[rid] = _fit_coefficients(
                    z_sub,
                    c_sub,
                    y_fit[idx],
                    reference=global_coef,
                    min_points=_REGIME_MIN_FIT_POINTS,
                )
                z_lo = float(np.nanpercentile(z_sub, 1.0))
                z_hi = float(np.nanpercentile(z_sub, 99.0))
                c_lo = float(np.nanpercentile(c_sub, 1.0))
                c_hi = float(np.nanpercentile(c_sub, 99.0))
                if np.isfinite(lower):
                    z_lo = max(z_lo, lower)
                if np.isfinite(upper):
                    z_hi = min(z_hi, upper)
                if z_lo >= z_hi:
                    z_lo, z_hi = float(np.nanmin(z_sub)), float(np.nanmax(z_sub))
                    if z_lo == z_hi:
                        z_hi = z_lo + 1.0
            else:
                model_by_regime[rid] = global_coef.copy()
                c_all = fit_colors[band]
                z_min = float(np.nanmin(fit_z)) if np.isfinite(fit_z).any() else -1.0
                z_max = float(np.nanmax(fit_z)) if np.isfinite(fit_z).any() else 12.0
                c_lo = float(np.nanpercentile(c_all, 1.0)) if np.isfinite(c_all).any() else -10.0
                c_hi = float(np.nanpercentile(c_all, 99.0)) if np.isfinite(c_all).any() else 10.0
                z_lo = lower if lower <= upper else z_min
                z_hi = upper if upper >= lower else z_max

            if not np.isfinite(z_lo):
                z_lo = lower
            if not np.isfinite(z_hi):
                z_hi = upper
            if not np.isfinite(c_lo):
                c_lo = -10.0
            if not np.isfinite(c_hi):
                c_hi = 10.0
            if z_lo >= z_hi:
                z_hi = z_lo + 1.0
            if c_lo >= c_hi:
                c_hi = c_lo + 1.0

            hull_by_regime[rid] = (z_lo, z_hi, c_lo, c_hi)

        band_models[band] = model_by_regime
        band_hulls[band] = hull_by_regime

    out = pd.DataFrame(index=raw.index)
    out["k_corr_regime"] = pd.Series(regime, index=raw.index).astype("Int8")

    corrected_mags = {}

    for band in _BANDS:
        m = phot[band]
        c = colors_raw[band]
        kcorr = np.full(len(raw), np.nan, dtype=float)
        inside_hull = np.zeros(len(raw), dtype=bool)
        clipped_by_edge = np.zeros(len(raw), dtype=bool)

        for rid in (0, 1, 2):
            idx = np.flatnonzero(regime == rid)
            if idx.size == 0:
                continue

            idx_sub = idx[np.isfinite(z_raw[idx]) & np.isfinite(c[idx]) & np.isfinite(m[idx])]
            if idx_sub.size == 0:
                continue

            z_vals = z_raw[idx_sub]
            c_vals = c[idx_sub]
            m_vals = m[idx_sub]

            z_lo, z_hi, c_lo, c_hi = band_hulls[band][rid]
            z_clip = np.clip(z_vals, z_lo, z_hi)
            c_clip = np.clip(c_vals, c_lo, c_hi)
            clipped = (z_vals != z_clip) | (c_vals != c_clip)

            pred = _poly_predict(band_models[band][rid], z_clip, c_clip)
            corrected_vals = m_vals - pred

            kcorr[idx_sub] = corrected_vals
            inside_hull[idx_sub] = ~clipped
            clipped_by_edge[idx_sub] = clipped

        resid, zscore, low_support = _compute_regime_local_residuals(kcorr, z_raw, c, regime)
        corrected_mags[band] = kcorr

        out[f"{band}_kcorr"] = kcorr
        out[f"{band}_kcorr_resid"] = resid
        out[f"{band}_kcorr_zscore"] = zscore
        out[f"{band}_kcorr_inside_fit_hull"] = inside_hull
        out[f"{band}_kcorr_clipped_by_edge"] = clipped_by_edge
        out[f"{band}_kcorr_low_support_bin"] = low_support

    slope_pairs = (("u", "g"), ("g", "r"), ("r", "i"), ("i", "z"))
    slope_arrays = []

    for left, right in slope_pairs:
        slope = corrected_mags[left] - corrected_mags[right]
        base_color = colors_raw[left]
        slope_resid, slope_z, slope_low = _compute_regime_local_residuals(slope, z_raw, base_color, regime)

        out[f"kcorr_{left}_{right}_slope"] = slope
        out[f"kcorr_{left}_{right}_slope_resid"] = slope_resid
        out[f"kcorr_{left}_{right}_slope_zscore"] = slope_z
        out[f"kcorr_{left}_{right}_low_support_bin"] = slope_low
        slope_arrays.append(slope_resid)

    slope_matrix = np.column_stack(slope_arrays)
    out["kcorr_slope_dispersion"] = np.nanstd(slope_matrix, axis=1)
    out["kcorr_slope_roughness"] = (
        np.abs(slope_arrays[0] - slope_arrays[1])
        + np.abs(slope_arrays[1] - slope_arrays[2])
        + np.abs(slope_arrays[2] - slope_arrays[3])
    )

    return out


FEATURE_GROUPS = [
    {
        "name": "k_correction_residual_manifold",
        "fn": add_k_correction_residual_manifold,
        "depends_on": [],
        "description": "Fits redshift- and color-dependent K-correction residual surfaces per regime and emits regime-hardened corrected-magnitude residual diagnostics.",
    }
]