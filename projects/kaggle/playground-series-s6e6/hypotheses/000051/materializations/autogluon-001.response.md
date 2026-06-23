import numpy as np
import pandas as pd

REGIME_BOUNDS = ((0.0, 0.5), (0.5, 2.0), (2.0, np.inf))
BANDS = ("u", "g", "r", "i", "z")
COLOR_KEYS = ("c_u", "c_g", "c_r", "c_i", "c_z")
HUBER_K = 1.345
HUBER_ITERS = 12
REGIME_HULL_BINS = 16
MIN_BIN_SUPPORT = 24
AUX_FIT_WEIGHT = 0.35
MAG_LOW_SENTINEL = -1.0e4
MAG_HIGH_SENTINEL = 1.0e4


def _coerce_numeric(values):
    return pd.to_numeric(values, errors="coerce").astype(float)


def _sanitize_magnitude(values):
    arr = np.asarray(values, dtype=float)
    bad = ~np.isfinite(arr)
    bad |= arr <= MAG_LOW_SENTINEL
    bad |= arr >= MAG_HIGH_SENTINEL
    if np.any(bad):
        arr = arr.copy()
        arr[bad] = np.nan
    return arr


def _assign_regime(redshift):
    z = np.asarray(redshift, dtype=float)
    regime = np.full(len(z), -1, dtype=np.int8)
    valid = np.isfinite(z)
    regime[(z >= 0.0) & (z < 0.5) & valid] = 0
    regime[(z >= 0.5) & (z < 2.0) & valid] = 1
    regime[(z >= 2.0) & valid] = 2
    return regime


def _build_design_matrix(z, color):
    z = np.asarray(z, dtype=float)
    color = np.asarray(color, dtype=float)
    z2 = z * z
    c = color
    c2 = c * c
    return np.column_stack(
        (
            z,
            z2,
            z * c,
            z2 * c,
            z * c2,
            z2 * c2,
        )
    )


def _clip_coefficients(beta, scale):
    beta = np.where(np.isfinite(beta), beta, 0.0)
    absb = np.abs(beta)
    med = np.nanmedian(absb) if absb.size else 0.0
    mad = np.nanmedian(np.abs(absb - med)) if absb.size else 0.0
    base = scale if np.isfinite(scale) and scale > 0 else 1.0
    limit = max(base * 8.0, med + 6.0 * (mad if mad > 0 else 1.0), 1e-3)
    return np.clip(beta, -limit, limit)


def _fit_huber_regression(z, color, target, base_weight):
    z = np.asarray(z, dtype=float)
    color = np.asarray(color, dtype=float)
    target = np.asarray(target, dtype=float)
    base_weight = np.asarray(base_weight, dtype=float)

    valid = np.isfinite(z) & np.isfinite(color) & np.isfinite(target) & np.isfinite(base_weight)
    if np.count_nonzero(valid) < 8:
        return np.zeros(6, dtype=float)

    z = z[valid]
    color = color[valid]
    target = target[valid]
    base_weight = np.clip(base_weight[valid], 0.0, 1.0)

    X = _build_design_matrix(z, color)
    weights = base_weight.copy()
    beta = np.zeros(6, dtype=float)
    scale = 1.0

    for _ in range(HUBER_ITERS):
        sw = np.sqrt(np.clip(weights, 1e-12, None))
        Xw = X * sw[:, None]
        yw = target * sw
        XtX = Xw.T @ Xw + np.eye(6) * 1e-8
        beta_new = np.linalg.solve(XtX, Xw.T @ yw)

        pred = X @ beta_new
        resid = target - pred
        mad = np.nanmedian(np.abs(resid))
        s = mad * 1.4826 if mad > 0 else np.nanstd(resid)
        if not np.isfinite(s) or s <= 0:
            beta = beta_new
            break

        u = resid / (s + 1e-12)
        rw = HUBER_K / (np.abs(u) + 1e-12)
        rw = np.clip(rw, 0.0, 1.0)
        weights = base_weight * rw
        beta = beta_new
        scale = s

        if np.max(np.abs(rw - 1.0)) < 1e-4:
            break

    return _clip_coefficients(beta, scale)


def _build_bin_edges(values, n_bins):
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return np.array([0.0, 1.0])

    if v.size < 6:
        lo = np.min(v)
        hi = np.max(v)
        if lo == hi:
            return np.array([lo - 1.0, lo, lo + 1.0])
        return np.array([lo, (lo + hi) / 2.0, hi])

    probs = np.linspace(0.0, 1.0, min(n_bins, max(3, int(np.sqrt(v.size)))) + 1)
    edges = np.quantile(v, probs)
    edges = np.unique(edges.astype(float))
    if edges.size < 3:
        lo = np.min(v)
        hi = np.max(v)
        if lo == hi:
            return np.array([lo - 1.0, lo, lo + 1.0])
        return np.array([lo, (lo + hi) / 2.0, hi])
    return edges


def _bin_indices(values, edges):
    values = np.asarray(values, dtype=float)
    edges = np.asarray(edges, dtype=float)
    if edges.size < 2:
        return np.full(values.shape, -1, dtype=np.int32)

    idx = np.searchsorted(edges, values, side="right") - 1
    idx[~np.isfinite(values)] = -1
    idx[idx < 0] = -1
    idx[idx > (edges.size - 2)] = -1
    return idx.astype(np.int32)


def _bin_support_maps(z_bin, c_bin, value):
    z_bin = np.asarray(z_bin, dtype=np.int64)
    c_bin = np.asarray(c_bin, dtype=np.int64)
    value = np.asarray(value, dtype=float)
    n = len(value)

    index = pd.MultiIndex.from_arrays([z_bin, c_bin], names=("z_bin", "c_bin"))
    valid = (z_bin >= 0) & (c_bin >= 0) & np.isfinite(value)

    if not np.any(valid):
        return np.zeros(n, dtype=float), np.full(n, np.nan), np.full(n, np.nan)

    idx_valid = pd.MultiIndex.from_arrays([z_bin[valid], c_bin[valid]])
    vals = pd.Series(value[valid], index=idx_valid)

    counts = vals.groupby(level=[0, 1]).size()
    med = vals.groupby(level=[0, 1]).median()
    mad = vals.groupby(level=[0, 1]).apply(lambda s: np.nanmedian(np.abs(s - np.nanmedian(s)) + 1e-12)

    support = counts.reindex(index).fillna(0).to_numpy(dtype=float)
    med_map = med.reindex(index).to_numpy(dtype=float)
    mad_map = mad.reindex(index).to_numpy(dtype=float)

    return support, med_map, mad_map


def _mad_series(values):
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return np.nan
    med = np.nanmedian(v)
    return np.nanmedian(np.abs(v - med))


def _extract_aux(aux):
    if not isinstance(aux, pd.DataFrame):
        return None
    required = ("u", "g", "r", "i", "z", "redshift")
    if not set(required).issubset(aux.columns):
        return None

    payload = {}
    for col in required:
        payload[col] = _sanitize_magnitude(_coerce_numeric(aux[col]).to_numpy())

    z = payload["redshift"]
    if not np.isfinite(z).any():
        return None

    z = z.copy()
    z[np.isfinite(z) & (z < 0.0)] = 0.0
    payload["redshift"] = z

    payload["c_u"] = _sanitize_magnitude(payload["u"] - payload["g"])
    payload["c_g"] = _sanitize_magnitude(payload["g"] - payload["r"])
    payload["c_r"] = _sanitize_magnitude(payload["r"] - payload["i"])
    payload["c_i"] = _sanitize_magnitude(payload["i"] - payload["z"])
    payload["c_z"] = _sanitize_magnitude(payload["g"] - payload["r"])
    payload["regime"] = _assign_regime(payload["redshift"])
    return payload


def add_k_correction_residual_manifold(raw, deps, aux):
    idx = raw.index
    n = len(raw)

    z = _sanitize_magnitude(_coerce_numeric(raw["redshift"]).to_numpy())
    z[np.isfinite(z) & (z < 0.0)] = 0.0

    mags = {band: _sanitize_magnitude(_coerce_numeric(raw[band]).to_numpy()) for band in BANDS}
    colors = {
        "u": _sanitize_magnitude(mags["u"] - mags["g"]),
        "g": _sanitize_magnitude(mags["g"] - mags["r"]),
        "r": _sanitize_magnitude(mags["r"] - mags["i"]),
        "i": _sanitize_magnitude(mags["i"] - mags["z"]),
        "z": _sanitize_magnitude(mags["g"] - mags["r"]),
    }

    aux_payload = _extract_aux(aux)
    regime = _assign_regime(z)
    features = {"kcor_regime": regime.astype(np.int8)}

    rest_mag = {}
    resid_mag = {}

    for band, color_key in zip(BANDS, COLOR_KEYS):
        mag = mags[band]
        color = colors[band]

        finite_core = np.isfinite(mag) & np.isfinite(color) & np.isfinite(z)

        kcorr = np.full(n, np.nan)
        rest = np.full(n, np.nan)
        coeff = np.zeros((3, 6), dtype=float)

        for ridx in range(len(REGIME_BOUNDS)):
            fit_mask = finite_core & (regime == ridx)
            fit_z = z[fit_mask]
            fit_c = color[fit_mask]
            fit_y = mag[fit_mask]
            fit_w = np.ones(len(fit_z), dtype=float)

            if aux_payload is not None:
                a_mask = aux_payload["regime"] == ridx
                a_mask &= np.isfinite(aux_payload["redshift"])
                a_mask &= np.isfinite(aux_payload[band])
                a_mask &= np.isfinite(aux_payload[color_key])

                if np.any(a_mask):
                    fit_z = np.concatenate((fit_z, aux_payload["redshift"][a_mask]))
                    fit_c = np.concatenate((fit_c, aux_payload[color_key][a_mask]))
                    fit_y = np.concatenate((fit_y, aux_payload[band][a_mask]))
                    fit_w = np.concatenate((fit_w, np.full(np.count_nonzero(a_mask), AUX_FIT_WEIGHT)))

            beta = _fit_huber_regression(fit_z, fit_c, fit_y, fit_w)
            coeff[ridx] = beta

            pred_mask = finite_core & (regime == ridx)
            if np.any(pred_mask):
                xpred = _build_design_matrix(z[pred_mask], color[pred_mask])
                pred = xpred @ beta
                kcorr[pred_mask] = pred
                rest[pred_mask] = mag[pred_mask] - pred

        support_count = np.zeros(n, dtype=float)
        in_hull = np.zeros(n, dtype=bool)
        clipped_edge = np.zeros(n, dtype=bool)
        low_support = np.zeros(n, dtype=bool)
        resid = np.full(n, np.nan)
        zscore = np.full(n, np.nan)

        for ridx in range(len(REGIME_BOUNDS)):
            reg_mask = regime == ridx
            if not np.any(reg_mask):
                continue

            edge_mask = reg_mask & np.isfinite(z) & np.isfinite(color)
            if np.any(edge_mask):
                z_edges = _build_bin_edges(z[edge_mask], REGIME_HULL_BINS)
                c_edges = _build_bin_edges(color[edge_mask], REGIME_HULL_BINS)
            else:
                z_edges = np.array([0.0, 1.0], dtype=float)
                c_edges = np.array([0.0, 1.0], dtype=float)

            reg_idx = np.flatnonzero(reg_mask)
            z_codes = _bin_indices(z[reg_idx], z_edges)
            c_codes = _bin_indices(color[reg_idx], c_edges)

            support_l, med_l, mad_l = _bin_support_maps(z_codes, c_codes, rest[reg_idx])

            max_z_bin = max(0, z_edges.size - 2)
            max_c_bin = max(0, c_edges.size - 2)

            support_count[reg_idx] = support_l
            in_hull[reg_idx] = support_l > 0
            clipped_local = (z_codes >= 0) & (c_codes >= 0) & (
                (z_codes == 0)
                | (z_codes == max_z_bin)
                | (c_codes == 0)
                | (c_codes == max_c_bin)
            )
            clipped_edge[reg_idx] = clipped_local
            low_support[reg_idx] = support_l < MIN_BIN_SUPPORT

            resid_local = np.full(len(reg_idx), np.nan)
            zscore_local = np.full(len(reg_idx), np.nan)

            valid = np.isfinite(rest[reg_idx]) & np.isfinite(med_l)
            if np.any(valid):
                resid_local[valid] = rest[reg_idx][valid] - med_l[valid]
                gm = _mad_series(rest[reg_idx][np.isfinite(rest[reg_idx])])
                if not np.isfinite(gm) or gm <= 1e-12:
                    gm = 1.0
                denom = np.where(np.isfinite(mad_l) & (mad_l > 1e-12), mad_l, gm)
                zscore_local[valid] = resid_local[valid] / denom[valid]

            resid[reg_idx] = resid_local
            zscore[reg_idx] = zscore_local

        features[f"kcor_{band}_kcorr"] = kcorr
        features[f"kcor_{band}_rest"] = rest
        features[f"kcor_{band}_resid"] = resid
        features[f"kcor_{band}_zscore"] = zscore
        features[f"kcor_{band}_support_count"] = support_count
        features[f"kcor_{band}_in_fit_hull"] = in_hull
        features[f"kcor_{band}_clipped_by_edge"] = clipped_edge
        features[f"kcor_{band}_low_support"] = low_support

        rest_mag[band] = rest
        resid_mag[band] = resid

    slope_ug_rest = rest_mag["u"] - rest_mag["g"]
    slope_gr_rest = rest_mag["g"] - rest_mag["r"]
    slope_ri_rest = rest_mag["r"] - rest_mag["i"]
    slope_iz_rest = rest_mag["i"] - rest_mag["z"]

    slope_ug_resid = resid_mag["u"] - resid_mag["g"]
    slope_gr_resid = resid_mag["g"] - resid_mag["r"]
    slope_ri_resid = resid_mag["r"] - resid_mag["i"]
    slope_iz_resid = resid_mag["i"] - resid_mag["z"]

    features["kcor_rest_slope_ug"] = slope_ug_rest
    features["kcor_rest_slope_gr"] = slope_gr_rest
    features["kcor_rest_slope_ri"] = slope_ri_rest
    features["kcor_rest_slope_iz"] = slope_iz_rest

    features["kcor_resid_slope_ug"] = slope_ug_resid
    features["kcor_resid_slope_gr"] = slope_gr_resid
    features["kcor_resid_slope_ri"] = slope_ri_resid
    features["kcor_resid_slope_iz"] = slope_iz_resid

    features["kcor_rest_slope_curvature"] = np.nanmean(
        np.vstack(
            (
                np.abs(slope_ug_rest - slope_gr_rest),
                np.abs(slope_gr_rest - slope_ri_rest),
                np.abs(slope_ri_rest - slope_iz_rest),
            )
        ),
        axis=0,
    )
    features["kcor_resid_slope_curvature"] = np.nanmean(
        np.vstack(
            (
                np.abs(slope_ug_resid - slope_gr_resid),
                np.abs(slope_gr_resid - slope_ri_resid),
                np.abs(slope_ri_resid - slope_iz_resid),
            )
        ),
        axis=0,
    )

    return pd.DataFrame(features, index=idx)


FEATURE_GROUPS = [
    {
        "name": "k_correction_residual_manifold",
        "fn": add_k_correction_residual_manifold,
        "depends_on": [],
        "description": "Fits regime-conditioned redshift-color K-correction surfaces, computes pseudo-rest magnitudes, local residual/z-score features, regime hull support diagnostics, and corrected-color slope diagnostics.",
    }
]