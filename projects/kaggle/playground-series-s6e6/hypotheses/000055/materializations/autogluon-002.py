import numpy as np
import pandas as pd

_BANDS = ("u", "g", "r", "i", "z")
_SOFTENING = (1.4e-10, 0.9e-10, 1.2e-10, 1.8e-10, 7.4e-10)
_T_VALUES = (-2.0, -1.0, 0.0, 1.0, 2.0)
_REDSHIFT_QUANTILES = 6
_EFF_THRESHOLD = 1e-6
_ARG_CLIP = 35.0


def _prepare_redshift_bins(redshift, n_bins):
    redshift = pd.to_numeric(redshift, errors="coerce").replace([np.inf, -np.inf], np.nan)
    if redshift.empty:
        return pd.Series([], dtype="Int64")

    fallback = float(redshift.median()) if redshift.notna().any() else 0.0
    redshift = redshift.fillna(fallback)

    if redshift.nunique(dropna=True) <= 1:
        return pd.Series(0, index=redshift.index, dtype="Int64")

    try:
        bins = pd.qcut(redshift, q=n_bins, labels=False, duplicates="drop")
    except (ValueError, TypeError):
        return pd.Series(0, index=redshift.index, dtype="Int64")

    bins = pd.Series(bins, index=redshift.index)
    return bins.astype("Int64").fillna(0)


def _pair_scale(weight_a, weight_b):
    denom = weight_a + weight_b
    return np.sqrt(np.where(denom > 0.0, (weight_a * weight_b) / denom, 0.0))


def _fit_weighted_quadratic(q, w):
    q0, q1, q2, q3, q4 = q[:, 0], q[:, 1], q[:, 2], q[:, 3], q[:, 4]
    w0, w1, w2, w3, w4 = w[:, 0], w[:, 1], w[:, 2], w[:, 3], w[:, 4]

    s0 = w0 + w1 + w2 + w3 + w4
    s1 = -2.0 * w0 - w1 + w3 + 2.0 * w4
    s2 = 4.0 * w0 + w1 + w3 + 4.0 * w4
    s3 = -8.0 * w0 - w1 + w3 + 8.0 * w4
    s4 = 16.0 * w0 + w1 + w3 + 16.0 * w4

    t0 = q0 * w0 + q1 * w1 + q2 * w2 + q3 * w3 + q4 * w4
    t1 = -2.0 * q0 * w0 - q1 * w1 + q3 * w3 + 2.0 * q4 * w4
    t2 = 4.0 * q0 * w0 + q1 * w1 + q3 * w3 + 4.0 * q4 * w4

    det = s0 * (s2 * s4 - s3 * s3) - s1 * (s1 * s4 - s2 * s3) + s2 * (s1 * s3 - s2 * s2)

    c0 = np.zeros_like(s0, dtype=np.float64)
    c1 = np.zeros_like(s0, dtype=np.float64)
    c2 = np.zeros_like(s0, dtype=np.float64)

    c0_num = t0 * (s2 * s4 - s3 * s3) - s1 * (t1 * s4 - s3 * t2) + s2 * (t1 * s3 - s2 * t2)
    c1_num = s0 * (t1 * s4 - s3 * t2) - t0 * (s1 * s4 - s2 * s3) + s2 * (s1 * t2 - t1 * s2)
    c2_num = s0 * (s2 * t2 - s3 * t1) - s1 * (s1 * t2 - s2 * t1) + t0 * (s1 * s3 - s2 * s2)

    det_good = np.abs(det) > 1e-12
    np.divide(c0_num, det, out=c0, where=det_good)
    np.divide(c1_num, det, out=c1, where=det_good)
    np.divide(c2_num, det, out=c2, where=det_good)

    fallback = np.divide(t0, s0, out=np.zeros_like(s0), where=s0 > 0.0)
    fallback_mask = ~det_good
    c0[fallback_mask] = np.where(s0[fallback_mask] > 0.0, fallback[fallback_mask], 0.0)

    t = np.array(_T_VALUES, dtype=np.float64)
    tt = t * t
    predicted = c0[:, None] + c1[:, None] * t[None, :] + c2[:, None] * tt[None, :]
    residual = q - predicted
    return c0, c1, c2, residual


def add_asinh_jacobian_weighted_shape(raw, deps, aux):
    _ = deps
    _ = aux

    index = raw.index
    n = len(raw)

    mags = raw.loc[:, list(_BANDS)].to_numpy(dtype=np.float64, copy=False)

    q = np.empty_like(mags, dtype=np.float64)
    scale = -(np.log(10.0) / 2.5)

    for bi, b0 in enumerate(_SOFTENING):
        arg = scale * mags[:, bi] - np.log(b0)
        arg = np.clip(arg, -_ARG_CLIP, _ARG_CLIP)
        q[:, bi] = np.sinh(arg) / (2.0 * b0)

    w = np.empty_like(q, dtype=np.float64)
    for bi, b0 in enumerate(_SOFTENING):
        band_abs = np.abs(q[:, bi])
        w[:, bi] = band_abs / (band_abs + (2.0 * b0))

    n_eff = np.sum(w, axis=1)
    low_confidence = (np.any(~np.isfinite(q), axis=1)) | (n_eff <= _EFF_THRESHOLD)

    if low_confidence.any():
        q_cols = ["q_u", "q_g", "q_r", "q_i", "q_z"]
        q_df = pd.DataFrame(q, columns=q_cols, index=index)

        redshift_bin = _prepare_redshift_bins(raw["redshift"], _REDSHIFT_QUANTILES)
        spectral_key = raw["spectral_type"].where(raw["spectral_type"].notna(), "~missing").astype("string")
        galaxy_key = raw["galaxy_population"].where(raw["galaxy_population"].notna(), "~missing").astype("string")

        cell_medians = (
            q_df.assign(
                redshift_bin=redshift_bin.astype("Int64"),
                spectral_type=spectral_key,
                galaxy_population=galaxy_key,
            )
            .groupby(["redshift_bin", "spectral_type", "galaxy_population"], dropna=False, observed=False)[q_cols]
            .median()
        )

        global_medians = np.nanmedian(q, axis=0)
        global_medians = np.where(np.isfinite(global_medians), global_medians, 0.0)

        key_index = pd.MultiIndex.from_arrays(
            [redshift_bin.astype("Int64"), spectral_key, galaxy_key],
            names=["redshift_bin", "spectral_type", "galaxy_population"],
        )
        fallback = np.column_stack([cell_medians[col].reindex(key_index).to_numpy(dtype=np.float64) for col in q_cols])
        fallback = np.where(np.isnan(fallback), global_medians, fallback)
        fallback = np.where(np.isfinite(fallback), fallback, global_medians)

        q[low_confidence] = fallback[low_confidence]

        for bi, b0 in enumerate(_SOFTENING):
            band_abs = np.abs(q[:, bi])
            w[:, bi] = band_abs / (band_abs + (2.0 * b0))
        n_eff = np.sum(w, axis=1)

    s0, s1, s2, residual = _fit_weighted_quadratic(q, w)

    weighted_ss = np.sum(w * residual * residual, axis=1)
    weighted_mae_num = np.sum(w * np.abs(residual), axis=1)
    weighted_residual_rms = np.sqrt(np.divide(weighted_ss, n_eff, out=np.zeros(n), where=n_eff > 0.0))
    weighted_residual_mae = np.divide(weighted_mae_num, n_eff, out=np.zeros(n), where=n_eff > 0.0)

    q_u, q_g, q_r, q_i, q_z = q[:, 0], q[:, 1], q[:, 2], q[:, 3], q[:, 4]
    w_u, w_g, w_r, w_i, w_z = w[:, 0], w[:, 1], w[:, 2], w[:, 3], w[:, 4]

    scale_ug = _pair_scale(w_u, w_g)
    scale_gr = _pair_scale(w_g, w_r)
    scale_ri = _pair_scale(w_r, w_i)
    scale_iz = _pair_scale(w_i, w_z)

    c_ug = scale_ug * (q_u - q_g)
    c_gr = scale_gr * (q_g - q_r)
    c_ri = scale_ri * (q_r - q_i)
    c_iz = scale_iz * (q_i - q_z)

    d1 = scale_ug * (q_u - 2.0 * q_g + q_r)
    d2 = scale_gr * (q_g - 2.0 * q_r + q_i)
    d3 = scale_ri * (q_r - 2.0 * q_i + q_z)

    frac_hi = np.mean(w > 0.6, axis=1)
    frac_mid = np.mean(w > 0.2, axis=1)
    min_w = np.min(w, axis=1)
    max_w = np.max(w, axis=1)

    adj_diffs = np.abs(np.diff(q, axis=1))
    pair_mask = (w[:, :-1] > 0.1) & (w[:, 1:] > 0.1)
    pair_sum = (adj_diffs * pair_mask).sum(axis=1)
    pair_count = pair_mask.sum(axis=1).astype(np.float64)
    mean_adj_gap = np.divide(pair_sum, pair_count, out=np.zeros(n), where=pair_count > 0.0)

    low_confidence = (low_confidence | ~np.isfinite(n_eff) | (n_eff <= _EFF_THRESHOLD)).astype(np.int8)

    return pd.DataFrame(
        {
            "q_u": q_u,
            "q_g": q_g,
            "q_r": q_r,
            "q_i": q_i,
            "q_z": q_z,
            "w_u": w_u,
            "w_g": w_g,
            "w_r": w_r,
            "w_i": w_i,
            "w_z": w_z,
            "n_eff": n_eff,
            "low_confidence": low_confidence,
            "s0": s0,
            "s1": s1,
            "s2": s2,
            "weighted_residual_rms": weighted_residual_rms,
            "weighted_residual_mae": weighted_residual_mae,
            "c_ug": c_ug,
            "c_gr": c_gr,
            "c_ri": c_ri,
            "c_iz": c_iz,
            "d1": d1,
            "d2": d2,
            "d3": d3,
            "frac_hi": frac_hi,
            "frac_mid": frac_mid,
            "min_w": min_w,
            "max_w": max_w,
            "mean_adj_gap": mean_adj_gap,
        },
        index=index,
    )


FEATURE_GROUPS = [
    {
        "name": "asinh_jacobian_weighted_shape",
        "fn": add_asinh_jacobian_weighted_shape,
        "depends_on": [],
        "description": "Build luptitude-derived weighted spectral-shape features with reliability-aware polynomial, contrast, and curvature descriptors.",
    }
]