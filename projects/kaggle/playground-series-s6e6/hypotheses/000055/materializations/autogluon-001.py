import numpy as np
import pandas as pd

_BANDS = ("u", "g", "r", "i", "z")
_BAND_SOFTENING = (
    ("u", 1.4e-10),
    ("g", 0.9e-10),
    ("r", 1.2e-10),
    ("i", 1.8e-10),
    ("z", 7.4e-10),
)
_BAND_ORDER = (-2.0, -1.0, 0.0, 1.0, 2.0)
_LOG10_OVER_2P5 = 0.9210340371976183
_LOW_CONFIDENCE_THRESHOLD = 0.5
_MAG_VALID_MIN = -120.0
_MAG_VALID_MAX = 400.0
_REDSHIFT_BINS = 11


def _as_float_matrix(frame, columns):
    arrays = []
    for col in columns:
        arrays.append(
            pd.to_numeric(frame[col], errors="coerce").to_numpy(dtype=np.float64, copy=False)
        )
    if len(arrays) == 0:
        return np.empty((len(frame), 0), dtype=np.float64)
    return np.column_stack(arrays)


def _weighted_contrast(q, w, left, right):
    qa = q[:, left]
    qb = q[:, right]
    wa = w[:, left]
    wb = w[:, right]
    denom = wa + wb
    scale = np.zeros_like(qa, dtype=np.float64)
    valid = denom > 0.0
    scale[valid] = np.sqrt((wa[valid] * wb[valid]) / denom[valid])
    return (qa - qb) * scale


def _compute_base_features(frame):
    n = len(frame)
    mags = _as_float_matrix(frame, _BANDS)

    soften_map = {band: coeff for band, coeff in _BAND_SOFTENING}
    softening = np.array([soften_map[band] for band in _BANDS], dtype=np.float64)

    valid = np.isfinite(mags)
    valid &= mags >= _MAG_VALID_MIN
    valid &= mags <= _MAG_VALID_MAX
    safe_mags = np.where(valid, mags, 0.0)

    arg = -_LOG10_OVER_2P5 * safe_mags - np.log(softening)
    x = np.sinh(arg)
    q = 2.0 * softening * x
    w = np.abs(x) / (1.0 + np.abs(x))

    q = np.where(valid, q, 0.0)
    w = np.where(valid, w, 0.0)
    q = np.where(np.isfinite(q), q, 0.0)
    w = np.where(np.isfinite(w), w, 0.0)

    n_eff = w.sum(axis=1)
    has_weight = n_eff > 0.0
    mu = np.zeros(n, dtype=np.float64)
    mu[has_weight] = (w[has_weight] * q[has_weight]).sum(axis=1) / n_eff[has_weight]

    t = np.array(_BAND_ORDER, dtype=np.float64)
    t2 = t * t
    t3 = t2 * t
    t4 = t2 * t2

    resid = q - mu[:, None]

    A11 = (w * t2).sum(axis=1)
    A12 = (w * t3).sum(axis=1)
    A22 = (w * t4).sum(axis=1)
    det = A11 * A22 - A12 * A12

    b1 = (w * resid * t).sum(axis=1)
    b2 = (w * resid * t2).sum(axis=1)

    slope = np.full(n, np.nan, dtype=np.float64)
    curvature = np.full(n, np.nan, dtype=np.float64)
    fit_ok = has_weight & (np.abs(det) > 1e-12)
    slope[fit_ok] = (A22[fit_ok] * b1[fit_ok] - A12[fit_ok] * b2[fit_ok]) / det[fit_ok]
    curvature[fit_ok] = (A11[fit_ok] * b2[fit_ok] - A12[fit_ok] * b1[fit_ok]) / det[fit_ok]

    contrast_ug = _weighted_contrast(q, w, 0, 1)
    contrast_gr = _weighted_contrast(q, w, 1, 2)
    contrast_ri = _weighted_contrast(q, w, 2, 3)
    contrast_iz = _weighted_contrast(q, w, 3, 4)

    frac_reliable = (w > 0.6).sum(axis=1) / len(_BANDS)
    weak_band = w.min(axis=1)
    reliable_band_count = (w > 0.2).sum(axis=1).astype(np.int8)

    return pd.DataFrame(
        {
            "q_u": q[:, 0],
            "q_g": q[:, 1],
            "q_r": q[:, 2],
            "q_i": q[:, 3],
            "q_z": q[:, 4],
            "w_u": w[:, 0],
            "w_g": w[:, 1],
            "w_r": w[:, 2],
            "w_i": w[:, 3],
            "w_z": w[:, 4],
            "mu_flux": mu,
            "shape_slope": slope,
            "shape_curvature": curvature,
            "contrast_ug": contrast_ug,
            "contrast_gr": contrast_gr,
            "contrast_ri": contrast_ri,
            "contrast_iz": contrast_iz,
            "n_eff": n_eff,
            "frac_reliable": frac_reliable,
            "weak_band": weak_band,
            "reliable_band_count": reliable_band_count,
        },
        index=frame.index,
    )


def _combine_aux_frame(raw, aux):
    required = _BANDS + ("redshift",)
    if isinstance(aux, pd.DataFrame) and (not aux.empty) and all(col in aux.columns for col in required):
        return pd.concat([raw.loc[:, required], aux.loc[:, required]], axis=0)
    return raw.loc[:, required]


def _build_redshift_edges(redshift_values):
    z = pd.to_numeric(redshift_values, errors="coerce").to_numpy(dtype=np.float64, copy=False)
    finite = np.isfinite(z)

    if not finite.any():
        return np.array([0.0, 1.0], dtype=np.float64)

    zmin = float(np.nanmin(z[finite]))
    zmax = float(np.nanmax(z[finite]))

    if zmin == zmax:
        span = 1.0
        if abs(zmax) > 0.0:
            span = max(abs(zmax) * 0.05, 1e-6)
        return np.array([zmin - span, zmax + span], dtype=np.float64)

    return np.linspace(zmin, zmax, num=_REDSHIFT_BINS, dtype=np.float64)


def _impute_low_confidence_features(features, raw, aux):
    low_confidence = features["n_eff"] <= _LOW_CONFIDENCE_THRESHOLD
    features["low_confidence"] = low_confidence.astype(np.uint8)

    if not low_confidence.any():
        return features

    source = _combine_aux_frame(raw, aux)
    source_features = _compute_base_features(source)

    edges = _build_redshift_edges(source["redshift"])
    raw_z = pd.to_numeric(raw["redshift"], errors="coerce")
    source_z = pd.to_numeric(source["redshift"], errors="coerce")

    raw_bins = pd.cut(raw_z, bins=edges, include_lowest=True, duplicates="drop")
    source_bins = pd.cut(source_z, bins=edges, include_lowest=True, duplicates="drop")

    impute_cols = (
        "mu_flux",
        "shape_slope",
        "shape_curvature",
        "contrast_ug",
        "contrast_gr",
        "contrast_ri",
        "contrast_iz",
    )
    pool = source_features.loc[:, impute_cols].copy()
    pool["redshift_bin"] = source_bins

    grouped_medians = pool.groupby("redshift_bin", dropna=True)[list(impute_cols)].median(numeric_only=True)
    global_medians = pool[impute_cols].median(numeric_only=True)

    for col in impute_cols:
        fallback = raw_bins.map(grouped_medians[col])
        fallback = fallback.fillna(global_medians.get(col, 0.0)).fillna(0.0)
        features.loc[low_confidence, col] = fallback.loc[low_confidence].to_numpy(dtype=np.float64)

    return features


def add_asinh_jacobian_weighted_shape(raw, deps, aux):
    base = _compute_base_features(raw)
    return _impute_low_confidence_features(base, raw, aux)


FEATURE_GROUPS = [
    {
        "name": "asinh_jacobian_weighted_shape",
        "fn": add_asinh_jacobian_weighted_shape,
        "depends_on": [],
        "description": "Builds weighted asinh-flux shape descriptors from u,g,r,i,z magnitudes with low-SNR suppression and redshift-bin fallback imputation.",
    },
]