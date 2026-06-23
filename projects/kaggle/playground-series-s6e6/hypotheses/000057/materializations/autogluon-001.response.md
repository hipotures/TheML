import numpy as np
import pandas as pd

def _safe_float_array(raw, col, length):
    if not isinstance(raw, pd.DataFrame) or col not in raw.columns:
        return np.full(length, np.nan, dtype=np.float64)
    return pd.to_numeric(raw[col], errors="coerce").to_numpy(dtype=np.float64)


def _sigmoid(x):
    x = np.asarray(x, dtype=np.float64)
    x = np.clip(x, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-x))


def _gaussian(x, center, width):
    x = np.asarray(x, dtype=np.float64)
    width = np.maximum(np.asarray(width, dtype=np.float64), 1e-6)
    return np.exp(-0.5 * ((x - center) / width) ** 2.0)


def _extract_aux_color_cubes(aux):
    if not isinstance(aux, pd.DataFrame) or aux.empty:
        return np.empty((0, 3), dtype=np.float64), np.empty((0, 3), dtype=np.float64)

    required = ("u", "g", "r", "i", "z")
    if not all(col in aux.columns for col in required):
        return np.empty((0, 3), dtype=np.float64), np.empty((0, 3), dtype=np.float64)

    aux_use = aux
    if "class" in aux_use.columns:
        cls = aux_use["class"].astype(str).str.upper()
        star_like = cls.str.contains("STAR", na=False)
        if star_like.any():
            aux_use = aux_use.loc[star_like]

    mags = np.column_stack(
        [
            pd.to_numeric(aux_use["u"], errors="coerce").to_numpy(dtype=np.float64),
            pd.to_numeric(aux_use["g"], errors="coerce").to_numpy(dtype=np.float64),
            pd.to_numeric(aux_use["r"], errors="coerce").to_numpy(dtype=np.float64),
            pd.to_numeric(aux_use["i"], errors="coerce").to_numpy(dtype=np.float64),
            pd.to_numeric(aux_use["z"], errors="coerce").to_numpy(dtype=np.float64),
        ]
    )

    finite = np.isfinite(mags).all(axis=1) & (np.max(np.abs(mags), axis=1) < 200.0)
    mags = mags[finite]
    if mags.shape[0] == 0:
        return np.empty((0, 3), dtype=np.float64), np.empty((0, 3), dtype=np.float64)

    v1 = np.column_stack((mags[:, 0] - mags[:, 1], mags[:, 1] - mags[:, 2], mags[:, 2] - mags[:, 3]))
    v2 = np.column_stack((mags[:, 1] - mags[:, 2], mags[:, 2] - mags[:, 3], mags[:, 3] - mags[:, 4]))
    return v1, v2


def _fit_locus_model(vectors):
    vectors = np.asarray(vectors, dtype=np.float64)
    if vectors.size == 0:
        return np.zeros(3, dtype=np.float64), np.array([1.0, 0.0, 0.0], dtype=np.float64), np.eye(3, dtype=np.float64), np.zeros(
            3, dtype=np.float64
        )

    finite = np.isfinite(vectors).all(axis=1)
    x = vectors[finite]
    if x.shape[0] == 0:
        return np.zeros(3, dtype=np.float64), np.array([1.0, 0.0, 0.0], dtype=np.float64), np.eye(3, dtype=np.float64), np.zeros(
            3, dtype=np.float64
        )

    mean = np.nanmedian(x, axis=0)
    mean = np.where(np.isfinite(mean), mean, 0.0)

    mad = np.nanmedian(np.abs(x - mean), axis=0)
    mad = np.where(np.isfinite(mad) & (mad > 0.0), mad, 1e-3)
    clipped = np.clip(x, mean - 6.0 * mad, mean + 6.0 * mad)
    centered = clipped - mean

    if centered.shape[0] >= 2 and np.isfinite(centered).all():
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        if vt.ndim == 2 and vt.size > 0:
            axis = vt[0].copy()
        else:
            axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    else:
        axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)

    axis_norm = np.linalg.norm(axis)
    if (not np.isfinite(axis_norm)) or axis_norm <= 1e-12:
        axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        axis_norm = 1.0
    axis = axis / axis_norm

    residual = centered - np.outer(np.dot(centered, axis), axis)
    residual = residual[np.isfinite(residual).all(axis=1)]

    if residual.shape[0] >= 2:
        resid_mean = np.nanmedian(residual, axis=0)
        resid_mean = np.where(np.isfinite(resid_mean), resid_mean, 0.0)
        resid_mad = np.nanmedian(np.abs(residual - resid_mean), axis=0)
        resid_mad = np.where(np.isfinite(resid_mad) & (resid_mad > 0.0), resid_mad, 1e-3)
        residual_clipped = np.clip(residual, resid_mean - 6.0 * resid_mad, resid_mean + 6.0 * resid_mad)
        residual_centered = residual_clipped - resid_mean
        residual_cov = np.cov(residual_centered, rowvar=False, bias=True)
        if residual_cov.shape != (3, 3):
            residual_cov = np.eye(3, dtype=np.float64)
    else:
        resid_mean = np.zeros(3, dtype=np.float64)
        residual_cov = np.eye(3, dtype=np.float64) * 1e-3

    residual_cov = 0.5 * (residual_cov + residual_cov.T)
    residual_cov = np.nan_to_num(residual_cov, nan=0.0, posinf=0.0, neginf=0.0)
    residual_cov = residual_cov + np.eye(3, dtype=np.float64) * (0.03 ** 2)

    inv_cov = np.linalg.pinv(residual_cov)
    if not np.isfinite(inv_cov).all():
        inv_cov = np.linalg.pinv(np.eye(3, dtype=np.float64) + np.eye(3, dtype=np.float64) * (0.03 ** 2))

    return mean, axis, inv_cov, resid_mean


def _mahalanobis_orthogonal_distance(query_vectors, model):
    query_vectors = np.asarray(query_vectors, dtype=np.float64)
    if query_vectors.size == 0:
        return np.empty((0,), dtype=np.float64)

    mean, axis, inv_cov, resid_mean = model
    q = np.where(np.isfinite(query_vectors), query_vectors, mean)
    centered = q - mean
    projection = np.outer(np.dot(centered, axis), axis)
    residual = centered - projection - resid_mean

    qval = np.einsum("ij,ij->i", residual @ inv_cov, residual)
    qval = np.where(np.isfinite(qval), qval, 0.0)
    qval = np.maximum(qval, 0.0)
    return np.sqrt(qval)


def add_quasar_stellar_locus_conflict(raw, deps, aux):
    n = len(raw)
    idx = raw.index

    u = _safe_float_array(raw, "u", n)
    g = _safe_float_array(raw, "g", n)
    r = _safe_float_array(raw, "r", n)
    i = _safe_float_array(raw, "i", n)
    z = _safe_float_array(raw, "z", n)
    redshift = _safe_float_array(raw, "redshift", n)

    c_ug = u - g
    c_gr = g - r
    c_ri = r - i
    c_iz = i - z

    raw_v1 = np.column_stack((c_ug, c_gr, c_ri))
    raw_v2 = np.column_stack((c_gr, c_ri, c_iz))

    aux_v1, aux_v2 = _extract_aux_color_cubes(aux)

    fit_v1 = raw_v1[np.isfinite(raw_v1).all(axis=1)]
    fit_v2 = raw_v2[np.isfinite(raw_v2).all(axis=1)]
    if aux_v1.shape[0]:
        fit_v1 = np.concatenate([fit_v1, aux_v1], axis=0)
    if aux_v2.shape[0]:
        fit_v2 = np.concatenate([fit_v2, aux_v2], axis=0)

    model_v1 = _fit_locus_model(fit_v1)
    model_v2 = _fit_locus_model(fit_v2)

    d1 = _mahalanobis_orthogonal_distance(raw_v1, model_v1)
    d2 = _mahalanobis_orthogonal_distance(raw_v2, model_v2)
    d1 = np.where(np.isfinite(d1), d1, 0.0)
    d2 = np.where(np.isfinite(d2), d2, 0.0)

    s1 = np.clip(d1 - 4.0, 0.0, 1.0)
    s2 = np.clip(d2 - 4.0, 0.0, 1.0)

    z_scale = 1.0 + np.clip(np.abs(redshift), 0.0, 5.0)
    z_scale = np.where(np.isfinite(z_scale) & (z_scale > 1e-6), z_scale, 1.0)
    d1_z = np.clip(d1 / z_scale, 0.0, 1.0)
    d2_z = np.clip(d2 / z_scale, 0.0, 1.0)

    i_safe = np.where(np.isfinite(i), i, 20.2)
    red_safe = np.where(np.isfinite(redshift), redshift, 0.0)
    c_ug_safe = np.where(np.isfinite(c_ug), c_ug, 0.0)
    c_gr_safe = np.where(np.isfinite(c_gr), c_gr, 0.0)
    c_ri_safe = np.where(np.isfinite(c_ri), c_ri, 0.0)
    c_iz_safe = np.where(np.isfinite(c_iz), c_iz, 0.0)

    uv_like = np.clip((0.35 - c_ug_safe) / 0.95, 0.0, 1.0)
    uv_like = np.clip(uv_like * _sigmoid((20.2 - i_safe) / 0.55), 0.0, 1.0) * (0.5 + 0.5 * s1)

    midz_mid_ug = _gaussian(c_ug_safe, 1.05, 0.35)
    midz_mid_gr = _gaussian(c_gr_safe, 0.10, 0.10)
    midz_z = np.exp(-((red_safe - 2.65) ** 2.0) / (2.0 * 0.95 ** 2.0))
    midz_bridge = np.clip(midz_mid_ug * midz_mid_gr * midz_z * (0.6 + 0.4 * s2), 0.0, 1.0)

    highz_red = np.clip((c_ug_safe - 1.1) / 1.8, 0.0, 1.0)
    highz_outlier = np.clip(d2 / 8.0, 0.0, 1.0)
    highz_regime = np.clip(highz_red * highz_outlier * _sigmoid((20.2 - i_safe) / 0.45) * _sigmoid((red_safe - 1.5) / 0.6), 0.0, 1.0)

    wd_strip = _gaussian(c_ug_safe, 0.10, 0.30) * _gaussian(c_gr_safe, -0.08, 0.14) * _gaussian(c_ri_safe, 0.02, 0.20)
    a_star_strip = _gaussian(c_ug_safe, 1.05, 0.30) * _gaussian(c_gr_safe, 0.16, 0.11) * _gaussian(c_ri_safe, 0.05, 0.12) * _gaussian(c_iz_safe, 0.06, 0.14)
    lowz_strip_gate = _sigmoid((0.9 - red_safe) / 0.30)
    contaminant_penalty = np.clip((0.65 * wd_strip + 0.35 * a_star_strip) * lowz_strip_gate, 0.0, 1.0)

    net_support = np.clip(0.52 * s1 + 0.48 * s2 + 0.20 * uv_like + 0.20 * midz_bridge + 0.15 * highz_regime - 0.70 * contaminant_penalty, 0.0, 1.0)

    return pd.DataFrame(
        {
            "quasar_locus_outlier1": s1,
            "quasar_locus_outlier2": s2,
            "quasar_locus_d1_zscaled": d1_z,
            "quasar_locus_d2_zscaled": d2_z,
            "quasar_locus_uv_like": uv_like,
            "quasar_locus_midz_bridge": midz_bridge,
            "quasar_locus_highz_regime": highz_regime,
            "quasar_locus_contaminant_penalty": contaminant_penalty,
            "quasar_locus_net_support": net_support,
        },
        index=idx,
    )


FEATURE_GROUPS = [
    {
        "name": "quasar_stellar_locus_conflict",
        "fn": add_quasar_stellar_locus_conflict,
        "depends_on": [],
        "description": "Build redshift-aware quasar-vs-stellar manifold conflict features from orthogonalized 3D color geometry with blue/outlier, mid-z, high-z, and contaminant-suppression terms.",
    }
]