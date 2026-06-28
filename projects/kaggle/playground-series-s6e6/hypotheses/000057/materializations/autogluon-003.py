import numpy as np
import pandas as pd


MAG_COLS = ("u", "g", "r", "i", "z")
COLOR_BOUNDS = (-8.0, 8.0)
MAG_Q_LOW = 0.001
MAG_Q_HIGH = 0.999
DIST_CLIP = (0.0, 30.0)
BOUNDED_CLIP = (0.0, 1.0)


def _as_float_series(raw, name):
    if name in raw.columns:
        return pd.to_numeric(raw[name], errors="coerce").astype("float64")
    return pd.Series(0.0, index=raw.index, dtype="float64")


def _finite_quantile(values, q, default):
    arr = np.asarray(values, dtype="float64")
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float(default)
    return float(np.quantile(arr, q))


def _robust_clip_series(s, low_q, high_q):
    values = s.to_numpy(dtype="float64", copy=False)
    low = _finite_quantile(values, low_q, np.nanmin(values) if np.isfinite(values).any() else 0.0)
    high = _finite_quantile(values, high_q, np.nanmax(values) if np.isfinite(values).any() else 1.0)
    if not np.isfinite(low):
        low = 0.0
    if not np.isfinite(high) or high <= low:
        high = low + 1.0
    return s.clip(lower=low, upper=high)


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50.0, 50.0)))


def _winsorized_cov(x):
    arr = np.asarray(x, dtype="float64")
    if arr.ndim != 2 or arr.shape[0] < 4:
        return np.eye(arr.shape[1] if arr.ndim == 2 else 1, dtype="float64")
    lo = np.nanquantile(arr, 0.05, axis=0)
    hi = np.nanquantile(arr, 0.95, axis=0)
    clipped = np.clip(arr, lo, hi)
    cov = np.cov(clipped, rowvar=False)
    cov = np.asarray(cov, dtype="float64")
    if cov.ndim == 0:
        cov = np.array([[float(cov)]], dtype="float64")
    cov = np.nan_to_num(cov, nan=0.0, posinf=0.0, neginf=0.0)
    return cov


def _locus_distance(v):
    arr = np.asarray(v, dtype="float64")
    valid = np.isfinite(arr).all(axis=1)
    fit = arr[valid]
    if fit.shape[0] < 8:
        fit = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

    mu = np.nanmedian(fit, axis=0)
    centered = fit - mu
    cov_main = _winsorized_cov(centered)

    try:
        eigvals, eigvecs = np.linalg.eigh(cov_main)
        p = eigvecs[:, int(np.argmax(eigvals))]
    except np.linalg.LinAlgError:
        p = np.zeros(arr.shape[1], dtype="float64")
        p[0] = 1.0

    p_norm = np.linalg.norm(p)
    if not np.isfinite(p_norm) or p_norm <= 0.0:
        p = np.zeros(arr.shape[1], dtype="float64")
        p[0] = 1.0
    else:
        p = p / p_norm

    centered_all = np.nan_to_num(arr - mu, nan=0.0, posinf=0.0, neginf=0.0)
    projection = np.outer(centered_all @ p, p)
    residual_all = centered_all - projection

    residual_fit = centered - np.outer(centered @ p, p)
    residual_cov = _winsorized_cov(residual_fit)
    diag_cov = np.diag(np.diag(residual_cov))
    reg_cov = (
        0.85 * residual_cov
        + 0.15 * diag_cov
        + (0.03 ** 2) * np.eye(arr.shape[1], dtype="float64")
        + 1e-6 * np.eye(arr.shape[1], dtype="float64")
    )

    try:
        inv_cov = np.linalg.pinv(reg_cov)
    except np.linalg.LinAlgError:
        inv_cov = np.eye(arr.shape[1], dtype="float64")

    dist2 = np.einsum("ij,jk,ik->i", residual_all, inv_cov, residual_all)
    dist = np.sqrt(np.maximum(dist2, 0.0))
    return np.clip(np.nan_to_num(dist, nan=0.0, posinf=DIST_CLIP[1], neginf=0.0), DIST_CLIP[0], DIST_CLIP[1])


def _robust_z_and_tail(dist):
    finite = dist[np.isfinite(dist)]
    if finite.size == 0:
        finite = np.array([0.0], dtype="float64")
    med = float(np.median(finite))
    mad = float(np.median(np.abs(finite - med)))
    scale = 1.4826 * mad + 1e-6
    p95 = float(np.quantile(finite, 0.95))
    p99 = float(np.quantile(finite, 0.99))
    z = (dist - med) / scale
    outlier = np.clip((dist - p95) / (p99 - p95 + 1e-6), 0.0, 1.0)
    return np.nan_to_num(z, nan=0.0, posinf=30.0, neginf=-30.0), np.nan_to_num(outlier, nan=0.0, posinf=1.0, neginf=0.0)


def add_quasar_stellar_locus_conflict(raw, deps, aux):
    u = _robust_clip_series(_as_float_series(raw, "u"), MAG_Q_LOW, MAG_Q_HIGH)
    g = _robust_clip_series(_as_float_series(raw, "g"), MAG_Q_LOW, MAG_Q_HIGH)
    r = _robust_clip_series(_as_float_series(raw, "r"), MAG_Q_LOW, MAG_Q_HIGH)
    i = _robust_clip_series(_as_float_series(raw, "i"), MAG_Q_LOW, MAG_Q_HIGH)
    zmag = _robust_clip_series(_as_float_series(raw, "z"), MAG_Q_LOW, MAG_Q_HIGH)
    redshift = _as_float_series(raw, "redshift").replace([np.inf, -np.inf], np.nan).fillna(0.0)

    c_ug = (u - g).clip(*COLOR_BOUNDS)
    c_gr = (g - r).clip(*COLOR_BOUNDS)
    c_ri = (r - i).clip(*COLOR_BOUNDS)
    c_iz = (i - zmag).clip(*COLOR_BOUNDS)

    v1 = np.column_stack((c_ug.to_numpy(), c_gr.to_numpy(), c_ri.to_numpy()))
    v2 = np.column_stack((c_gr.to_numpy(), c_ri.to_numpy(), c_iz.to_numpy()))

    d1 = _locus_distance(v1)
    d2 = _locus_distance(v2)
    z1, o1 = _robust_z_and_tail(d1)
    z2, o2 = _robust_z_and_tail(d2)

    c_ug_arr = c_ug.to_numpy(dtype="float64")
    c_gr_arr = c_gr.to_numpy(dtype="float64")
    c_ri_arr = c_ri.to_numpy(dtype="float64")
    i_arr = i.to_numpy(dtype="float64")
    red_arr = redshift.to_numpy(dtype="float64")

    blue_uv = np.clip(
        _sigmoid((-c_ug_arr - 0.05) / 0.20)
        * _sigmoid((0.45 - c_gr_arr) / 0.18)
        * _sigmoid((20.5 - i_arr) / 0.9),
        0.0,
        1.0,
    )
    mid_bridge = np.clip(
        np.exp(-0.5 * (((c_ug_arr - 0.90) ** 2) / (0.22 ** 2) + ((c_gr_arr - 0.18) ** 2) / (0.14 ** 2)))
        * _sigmoid((red_arr - 0.4) / 0.35)
        * _sigmoid((1.9 - red_arr) / 0.7),
        0.0,
        1.0,
    )
    high_z = np.clip(
        np.exp(-0.5 * (((c_ug_arr - 1.60) ** 2) / (0.35 ** 2) + ((c_ri_arr - 0.25) ** 2) / (0.25 ** 2)))
        * _sigmoid((red_arr - 2.1) / 0.4)
        * _sigmoid((4.2 - red_arr) / 1.1)
        * _sigmoid((21.0 - i_arr) / 1.0),
        0.0,
        1.0,
    )

    q_af = np.exp(
        -0.5
        * (
            ((c_ug_arr - 0.55) ** 2) / (0.14 ** 2)
            + ((c_gr_arr - 0.10) ** 2) / (0.09 ** 2)
            + ((c_ri_arr - 0.05) ** 2) / (0.09 ** 2)
        )
    )
    q_wd = np.exp(
        -0.5
        * (
            ((c_ug_arr - 1.05) ** 2) / (0.18 ** 2)
            + ((c_gr_arr - 0.30) ** 2) / (0.12 ** 2)
            + ((c_ri_arr - 0.12) ** 2) / (0.11 ** 2)
        )
    )
    mimic_penalty = np.clip(
        (q_af + q_wd)
        * _sigmoid((0.35 - np.abs(red_arr - 2.7)) / 0.35)
        * _sigmoid((20.2 - i_arr) / 0.7),
        0.0,
        1.0,
    )

    geom_signal = np.clip(0.42 * o1 + 0.38 * o2 + 0.10 * blue_uv + 0.10 * high_z, 0.0, 1.0)
    quasar_conflict = np.clip(geom_signal + 0.20 * mid_bridge - 0.55 * mimic_penalty, -1.0, 1.0)

    data = {
        "d1": d1,
        "d2": d2,
        "z1": z1,
        "z2": z2,
        "o1": o1,
        "o2": o2,
        "blue_uv": blue_uv,
        "mid_bridge": mid_bridge,
        "high_z": high_z,
        "mimic_penalty": mimic_penalty,
        "geom_signal": geom_signal,
        "quasar_conflict": quasar_conflict,
    }
    out = pd.DataFrame(data, index=raw.index)
    return out.replace([np.inf, -np.inf], 0.0).fillna(0.0)


FEATURE_GROUPS = [
    {
        "name": "quasar_stellar_locus_conflict",
        "fn": add_quasar_stellar_locus_conflict,
        "depends_on": [],
        "description": "Robust color-locus conflict features with redshift-aware quasar corridor support and compact stellar-mimic suppression.",
    }
]