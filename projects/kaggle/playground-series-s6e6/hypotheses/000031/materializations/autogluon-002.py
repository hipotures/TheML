import numpy as np
import pandas as pd

NUM_EQUAL_FREQ_BINS = 30
MIN_BIN_ROWS = 5000
MAD_SCALE = 1.4826
SCALE_FLOOR = 1e-6
CLIP_PERCENTILE = 99.5
OVERLAP_GAUSS_CENTER = 2.7
OVERLAP_GAUSS_SIGMA = 0.45
RATIO_DENOM_EPS = 1e-4


def _to_float(values):
    return np.asarray(values, dtype=np.float64)


def _robust_mad(values):
    med = np.nanmedian(values)
    mad = np.nanmedian(np.abs(values - med))
    return med, mad


def _empty_cube_param():
    return {
        "valid": False,
        "center": np.zeros(3, dtype=np.float64),
        "color_scale": np.ones(3, dtype=np.float64),
        "components": np.eye(3, dtype=np.float64),
        "scale_t": SCALE_FLOOR,
        "scale_q2": SCALE_FLOOR,
        "scale_q3": SCALE_FLOOR,
        "scale_d": SCALE_FLOOR,
        "clip_t": 0.0,
        "clip_q2": 0.0,
        "clip_q3": 0.0,
        "clip_d": 0.0,
    }


def _copy_cube_param(p):
    return {
        "valid": bool(p["valid"]),
        "center": np.array(p["center"], copy=True),
        "color_scale": np.array(p["color_scale"], copy=True),
        "components": np.array(p["components"], copy=True),
        "scale_t": float(p["scale_t"]),
        "scale_q2": float(p["scale_q2"]),
        "scale_q3": float(p["scale_q3"]),
        "scale_d": float(p["scale_d"]),
        "clip_t": float(p["clip_t"]),
        "clip_q2": float(p["clip_q2"]),
        "clip_q3": float(p["clip_q3"]),
        "clip_d": float(p["clip_d"]),
    }


def _build_equal_freq_edges(redshift_nonneg):
    z = np.asarray(redshift_nonneg, dtype=np.float64)
    z = np.maximum(z, 0.0)
    zmin = float(np.nanmin(z))
    zmax = float(np.nanmax(z))
    if not np.isfinite(zmin) or not np.isfinite(zmax):
        zmin = 0.0
        zmax = 1.0
    if zmax <= zmin:
        zmax = zmin + 1.0

    probs = np.linspace(0.0, 1.0, NUM_EQUAL_FREQ_BINS + 1)
    quant = np.quantile(z, probs, interpolation="linear")
    edges = [float(quant[0])]
    for v in quant[1:]:
        if float(v) > edges[-1] + 1e-12:
            edges.append(float(v))

    if len(edges) < 2:
        edges = [zmin, zmax]

    edges = np.array(edges, dtype=np.float64)
    return _merge_small_bins(z, edges)


def _merge_small_bins(z, edges):
    edges = np.array(edges, dtype=np.float64)
    if edges.size < 2:
        return np.array([0.0, 1.0], dtype=np.float64)

    while True:
        if edges.size <= 2:
            break

        bin_idx = np.clip(np.searchsorted(edges, z, side="right") - 1, 0, edges.size - 2)
        counts = np.bincount(bin_idx, minlength=edges.size - 1)
        small_bins = np.flatnonzero(counts < MIN_BIN_ROWS)

        if small_bins.size == 0:
            break

        b = int(small_bins[0])
        if b == 0:
            remove = 1
        elif b == counts.size - 1:
            remove = b
        else:
            left = counts[b - 1]
            right = counts[b + 1]
            remove = b if left <= right else b + 1

        if remove <= 0 or remove >= edges.size - 1:
            break

        edges = np.delete(edges, remove)

    if edges.size < 2:
        edges = np.array([np.nanmin(z), np.nanmax(z)], dtype=np.float64)
        if not np.isfinite(edges[0]) or not np.isfinite(edges[1]) or edges[1] <= edges[0]:
            edges[1] = edges[0] + 1.0

    return edges


def _nearest_index(reference_centers, candidates, idx):
    c = np.asarray(candidates, dtype=np.int64)
    deltas = np.abs(reference_centers[c] - reference_centers[idx])
    return int(c[int(np.argmin(deltas))])


def _fit_bin_cube_params(colors, row_idx):
    p = _empty_cube_param()
    if row_idx.size == 0:
        return p

    x = colors[row_idx]
    if x.size == 0 or x.shape[0] < 3:
        return p

    center, mad = _robust_mad(x)
    if not np.all(np.isfinite(center)) or not np.all(np.isfinite(mad)):
        return p
    if np.any(mad <= 0.0):
        return p

    color_scale = np.maximum(mad * MAD_SCALE, SCALE_FLOOR)
    x_std = (x - center) / color_scale

    cov = np.cov(x_std, rowvar=False, bias=True)
    if cov.shape != (3, 3) or not np.all(np.isfinite(cov)):
        return p

    eigvals, eigvecs = np.linalg.eigh(cov)
    if not np.all(np.isfinite(eigvals)) or not np.all(np.isfinite(eigvecs)):
        return p

    order = np.argsort(eigvals)[::-1]
    components = eigvecs[:, order].T
    if not np.all(np.isfinite(components)):
        return p

    proj = x_std @ components.T
    t = proj[:, 0]
    q2 = proj[:, 1]
    q3 = proj[:, 2]
    d = np.sqrt(q2 * q2 + q3 * q3)

    med_t, mad_t = _robust_mad(t)
    med_q2, mad_q2 = _robust_mad(q2)
    med_q3, mad_q3 = _robust_mad(q3)
    med_d, mad_d = _robust_mad(d)

    if not np.isfinite(med_t) or not np.isfinite(med_q2) or not np.isfinite(med_q3) or not np.isfinite(med_d):
        return p
    if mad_t <= 0.0 or mad_q2 <= 0.0 or mad_q3 <= 0.0 or mad_d <= 0.0:
        return p

    scale_t = max(mad_t * MAD_SCALE, SCALE_FLOOR)
    scale_q2 = max(mad_q2 * MAD_SCALE, SCALE_FLOOR)
    scale_q3 = max(mad_q3 * MAD_SCALE, SCALE_FLOOR)
    scale_d = max(mad_d * MAD_SCALE, SCALE_FLOOR)

    t_hat = t / scale_t
    q2_hat = q2 / scale_q2
    q3_hat = q3 / scale_q3
    d_hat = d / scale_d

    clip_t = float(np.nanpercentile(np.abs(t_hat), CLIP_PERCENTILE))
    clip_q2 = float(np.nanpercentile(np.abs(q2_hat), CLIP_PERCENTILE))
    clip_q3 = float(np.nanpercentile(np.abs(q3_hat), CLIP_PERCENTILE))
    clip_d = float(np.nanpercentile(np.abs(d_hat), CLIP_PERCENTILE))

    p["valid"] = True
    p["center"] = center
    p["color_scale"] = color_scale
    p["components"] = components
    p["scale_t"] = scale_t
    p["scale_q2"] = scale_q2
    p["scale_q3"] = scale_q3
    p["scale_d"] = scale_d
    p["clip_t"] = clip_t
    p["clip_q2"] = clip_q2
    p["clip_q3"] = clip_q3
    p["clip_d"] = clip_d
    return p


def _prepare_cube_params(colors, bin_idx, n_bins, bin_centers, merged_from_small):
    base = []
    for b in range(n_bins):
        rows = np.flatnonzero(bin_idx == b)
        base.append(_fit_bin_cube_params(colors, rows))

    # Deterministic sign convention with continuity across adjacent bins.
    ones = np.array([1.0, 1.0, 1.0], dtype=np.float64)
    prev = None
    for b in range(n_bins):
        if not base[b]["valid"]:
            continue
        comps = np.array(base[b]["components"], copy=True)

        if np.dot(comps[0], ones) < 0.0:
            comps = -comps

        if prev is not None:
            for k in range(3):
                if np.dot(comps[k], prev[k]) < 0.0:
                    comps[k] = -comps[k]

        base[b]["components"] = comps
        prev = comps

    valid_indices = [i for i in range(n_bins) if base[i]["valid"] and (not merged_from_small[i])]
    if len(valid_indices) == 0:
        fallback_flags = np.ones(n_bins, dtype=np.int8)
        return [_copy_cube_param(_empty_cube_param()) for _ in range(n_bins)], fallback_flags

    resolved = []
    fallback_flags = np.zeros(n_bins, dtype=np.int8)
    for b in range(n_bins):
        needs_fallback = (not base[b]["valid"]) or bool(merged_from_small[b])
        if not needs_fallback:
            resolved.append(_copy_cube_param(base[b]))
            continue

        nearest = _nearest_index(bin_centers, np.array(valid_indices, dtype=np.int64), b)
        resolved.append(_copy_cube_param(base[nearest]))
        fallback_flags[b] = 1

    return resolved, fallback_flags


def _build_cube_feature_block(prefix, colors, overlap_weight, bin_idx, params):
    n = colors.shape[0]
    t_hat = np.zeros(n, dtype=np.float64)
    q2_hat = np.zeros(n, dtype=np.float64)
    q3_hat = np.zeros(n, dtype=np.float64)
    d_hat = np.zeros(n, dtype=np.float64)

    t_hat_clipped = np.zeros(n, dtype=np.float64)
    q2_hat_clipped = np.zeros(n, dtype=np.float64)
    q3_hat_clipped = np.zeros(n, dtype=np.float64)
    d_hat_clipped = np.zeros(n, dtype=np.float64)

    sign2 = np.zeros(n, dtype=np.float64)
    sign3 = np.zeros(n, dtype=np.float64)

    for b, p in enumerate(params):
        idx = np.flatnonzero(bin_idx == b)
        if idx.size == 0:
            continue

        x = colors[idx]
        x_std = (x - p["center"]) / p["color_scale"]
        proj = x_std @ p["components"].T

        t = proj[:, 0]
        q2 = proj[:, 1]
        q3 = proj[:, 2]
        d = np.sqrt(q2 * q2 + q3 * q3)

        t_u = t / p["scale_t"]
        q2_u = q2 / p["scale_q2"]
        q3_u = q3 / p["scale_q3"]
        d_u = d / p["scale_d"]

        t_hat[idx] = t_u
        q2_hat[idx] = q2_u
        q3_hat[idx] = q3_u
        d_hat[idx] = d_u

        t_hat_clipped[idx] = np.clip(t_u, -p["clip_t"], p["clip_t"])
        q2_hat_clipped[idx] = np.clip(q2_u, -p["clip_q2"], p["clip_q2"])
        q3_hat_clipped[idx] = np.clip(q3_u, -p["clip_q3"], p["clip_q3"])
        d_hat_clipped[idx] = np.clip(d_u, -p["clip_d"], p["clip_d"])

        sign2[idx] = np.sign(q2_u)
        sign3[idx] = np.sign(q3_u)

    ratio_t_over_q2 = t_hat / (np.abs(q2_hat) + RATIO_DENOM_EPS)
    ratio_t_over_q3 = t_hat / (np.abs(q3_hat) + RATIO_DENOM_EPS)
    g_d_hat = overlap_weight * d_hat
    g_absq2_hat = overlap_weight * np.abs(q2_hat)

    return {
        f"{prefix}_t_hat": t_hat,
        f"{prefix}_q2_hat": q2_hat,
        f"{prefix}_q3_hat": q3_hat,
        f"{prefix}_d_hat": d_hat,
        f"{prefix}_t_hat_clipped": t_hat_clipped,
        f"{prefix}_q2_hat_clipped": q2_hat_clipped,
        f"{prefix}_q3_hat_clipped": q3_hat_clipped,
        f"{prefix}_d_hat_clipped": d_hat_clipped,
        f"{prefix}_sign2": sign2,
        f"{prefix}_sign3": sign3,
        f"{prefix}_ratio_t_over_abs_q2": ratio_t_over_q2,
        f"{prefix}_ratio_t_over_abs_q3": ratio_t_over_q3,
        f"{prefix}_g_d_hat": g_d_hat,
        f"{prefix}_g_abs_q2_hat": g_absq2_hat,
    }


def add_redshift_adaptive_color_tube_residuals(raw, deps, aux):
    z = _to_float(raw["redshift"])
    z = np.maximum(z, 0.0)

    edges = _build_equal_freq_edges(z)

    z_clamped = np.clip(z, edges[0], edges[-1])
    bin_idx = np.clip(np.searchsorted(edges, z_clipped, side="right") - 1, 0, edges.size - 2).astype(np.int64)
    n_bins = edges.size - 1

    bin_counts = np.bincount(bin_idx, minlength=n_bins)
    merged_from_small = bin_counts < MIN_BIN_ROWS
    bin_centers = 0.5 * (edges[:-1] + edges[1:])

    u = _to_float(raw["u"])
    g = _to_float(raw["g"])
    r = _to_float(raw["r"])
    i = _to_float(raw["i"])
    z_band = _to_float(raw["z"])

    c1 = u - g
    c2 = g - r
    c3 = r - i
    c4 = i - z_band

    ugri = np.column_stack((c1, c2, c3))
    griz = np.column_stack((c2, c3, c4))

    ugri_params, ugri_fallback_by_bin = _prepare_cube_params(ugri, bin_idx, n_bins, bin_centers, merged_from_small)
    griz_params, griz_fallback_by_bin = _prepare_cube_params(griz, bin_idx, n_bins, bin_centers, merged_from_small)

    overlap_weight = np.exp(-0.5 * ((z - OVERLAP_GAUSS_CENTER) / OVERLAP_GAUSS_SIGMA) ** 2)

    ugri_feats = _build_cube_feature_block("ugri", ugri, overlap_weight, bin_idx, ugri_params)
    griz_feats = _build_cube_feature_block("griz", griz, overlap_weight, bin_idx, griz_params)

    fallback_ugri = ugri_fallback_by_bin[bin_idx].astype(np.int8)
    fallback_griz = griz_fallback_by_bin[bin_idx].astype(np.int8)

    features = {}
    features.update(ugri_feats)
    features.update(griz_feats)
    features["ugri_fallback_from_bin"] = fallback_ugri
    features["griz_fallback_from_bin"] = fallback_griz
    return pd.DataFrame(features, index=raw.index)


FEATURE_GROUPS = [
    {
        "name": "redshift_adaptive_color_tube_residuals",
        "fn": add_redshift_adaptive_color_tube_residuals,
        "depends_on": [],
        "description": "Builds redshift-adaptive PCA color-tube residual coordinates in Ugri and Griz spaces with continuity-stable orientation and overlap-window emphasized residual features.",
    }
]