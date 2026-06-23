import numpy as np
import pandas as pd

_BIN_COUNT = 20
_MIN_BIN_ROWS = 5000
_MAD_SCALE = 1.4826
_GATE_CENTER = 2.7
_GATE_WIDTH = 0.35
_EPS = 1e-6


def _build_redshift_bins(z_plus, num_bins=_BIN_COUNT, min_rows=_MIN_BIN_ROWS):
    z_plus = np.asarray(z_plus, dtype=np.float64)
    n = z_plus.size
    if n == 0:
        return np.array([0.0, 1.0], dtype=np.float64), np.zeros(0, dtype=np.int64)

    finite = np.isfinite(z_plus)
    if not finite.any():
        return np.array([0.0, 1.0], dtype=np.float64), np.zeros(n, dtype=np.int64)

    finite_vals = z_plus[finite]
    quantiles = np.linspace(0.0, 1.0, num_bins + 1)
    edges = np.quantile(finite_vals, quantiles)
    edges = np.asarray(edges, dtype=np.float64)
    edges = np.maximum.accumulate(edges)
    edges[0] -= 1e-9
    edges[-1] += 1e-9

    merged = [edges[0]]
    for e in edges[1:]:
        if e > merged[-1]:
            merged.append(e)
    if len(merged) < 2:
        z_min = np.nanmin(finite_vals)
        z_max = np.nanmax(finite_vals)
        if not np.isfinite(z_min) or not np.isfinite(z_max):
            z_min, z_max = 0.0, 1.0
        if z_min == z_max:
            z_min -= 1.0
            z_max += 1.0
        merged = [z_min, z_max]

    edges = np.array(merged, dtype=np.float64)
    n_edges = edges.size - 1
    if n_edges <= 0:
        return np.array([0.0, 1.0], dtype=np.float64), np.zeros(n, dtype=np.int64)

    initial_labels = np.searchsorted(edges, z_plus, side="right") - 1
    in_initial = np.isfinite(z_plus) & (z_plus >= edges[0]) & (z_plus <= edges[-1])
    initial_labels = np.where(in_initial, initial_labels, -1).astype(np.int64)

    counts = np.bincount(initial_labels[initial_labels >= 0], minlength=n_edges)

    segments = []
    i = 0
    while i < n_edges:
        start = i
        end = i
        total = counts[i]
        while total < min_rows and end + 1 < n_edges:
            end += 1
            total += counts[end]
        if end == n_edges - 1 and total < min_rows and segments:
            prev_start, prev_end, prev_total = segments.pop()
            segments.append((prev_start, end, prev_total + total))
        else:
            segments.append((start, end, total))
        i = end + 1

    if not segments:
        return np.array([edges[0], edges[-1]], dtype=np.float64), np.zeros(n, dtype=np.int64)

    final_edges = [edges[segments[0][0]]]
    for _start, end, _total in segments:
        final_edges.append(edges[end + 1])
    final_edges = np.array(final_edges, dtype=np.float64)

    final_labels = np.searchsorted(final_edges, z_plus, side="right") - 1
    in_final = np.isfinite(z_plus) & (z_plus >= final_edges[0]) & (z_plus <= final_edges[-1])
    final_labels = np.where(in_final, final_labels, -1).astype(np.int64)

    if (final_labels < 0).any():
        centers = 0.5 * (final_edges[:-1] + final_edges[1:])
        missing = np.where(final_labels < 0)[0]
        fill = z_plus[missing]
        fill = np.where(np.isfinite(fill), fill, np.nanmean(centers))
        nearest = np.argmin(np.abs(fill[:, None] - centers[None, :]), axis=1).astype(np.int64)
        final_labels[missing] = nearest

    return final_labels, final_edges


def _compute_cube_geometry(x_block):
    x = np.asarray(x_block, dtype=np.float64)
    n = x.shape[0]

    center = np.nanmedian(x, axis=0)
    mad = np.nanmedian(np.abs(x - center), axis=0)
    scale = _MAD_SCALE * mad
    scale = np.where(np.isfinite(scale) & (scale > 0), scale, 1.0)
    xs = (x - center) / scale

    if n > 1:
        cov = np.dot(xs.T, xs) / float(n - 1)
    elif n == 1:
        cov = np.outer(xs[0], xs[0])
    else:
        return (
            np.array([], dtype=np.float64),
            np.array([], dtype=np.float64),
            np.array([], dtype=np.float64),
            np.array([], dtype=np.float64),
            np.array([], dtype=np.float64),
        )

    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvecs = eigvecs[:, order]
    v1 = eigvecs[:, 0]
    v2 = eigvecs[:, 1] if eigvecs.shape[1] > 1 else eigvecs[:, 0]

    t = np.dot(xs, v1)
    residual = xs - np.outer(t, v1)
    d = np.linalg.norm(residual, axis=1)
    s = np.sign(np.dot(xs, v2))

    t_med = np.nanmedian(t)
    t_mad = _MAD_SCALE * np.nanmedian(np.abs(t - t_med))
    if not (np.isfinite(t_mad) and t_mad > 0):
        t_mad = 1.0
    t_ratio = t / (t_mad + _EPS)

    d_med = np.nanmedian(d)
    d_mad = _MAD_SCALE * np.nanmedian(np.abs(d - d_med))
    if not (np.isfinite(d_mad) and d_mad > 0):
        d_mad = 1.0
    d_norm = d / d_mad
    if n > 1:
        upper = np.nanpercentile(d_norm, 99.5)
        if not np.isfinite(upper) or upper <= 0:
            upper = np.nanmax(d_norm) if np.isfinite(np.nanmax(d_norm)) else 0.0
    else:
        upper = d_norm[0]
        if not np.isfinite(upper) or upper < 0:
            upper = 0.0
    d_norm = np.clip(d_norm, 0.0, upper)

    return t, d, d_norm, s, t_ratio


def add_redshift_adaptive_color_tube_residuals(raw, deps, aux):
    n = raw.shape[0]
    if n == 0:
        return pd.DataFrame(index=raw.index)

    redshift = np.asarray(raw["redshift"], dtype=np.float64)
    z_plus = np.maximum(redshift, 0.0)

    u = np.asarray(raw["u"], dtype=np.float64)
    g = np.asarray(raw["g"], dtype=np.float64)
    r = np.asarray(raw["r"], dtype=np.float64)
    i = np.asarray(raw["i"], dtype=np.float64)
    z = np.asarray(raw["z"], dtype=np.float64)

    c1 = u - g
    c2 = g - r
    c3 = r - i
    c4 = i - z

    ugri = np.column_stack((c1, c2, c3))
    griz = np.column_stack((c2, c3, c4))

    bin_ids, bin_edges = _build_redshift_bins(z_plus)
    bin_count = max(1, bin_edges.size - 1)

    t_ugri = np.full(n, np.nan, dtype=np.float64)
    d_ugri = np.full(n, np.nan, dtype=np.float64)
    dnorm_ugri = np.full(n, np.nan, dtype=np.float64)
    s_ugri = np.full(n, np.nan, dtype=np.float64)
    trate_ugri = np.full(n, np.nan, dtype=np.float64)

    t_griz = np.full(n, np.nan, dtype=np.float64)
    d_griz = np.full(n, np.nan, dtype=np.float64)
    dnorm_griz = np.full(n, np.nan, dtype=np.float64)
    s_griz = np.full(n, np.nan, dtype=np.float64)
    trate_griz = np.full(n, np.nan, dtype=np.float64)

    for b in range(bin_count):
        mask = bin_ids == b
        if not np.any(mask):
            continue

        t_u, d_u, dnorm_u, s_u, tr_u = _compute_cube_geometry(ugri[mask])
        t_g, d_g, dnorm_g, s_g, tr_g = _compute_cube_geometry(griz[mask])

        t_ugri[mask] = t_u
        d_ugri[mask] = d_u
        dnorm_ugri[mask] = dnorm_u
        s_ugri[mask] = s_u
        trate_ugri[mask] = tr_u

        t_griz[mask] = t_g
        d_griz[mask] = d_g
        dnorm_griz[mask] = dnorm_g
        s_griz[mask] = s_g
        trate_griz[mask] = tr_g

    gate = np.exp(-((z_plus - _GATE_CENTER) ** 2) / (2.0 * (_GATE_WIDTH ** 2)))
    gate = np.where(np.isfinite(gate), gate, 0.0)

    feats = pd.DataFrame(
        {
            "tube_t_ugri": t_ugri,
            "tube_t_ratio_ugri": trate_ugri,
            "tube_d_ugri": d_ugri,
            "tube_d_norm_ugri": dnorm_ugri,
            "tube_s_ugri": s_ugri,
            "tube_t_griz": t_griz,
            "tube_t_ratio_griz": trate_griz,
            "tube_d_griz": d_griz,
            "tube_d_norm_griz": dnorm_griz,
            "tube_s_griz": s_griz,
            "tube_d_norm_ugri_gated_z27": dnorm_ugri * gate,
            "tube_d_norm_griz_gated_z27": dnorm_griz * gate,
        },
        index=raw.index,
    )

    return feats


FEATURE_GROUPS = [
    {
        "name": "redshift_adaptive_color_tube_residuals",
        "fn": add_redshift_adaptive_color_tube_residuals,
        "depends_on": [],
        "description": "Build redshift-adaptive local color-manifold tube features using robust PCA residual geometry and z≈2.7 overlap gating.",
    }
]