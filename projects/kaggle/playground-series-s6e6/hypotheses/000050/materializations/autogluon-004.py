import numpy as np
import pandas as pd


MS_MIN_COLOR = 0.20
MS_MAX_COLOR = 4.00
MS_BLUE_TURNOFF = 0.45
MS_N_BINS = 80
MS_MIN_BIN_N = 500
MS_MAD_FLOOR = 1e-4
MS_Z_CLIP = 10.0
MS_LOG_D_MIN = 0.0
MS_LOG_D_MAX = 10.0
MS_QUANTILE_COUNT = 101

MS_STAT_FEATURES = (
    "mu_ms",
    "log_d_pc",
    "M_u",
    "M_g",
    "M_r_ms",
    "M_i",
    "M_z",
    "abs_ug",
    "abs_gr",
    "abs_ri",
    "abs_iz",
    "delta_gr",
    "delta_ri",
    "delta_iz",
)


def _fit_mask_from_aux(raw, aux):
    if aux is None or len(aux) == 0:
        return pd.Series(True, index=raw.index)

    for col in ("fit_mask", "is_fit", "is_train", "__is_train__", "train_mask"):
        if col in aux.columns:
            mask = aux[col].reindex(raw.index)
            return mask.fillna(False).astype(bool)

    return pd.Series(True, index=raw.index)


def _safe_numeric(raw, col):
    return pd.to_numeric(raw[col], errors="coerce").astype(float)


def _scaled_mad(values):
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return 1.0
    med = np.median(arr)
    mad = np.median(np.abs(arr - med))
    return max(1.4826 * mad, MS_MAD_FLOOR)


def _global_stats(frame, columns):
    stats = {}
    for col in columns:
        vals = np.asarray(frame[col], dtype=float)
        finite = vals[np.isfinite(vals)]
        if finite.size == 0:
            stats[col] = (0.0, 1.0)
        else:
            stats[col] = (float(np.median(finite)), float(_scaled_mad(finite)))
    return stats


def _make_bin_edges(x_fit):
    x = pd.Series(x_fit[np.isfinite(x_fit)])
    if x.empty:
        return [MS_MIN_COLOR, MS_MAX_COLOR]

    quantiles = np.linspace(0.0, 1.0, MS_N_BINS + 1)
    edges = np.quantile(x.to_numpy(dtype=float), quantiles)
    edges = np.unique(edges)

    if edges.size < 2:
        center = float(edges[0]) if edges.size == 1 else float(np.nanmedian(x))
        lo = max(MS_MIN_COLOR, center - 1e-6)
        hi = min(MS_MAX_COLOR, center + 1e-6)
        if lo >= hi:
            lo = MS_MIN_COLOR
            hi = MS_MAX_COLOR
        return [lo, hi]

    edges[0] = MS_MIN_COLOR
    edges[-1] = MS_MAX_COLOR
    return edges.tolist()


def _assign_bins(x_values, edges):
    bins = np.searchsorted(np.asarray(edges, dtype=float), x_values, side="right") - 1
    return np.clip(bins, 0, max(len(edges) - 2, 0))


def _neighbor_indices(bin_id, n_bins, counts):
    selected = [bin_id]
    total = int(counts[bin_id]) if 0 <= bin_id < n_bins else 0
    radius = 1

    while total < MS_MIN_BIN_N and (bin_id - radius >= 0 or bin_id + radius < n_bins):
        left = bin_id - radius
        right = bin_id + radius

        if left >= 0:
            selected.append(left)
            total += int(counts[left])
        if total >= MS_MIN_BIN_N:
            break
        if right < n_bins:
            selected.append(right)
            total += int(counts[right])

        radius += 1

    if total < MS_MIN_BIN_N:
        return None
    return selected


def _empirical_percentile(values, quantiles):
    arr = np.asarray(values, dtype=float)
    q = np.asarray(quantiles, dtype=float)
    if q.size == 0 or not np.isfinite(q).any():
        return np.full(arr.shape, 0.5, dtype=float)

    q = q[np.isfinite(q)]
    if q.size == 1:
        return np.where(arr < q[0], 0.0, np.where(arr > q[0], 1.0, 0.5))

    probs = np.linspace(0.0, 1.0, q.size)
    return np.interp(arr, q, probs, left=0.0, right=1.0)


def add_main_sequence_parallax_plausibility(raw, deps, aux):
    idx = raw.index

    u = _safe_numeric(raw, "u")
    g = _safe_numeric(raw, "g")
    r = _safe_numeric(raw, "r")
    i = _safe_numeric(raw, "i")
    z = _safe_numeric(raw, "z")

    x = g - i
    x_c = x.clip(MS_MIN_COLOR, MS_MAX_COLOR)

    m_r_ms = (
        -5.06
        + 14.32 * x_c
        - 12.97 * np.power(x_c, 2)
        + 6.127 * np.power(x_c, 3)
        - 1.267 * np.power(x_c, 4)
        + 0.0967 * np.power(x_c, 5)
    )

    mu_ms = r - m_r_ms
    log_d_pc = ((mu_ms + 5.0) / 5.0).clip(MS_LOG_D_MIN, MS_LOG_D_MAX)
    d_pc_clipped = np.power(10.0, log_d_pc)

    m_u = u - mu_ms
    m_g = g - mu_ms
    m_i = i - mu_ms
    m_z = z - mu_ms

    abs_ug = m_u - m_g
    abs_gr = m_g - m_r_ms
    abs_ri = m_r_ms - m_i
    abs_iz = m_i - m_z

    delta_gr = abs_gr - (g - r)
    delta_ri = abs_ri - (r - i)
    delta_iz = abs_iz - (i - z)

    base = pd.DataFrame(
        {
            "gi_color": x,
            "gi_color_clipped": x_c,
            "gi_below_ms_range": (x < MS_MIN_COLOR).astype(np.int8),
            "gi_above_ms_range": (x > MS_MAX_COLOR).astype(np.int8),
            "gi_blue_turnoff": (x < MS_BLUE_TURNOFF).astype(np.int8),
            "M_r_ms": m_r_ms,
            "mu_ms": mu_ms,
            "log_d_pc": log_d_pc,
            "d_pc_clipped": d_pc_clipped,
            "M_u": m_u,
            "M_g": m_g,
            "M_i": m_i,
            "M_z": m_z,
            "abs_ug": abs_ug,
            "abs_gr": abs_gr,
            "abs_ri": abs_ri,
            "abs_iz": abs_iz,
            "delta_gr": delta_gr,
            "delta_ri": delta_ri,
            "delta_iz": delta_iz,
        },
        index=idx,
    )

    fit_mask = _fit_mask_from_aux(raw, aux).reindex(idx).fillna(False).astype(bool)
    fit_frame = base.loc[fit_mask]
    if fit_frame.empty:
        fit_frame = base

    global_stats = _global_stats(fit_frame, MS_STAT_FEATURES)
    global_mu = np.asarray(fit_frame["mu_ms"], dtype=float)
    global_mu = global_mu[np.isfinite(global_mu)]
    if global_mu.size == 0:
        global_mu_quantiles = np.array([0.0])
    else:
        global_mu_quantiles = np.quantile(
            global_mu,
            np.linspace(0.0, 1.0, MS_QUANTILE_COUNT),
        )

    x_fit = np.asarray(fit_frame["gi_color_clipped"], dtype=float)
    edges = _make_bin_edges(x_fit)
    train_bins = _assign_bins(x_fit, edges)
    all_bins = _assign_bins(np.asarray(base["gi_color_clipped"], dtype=float), edges)
    n_bins = max(len(edges) - 1, 1)
    counts = np.bincount(train_bins, minlength=n_bins)

    bin_stats = []
    bin_mu_quantiles = []

    for bin_id in range(n_bins):
        neighbors = _neighbor_indices(bin_id, n_bins, counts)
        if neighbors is None:
            stat_frame = fit_frame
            mu_quantiles = global_mu_quantiles
        else:
            neighbor_mask = np.isin(train_bins, np.asarray(neighbors, dtype=int))
            stat_frame = fit_frame.iloc[np.flatnonzero(neighbor_mask)]
            finite_mu = np.asarray(stat_frame["mu_ms"], dtype=float)
            finite_mu = finite_mu[np.isfinite(finite_mu)]
            if finite_mu.size == 0:
                mu_quantiles = global_mu_quantiles
            else:
                mu_quantiles = np.quantile(
                    finite_mu,
                    np.linspace(0.0, 1.0, MS_QUANTILE_COUNT),
                )

        bin_stats.append(_global_stats(stat_frame, MS_STAT_FEATURES))
        bin_mu_quantiles.append(mu_quantiles)

    out = base.copy()

    for col in MS_STAT_FEATURES:
        med = np.empty(len(base), dtype=float)
        mad = np.empty(len(base), dtype=float)

        for bin_id in range(n_bins):
            rows = all_bins == bin_id
            if np.any(rows):
                med_val, mad_val = bin_stats[bin_id].get(col, global_stats[col])
                med[rows] = med_val
                mad[rows] = mad_val

        z_col = (np.asarray(base[col], dtype=float) - med) / np.maximum(mad, MS_MAD_FLOOR)
        out[col + "_within_gi_mad_z"] = np.clip(z_col, -MS_Z_CLIP, MS_Z_CLIP)

    mu_percentile = np.empty(len(base), dtype=float)
    mu_values = np.asarray(base["mu_ms"], dtype=float)

    for bin_id in range(n_bins):
        rows = all_bins == bin_id
        if np.any(rows):
            mu_percentile[rows] = _empirical_percentile(
                mu_values[rows],
                bin_mu_quantiles[bin_id],
            )

    out["mu_ms_within_gi_percentile"] = np.clip(mu_percentile, 0.0, 1.0)

    return out


FEATURE_GROUPS = [
    {
        "name": "main_sequence_parallax_plausibility",
        "fn": add_main_sequence_parallax_plausibility,
        "depends_on": [],
        "description": "Main-sequence photometric-parallax plausibility features from ugriz colors, implied distances, and color-neighborhood residuals.",
    }
]