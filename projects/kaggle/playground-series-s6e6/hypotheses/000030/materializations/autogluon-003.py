import numpy as np
import pandas as pd

_CUBE_BIN_WIDTH = 0.08
_BLEND_COUNT = 30.0
_BLEND_SCALE = 80.0
_POPULATION_BIN_MIN = 40
_SCALE_FLOOR = 0.015
_MAD_NORMALIZER = 0.6744897501960817
_OUTLIER_CENTER = 3.9
_OUTLIER_SCALE = 0.45
_DEFAULT_AXIS = (1.0, 0.0, 0.0)
_MISSING_POPULATION = "__MISSING__"


def _sigmoid(x):
    x = np.asarray(x, dtype=float)
    return 1.0 / (1.0 + np.exp(-np.clip(x, -80.0, 80.0)))


def _soft_box(v, left, right, tau):
    return _sigmoid((v - left) / tau) * _sigmoid((right - v) / tau)


def _normalize(vec, fallback_axis):
    v = np.asarray(vec, dtype=float)
    n = np.linalg.norm(v)
    if (not np.isfinite(n)) or n <= 0.0:
        return np.asarray(fallback_axis, dtype=float)
    return v / n


def _mad_scale(sample, center):
    mad = np.nanmedian(np.abs(sample - center), axis=0) / _MAD_NORMALIZER
    mad = np.where(np.isfinite(mad), mad, _SCALE_FLOOR)
    return np.maximum(mad, _SCALE_FLOOR)


def _principal_axis(standardized):
    if standardized.shape[0] < 2 or standardized.shape[1] != 3:
        return np.asarray(_DEFAULT_AXIS, dtype=float)

    data = np.nan_to_num(standardized, nan=0.0, posinf=0.0, neginf=0.0)
    if not np.isfinite(data).all():
        return np.asarray(_DEFAULT_AXIS, dtype=float)
    if np.allclose(data, 0.0):
        return np.asarray(_DEFAULT_AXIS, dtype=float)

    try:
        cov = np.cov(data, rowvar=False, bias=False)
    except Exception:
        return np.asarray(_DEFAULT_AXIS, dtype=float)

    if cov.shape != (3, 3) or not np.isfinite(cov).all() or np.allclose(cov, 0.0):
        return np.asarray(_DEFAULT_AXIS, dtype=float)

    try:
        eigvals, eigvecs = np.linalg.eigh(cov)
    except Exception:
        return np.asarray(_DEFAULT_AXIS, dtype=float)

    if not np.isfinite(eigvals).all():
        return np.asarray(_DEFAULT_AXIS, dtype=float)

    axis = eigvecs[:, int(np.argmax(eigvals))]
    return _normalize(axis, _DEFAULT_AXIS)


def _bin_cube_stats(values, bin_index, n_bins):
    counts = np.zeros(n_bins, dtype=np.int64)
    mu = np.full((n_bins, 3), np.nan, dtype=float)
    scale = np.full((n_bins, 3), np.nan, dtype=float)
    axis = np.zeros((n_bins, 3), dtype=float)
    default_axis = np.asarray(_DEFAULT_AXIS, dtype=float)
    axis[:] = default_axis

    for b in range(n_bins):
        bmask = bin_index == b
        if not np.any(bmask):
            continue
        block = np.asarray(values[bmask], dtype=float)
        block = block[np.isfinite(block).all(axis=1)]
        if block.size == 0:
            continue

        c = np.nanmedian(block, axis=0)
        s = _mad_scale(block, c)
        mu[b] = c
        scale[b] = s
        counts[b] = block.shape[0]
        axis[b] = _principal_axis((block - c) / s)

    return counts, mu, scale, axis


def _clip_by_percentiles(values):
    arr = np.asarray(values, dtype=float)
    finite = np.isfinite(arr)
    if not finite.any():
        return np.array(arr, dtype=float), 0.0, 1.0

    lo, hi = np.nanpercentile(arr[finite], [1.0, 99.0])
    lo = float(lo)
    hi = float(hi)
    if (not np.isfinite(lo)) or (not np.isfinite(hi)) or hi <= lo:
        lo = float(np.nanmin(arr[finite]))
        hi = float(np.nanmax(arr[finite]))
        if (not np.isfinite(lo)) or (not np.isfinite(hi)) or hi <= lo:
            lo, hi = 0.0, 1.0
        elif hi <= lo:
            hi = lo + 1.0

    clipped = np.clip(arr, lo, hi)
    return clipped.astype(float), lo, hi


def _cube_outlier_scores(colors_3d, bin_axis, axis_low, axis_high, population_keys, bin_width):
    n = len(colors_3d)
    scores = np.full(n, np.nan, dtype=float)

    colors = np.asarray(colors_3d, dtype=float)
    axis = np.asarray(bin_axis, dtype=float)
    pop_keys = np.asarray(population_keys, dtype=object)

    finite = np.isfinite(colors).all(axis=1) & np.isfinite(axis)
    if not finite.any():
        return scores

    colors_f = colors[finite]
    axis_f = axis[finite]
    pop_f = pop_keys[finite]

    lo = float(axis_low)
    hi = float(axis_high)
    if (not np.isfinite(lo)) or (not np.isfinite(hi)) or hi <= lo:
        lo = float(np.nanmin(axis_f))
        hi = float(np.nanmax(axis_f))
        if (not np.isfinite(lo)) or (not np.isfinite(hi)) or hi <= lo:
            lo, hi = 0.0, 1.0

    n_bins = int(np.ceil((hi - lo) / bin_width))
    if n_bins < 1 or not np.isfinite(n_bins):
        n_bins = 1

    axis_f = np.clip(axis_f, lo, hi)
    bin_idx = np.floor((axis_f - lo) / bin_width).astype(np.int64)
    if bin_idx.size:
        bin_idx = np.clip(bin_idx, 0, n_bins - 1)

    g_counts, g_mu, g_scale, g_axis = _bin_cube_stats(colors_f, bin_idx, n_bins)

    pop_stats = {}
    for pop in np.unique(pop_f):
        mask = pop_f == pop
        if not np.any(mask):
            continue
        pop_stats[pop] = _bin_cube_stats(colors_f[mask], bin_idx[mask], n_bins)

    finite_idx = np.nonzero(finite)[0]
    scores_f = np.full(len(finite_idx), np.nan, dtype=float)

    for i_local, i_global in enumerate(finite_idx):
        b = bin_idx[i_local]
        mu_g = g_mu[b]
        s_g = g_scale[b]
        t_g = g_axis[b]

        if not (np.isfinite(mu_g).all() and np.isfinite(s_g).all() and np.isfinite(t_g).all()):
            continue

        pop = pop_f[i_local]
        mu = mu_g
        s = s_g
        t = t_g

        if pop in pop_stats:
            p_counts, p_mu, p_scale, p_axis = pop_stats[pop]
            n_pop = int(p_counts[b])

            if n_pop >= _POPULATION_BIN_MIN:
                if np.isfinite(p_mu[b]).all() and np.isfinite(p_scale[b]).all() and np.isfinite(p_axis[b]).all():
                    mu = p_mu[b]
                    s = p_scale[b]
                    t = p_axis[b]

            elif n_pop > 0:
                if np.isfinite(p_mu[b]).all() and np.isfinite(p_scale[b]).all() and np.isfinite(p_axis[b]).all():
                    w = (float(n_pop) - _BLEND_COUNT) / _BLEND_SCALE
                    if w < 0.0:
                        w = 0.0
                    elif w > 1.0:
                        w = 1.0
                    mu = (w * p_mu[b]) + ((1.0 - w) * mu_g)
                    s = np.sqrt((w * (p_scale[b] ** 2)) + ((1.0 - w) * (s_g ** 2)))
                    s = np.maximum(s, _SCALE_FLOOR)
                    t = (w * p_axis[b]) + ((1.0 - w) * t_g)
                    t = _normalize(t, t_g)

        x = colors_f[i_local]
        r = (x - mu) / s
        p = float(np.dot(r, t))
        q = r - (p * t)
        d = float(np.linalg.norm(q))
        scores_f[i_local] = _sigmoid((d - _OUTLIER_CENTER) / _OUTLIER_SCALE)

    scores[finite] = scores_f
    return scores


def add_sdss_quasar_targeting_surface(raw, deps, aux):
    raw_df = raw

    u = raw_df["u"].to_numpy(dtype=float)
    g = raw_df["g"].to_numpy(dtype=float)
    r = raw_df["r"].to_numpy(dtype=float)
    i = raw_df["i"].to_numpy(dtype=float)
    z = raw_df["z"].to_numpy(dtype=float)
    redshift = raw_df["redshift"].to_numpy(dtype=float)

    c1_raw = u - g
    c2_raw = g - r
    c3_raw = r - i
    c4_raw = i - z

    c1, c1_lo, c1_hi = _clip_by_percentiles(c1_raw)
    c2, c2_lo, c2_hi = _clip_by_percentiles(c2_raw)
    c3, c3_lo, c3_hi = _clip_by_percentiles(c3_raw)
    c4, c4_lo, c4_hi = _clip_by_percentiles(c4_raw)

    pop_src = raw_df["galaxy_population"].to_numpy()
    population_keys = np.array(
        [
            _MISSING_POPULATION if pd.isna(v) else str(v)
            for v in pop_src
        ],
        dtype=object,
    )

    cube_a = np.column_stack((c1, c2, c3))
    cube_b = np.column_stack((c2, c3, c4))

    oa = _cube_outlier_scores(cube_a, c3, c3_lo, c3_hi, population_keys, _CUBE_BIN_WIDTH)
    ob = _cube_outlier_scores(cube_b, c2, c2_lo, c2_hi, population_keys, _CUBE_BIN_WIDTH)
    omax = np.maximum(oa, ob)

    uvx = _sigmoid((20.2 - i) / 1.0) * _sigmoid((0.60 - c1_raw) / 0.05)
    mid_z = (
        _soft_box(c1_raw, 0.65, 1.5, 0.07)
        * _soft_box(c2_raw, 0.0, 0.20, 0.06)
        * _sigmoid((redshift - 2.35) / 0.35)
        * _sigmoid((3.20 - redshift) / 0.35)
    )
    hi_z = (
        _sigmoid((redshift - 3.0) / 0.35)
        * _sigmoid((c1_raw - 1.55) / 0.08)
        * _sigmoid((20.2 - i) / 1.0)
    )
    wd = (
        _soft_box(c2_raw, -0.8, -0.2, 0.10)
        * _soft_box(c3_raw, -0.6, -0.2, 0.10)
        * _soft_box(c4_raw, -1.0, 0.0, 0.10)
    )
    mwd = _soft_box(c2_raw, 0.0, 1.6, 0.10) * _soft_box(c3_raw, 0.6, 2.0, 0.10)
    a_term = _soft_box(c1_raw, 0.9, 1.5, 0.06) * _soft_box(c2_raw, -0.35, 0.0, 0.06)
    blue = _sigmoid((0.90 - c1_raw) / 0.08) * _sigmoid((0.80 - c2_raw) / 0.08) * _sigmoid((i - 19.0) / 1.0)

    return pd.DataFrame(
        {
            "sdss_OA": oa,
            "sdss_OB": ob,
            "sdss_Omax": omax,
            "sdss_UVX": uvx,
            "sdss_mid_z": mid_z,
            "sdss_hi_z": hi_z,
            "sdss_WD": wd,
            "sdss_MWD": mwd,
            "sdss_A": a_term,
            "sdss_BLUE": blue,
        },
        index=raw_df.index,
    )


FEATURE_GROUPS = [
    {
        "name": "sdss_quasar_targeting_surface",
        "fn": add_sdss_quasar_targeting_surface,
        "depends_on": [],
        "description": "Builds smooth SDSS-inspired two-cube color-geometry outlier scores with population-aware bin blending and continuous inclusion/exclusion priors.",
    }
]