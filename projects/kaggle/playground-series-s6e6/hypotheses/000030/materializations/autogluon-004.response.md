import numpy as np
import pandas as pd


_COLOR_LOWER_Q = 0.005
_COLOR_UPPER_Q = 0.995
_BIN_STEP = 0.08
_KERNEL_WIDTH = 0.12
_MIN_SCALE = 0.015
_CORE_RADIUS_Q = 0.80
_CORE_RADIUS_CAP = 3.0
_DISTANCE_CLIP = 25.0


def _sigmoid(x):
    x = np.clip(np.asarray(x, dtype=np.float64), -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-x))


def _box(v, lower, upper, tau):
    v = np.asarray(v, dtype=np.float64)
    return _sigmoid((v - lower) / tau) * _sigmoid((upper - v) / tau)


def _safe_series(raw, name, default=0.0):
    if name in raw.columns:
        return pd.to_numeric(raw[name], errors="coerce").to_numpy(dtype=np.float64)
    return np.full(len(raw), default, dtype=np.float64)


def _weighted_quantile(values, quantile, weights=None):
    values = np.asarray(values, dtype=np.float64)
    mask = np.isfinite(values)
    if weights is None:
        clean = values[mask]
        if clean.size == 0:
            return 0.0
        return float(np.quantile(clean, quantile))

    weights = np.asarray(weights, dtype=np.float64)
    mask &= np.isfinite(weights) & (weights > 0.0)
    clean = values[mask]
    clean_weights = weights[mask]
    if clean.size == 0 or clean_weights.sum() <= 0.0:
        return _weighted_quantile(values, quantile, None)

    order = np.argsort(clean)
    clean = clean[order]
    clean_weights = clean_weights[order]
    cumulative = np.cumsum(clean_weights)
    cutoff = quantile * cumulative[-1]
    return float(clean[np.searchsorted(cumulative, cutoff, side="left")])


def _weighted_median_axis(x, weights=None):
    return np.array([_weighted_quantile(x[:, j], 0.5, weights) for j in range(x.shape[1])], dtype=np.float64)


def _weighted_cov(x, weights):
    weights = np.asarray(weights, dtype=np.float64)
    weights = np.where(np.isfinite(weights) & (weights > 0.0), weights, 0.0)
    total = weights.sum()
    if total <= 0.0:
        return np.eye(x.shape[1], dtype=np.float64)

    mean = np.sum(x * weights[:, None], axis=0) / total
    centered = x - mean
    denom = total - np.sum(weights * weights) / total
    if denom <= 1.0e-12:
        denom = total
    cov = (centered * weights[:, None]).T @ centered / denom
    cov = np.nan_to_num(cov, nan=0.0, posinf=0.0, neginf=0.0)
    cov += np.eye(x.shape[1], dtype=np.float64) * 1.0e-6
    return cov


def _tube_params(x, weights=None, fallback=None, shrink_neff=None, global_direction=None):
    if x.shape[0] == 0:
        return fallback

    if weights is None:
        weights = np.ones(x.shape[0], dtype=np.float64)
    else:
        weights = np.asarray(weights, dtype=np.float64)
        weights = np.where(np.isfinite(weights) & (weights > 0.0), weights, 0.0)

    effective = float(weights.sum()) if shrink_neff is None else float(shrink_neff)
    if weights.sum() <= 0.0:
        weights = np.ones(x.shape[0], dtype=np.float64)
        effective = float(x.shape[0]) if shrink_neff is None else float(shrink_neff)

    center = _weighted_median_axis(x, weights)
    mad = np.array(
        [_weighted_quantile(np.abs(x[:, j] - center[j]), 0.5, weights) for j in range(x.shape[1])],
        dtype=np.float64,
    )
    scale = np.maximum(1.4826 * mad, _MIN_SCALE)
    radius = np.sqrt(np.sum(((x - center) / scale) ** 2, axis=1))
    radius_cut = max(_weighted_quantile(radius, _CORE_RADIUS_Q, weights), _CORE_RADIUS_CAP)
    core_mask = radius <= radius_cut
    if core_mask.sum() < max(8, min(30, x.shape[0])):
        core_mask = np.ones(x.shape[0], dtype=bool)

    core_x = x[core_mask]
    core_w = weights[core_mask]
    center = _weighted_median_axis(core_x, core_w)
    mad = np.array(
        [_weighted_quantile(np.abs(core_x[:, j] - center[j]), 0.5, core_w) for j in range(core_x.shape[1])],
        dtype=np.float64,
    )
    scale = np.maximum(1.4826 * mad, _MIN_SCALE)

    z = (core_x - center) / scale
    cov = _weighted_cov(z, core_w)
    try:
        eigvals, eigvecs = np.linalg.eigh(cov)
        direction = eigvecs[:, int(np.argmax(eigvals))]
        if not np.all(np.isfinite(direction)) or np.linalg.norm(direction) <= 0.0:
            direction = None
    except np.linalg.LinAlgError:
        direction = None

    if direction is None:
        direction = fallback["direction"].copy() if fallback is not None else np.ones(x.shape[1], dtype=np.float64)

    direction = direction / max(np.linalg.norm(direction), 1.0e-12)
    orient = global_direction
    if orient is None and fallback is not None:
        orient = fallback["direction"]
    if orient is not None and float(np.dot(direction, orient)) < 0.0:
        direction = -direction

    params = {"center": center, "scale": scale, "direction": direction}
    if fallback is not None:
        shrink = np.clip((effective - 30.0) / 100.0, 0.0, 1.0)
        if effective >= 80.0:
            shrink = max(shrink, 0.5)
        params = {
            "center": shrink * params["center"] + (1.0 - shrink) * fallback["center"],
            "scale": np.maximum(shrink * params["scale"] + (1.0 - shrink) * fallback["scale"], _MIN_SCALE),
            "direction": shrink * params["direction"] + (1.0 - shrink) * fallback["direction"],
        }
        params["direction"] = params["direction"] / max(np.linalg.norm(params["direction"]), 1.0e-12)
        if float(np.dot(params["direction"], fallback["direction"])) < 0.0:
            params["direction"] = -params["direction"]

    return params


def _tube_distance(x, params):
    z = (x - params["center"]) / params["scale"]
    direction = params["direction"]
    projection = np.sum(z * direction[None, :], axis=1)
    residual = z - projection[:, None] * direction[None, :]
    return np.clip(np.sqrt(np.sum(residual * residual, axis=1)), 0.0, _DISTANCE_CLIP)


def _bin_centers(index_values):
    finite = index_values[np.isfinite(index_values)]
    if finite.size == 0:
        return np.array([0.0], dtype=np.float64)
    low = float(np.min(finite))
    high = float(np.max(finite))
    if high <= low:
        return np.array([low], dtype=np.float64)
    count = int(np.floor((high - low) / _BIN_STEP)) + 1
    centers = low + np.arange(count + 1, dtype=np.float64) * _BIN_STEP
    return centers


def _local_tube_distances(x, index_values, strata_key=None):
    valid = np.all(np.isfinite(x), axis=1) & np.isfinite(index_values)
    distances = np.zeros(x.shape[0], dtype=np.float64)
    weight_sums = np.zeros(x.shape[0], dtype=np.float64)

    fallback = _tube_params(x[valid], None, None, None, None)
    if fallback is None:
        return np.zeros(x.shape[0], dtype=np.float64)

    centers = _bin_centers(index_values[valid])
    global_direction = fallback["direction"]

    for center_value in centers:
        weights_all = np.maximum(0.0, 1.0 - np.abs(index_values - center_value) / _KERNEL_WIDTH)
        fit_mask = valid & (weights_all > 0.0)
        if not np.any(fit_mask):
            continue

        weights_fit = weights_all[fit_mask]
        params = _tube_params(
            x[fit_mask],
            weights_fit,
            fallback=fallback,
            shrink_neff=float(weights_fit.sum()),
            global_direction=global_direction,
        )

        eval_mask = fit_mask
        distance = _tube_distance(x[eval_mask], params)
        distances[eval_mask] += weights_all[eval_mask] * distance
        weight_sums[eval_mask] += weights_all[eval_mask]

    global_dist = _tube_distance(x, fallback)
    out = np.where(weight_sums > 0.0, distances / np.maximum(weight_sums, 1.0e-12), global_dist)

    if strata_key is None:
        return out

    strata_distances = np.zeros(x.shape[0], dtype=np.float64)
    strata_weight_sums = np.zeros(x.shape[0], dtype=np.float64)
    keys = pd.Series(strata_key).astype("string").fillna("__missing__").to_numpy()

    for key in pd.unique(keys):
        key_mask = keys == key
        key_valid = valid & key_mask
        if key_valid.sum() < 30:
            continue

        key_fallback = _tube_params(
            x[key_valid],
            None,
            fallback=fallback,
            shrink_neff=float(key_valid.sum()),
            global_direction=global_direction,
        )

        for center_value in centers:
            weights_all = np.maximum(0.0, 1.0 - np.abs(index_values - center_value) / _KERNEL_WIDTH)
            fit_mask = key_valid & (weights_all > 0.0)
            if not np.any(fit_mask):
                continue

            weights_fit = weights_all[fit_mask]
            params = _tube_params(
                x[fit_mask],
                weights_fit,
                fallback=key_fallback,
                shrink_neff=float(weights_fit.sum()),
                global_direction=global_direction,
            )

            eval_mask = fit_mask
            distance = _tube_distance(x[eval_mask], params)
            strata_distances[eval_mask] += weights_all[eval_mask] * distance
            strata_weight_sums[eval_mask] += weights_all[eval_mask]

    return np.where(strata_weight_sums > 0.0, strata_distances / np.maximum(strata_weight_sums, 1.0e-12), out)


def add_sdss_quasar_targeting_surface(raw, deps, aux):
    index = raw.index

    u = _safe_series(raw, "u")
    g = _safe_series(raw, "g")
    r = _safe_series(raw, "r")
    i_mag = _safe_series(raw, "i")
    z_mag = _safe_series(raw, "z")
    redshift = _safe_series(raw, "redshift")

    c1 = u - g
    c2 = g - r
    c3 = r - i_mag
    c4 = i_mag - z_mag

    colors = np.column_stack([c1, c2, c3, c4])
    clipped = np.empty_like(colors)
    for j in range(colors.shape[1]):
        col = colors[:, j]
        finite = col[np.isfinite(col)]
        if finite.size:
            lower = float(np.quantile(finite, _COLOR_LOWER_Q))
            upper = float(np.quantile(finite, _COLOR_UPPER_Q))
            clipped[:, j] = np.clip(col, lower, upper)
        else:
            clipped[:, j] = 0.0

    c1c = clipped[:, 0]
    c2c = clipped[:, 1]
    c3c = clipped[:, 2]
    c4c = clipped[:, 3]

    cube_a = np.column_stack([c1c, c2c, c3c])
    cube_b = np.column_stack([c2c, c3c, c4c])

    spectral = raw["spectral_type"] if "spectral_type" in raw.columns else pd.Series("__missing__", index=index)
    population = raw["galaxy_population"] if "galaxy_population" in raw.columns else pd.Series("__missing__", index=index)
    strata = spectral.astype("string").fillna("__missing__") + "|" + population.astype("string").fillna("__missing__")

    d_a = _local_tube_distances(cube_a, c3c)
    d_b = _local_tube_distances(cube_b, c2c)
    d_a_stratum = _local_tube_distances(cube_a, c3c, strata)
    d_b_stratum = _local_tube_distances(cube_b, c2c, strata)

    o_a = _sigmoid((d_a - 4.0) / 0.45)
    o_b = _sigmoid((d_b - 4.0) / 0.45)
    o_a_stratum = _sigmoid((d_a_stratum - 4.0) / 0.45)
    o_b_stratum = _sigmoid((d_b_stratum - 4.0) / 0.45)
    o_max = np.maximum.reduce([o_a, o_b, o_a_stratum, o_b_stratum])
    o_diff_ab = o_a - o_b
    o_diff_stratum_ab = o_a_stratum - o_b_stratum

    low_mag = _sigmoid((i_mag - 15.0) / 0.25) * _sigmoid((19.1 - i_mag) / 0.60)
    high_mag = _sigmoid((i_mag - 15.0) / 0.25) * _sigmoid((20.2 - i_mag) / 0.60)
    low_z = _sigmoid((2.25 - redshift) / 0.35)
    mid_z_redshift = _sigmoid((redshift - 2.45) / 0.30) * _sigmoid((3.05 - redshift) / 0.30)
    high_z_redshift = _sigmoid((redshift - 3.0) / 0.35)

    uvx = low_mag * low_z * _sigmoid((0.60 - c1c) / 0.05)
    midz = low_mag * mid_z_redshift * _box(c1c, 0.65, 1.50, 0.07) * _box(c2c, 0.00, 0.20, 0.06)
    hiz = high_mag * high_z_redshift * _sigmoid((c1c - 1.50) / 0.08)

    wd = _box(c2c, -0.8, -0.2, 0.10) * _box(c3c, -0.6, -0.2, 0.10) * _box(c4c, -1.0, 0.0, 0.10)
    mwd = _box(c2c, 0.0, 1.6, 0.10) * _box(c3c, 0.6, 2.0, 0.10)
    a_rej = _box(c1c, 0.9, 1.5, 0.06) * _box(c2c, -0.35, 0.0, 0.06)
    blue_rej = _sigmoid((0.90 - c1c) / 0.08) * _sigmoid((0.80 - c2c) / 0.08) * _sigmoid((i_mag - 19.0) / 1.0)
    bright_rej = _sigmoid((15.0 - i_mag) / 0.25)

    low_qso = (
        low_mag
        * (1.0 - (1.0 - o_a) * (1.0 - uvx) * (1.0 - midz))
        * (1.0 - wd)
        * (1.0 - mwd)
        * (1.0 - a_rej)
        * (1.0 - bright_rej)
    )
    high_qso = (
        high_mag
        * (1.0 - (1.0 - o_b) * (1.0 - hiz))
        * (1.0 - blue_rej)
        * (1.0 - wd)
        * (1.0 - mwd)
        * (1.0 - a_rej)
        * (1.0 - bright_rej)
    )
    any_qso = 1.0 - (1.0 - low_qso) * (1.0 - high_qso)

    return pd.DataFrame(
        {
            "color_u_g_clip": c1c,
            "color_g_r_clip": c2c,
            "color_r_i_clip": c3c,
            "color_i_z_clip": c4c,
            "tube_dist_a": d_a,
            "tube_dist_b": d_b,
            "tube_dist_a_stratum": d_a_stratum,
            "tube_dist_b_stratum": d_b_stratum,
            "outlier_a": o_a,
            "outlier_b": o_b,
            "outlier_a_stratum": o_a_stratum,
            "outlier_b_stratum": o_b_stratum,
            "outlier_max": o_max,
            "outlier_diff_ab": o_diff_ab,
            "outlier_diff_stratum_ab": o_diff_stratum_ab,
            "low_mag_gate": low_mag,
            "high_mag_gate": high_mag,
            "low_redshift_gate": low_z,
            "mid_redshift_gate": mid_z_redshift,
            "high_redshift_gate": high_z_redshift,
            "uvx_surface": uvx,
            "midz_surface": midz,
            "hiz_surface": hiz,
            "wd_rejection": wd,
            "mwd_rejection": mwd,
            "a_rejection": a_rej,
            "blue_rejection": blue_rej,
            "bright_rejection": bright_rej,
            "low_qso_union": low_qso,
            "high_qso_union": high_qso,
            "any_qso_union": any_qso,
        },
        index=index,
    )


FEATURE_GROUPS = [
    {
        "name": "sdss_quasar_targeting_surface",
        "fn": add_sdss_quasar_targeting_surface,
        "depends_on": [],
        "description": "Fold-safe SDSS-inspired quasar targeting surfaces from smooth stellar-locus tube distances and soft color-redshift gates.",
    }
]