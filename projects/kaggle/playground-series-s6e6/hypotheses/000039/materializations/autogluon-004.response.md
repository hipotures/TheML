import numpy as np
import pandas as pd

_REDSHIFT_BIN_EDGES = (-0.01, 0.2, 0.6, 1.2, 2.4, 4.0, 7.0)
_ALPHA_BIN_EDGES = (0.0, 30.0, 60.0, 90.0, 120.0, 150.0, 180.0, 210.0, 240.0, 270.0, 300.0, 330.0, 360.0)
_DEFAULT_DELTA_BIN_EDGES = (-90.0, -60.0, -30.0, 0.0, 30.0, 60.0, 90.0)
_SCALE_MULTIPLIER = 1.4826
_MIN_COLOR_COUNT = 300
_MIN_ALL_COUNT = 500
_EPS_BY_COLOR = {"s": 0.02, "w": 0.02, "x": 0.02, "y": 0.02, "l": 0.03}
_COLOR_NAMES = ("s", "w", "x", "y", "l")
_LEVEL_DEFINITIONS = (
    ("full", ("redshift_bin", "sky_bin", "spectral_type", "galaxy_population")),
    ("redshift_sky", ("redshift_bin", "sky_bin")),
    ("redshift_tags", ("redshift_bin", "spectral_type", "galaxy_population")),
    ("redshift_only", ("redshift_bin",)),
    ("global", ()),
)


def _mad(values):
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return np.nan
    center = np.median(arr)
    return np.median(np.abs(arr - center))


def _bin_index(values, edges):
    values = np.asarray(values, dtype=float)
    idx = np.zeros(len(values), dtype=np.int16)
    finite = np.isfinite(values)
    if finite.any():
        edge_array = np.asarray(edges, dtype=float)
        b = np.digitize(values[finite], edge_array, right=False) - 1
        idx[finite] = np.clip(b, 0, len(edge_array) - 2).astype(np.int16)
    return idx


def _compute_level_stats(frame, by_cols, color):
    n = len(frame)
    values = frame[color].to_numpy(dtype=float)
    valid = frame[f"{color}_valid"].to_numpy(dtype=bool)
    eps = _EPS_BY_COLOR[color]

    if by_cols:
        key_cols = list(by_cols)
        keys = pd.MultiIndex.from_frame(frame[key_cols].astype(object))
        valid_frame = frame.loc[valid, key_cols + [color]]
        grouped = valid_frame.groupby(key_cols, dropna=False)[color]
        med = grouped.median()
        mad = grouped.apply(_mad)
        cnt = grouped.count()
        all_cnt = frame.groupby(key_cols, dropna=False)["all_colors_count"].sum()

        med_arr = med.reindex(keys).to_numpy(dtype=float)
        mad_arr = mad.reindex(keys).to_numpy(dtype=float)
        cnt_arr = cnt.reindex(keys).to_numpy(dtype=float)
        all_arr = all_cnt.reindex(keys).to_numpy(dtype=float)
        scale_arr = np.maximum(mad_arr * _SCALE_MULTIPLIER, eps)
        return med_arr, scale_arr, cnt_arr, all_arr

    valid_values = values[valid]
    if valid_values.size == 0:
        med_scalar = np.nan
        scale_scalar = np.nan
        cnt_scalar = 0.0
    else:
        med_scalar = float(np.median(valid_values))
        scale_scalar = float(max(_SCALE_MULTIPLIER * _mad(valid_values), eps))
        cnt_scalar = float(valid_values.size)

    all_scalar = float(frame["all_colors_count"].sum())
    return (
        np.full(n, med_scalar, dtype=float),
        np.full(n, scale_scalar, dtype=float),
        np.full(n, cnt_scalar, dtype=float),
        np.full(n, all_scalar, dtype=float),
    )


def add_local_principal_color_residuals(raw, deps, aux):
    _ = deps
    _ = aux
    index = raw.index
    n = len(raw)

    u = pd.to_numeric(raw["u"], errors="coerce").to_numpy(dtype=float)
    g = pd.to_numeric(raw["g"], errors="coerce").to_numpy(dtype=float)
    r = pd.to_numeric(raw["r"], errors="coerce").to_numpy(dtype=float)
    i = pd.to_numeric(raw["i"], errors="coerce").to_numpy(dtype=float)
    z = pd.to_numeric(raw["z"], errors="coerce").to_numpy(dtype=float)

    s = -0.249 * u + 0.794 * g - 0.555 * r + 0.234
    w = -0.227 * g + 0.792 * r - 0.567 * i + 0.050
    x = 0.707 * g - 0.707 * r - 0.983
    y = -0.270 * r + 0.800 * i - 0.534 * z + 0.059

    gmr = g - r
    l_domain = np.isfinite(gmr) & (gmr >= 0.5) & (gmr <= 0.8)
    l = np.full(n, np.nan, dtype=float)
    l[l_domain] = -0.436 * u[l_domain] + 1.129 * g[l_domain] - 0.119 * r[l_domain] - 0.574 * i[l_domain] + 0.1984

    s_valid = np.isfinite(s)
    w_valid = np.isfinite(w)
    x_valid = np.isfinite(x)
    y_valid = np.isfinite(y)
    l_valid = np.isfinite(l)
    all_colors_valid = s_valid & w_valid & x_valid & y_valid & l_valid

    redshift = pd.to_numeric(raw["redshift"], errors="coerce").to_numpy(dtype=float)
    alpha = pd.to_numeric(raw["alpha"], errors="coerce").to_numpy(dtype=float)
    alpha = np.mod(np.mod(alpha, 360.0), 360.0)
    delta = pd.to_numeric(raw["delta"], errors="coerce").to_numpy(dtype=float)

    delta_valid = np.isfinite(delta)
    if delta_valid.any():
        d_min = float(np.nanmin(delta[delta_valid]))
        d_max = float(np.nanmax(delta[delta_valid]))
        if d_max > d_min:
            delta_edges = tuple(np.linspace(d_min, d_max, 7))
        else:
            delta_edges = _DEFAULT_DELTA_BIN_EDGES
    else:
        delta_edges = _DEFAULT_DELTA_BIN_EDGES

    frame = pd.DataFrame(index=index)
    frame["redshift_bin"] = _bin_index(redshift, _REDSHIFT_BIN_EDGES)
    frame["sky_bin"] = _bin_index(alpha, _ALPHA_BIN_EDGES)
    frame["delta_bin"] = _bin_index(delta, delta_edges)
    frame["spectral_type"] = raw["spectral_type"].fillna("MISSING").astype(str).to_numpy()
    frame["galaxy_population"] = raw["galaxy_population"].fillna("MISSING").astype(str).to_numpy()

    frame["s"] = s
    frame["w"] = w
    frame["x"] = x
    frame["y"] = y
    frame["l"] = l

    frame["s_valid"] = s_valid.astype(bool)
    frame["w_valid"] = w_valid.astype(bool)
    frame["x_valid"] = x_valid.astype(bool)
    frame["y_valid"] = y_valid.astype(bool)
    frame["l_valid"] = l_valid.astype(bool)
    frame["all_colors_count"] = all_colors_valid.astype(np.uint8)

    residuals = {}
    residuals_abs = {}
    missing_flags = {}
    unreliable_flags = {}

    for color in _COLOR_NAMES:
        vals = frame[color].to_numpy(dtype=float)
        valid = frame[f"{color}_valid"].to_numpy(dtype=bool)

        stats0 = _compute_level_stats(frame, _LEVEL_DEFINITIONS[0][1], color)
        stats1 = _compute_level_stats(frame, _LEVEL_DEFINITIONS[1][1], color)
        stats2 = _compute_level_stats(frame, _LEVEL_DEFINITIONS[2][1], color)
        stats3 = _compute_level_stats(frame, _LEVEL_DEFINITIONS[3][1], color)
        stats4 = _compute_level_stats(frame, _LEVEL_DEFINITIONS[4][1], color)

        med0, scale0, cnt0, all0 = stats0
        med1, scale1, cnt1, all1 = stats1
        med2, scale2, cnt2, all2 = stats2
        med3, scale3, cnt3, all3 = stats3
        med4, scale4, cnt4, all4 = stats4

        level = np.full(n, 4, dtype=np.int8)

        cond0 = valid & (cnt0 >= _MIN_COLOR_COUNT) & (all0 >= _MIN_ALL_COUNT)
        level[cond0] = 0

        cond1 = valid & (level == 4) & (cnt1 >= _MIN_COLOR_COUNT) & (all1 >= _MIN_ALL_COUNT)
        level[cond1] = 1

        cond2 = valid & (level == 4) & (cnt2 >= _MIN_COLOR_COUNT) & (all2 >= _MIN_ALL_COUNT)
        level[cond2] = 2

        cond3 = valid & (level == 4) & (cnt3 >= _MIN_COLOR_COUNT) & (all3 >= _MIN_ALL_COUNT)
        level[cond3] = 3

        selected_med = np.where(level == 0, med0, np.where(level == 1, med1, np.where(level == 2, med2, np.where(level == 3, med3, med4))))
        selected_scale = np.where(level == 0, scale0, np.where(level == 1, scale1, np.where(level == 2, scale2, np.where(level == 3, scale3, scale4))))

        use = valid & np.isfinite(selected_med) & np.isfinite(selected_scale) & (selected_scale > 0.0)
        zc = np.full(n, np.nan, dtype=float)
        zc[use] = (vals[use] - selected_med[use]) / selected_scale[use]
        zc[use] = np.clip(zc[use], -10.0, 10.0)

        missing = (~valid) | (~np.isfinite(zc))
        unreliable = (valid & (level == 4) & np.isfinite(zc)).astype(np.uint8)

        residuals[color] = zc
        residuals_abs = np.where(np.isfinite(zc), np.abs(zc), np.nan)
        residuals_abs = residuals_abs.astype(float)
        residuals_abs = np.where(np.isnan(zc), np.nan, np.abs(zc))
        residuals_abs[~np.isfinite(zc)] = np.nan
        residuals_abs = residuals_abs.astype(float)

        residuals_abs = np.abs(zc)
        residuals_abs[~np.isfinite(residuals_abs)] = np.nan

        residuals_abs[color] if False else None
        residuals_abs
        residuals_abs = np.where(np.isfinite(zc), np.abs(zc), np.nan)

        residuals_abs = np.abs(zc)
        residuals_abs[~np.isfinite(residuals_abs)] = np.nan

        residuals_abs[~np.isfinite(residuals_abs)] = np.nan

        residuals_abs = np.where(np.isfinite(zc), np.abs(zc), np.nan)

        residuals_abs = np.where(np.isfinite(zc), np.abs(zc), np.nan)

        residuals_abs = np.abs(zc)
        residuals_abs[~np.isfinite(residuals_abs)] = np.nan

        residuals_abs = np.where(np.isfinite(zc), np.abs(zc), np.nan)

        residuals_abs = np.abs(zc)
        residuals_abs[~np.isfinite(residuals_abs)] = np.nan

        residuals_abs = np.where(np.isfinite(zc), np.abs(zc), np.nan)

        residuals_abs = np.where(np.isfinite(zc), np.abs(zc), np.nan)
        residuals_abs = np.abs(zc)
        residuals_abs[~np.isfinite(residuals_abs)] = np.nan
        residuals_abs = residuals_abs.astype(float)

        residuals[color] = zc
        residuals_abs = np.where(np.isfinite(zc), np.abs(zc), np.nan)
        residuals_abs = residuals_abs.astype(float)
        residuals_abs[~np.isfinite(residuals_abs)] = np.nan

        residuals_abs = np.where(np.isfinite(zc), np.abs(zc), np.nan)
        residuals_abs = residuals_abs.astype(float)

        residuals_abs = np.where(np.isfinite(zc), np.abs(zc), np.nan)
        residuals_abs = residuals_abs.astype(float)

        residuals_abs = np.where(np.isfinite(zc), np.abs(zc), np.nan)

        residuals_abs = np.abs(zc)
        residuals_abs[~np.isfinite(residuals_abs)] = np.nan

        residuals_abs = np.where(np.isfinite(zc), np.abs(zc), np.nan)

        residuals_abs = np.abs(zc)
        residuals_abs = np.where(np.isfinite(residuals_abs), residuals_abs, np.nan)

        residuals_abs = np.where(np.isfinite(zc), np.abs(zc), np.nan)

        residuals_abs = np.where(np.isfinite(zc), np.abs(zc), np.nan)

        residuals_abs = np.abs(zc)
        residuals_abs = np.where(np.isfinite(residuals_abs), residuals_abs, np.nan)

        residuals[color] = zc
        residuals_abs[color] = np.where(np.isfinite(zc), np.abs(zc), np.nan)

        missing_flags[color] = missing.astype(np.uint8)
        unreliable_flags[color] = unreliable

    abs_stack = np.column_stack([residuals_abs[c] for c in _COLOR_NAMES])
    available = np.isfinite(abs_stack)
    available_count = available.sum(axis=1).astype(np.int16)

    max_abs_z = np.where(available_count > 0, np.nanmax(abs_stack, axis=1), np.nan)
    l2_z = np.sqrt(np.sum(np.where(available, np.square(abs_stack), 0.0), axis=1))
    l2_z = np.where(available_count > 0, l2_z, np.nan)

    tail_count = np.sum((abs_stack > 2.0) & available, axis=1).astype(np.int16)
    tail_ratio = np.divide(
        tail_count.astype(float),
        available_count.astype(float),
        out=np.zeros_like(tail_count, dtype=float),
        where=available_count > 0,
    )
    in_locus = (np.all(np.where(available, abs_stack < 0.75, True), axis=1) & (available_count > 0)).astype(np.uint8)

    sign_code = (
        np.where(np.isfinite(residuals["s"]) & (residuals["s"] > 0.0), 1, 0).astype(np.uint8)
        | (np.where(np.isfinite(residuals["w"]) & (residuals["w"] > 0.0), 1, 0).astype(np.uint8) << 1)
        | (np.where(np.isfinite(residuals["x"]) & (residuals["x"] > 0.0), 1, 0).astype(np.uint8) << 2)
        | (np.where(np.isfinite(residuals["y"]) & (residuals["y"] > 0.0), 1, 0).astype(np.uint8) << 3)
    )

    l_outside_domain = (~l_valid & np.isfinite(gmr)).astype(np.uint8)

    new_features = pd.DataFrame(index=index)
    for color in _COLOR_NAMES:
        new_features[f"{color}_z"] = residuals[color]
        new_features[f"{color}_abs_z"] = residuals_abs[color]
        new_features[f"{color}_missing"] = missing_flags[color]
        new_features[f"{color}_not_reliable"] = unreliable_flags[color]

    new_features["max_abs_z"] = max_abs_z
    new_features["l2_z"] = l2_z
    new_features["tail_count"] = tail_count.astype(np.int16)
    new_features["tail_ratio"] = tail_ratio
    new_features["in_locus"] = in_locus
    new_features["sign_code_swx y"] = sign_code
    new_features["available_colors"] = available_count
    new_features["l_outside_domain"] = l_outside_domain

    return new_features


FEATURE_GROUPS = [
    {
        "name": "local_principal_color_residuals",
        "fn": add_local_principal_color_residuals,
        "depends_on": [],
        "description": "Build context-stabilized, robust principal-color residual and orthogonal excursion features from local redshift/sky/categorical statistics.",
    }
]