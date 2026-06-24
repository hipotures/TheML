import numpy as np
import pandas as pd

_MS_PP_POLY_COEFFS = (0.0967, -1.267, 6.127, -12.97, 14.32, -5.06)
_MS_PP_N_BINS = 80
_MS_PP_MIN_BIN_N = 500
_MS_PP_EXPANSION_STEPS = (0, 1, 2, 3)
_MS_PP_MAD_SCALE = 1.4826
_MS_PP_EPSILON = 1e-6
_MS_PP_Z_CLIP = 10.0
_MS_PP_STAT_FEATURES = (
    "mu",
    "log_d",
    "abscolor_u_g",
    "abscolor_g_r",
    "abscolor_r_i",
    "abscolor_i_z",
)


def _main_sequence_Mr(x_c):
    c5, c4, c3, c2, c1, c0 = _MS_PP_POLY_COEFFS
    return (((((c5 * x_c + c4) * x_c + c3) * x_c + c2) * x_c + c1) * x_c + c0)


def _median_and_mad(values):
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.nan, np.nan
    med = float(np.median(finite))
    mad = float(np.median(np.abs(finite - med)) * _MS_PP_MAD_SCALE)
    return med, mad


def _robust_stats_for_indices(feature_matrix, indices):
    med = np.full(feature_matrix.shape[1], np.nan, dtype=float)
    mad = np.full(feature_matrix.shape[1], np.nan, dtype=float)
    if indices.size == 0:
        return med, mad

    for i in range(feature_matrix.shape[1]):
        med[i], mad[i] = _median_and_mad(feature_matrix[:, i][indices])

    return med, mad


def add_main_sequence_parallax_plausibility(raw, deps, aux):
    x = raw["g"].to_numpy(dtype=float) - raw["i"].to_numpy(dtype=float)
    x_below = x < 0.20
    x_above = x > 4.00
    x_turnoff_zone = x < 0.45
    x_c = np.clip(x, 0.20, 4.00)

    M_r = _main_sequence_Mr(x_c)
    mu = raw["r"].to_numpy(dtype=float) - M_r
    d_pc = np.power(10.0, (mu + 5.0) / 5.0)
    d_pc = np.clip(d_pc, 1.0, 1.0e10)
    log_d = np.log10(d_pc)

    M_u = raw["u"].to_numpy(dtype=float) - mu
    M_g = raw["g"].to_numpy(dtype=float) - mu
    M_i = raw["i"].to_numpy(dtype=float) - mu
    M_z = raw["z"].to_numpy(dtype=float) - mu

    abscolor_u_g = M_u - M_g
    abscolor_g_r = M_g - M_r
    abscolor_r_i = M_r - M_i
    abscolor_i_z = M_i - M_z

    feature_matrix = np.column_stack(
        (
            mu,
            log_d,
            abscolor_u_g,
            abscolor_g_r,
            abscolor_r_i,
            abscolor_i_z,
        )
    )
    n_rows = feature_matrix.shape[0]
    index = raw.index

    if n_rows == 0:
        return pd.DataFrame(
            {
                "x": np.array([], dtype=float),
                "x_c": np.array([], dtype=float),
                "x_below_0p20": np.array([], dtype=bool),
                "x_above_4p00": np.array([], dtype=bool),
                "x_turnoff_zone": np.array([], dtype=bool),
                "M_r": np.array([], dtype=float),
                "mu": np.array([], dtype=float),
                "log_d": np.array([], dtype=float),
                "abscolor_u_g": np.array([], dtype=float),
                "abscolor_g_r": np.array([], dtype=float),
                "abscolor_r_i": np.array([], dtype=float),
                "abscolor_i_z": np.array([], dtype=float),
                "z_mu": np.array([], dtype=float),
                "z_log_d": np.array([], dtype=float),
                "z_abscolor_u_g": np.array([], dtype=float),
                "z_abscolor_g_r": np.array([], dtype=float),
                "z_abscolor_r_i": np.array([], dtype=float),
                "z_abscolor_i_z": np.array([], dtype=float),
                "mu_within_bin_percentile": np.array([], dtype=float),
                "used_bin_n": np.array([], dtype=int),
            },
            index=index,
        )

    try:
        x_c_series = pd.Series(x_c, index=index, name="x_c")
        bin_series = pd.qcut(
            x_c_series,
            q=_MS_PP_N_BINS,
            labels=False,
            duplicates="drop",
        )
    except (ValueError, TypeError):
        bin_series = pd.Series(np.zeros(n_rows, dtype="float64"), index=index)

    bin_arr = np.full(n_rows, -1, dtype=int)
    valid_bins = bin_series.notna().to_numpy()
    if valid_bins.any():
        bin_arr[valid_bins] = bin_series.to_numpy(dtype=float)[valid_bins].astype(int)

    n_bins = 0
    if valid_bins.any():
        n_bins = int(np.max(bin_arr[valid_bins])) + 1

    bin_members = [np.flatnonzero(bin_arr == b) for b in range(n_bins)]
    bin_counts = np.array([len(m) for m in bin_members], dtype=int)
    bin_prefix = np.zeros(n_bins + 1, dtype=int)
    if n_bins > 0:
        bin_prefix[1:] = np.cumsum(bin_counts)

    global_medians, global_mads = _robust_stats_for_indices(feature_matrix, np.arange(n_rows, dtype=int))

    window_stats_medians = [global_medians]
    window_stats_mads = [global_mads]
    window_stats_counts = [int(n_rows)]
    window_key_to_id = {("global",): 0}

    bin_window_id = np.zeros(n_bins, dtype=int)
    bin_window_count = np.full(n_bins, int(n_rows), dtype=int)
    window_cache = {}

    for b in range(n_bins):
        selected_id = 0
        selected_count = int(n_rows)
        for expand in _MS_PP_EXPANSION_STEPS:
            left = max(0, b - expand)
            right = min(n_bins - 1, b + expand)
            candidate_count = int(bin_prefix[right + 1] - bin_prefix[left])

            if candidate_count >= _MS_PP_MIN_BIN_N:
                key = (left, right)
                if key not in window_cache:
                    idx_parts = [bin_members[k] for k in range(left, right + 1) if bin_members[k].size]
                    if idx_parts:
                        idx = np.concatenate(idx_parts)
                    else:
                        idx = np.empty(0, dtype=int)
                    m, d = _robust_stats_for_indices(feature_matrix, idx)
                    window_cache[key] = (len(window_stats_medians), m, d, int(idx.size))
                    window_stats_medians.append(m)
                    window_stats_mads.append(d)
                    window_stats_counts.append(int(idx.size))
                selected_id = window_cache[key][0]
                selected_count = candidate_count
                break

        bin_window_id[b] = selected_id
        bin_window_count[b] = selected_count

    row_window_id = np.zeros(n_rows, dtype=int)
    row_window_count = np.full(n_rows, int(n_rows), dtype=int)
    for b in range(n_bins):
        rows = bin_members[b]
        if rows.size:
            row_window_id[rows] = bin_window_id[b]
            row_window_count[rows] = bin_window_count[b]

    med_grid = np.asarray(window_stats_medians, dtype=float)
    mad_grid = np.asarray(window_stats_mads, dtype=float)

    med_by_row = med_grid[row_window_id]
    mad_by_row = mad_grid[row_window_id]
    z = (feature_matrix - med_by_row) / (mad_by_row + _MS_PP_EPSILON)
    z = np.clip(z, -_MS_PP_Z_CLIP, _MS_PP_Z_CLIP)

    mu_pct = pd.Series(mu, index=index, dtype=float).groupby(
        pd.Series(row_window_id, index=index, dtype=int)
    ).rank(pct=True, method="average")

    return pd.DataFrame(
        {
            "x": x,
            "x_c": x_c,
            "x_below_0p20": x_below,
            "x_above_4p00": x_above,
            "x_turnoff_zone": x_turnoff_zone,
            "M_r": M_r,
            "mu": mu,
            "log_d": log_d,
            "abscolor_u_g": abscolor_u_g,
            "abscolor_g_r": abscolor_g_r,
            "abscolor_r_i": abscolor_r_i,
            "abscolor_i_z": abscolor_i_z,
            "z_mu": z[:, 0],
            "z_log_d": z[:, 1],
            "z_abscolor_u_g": z[:, 2],
            "z_abscolor_g_r": z[:, 3],
            "z_abscolor_r_i": z[:, 4],
            "z_abscolor_i_z": z[:, 5],
            "mu_within_bin_percentile": mu_pct.to_numpy(),
            "used_bin_n": row_window_count.astype(float),
        },
        index=index,
    )


FEATURE_GROUPS = [
    {
        "name": "main_sequence_parallax_plausibility",
        "fn": add_main_sequence_parallax_plausibility,
        "depends_on": [],
        "description": "Builds main-sequence-plausibility residuals from ugriz photometry using quantile-binned robust scaling with sparse-bin expansion and global fallback.",
    }
]