import numpy as np
import pandas as pd

PHOTOZ_COLOR_CONSISTENCY_GROUP = "photoz_color_consistency"
PRIMARY_GRID_BINS = 20
PRIMARY_GRID_MIN_COUNT = 30
PAIRWISE_GRID_BINS = 24
PAIRWISE_GRID_MIN_COUNT = 50


def _sanitize_series(values):
    series = pd.to_numeric(values, errors="coerce")
    arr = series.to_numpy(dtype=np.float64, copy=True)
    finite = np.isfinite(arr)
    if not finite.any():
        return pd.Series(np.zeros_like(arr, dtype=np.float64), index=series.index)
    lo = np.nanmin(arr[finite])
    hi = np.nanmax(arr[finite])
    arr[np.isposinf(arr)] = hi
    arr[np.isneginf(arr)] = lo
    arr[~np.isfinite(arr)] = lo
    return pd.Series(arr, index=series.index, name=series.name)


def _quantile_bin_codes(values, q_bins):
    if len(values) == 0:
        return pd.Series(np.array([], dtype=np.int32), index=values.index, name="qbin")
    if values.nunique(dropna=True) <= 1:
        return pd.Series(np.zeros(len(values), dtype=np.int32), index=values.index, name="qbin")

    try:
        codes = pd.qcut(values, q=q_bins, labels=False, duplicates="drop")
    except (ValueError, TypeError):
        return pd.Series(np.full(len(values), -1, dtype=np.int32), index=values.index, name="qbin")

    codes = codes.fillna(-1).astype(np.int32)
    return codes


def _build_lookup_table(redshift, bins, grid_columns, min_count):
    frame = bins.copy()
    frame["redshift"] = redshift.to_numpy(dtype=np.float64, copy=False)
    valid = frame[grid_columns].ge(0).all(axis=1) & np.isfinite(frame["redshift"])
    frame = frame.loc[valid, :]
    if frame.empty:
        out = pd.DataFrame(columns=grid_columns + ["median", "iqr", "count"])
        return out.set_index(grid_columns)

    grouped = (
        frame
        .groupby(grid_columns, sort=False)["redshift"]
        .agg(
            median="median",
            q25=lambda x: x.quantile(0.25),
            q75=lambda x: x.quantile(0.75),
            count="size",
        )
    )
    grouped = grouped[grouped["count"] >= min_count]
    if grouped.empty:
        out = pd.DataFrame(columns=grid_columns + ["median", "iqr", "count"])
        return out.set_index(grid_columns)

    grouped["iqr"] = grouped["q75"] - grouped["q25"]
    return grouped[["median", "iqr", "count"]]


def _lookup_from_table(bin_frame, lookup):
    n = len(bin_frame)
    if lookup.empty:
        nan = np.full(n, np.nan, dtype=np.float64)
        return nan, nan, nan

    idx = pd.MultiIndex.from_frame(bin_frame)
    aligned = lookup.reindex(idx)
    med = aligned["median"].to_numpy(dtype=np.float64)
    iqr = aligned["iqr"].to_numpy(dtype=np.float64)
    cnt = aligned["count"].to_numpy(dtype=np.float64)
    return med, iqr, cnt


def _global_redshift_stats(redshift):
    vals = redshift.to_numpy(dtype=np.float64, copy=False)
    finite = np.isfinite(vals)
    if not finite.any():
        return 0.0, 0.0, 0.0

    vals = vals[finite]
    med = float(np.median(vals))
    q25 = float(np.quantile(vals, 0.25))
    q75 = float(np.quantile(vals, 0.75))
    return med, float(q75 - q25), float(vals.size)


def add_photoz_color_consistency(raw, deps, aux):
    if raw.empty:
        return pd.DataFrame(index=raw.index)

    z = _sanitize_series(raw["redshift"])
    u = _sanitize_series(raw["u"])
    g = _sanitize_series(raw["g"])
    r = _sanitize_series(raw["r"])
    i = _sanitize_series(raw["i"])
    z_band = _sanitize_series(raw["z"])

    u_g = _sanitize_series(u - g)
    g_r = _sanitize_series(g - r)
    r_i = _sanitize_series(r - i)
    i_z = _sanitize_series(i - z_band)

    p_b1 = _quantile_bin_codes(g_r, PRIMARY_GRID_BINS)
    p_b2 = _quantile_bin_codes(r_i, PRIMARY_GRID_BINS)
    p_b3 = _quantile_bin_codes(r, PRIMARY_GRID_BINS)

    p_lookup = _build_lookup_table(
        z,
        pd.DataFrame({"g_r": p_b1, "r_i": p_b2, "r": p_b3}),
        ["g_r", "r_i", "r"],
        PRIMARY_GRID_MIN_COUNT,
    )
    p_med, p_iqr, p_cnt = _lookup_from_table(
        pd.DataFrame({"g_r": p_b1, "r_i": p_b2, "r": p_b3}),
        p_lookup,
    )

    b1 = _quantile_bin_codes(g_r, PAIRWISE_GRID_BINS)
    b2 = _quantile_bin_codes(r_i, PAIRWISE_GRID_BINS)
    b3 = _quantile_bin_codes(u_g, PAIRWISE_GRID_BINS)
    b4 = _quantile_bin_codes(i_z, PAIRWISE_GRID_BINS)

    f1_lookup = _build_lookup_table(
        z,
        pd.DataFrame({"g_r": b1, "r_i": b2}),
        ["g_r", "r_i"],
        PAIRWISE_GRID_MIN_COUNT,
    )
    f2_lookup = _build_lookup_table(
        z,
        pd.DataFrame({"u_g": b3, "g_r": b1}),
        ["u_g", "g_r"],
        PAIRWISE_GRID_MIN_COUNT,
    )
    f3_lookup = _build_lookup_table(
        z,
        pd.DataFrame({"r_i": b2, "i_z": b4}),
        ["r_i", "i_z"],
        PAIRWISE_GRID_MIN_COUNT,
    )

    f1_med, f1_iqr, f1_cnt = _lookup_from_table(
        pd.DataFrame({"g_r": b1, "r_i": b2}),
        f1_lookup,
    )
    f2_med, f2_iqr, f2_cnt = _lookup_from_table(
        pd.DataFrame({"u_g": b3, "g_r": b1}),
        f2_lookup,
    )
    f3_med, f3_iqr, f3_cnt = _lookup_from_table(
        pd.DataFrame({"r_i": b2, "i_z": b4}),
        f3_lookup,
    )

    global_med, global_iqr, global_cnt = _global_redshift_stats(z)
    n = len(raw)
    local_med = np.full(n, global_med, dtype=np.float64)
    local_iqr = np.full(n, global_iqr, dtype=np.float64)
    local_cnt = np.full(n, global_cnt, dtype=np.float64)

    is_primary = np.isfinite(p_med)
    is_pairwise = np.zeros(n, dtype=bool)

    local_med[is_primary] = p_med[is_primary]
    local_iqr[is_primary] = np.where(np.isfinite(p_iqr), p_iqr, global_iqr)[is_primary]
    local_cnt[is_primary] = np.where(np.isfinite(p_cnt), p_cnt, global_cnt)[is_primary]

    unresolved = ~is_primary
    use_f1 = unresolved & np.isfinite(f1_med)
    local_med[use_f1] = f1_med[use_f1]
    local_iqr[use_f1] = np.where(np.isfinite(f1_iqr), f1_iqr, global_iqr)[use_f1]
    local_cnt[use_f1] = np.where(np.isfinite(f1_cnt), f1_cnt, global_cnt)[use_f1]
    is_pairwise[use_f1] = True
    unresolved = unresolved & ~use_f1

    use_f2 = unresolved & np.isfinite(f2_med)
    local_med[use_f2] = f2_med[use_f2]
    local_iqr[use_f2] = np.where(np.isfinite(f2_iqr), f2_iqr, global_iqr)[use_f2]
    local_cnt[use_f2] = np.where(np.isfinite(f2_cnt), f2_cnt, global_cnt)[use_f2]
    is_pairwise[use_f2] = True
    unresolved = unresolved & ~use_f2

    use_f3 = unresolved & np.isfinite(f3_med)
    local_med[use_f3] = f3_med[use_f3]
    local_iqr[use_f3] = np.where(np.isfinite(f3_iqr), f3_iqr, global_iqr)[use_f3]
    local_cnt[use_f3] = np.where(np.isfinite(f3_cnt), f3_cnt, global_cnt)[use_f3]
    is_pairwise[use_f3] = True

    z_obs = z.to_numpy(dtype=np.float64, copy=False)
    residual = z_obs - local_med
    abs_residual = np.abs(residual)
    norm_residual = residual / np.maximum(local_iqr, 1e-3)

    is_global = ~(is_primary | is_pairwise)

    return pd.DataFrame(
        {
            "photoz_color_consistency_residual_signed": residual,
            "photoz_color_consistency_residual_abs": abs_residual,
            "photoz_color_consistency_residual_over_local_iqr": norm_residual,
            "photoz_color_consistency_local_log_count": np.log1p(local_cnt),
            "photoz_color_consistency_local_iqr": local_iqr,
            "photoz_color_consistency_source_primary": is_primary.astype(np.int8),
            "photoz_color_consistency_source_pairwise": is_pairwise.astype(np.int8),
            "photoz_color_consistency_source_global": is_global.astype(np.int8),
        },
        index=raw.index,
    )


FEATURE_GROUPS = [
    {
        "name": PHOTOZ_COLOR_CONSISTENCY_GROUP,
        "fn": add_photoz_color_consistency,
        "depends_on": [],
        "description": "Computes train-defined color-redshift lookup residual, ambiguity, and source-indicator features for photometric redshift consistency checks.",
    }
]