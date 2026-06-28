import numpy as np
import pandas as pd


REDSHIFT_BIN_EDGES = (-np.inf, 0.01, 0.1, 0.3, 0.5, 1.0, 2.0, np.inf)
COLOR_FEATURES = (
    "q_gr_ri",
    "q_ri_iz",
    "c_ug",
    "c_gr",
    "c_ri",
    "c_iz",
    "c_ur",
    "c_gi",
)
MIN_CONTEXT_COUNT = 60
MAD_SCALE = 1.4826
MAD_FLOOR = 1.0e-6
NSIDE = 32
N_LAT_BINS = 96
N_LON_BINS = 384


def _safe_numeric(frame, column):
    if column in frame.columns:
        return pd.to_numeric(frame[column], errors="coerce").astype(float)
    return pd.Series(np.nan, index=frame.index, dtype=float)


def _mad(values):
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.nan
    med = np.median(arr)
    return np.median(np.abs(arr - med))


def _make_stats(frame, group_cols):
    grouped = frame.groupby(group_cols, sort=False, observed=True)
    pieces = [grouped.size().rename("count")]
    for col in COLOR_FEATURES:
        pieces.append(grouped[col].median().rename(col + "_median"))
        pieces.append(grouped[col].agg(_mad).rename(col + "_mad"))
    out = pd.concat(pieces, axis=1).reset_index()
    return out


def _merge_stats(base, stats, on_cols, suffix):
    renamed = {}
    for col in stats.columns:
        if col not in on_cols:
            renamed[col] = col + suffix
    stats_use = stats.rename(columns=renamed)
    return base.merge(stats_use, on=on_cols, how="left", sort=False)


def _choose_first_available(frame, columns):
    result = pd.Series(np.nan, index=frame.index, dtype=float)
    for col in columns:
        result = result.where(result.notna(), frame[col])
    return result


def add_local_reddening_free_locus_offsets(raw, deps, aux):
    out_index = raw.index
    work = pd.DataFrame(index=out_index)

    alpha = _safe_numeric(raw, "alpha")
    delta = _safe_numeric(raw, "delta")
    redshift = _safe_numeric(raw, "redshift")

    u = _safe_numeric(raw, "u")
    g = _safe_numeric(raw, "g")
    r = _safe_numeric(raw, "r")
    i_mag = _safe_numeric(raw, "i")
    z = _safe_numeric(raw, "z")

    work["c_ug"] = u - g
    work["c_gr"] = g - r
    work["c_ri"] = r - i_mag
    work["c_iz"] = i_mag - z
    work["c_ur"] = u - r
    work["c_gi"] = g - i_mag
    work["q_gr_ri"] = work["c_gr"] - 1.582 * work["c_ri"]
    work["q_ri_iz"] = work["c_ri"] - 0.987 * work["c_iz"]

    lon = np.mod(alpha.to_numpy(dtype=float), 360.0)
    lat = np.clip(delta.to_numpy(dtype=float), -90.0, 90.0)
    lon_bin = np.floor(lon / 360.0 * N_LON_BINS).astype(int)
    lat_bin = np.floor((lat + 90.0) / 180.0 * N_LAT_BINS).astype(int)
    lon_bin = np.clip(lon_bin, 0, N_LON_BINS - 1)
    lat_bin = np.clip(lat_bin, 0, N_LAT_BINS - 1)

    work["sky_cell"] = lat_bin * N_LON_BINS + lon_bin
    work["context_cell"] = work["sky_cell"].astype(np.int64)

    rz = redshift.to_numpy(dtype=float)
    work["redshift_bin"] = pd.cut(
        rz,
        bins=list(REDSHIFT_BIN_EDGES),
        labels=False,
        right=False,
        include_lowest=True,
    ).astype(float)

    stats_input = work[list(COLOR_FEATURES) + ["context_cell", "redshift_bin"]].copy()

    local_bin_stats = _make_stats(stats_input, ["context_cell", "redshift_bin"])
    local_all_stats = _make_stats(stats_input, ["context_cell"])
    global_bin_stats = _make_stats(stats_input, ["redshift_bin"])

    full_global = {}
    full_global["count_full"] = float(len(stats_input))
    for col in COLOR_FEATURES:
        vals = stats_input[col].to_numpy(dtype=float)
        finite_vals = vals[np.isfinite(vals)]
        if finite_vals.size:
            full_global[col + "_median_full"] = float(np.median(finite_vals))
            full_global[col + "_mad_full"] = float(_mad(finite_vals))
        else:
            full_global[col + "_median_full"] = 0.0
            full_global[col + "_mad_full"] = 1.0

    merged = work[["context_cell", "redshift_bin"]].copy()
    merged["_row_order"] = np.arange(len(merged), dtype=np.int64)
    merged = _merge_stats(merged, local_bin_stats, ["context_cell", "redshift_bin"], "_local_bin")
    merged = _merge_stats(merged, local_all_stats, ["context_cell"], "_local_all")
    merged = _merge_stats(merged, global_bin_stats, ["redshift_bin"], "_global_bin")
    merged = merged.sort_values("_row_order").set_index(out_index)

    count_local_bin = merged["count_local_bin"].fillna(0.0)
    count_local_all = merged["count_local_all"].fillna(0.0)
    count_global_bin = merged["count_global_bin"].fillna(0.0)

    use_local_bin = count_local_bin >= MIN_CONTEXT_COUNT
    use_local_all = (~use_local_bin) & (count_local_all >= MIN_CONTEXT_COUNT)
    use_global_bin = (~use_local_bin) & (~use_local_all) & (count_global_bin >= MIN_CONTEXT_COUNT)

    result = pd.DataFrame(index=out_index)
    result["fallback_layer"] = np.select(
        [use_local_bin, use_local_all, use_global_bin],
        ["local_redshift_bin", "local_all_redshift", "global_redshift_bin"],
        default="full_global",
    )
    result["fallback_code"] = np.select(
        [use_local_bin, use_local_all, use_global_bin],
        [0, 1, 2],
        default=3,
    ).astype(np.int8)

    selected_count = np.select(
        [use_local_bin, use_local_all, use_global_bin],
        [count_local_bin, count_local_all, count_global_bin],
        default=full_global["count_full"],
    )
    result["locus_log1p_count"] = np.log1p(selected_count.astype(float))

    count_reference = merged[["redshift_bin", "count_global_bin"]].copy()
    count_reference["log_count"] = np.log1p(count_reference["count_global_bin"].fillna(full_global["count_full"]))
    bin_count_stats = count_reference.groupby("redshift_bin", sort=False, observed=True)["log_count"].agg(["median", _mad])
    bin_count_stats = bin_count_stats.rename(columns={"median": "count_log_median", "_mad": "count_log_mad"}).reset_index()
    merged_count = count_reference[["redshift_bin"]].merge(bin_count_stats, on="redshift_bin", how="left", sort=False)

    global_log_count = np.log1p(float(full_global["count_full"]))
    count_center = pd.Series(merged_count["count_log_median"].to_numpy(), index=out_index).fillna(global_log_count)
    count_mad = pd.Series(merged_count["count_log_mad"].to_numpy(), index=out_index).fillna(1.0)
    count_scale = np.maximum(count_mad.to_numpy(dtype=float) * MAD_SCALE, MAD_FLOOR)
    result["locus_log_count_robust_z"] = np.clip(
        (result["locus_log1p_count"].to_numpy(dtype=float) - count_center.to_numpy(dtype=float)) / count_scale,
        -8.0,
        8.0,
    )

    q_z_cols = []
    for col in COLOR_FEATURES:
        med = pd.Series(full_global[col + "_median_full"], index=out_index, dtype=float)
        mad = pd.Series(full_global[col + "_mad_full"], index=out_index, dtype=float)

        local_bin_ok = use_local_bin & merged[col + "_median_local_bin"].notna()
        local_all_ok = use_local_all & merged[col + "_median_local_all"].notna()
        global_bin_ok = use_global_bin & merged[col + "_median_global_bin"].notna()

        med = med.mask(global_bin_ok, merged[col + "_median_global_bin"])
        med = med.mask(local_all_ok, merged[col + "_median_local_all"])
        med = med.mask(local_bin_ok, merged[col + "_median_local_bin"])

        mad = mad.mask(global_bin_ok, merged[col + "_mad_global_bin"])
        mad = mad.mask(local_all_ok, merged[col + "_mad_local_all"])
        mad = mad.mask(local_bin_ok, merged[col + "_mad_local_bin"])

        scale = np.maximum(mad.to_numpy(dtype=float) * MAD_SCALE, MAD_FLOOR)
        z_score = np.clip((work[col].to_numpy(dtype=float) - med.to_numpy(dtype=float)) / scale, -8.0, 8.0)
        feature_name = "z_" + col
        result[feature_name] = z_score
        if col in ("q_gr_ri", "q_ri_iz"):
            q_z_cols.append(feature_name)

    result["q_locus_distance"] = np.sqrt(
        np.square(result[q_z_cols[0]].to_numpy(dtype=float))
        + np.square(result[q_z_cols[1]].to_numpy(dtype=float))
    )

    return result


FEATURE_GROUPS = [
    {
        "name": "local_reddening_free_locus_offsets",
        "fn": add_local_reddening_free_locus_offsets,
        "depends_on": [],
        "description": "Fold-safe sky-context reddening-free color locus residuals with robust fallback statistics.",
    }
]