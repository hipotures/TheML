import numpy as np
import pandas as pd

_BIN_COUNT = 24
_BIN_COUNT_MIN_POINTS = 120
_MAD_SCALE = 1.4826
_EPS = 1e-12
_RESIDUAL_CLIP = 8.0
_UG_SPLIT_BASE = 2.22
_POPULATION_RED = "Red_Sequence"
_POPULATION_BLUE = "Blue_Cloud"
_COLOR_FEATURES = ["c_ug", "c_gr", "c_ri", "c_iz", "c_ur"]

def _to_numeric_frame(frame, columns):
    numeric = pd.DataFrame(index=frame.index)
    for col in columns:
        numeric[col] = pd.to_numeric(frame[col], errors="coerce")
    return numeric

def _build_colors(frame):
    phot = _to_numeric_frame(frame, ("u", "g", "r", "i", "z"))
    return pd.DataFrame(
        {
            "c_ug": phot["u"] - phot["g"],
            "c_gr": phot["g"] - phot["r"],
            "c_ri": phot["r"] - phot["i"],
            "c_iz": phot["i"] - phot["z"],
            "c_ur": phot["u"] - phot["r"],
        },
        index=frame.index,
    )

def _mad(values):
    arr = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.nan
    med = np.nanmedian(arr)
    return float(np.nanmedian(np.abs(arr - med)) * _MAD_SCALE)

def _build_redshift_bins(redshift):
    z = pd.to_numeric(redshift, errors="coerce").astype(float)
    z_values = z.to_numpy(dtype=float)
    finite = np.isfinite(z_values)
    if not finite.any():
        return pd.Series(np.zeros(len(z), dtype=np.int16), index=z.index)

    z_max = float(np.nanmax(z_values[finite]))
    if z_max < 0.0:
        z_max = 0.0

    z_clipped = np.clip(z_values, 0.0, z_max)
    logz = pd.Series(np.log1p(z_clipped), index=z.index)
    logz_finite = logz[np.isfinite(logz)]

    if logz_finite.size <= 1:
        return pd.Series(np.zeros(len(z), dtype=np.int16), index=z.index)

    probs = np.linspace(0.0, 1.0, _BIN_COUNT + 1)
    edges = np.quantile(logz_finite.to_numpy(dtype=float), probs)
    edges = np.array(edges, dtype=float)
    edges = np.sort(np.unique(edges))
    if edges.size < 2:
        return pd.Series(np.zeros(len(z), dtype=np.int16), index=z.index)

    bins = pd.cut(
        logz,
        bins=edges,
        labels=False,
        include_lowest=True,
        duplicates="drop",
    )
    return bins.fillna(0).astype(np.int16)

def _rowwise_global_pop_stats(global_by_pop, global_all, population):
    aligned = global_by_pop.reindex(population.to_numpy(dtype=object))
    if aligned is None:
        aligned = pd.DataFrame(index=population.index, columns=_COLOR_FEATURES, dtype=float)
    else:
        if list(aligned.columns) != list(_COLOR_FEATURES):
            aligned = aligned.reindex(columns=_COLOR_FEATURES)
    aligned = aligned.fillna(global_all)
    return aligned.astype(float)

def _blend_by_cell_weight(local_df, global_df, cell_counts, n_min):
    local = local_df.to_numpy(dtype=float)
    global_vals = global_df.to_numpy(dtype=float)
    counts = cell_counts.to_numpy(dtype=float)
    weight = counts / (counts + float(n_min))
    weight = weight.reshape(-1, 1)

    local = np.where(np.isfinite(local), local, global_vals)
    blended = (weight * local) + ((1.0 - weight) * global_vals)
    return pd.DataFrame(blended, index=local_df.index, columns=local_df.columns)

def _residual_matrix(values, medians, scales):
    v = values.to_numpy(dtype=float)
    mu = medians.to_numpy(dtype=float)
    s = scales.to_numpy(dtype=float)
    s = np.where(np.abs(s) >= _EPS, s, np.nan)
    residual = (v - mu) / s
    residual = np.where(np.isfinite(residual), np.clip(residual, -_RESIDUAL_CLIP, _RESIDUAL_CLIP), 0.0)
    return residual

def _aux_global_color_stats(aux):
    if aux is None or len(aux) == 0:
        return None
    required = ("u", "g", "r", "i", "z")
    if not all(col in aux.columns for col in required):
        return None

    colors = _build_colors(aux)
    colors = colors.replace([np.inf, -np.inf], np.nan)
    finite_rows = np.isfinite(colors.to_numpy(dtype=float)).all(axis=1)
    colors = colors.loc[finite_rows]
    if colors.empty:
        return None

    med = colors.median()
    mad = colors.apply(_mad)
    if med.isna().all() or mad.isna().all():
        return None

    return {"med": med, "mad": mad, "n": float(colors.shape[0])}

def _blend_global_with_aux(global_med, global_scale, aux):
    aux_stats = _aux_global_color_stats(aux)
    if aux_stats is None:
        return global_med, global_scale

    n = float(aux_stats["n"])
    if not np.isfinite(n) or n <= 0.0:
        return global_med, global_scale

    aux_weight = min(0.2, n / (n + 30000.0))
    if aux_weight <= 0.0:
        return global_med, global_scale

    med = ((1.0 - aux_weight) * global_med) + (aux_weight * aux_stats["med"])
    scale = ((1.0 - aux_weight) * global_scale) + (aux_weight * aux_stats["mad"])
    return med, scale

def add_population_color_manifold_drifts(raw, deps, aux):
    colors = _build_colors(raw)
    redshift_bins = _build_redshift_bins(raw["redshift"])
    population = raw["galaxy_population"].astype("object")

    stats = colors.copy()
    stats["z_bin"] = redshift_bins
    stats["galaxy_population"] = population

    grouped = stats.groupby(["z_bin", "galaxy_population"], sort=False)
    local_median = grouped[_COLOR_FEATURES].median()
    local_scale = grouped[_COLOR_FEATURES].agg(_mad)

    global_by_pop_median = stats.groupby("galaxy_population", sort=False)[_COLOR_FEATURES].median()
    global_by_pop_scale = stats.groupby("galaxy_population", sort=False)[_COLOR_FEATURES].agg(_mad)

    global_all_median = stats[_COLOR_FEATURES].median()
    global_all_scale = stats[_COLOR_FEATURES].agg(_mad)
    global_all_median, global_all_scale = _blend_global_with_aux(global_all_median, global_all_scale, aux)

    row_key = pd.MultiIndex.from_arrays(
        [stats["z_bin"].to_numpy(dtype=int), population.to_numpy(dtype=object)],
        names=("z_bin", "galaxy_population"),
    )
    opp_pop = np.where(
        population.to_numpy(dtype=object) == _POPULATION_RED,
        _POPULATION_BLUE,
        np.where(population.to_numpy(dtype=object) == _POPULATION_BLUE, _POPULATION_RED, population.to_numpy(dtype=object)),
    )
    opp_key = pd.MultiIndex.from_arrays(
        [stats["z_bin"].to_numpy(dtype=int), opp_pop],
        names=("z_bin", "galaxy_population"),
    )

    cell_count_assigned = grouped.size().reindex(row_key).fillna(0.0)
    cell_count_opposite = grouped.size().reindex(opp_key).fillna(0.0)

    assigned_global_median = _rowwise_global_pop_stats(global_by_pop_median, global_all_median, population)
    assigned_global_scale = _rowwise_global_pop_stats(global_by_pop_scale, global_all_scale, population)
    opposite_global_median = _rowwise_global_pop_stats(global_by_pop_median, global_all_median, pd.Series(opp_pop, index=raw.index))
    opposite_global_scale = _rowwise_global_pop_stats(global_by_pop_scale, global_all_scale, pd.Series(opp_pop, index=raw.index))

    assigned_med = local_median.reindex(row_key)
    assigned_scale = local_scale.reindex(row_key)
    opposite_med = local_median.reindex(opp_key)
    opposite_scale = local_scale.reindex(opp_key)

    assigned_med = _blend_by_cell_weight(assigned_med, assigned_global_median, cell_count_assigned, _BIN_COUNT_MIN_POINTS)
    assigned_scale = _blend_by_cell_weight(assigned_scale, assigned_global_scale, cell_count_assigned, _BIN_COUNT_MIN_POINTS)
    opposite_med = _blend_by_cell_weight(opposite_med, opposite_global_median, cell_count_opposite, _BIN_COUNT_MIN_POINTS)
    opposite_scale = _blend_by_cell_weight(opposite_scale, opposite_global_scale, cell_count_opposite, _BIN_COUNT_MIN_POINTS)

    res_assigned = _residual_matrix(colors[_COLOR_FEATURES], assigned_med, assigned_scale)
    res_opposite = _residual_matrix(colors[_COLOR_FEATURES], opposite_med, opposite_scale)

    d_assigned = np.sum(res_assigned * res_assigned, axis=1)
    d_opposite = np.sum(res_opposite * res_opposite, axis=1)
    d_gap = d_assigned - d_opposite
    d_ratio = d_assigned / (d_opposite + 1.0)

    weight_assigned = (cell_count_assigned.to_numpy(dtype=float) / (cell_count_assigned.to_numpy(dtype=float) + _BIN_COUNT_MIN_POINTS))
    weight_opposite = (cell_count_opposite.to_numpy(dtype=float) / (cell_count_opposite.to_numpy(dtype=float) + _BIN_COUNT_MIN_POINTS))

    max_bin = int(np.nanmax(redshift_bins.to_numpy(dtype=float))) if len(redshift_bins) > 0 else 0
    if max_bin < 0:
        max_bin = 0
    bin_axis = pd.Index(range(max_bin + 1), name="z_bin")
    bin_lookup = pd.Series(redshift_bins.astype(int).to_numpy(dtype=int), index=raw.index)

    med_by_bin_ug = local_median["c_ug"].unstack("galaxy_population")
    med_by_bin_gr = local_median["c_gr"].unstack("galaxy_population")
    med_by_bin_ri = local_median["c_ri"].unstack("galaxy_population")
    med_by_bin_iz = local_median["c_iz"].unstack("galaxy_population")

    red_ug = med_by_bin_ug.get(_POPULATION_RED, pd.Series(index=med_by_bin_ug.index, dtype=float))
    blue_ug = med_by_bin_ug.get(_POPULATION_BLUE, pd.Series(index=med_by_bin_ug.index, dtype=float))
    red_gr = med_by_bin_gr.get(_POPULATION_RED, pd.Series(index=med_by_bin_gr.index, dtype=float))
    blue_gr = med_by_bin_gr.get(_POPULATION_BLUE, pd.Series(index=med_by_bin_gr.index, dtype=float))
    red_ri = med_by_bin_ri.get(_POPULATION_RED, pd.Series(index=med_by_bin_ri.index, dtype=float))
    blue_ri = med_by_bin_ri.get(_POPULATION_BLUE, pd.Series(index=med_by_bin_ri.index, dtype=float))
    red_iz = med_by_bin_iz.get(_POPULATION_RED, pd.Series(index=med_by_bin_iz.index, dtype=float))
    blue_iz = med_by_bin_iz.get(_POPULATION_BLUE, pd.Series(index=med_by_bin_iz.index, dtype=float))

    red_ug = red_ug.reindex(bin_axis)
    blue_ug = blue_ug.reindex(bin_axis)
    red_gr = red_gr.reindex(bin_axis)
    blue_gr = blue_gr.reindex(bin_axis)
    red_ri = red_ri.reindex(bin_axis)
    blue_ri = blue_ri.reindex(bin_axis)
    red_iz = red_iz.reindex(bin_axis)
    blue_iz = blue_iz.reindex(bin_axis)

    red_ug_global = float(global_by_pop_median.loc[_POPULATION_RED, "c_ug"]) if _POPULATION_RED in global_by_pop_median.index else float(global_all_median["c_ug"])
    blue_ug_global = float(global_by_pop_median.loc[_POPULATION_BLUE, "c_ug"]) if _POPULATION_BLUE in global_by_pop_median.index else float(global_all_median["c_ug"])
    red_gr_global = float(global_by_pop_median.loc[_POPULATION_RED, "c_gr"]) if _POPULATION_RED in global_by_pop_median.index else float(global_all_median["c_gr"])
    blue_gr_global = float(global_by_pop_median.loc[_POPULATION_BLUE, "c_gr"]) if _POPULATION_BLUE in global_by_pop_median.index else float(global_all_median["c_gr"])
    red_ri_global = float(global_by_pop_median.loc[_POPULATION_RED, "c_ri"]) if _POPULATION_RED in global_by_pop_median.index else float(global_all_median["c_ri"])
    blue_ri_global = float(global_by_pop_median.loc[_POPULATION_BLUE, "c_ri"]) if _POPULATION_BLUE in global_by_pop_median.index else float(global_all_median["c_ri"])
    red_iz_global = float(global_by_pop_median.loc[_POPULATION_RED, "c_iz"]) if _POPULATION_RED in global_by_pop_median.index else float(global_all_median["c_iz"])
    blue_iz_global = float(global_by_pop_median.loc[_POPULATION_BLUE, "c_iz"]) if _POPULATION_BLUE in global_by_pop_median.index else float(global_all_median["c_iz"])

    red_ug = red_ug.fillna(red_ug_global)
    blue_ug = blue_ug.fillna(blue_ug_global)
    red_gr = red_gr.fillna(red_gr_global)
    blue_gr = blue_gr.fillna(blue_gr_global)
    red_ri = red_ri.fillna(red_ri_global)
    blue_ri = blue_ri.fillna(blue_ri_global)
    red_iz = red_iz.fillna(red_iz_global)
    blue_iz = blue_iz.fillna(blue_iz_global)

    gap_ug = red_ug - blue_ug
    ug_split_gap_by_row = bin_lookup.map(gap_ug)
    ug_to_red = bin_lookup.map(red_ug)
    ug_to_blue = bin_lookup.map(blue_ug)
    gr_to_red = bin_lookup.map(red_gr)
    gr_to_blue = bin_lookup.map(blue_gr)
    ri_to_red = bin_lookup.map(red_ri)
    ri_to_blue = bin_lookup.map(blue_ri)
    iz_to_red = bin_lookup.map(red_iz)
    iz_to_blue = bin_lookup.map(blue_iz)

    m_ug_split = colors["c_ug"] - (_UG_SPLIT_BASE + ug_split_gap_by_row)
    m_gr_to_red = colors["c_gr"] - gr_to_red
    m_gr_to_blue = colors["c_gr"] - gr_to_blue
    m_ri_to_red = colors["c_ri"] - ri_to_red
    m_ri_to_blue = colors["c_ri"] - ri_to_blue
    m_iz_to_red = colors["c_iz"] - iz_to_red
    m_iz_to_blue = colors["c_iz"] - iz_to_blue

    features = pd.DataFrame(index=raw.index)
    features["population_color_manifold_dist_assigned"] = d_assigned
    features["population_color_manifold_dist_opposite"] = d_opposite
    features["population_color_manifold_dist_gap"] = d_gap
    features["population_color_manifold_dist_ratio"] = d_ratio
    features["population_color_manifold_bin_weight_assigned"] = weight_assigned
    features["population_color_manifold_bin_weight_opposite"] = weight_opposite

    features["population_color_manifold_ug_split_margin"] = m_ug_split
    features["population_color_manifold_m_gr_red"] = m_gr_to_red
    features["population_color_manifold_m_gr_blue"] = m_gr_to_blue
    features["population_color_manifold_m_ri_red"] = m_ri_to_red
    features["population_color_manifold_m_ri_blue"] = m_ri_to_blue
    features["population_color_manifold_m_iz_red"] = m_iz_to_red
    features["population_color_manifold_m_iz_blue"] = m_iz_to_blue
    features["population_color_manifold_ug_to_red"] = colors["c_ug"] - ug_to_red
    features["population_color_manifold_ug_to_blue"] = colors["c_ug"] - ug_to_blue

    for i, col in enumerate(_COLOR_FEATURES):
        features[f"population_color_manifold_res_assigned_{col}"] = res_assigned[:, i]
        features[f"population_color_manifold_res_opposite_{col}"] = res_opposite[:, i]

    return features

FEATURE_GROUPS = [
    {
        "name": "population_color_manifold_drifts",
        "fn": add_population_color_manifold_drifts,
        "depends_on": [],
        "description": "Builds redshift-binned red/blue manifold residual distances and signed color-offset features conditioned on galaxy_population.",
    }
]