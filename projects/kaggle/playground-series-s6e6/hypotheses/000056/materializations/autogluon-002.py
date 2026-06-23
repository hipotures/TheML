import math
import numpy as np
import pandas as pd

N_SIDE = 32
N_LATITUDE_BANDS = 2 * N_SIDE
N_LONGITUDE_BANDS = 6 * N_SIDE
N_CONTEXT_NEIGHBORS = 9
Q_GRI_COEFF = 1.582
Q_RIZ_COEFF = 0.987
REDSHIFT_BINS = (0.01, 0.10, 0.50, 1.0, 2.0)
MIN_CONTEXT_COUNT = 40
LOCAL_STD_CLIP = 8.0
MAD_FLOOR = 1e-8
INVALID_CONTEXT = -1
AREA_PER_CELL = (4.0 * math.pi) / (12.0 * (N_SIDE * N_SIDE))

def _to_float_vector(df, name):
    return pd.to_numeric(df[name], errors="coerce").to_numpy(dtype=np.float64)

def _locus_features(u, g, r, i, z):
    qgri = (g - r) - Q_GRI_COEFF * (r - i)
    qriz = (r - i) - Q_RIZ_COEFF * (i - z)
    ug = u - g
    gr = g - r
    ri = r - i
    iz = i - z
    ur = u - r
    gi = g - i
    return {
        "qgri": qgri,
        "qriz": qriz,
        "u_minus_g": ug,
        "g_minus_r": gr,
        "r_minus_i": ri,
        "i_minus_z": iz,
        "u_minus_r": ur,
        "g_minus_i": gi,
    }

def _redshift_strata(redshift):
    return np.digitize(redshift, REDSHIFT_BINS, right=False).astype(np.int8)

def _sky_context(alpha, delta):
    lon = np.mod(np.asarray(alpha, dtype=np.float64), 360.0)
    lon_idx = np.floor((lon / 360.0) * N_LONGITUDE_BANDS).astype(np.int32)
    lon_idx = np.clip(lon_idx, 0, N_LONGITUDE_BANDS - 1)

    sin_dec = np.sin(np.deg2rad(np.clip(np.asarray(delta, dtype=np.float64), -90.0, 90.0)))
    lat_idx = np.floor((sin_dec + 1.0) * 0.5 * N_LATITUDE_BANDS).astype(np.int32)
    lat_idx = np.clip(lat_idx, 0, N_LATITUDE_BANDS - 1)

    context_id = lat_idx * N_LONGITUDE_BANDS + lon_idx
    return context_id.astype(np.int32), lat_idx.astype(np.int32), lon_idx.astype(np.int32)

def _context_neighbor_matrix(lat_idx, lon_idx):
    mats = []
    for dlat in (-1, 0, 1):
        lat_shift = lat_idx + dlat
        valid = (lat_shift >= 0) & (lat_shift < N_LATITUDE_BANDS)
        lat_shift = np.where(valid, lat_shift, 0)
        for dlon in (-1, 0, 1):
            lon_shift = np.mod(lon_idx + dlon, N_LONGITUDE_BANDS)
            neighbor = np.where(valid, lat_shift * N_LONGITUDE_BANDS + lon_shift, INVALID_CONTEXT)
            mats.append(neighbor.astype(np.int32))
    return np.stack(mats, axis=1)

def _expand_context_memberships(context_neighbors, strat):
    flat_context = context_neighbors.ravel()
    valid = flat_context != INVALID_CONTEXT
    flat_context = flat_context[valid]
    flat_strat = np.repeat(strat, N_CONTEXT_NEIGHBORS)[valid]
    return flat_context.astype(np.int32), flat_strat.astype(np.int16)

def _median_mad_by_group(context_ids, strat_ids, values):
    idx = pd.MultiIndex.from_arrays([context_ids, strat_ids], names=("context_id", "z_bin"))
    s = pd.Series(values, index=idx, dtype=np.float64)

    med = s.groupby(level=["context_id", "z_bin"], observed=True).median()
    mad = (s - med.reindex(s.index)).abs().groupby(level=["context_id", "z_bin"], observed=True).median()
    cnt = s.groupby(level=["context_id", "z_bin"], observed=True).size()
    return med, mad, cnt

def _safe_global_stats(values):
    med = float(np.nanmedian(values))
    mad = float(np.nanmedian(np.abs(values - med)))
    if not np.isfinite(med):
        med = 0.0
    if (not np.isfinite(mad)) or (mad < MAD_FLOOR):
        mad = 1.0
    return med, mad

def add_local_reddening_free_locus_offsets(raw, deps, aux):
    del deps
    n = len(raw)
    index = raw.index

    alpha = _to_float_vector(raw, "alpha")
    delta = _to_float_vector(raw, "delta")
    u = _to_float_vector(raw, "u")
    g = _to_float_vector(raw, "g")
    r = _to_float_vector(raw, "r")
    i = _to_float_vector(raw, "i")
    z = _to_float_vector(raw, "z")
    redshift = _to_float_vector(raw, "redshift")

    valid_raw = (
        np.isfinite(alpha)
        & np.isfinite(delta)
        & np.isfinite(u)
        & np.isfinite(g)
        & np.isfinite(r)
        & np.isfinite(i)
        & np.isfinite(z)
        & np.isfinite(redshift)
    )

    raw_ctx = np.full(n, INVALID_CONTEXT, dtype=np.int32)
    raw_lat = np.full(n, -1, dtype=np.int16)
    raw_strat = np.full(n, -1, dtype=np.int8)
    if np.any(valid_raw):
        ctx, lat, lon = _sky_context(alpha[valid_raw], delta[valid_raw])
        raw_ctx[valid_raw] = ctx
        raw_lat[valid_raw] = lat
        raw_strat[valid_raw] = _redshift_strata(redshift[valid_raw])

    raw_features = _locus_features(u, g, r, i, z)

    s_alpha = alpha[valid_raw].copy()
    s_delta = delta[valid_raw].copy()
    s_u = u[valid_raw].copy()
    s_g = g[valid_raw].copy()
    s_r = r[valid_raw].copy()
    s_i = i[valid_raw].copy()
    s_z = z[valid_raw].copy()
    s_redshift = redshift[valid_raw].copy()

    if isinstance(aux, pd.DataFrame) and not aux.empty:
        aux_cols = ("alpha", "delta", "u", "g", "r", "i", "z", "redshift")
        if set(aux_cols).issubset(set(aux.columns)):
            a_alpha = _to_float_vector(aux, "alpha")
            a_delta = _to_float_vector(aux, "delta")
            a_u = _to_float_vector(aux, "u")
            a_g = _to_float_vector(aux, "g")
            a_r = _to_float_vector(aux, "r")
            a_i = _to_float_vector(aux, "i")
            a_z = _to_float_vector(aux, "z")
            a_redshift = _to_float_vector(aux, "redshift")

            aux_valid = (
                np.isfinite(a_alpha)
                & np.isfinite(a_delta)
                & np.isfinite(a_u)
                & np.isfinite(a_g)
                & np.isfinite(a_r)
                & np.isfinite(a_i)
                & np.isfinite(a_z)
                & np.isfinite(a_redshift)
            )

            if np.any(aux_valid):
                s_alpha = np.concatenate([s_alpha, a_alpha[aux_valid]], axis=0)
                s_delta = np.concatenate([s_delta, a_delta[aux_valid]], axis=0)
                s_u = np.concatenate([s_u, a_u[aux_valid]], axis=0)
                s_g = np.concatenate([s_g, a_g[aux_valid]], axis=0)
                s_r = np.concatenate([s_r, a_r[aux_valid]], axis=0)
                s_i = np.concatenate([s_i, a_i[aux_valid]], axis=0)
                s_z = np.concatenate([s_z, a_z[aux_valid]], axis=0)
                s_redshift = np.concatenate([s_redshift, a_redshift[aux_valid]], axis=0)

    if s_alpha.size == 0:
        return pd.DataFrame(
            {
                "local_reddening_free_qgri_std": np.full(n, np.nan, dtype=np.float64),
                "local_reddening_free_qriz_std": np.full(n, np.nan, dtype=np.float64),
                "local_reddening_free_locus_distance": np.full(n, np.nan, dtype=np.float64),
                "local_reddening_free_ur_std": np.full(n, np.nan, dtype=np.float64),
                "local_reddening_free_gi_std": np.full(n, np.nan, dtype=np.float64),
                "local_reddening_free_context_density": np.full(n, np.nan, dtype=np.float64),
                "local_reddening_free_context_density_z": np.full(n, np.nan, dtype=np.float64),
                "local_reddening_free_fallback_level": np.full(n, np.nan, dtype=np.float64),
            },
            index=index,
        )

    support_features = _locus_features(s_u, s_g, s_r, s_i, s_z)
    s_ctx, s_lat, s_lon = _sky_context(s_alpha, s_delta)
    s_strat = _redshift_strata(s_redshift)
    s_neighbors = _context_neighbor_matrix(s_lat, s_lon)
    rep_ctx, rep_strat = _expand_context_memberships(s_neighbors, s_strat)

    repeat_rep = np.tile(support_features["qgri"], N_CONTEXT_NEIGHBORS)
    rep_values = repeat_rep[np.flatnonzero((s_neighbors.ravel() != INVALID_CONTEXT))]
    med_cs_anchor, mad_cs_anchor, count_cs_anchor = _median_mad_by_group(rep_ctx, rep_strat, rep_values)

    rep_strat_all = np.full(rep_ctx.shape[0], -1, dtype=np.int16)
    _, _, count_ctx_anchor = _median_mad_by_group(rep_ctx, rep_strat_all, rep_values)

    raw_key = pd.MultiIndex.from_arrays((raw_ctx, raw_strat), names=("context_id", "z_bin"))
    raw_context_key = pd.MultiIndex.from_arrays(
        (raw_ctx, np.full(n, -1, dtype=np.int16)),
        names=("context_id", "z_bin"),
    )

    count_cs_row = count_cs_anchor.reindex(raw_key).to_numpy(dtype=np.float64)
    count_ctx_row = count_ctx_anchor.reindex(raw_context_key).to_numpy(dtype=np.float64)

    have_cs = np.isfinite(count_cs_row) & (count_cs_row >= MIN_CONTEXT_COUNT)
    have_ctx = np.isfinite(count_ctx_row) & (count_ctx_row >= MIN_CONTEXT_COUNT)
    fallback_density_count = np.where(
        have_cs,
        np.maximum(count_cs_row, 0.0),
        np.where(have_ctx, np.maximum(count_ctx_row, 0.0), float(len(s_alpha))),
    )

    context_cells = np.where(
        raw_ctx == INVALID_CONTEXT,
        np.nan,
        np.where((raw_lat == 0) | (raw_lat == (N_LATITUDE_BANDS - 1)), 6.0, 9.0),
    )
    context_density = np.log1p(fallback_density_count / (context_cells * AREA_PER_CELL))
    context_density = np.where(np.isfinite(context_density), context_density, np.nan)

    density_valid = np.isfinite(context_density)
    density_mean = float(np.nanmean(context_density))
    density_std = float(np.nanstd(context_density))
    if not np.isfinite(density_mean):
        density_mean = 0.0
    if (not np.isfinite(density_std)) or (density_std < MAD_FLOOR):
        density_std = 1.0
    context_density_z = (context_density - density_mean) / density_std

    fallback_level = np.where(
        have_cs,
        0.0,
        np.where(have_ctx, 1.0, 2.0),
    )
    fallback_level = np.where(valid_raw, fallback_level, np.nan)

    # standardized color-offset features
    raw_qgri = raw_features["qgri"]
    raw_qriz = raw_features["qriz"]
    raw_ur = raw_features["u_minus_r"]
    raw_gi = raw_features["g_minus_i"]

    std_qgri = np.full(n, np.nan, dtype=np.float64)
    std_qriz = np.full(n, np.nan, dtype=np.float64)
    std_ur = np.full(n, np.nan, dtype=np.float64)
    std_gi = np.full(n, np.nan, dtype=np.float64)

    # 1) qgri
    rep_values = np.tile(support_features["qgri"], N_CONTEXT_NEIGHBORS)
    rep_values = rep_values[np.flatnonzero((s_neighbors.ravel() != INVALID_CONTEXT))]
    med_cs, mad_cs, _ = _median_mad_by_group(rep_ctx, rep_strat, rep_values)
    rep_values_ctx_only = np.tile(support_features["qgri"], N_CONTEXT_NEIGHBORS)
    rep_values_ctx_only = rep_values_ctx_only[np.flatnonzero((s_neighbors.ravel() != INVALID_CONTEXT))]
    med_ctx, mad_ctx, _ = _median_mad_by_group(rep_ctx, rep_strat_all, rep_values_ctx_only)
    g_med, g_mad = _safe_global_stats(support_features["qgri"])

    med_cs_row = med_cs.reindex(raw_key).to_numpy(dtype=np.float64)
    mad_cs_row = mad_cs.reindex(raw_key).to_numpy(dtype=np.float64)
    med_ctx_row = med_ctx.reindex(raw_context_key).to_numpy(dtype=np.float64)
    mad_ctx_row = mad_ctx.reindex(raw_context_key).to_numpy(dtype=np.float64)

    use_cs = np.isfinite(med_cs_row) & np.isfinite(mad_cs_row) & (count_cs_row >= MIN_CONTEXT_COUNT)
    use_ctx = np.isfinite(med_ctx_row) & np.isfinite(mad_ctx_row) & (count_ctx_row >= MIN_CONTEXT_COUNT)

    med_row = np.where(use_cs, med_cs_row, med_ctx_row)
    mad_row = np.where(use_cs, mad_cs_row, mad_ctx_row)
    fallback_mask = ~use_cs & ~use_ctx
    med_row = np.where(fallback_mask, g_med, med_row)
    mad_row = np.where(fallback_mask, g_mad, mad_row)
    mad_row = np.where(np.isfinite(mad_row) & (mad_row > MAD_FLOOR), mad_row, g_mad)
    std_qgri = np.clip((raw_qgri - med_row) / mad_row, -LOCAL_STD_CLIP, LOCAL_STD_CLIP)

    # 2) qriz
    rep_values = np.tile(support_features["qriz"], N_CONTEXT_NEIGHBORS)
    rep_values = rep_values[np.flatnonzero((s_neighbors.ravel() != INVALID_CONTEXT))]
    med_cs, mad_cs, _ = _median_mad_by_group(rep_ctx, rep_strat, rep_values)
    rep_values_ctx_only = np.tile(support_features["qriz"], N_CONTEXT_NEIGHBORS)
    rep_values_ctx_only = rep_values_ctx_only[np.flatnonzero((s_neighbors.ravel() != INVALID_CONTEXT))]
    med_ctx, mad_ctx, _ = _median_mad_by_group(rep_ctx, rep_strat_all, rep_values_ctx_only)
    g_med, g_mad = _safe_global_stats(support_features["qriz"])

    med_cs_row = med_cs.reindex(raw_key).to_numpy(dtype=np.float64)
    mad_cs_row = mad_cs.reindex(raw_key).to_numpy(dtype=np.float64)
    med_ctx_row = med_ctx.reindex(raw_context_key).to_numpy(dtype=np.float64)
    mad_ctx_row = mad_ctx.reindex(raw_context_key).to_numpy(dtype=np.float64)

    use_cs = np.isfinite(med_cs_row) & np.isfinite(mad_cs_row) & (count_cs_row >= MIN_CONTEXT_COUNT)
    use_ctx = np.isfinite(med_ctx_row) & np.isfinite(mad_ctx_row) & (count_ctx_row >= MIN_CONTEXT_COUNT)

    med_row = np.where(use_cs, med_cs_row, med_ctx_row)
    mad_row = np.where(use_cs, mad_cs_row, mad_ctx_row)
    fallback_mask = ~use_cs & ~use_ctx
    med_row = np.where(fallback_mask, g_med, med_row)
    mad_row = np.where(fallback_mask, g_mad, mad_row)
    mad_row = np.where(np.isfinite(mad_row) & (mad_row > MAD_FLOOR), mad_row, g_mad)
    std_qriz = np.clip((raw_qriz - med_row) / mad_row, -LOCAL_STD_CLIP, LOCAL_STD_CLIP)

    # 3) u-r residual
    rep_values = np.tile(support_features["u_minus_r"], N_CONTEXT_NEIGHBORS)
    rep_values = rep_values[np.flatnonzero((s_neighbors.ravel() != INVALID_CONTEXT))]
    med_cs, mad_cs, _ = _median_mad_by_group(rep_ctx, rep_strat, rep_values)
    rep_values_ctx_only = np.tile(support_features["u_minus_r"], N_CONTEXT_NEIGHBORS)
    rep_values_ctx_only = rep_values_ctx_only[np.flatnonzero((s_neighbors.ravel() != INVALID_CONTEXT))]
    med_ctx, mad_ctx, _ = _median_mad_by_group(rep_ctx, rep_strat_all, rep_values_ctx_only)
    g_med, g_mad = _safe_global_stats(support_features["u_minus_r"])

    med_cs_row = med_cs.reindex(raw_key).to_numpy(dtype=np.float64)
    mad_cs_row = mad_cs.reindex(raw_key).to_numpy(dtype=np.float64)
    med_ctx_row = med_ctx.reindex(raw_context_key).to_numpy(dtype=np.float64)
    mad_ctx_row = mad_ctx.reindex(raw_context_key).to_numpy(dtype=np.float64)

    use_cs = np.isfinite(med_cs_row) & np.isfinite(mad_cs_row) & (count_cs_row >= MIN_CONTEXT_COUNT)
    use_ctx = np.isfinite(med_ctx_row) & np.isfinite(mad_ctx_row) & (count_ctx_row >= MIN_CONTEXT_COUNT)

    med_row = np.where(use_cs, med_cs_row, med_ctx_row)
    mad_row = np.where(use_cs, mad_cs_row, mad_ctx_row)
    fallback_mask = ~use_cs & ~use_ctx
    med_row = np.where(fallback_mask, g_med, med_row)
    mad_row = np.where(fallback_mask, g_mad, mad_row)
    mad_row = np.where(np.isfinite(mad_row) & (mad_row > MAD_FLOOR), mad_row, g_mad)
    std_ur = np.clip((raw_ur - med_row) / mad_row, -LOCAL_STD_CLIP, LOCAL_STD_CLIP)

    # 4) g-i residual
    rep_values = np.tile(support_features["g_minus_i"], N_CONTEXT_NEIGHBORS)
    rep_values = rep_values[np.flatnonzero((s_neighbors.ravel() != INVALID_CONTEXT))]
    med_cs, mad_cs, _ = _median_mad_by_group(rep_ctx, rep_strat, rep_values)
    rep_values_ctx_only = np.tile(support_features["g_minus_i"], N_CONTEXT_NEIGHBORS)
    rep_values_ctx_only = rep_values_ctx_only[np.flatnonzero((s_neighbors.ravel() != INVALID_CONTEXT))]
    med_ctx, mad_ctx, _ = _median_mad_by_group(rep_ctx, rep_strat_all, rep_values_ctx_only)
    g_med, g_mad = _safe_global_stats(support_features["g_minus_i"])

    med_cs_row = med_cs.reindex(raw_key).to_numpy(dtype=np.float64)
    mad_cs_row = mad_cs.reindex(raw_key).to_numpy(dtype=np.float64)
    med_ctx_row = med_ctx.reindex(raw_context_key).to_numpy(dtype=np.float64)
    mad_ctx_row = mad_ctx.reindex(raw_context_key).to_numpy(dtype=np.float64)

    use_cs = np.isfinite(med_cs_row) & np.isfinite(mad_cs_row) & (count_cs_row >= MIN_CONTEXT_COUNT)
    use_ctx = np.isfinite(med_ctx_row) & np.isfinite(mad_ctx_row) & (count_ctx_row >= MIN_CONTEXT_COUNT)

    med_row = np.where(use_cs, med_cs_row, med_ctx_row)
    mad_row = np.where(use_cs, mad_cs_row, mad_ctx_row)
    fallback_mask = ~use_cs & ~use_ctx
    med_row = np.where(fallback_mask, g_med, med_row)
    mad_row = np.where(fallback_mask, g_mad, mad_row)
    mad_row = np.where(np.isfinite(mad_row) & (mad_row > MAD_FLOOR), mad_row, g_mad)
    std_gi = np.clip((raw_gi - med_row) / mad_row, -LOCAL_STD_CLIP, LOCAL_STD_CLIP)

    locus_distance = np.sqrt(np.square(std_qgri) + np.square(std_qriz))

    return pd.DataFrame(
        {
            "local_reddening_free_qgri_std": std_qgri,
            "local_reddening_free_qriz_std": std_qriz,
            "local_reddening_free_locus_distance": locus_distance,
            "local_reddening_free_ur_std": std_ur,
            "local_reddening_free_gi_std": std_gi,
            "local_reddening_free_context_density": context_density,
            "local_reddening_free_context_density_z": context_density_z,
            "local_reddening_free_fallback_level": fallback_level,
        },
        index=index,
    )

FEATURE_GROUPS = [
    {
        "name": "local_reddening_free_locus_offsets",
        "fn": add_local_reddening_free_locus_offsets,
        "depends_on": [],
        "description": "Compute local sky-cell reddening-free locus offsets from neighborhood medians and MADs with sparse-context fallback.",
    }
]