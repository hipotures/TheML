import numpy as np
import pandas as pd

_CONTEXT_NSIDE = 32
_REDSHIFT_STRATA = (-np.inf, 0.01, 0.1, 0.3, 0.5, 1.0, 2.0, np.inf)
_N_STRATA = 7
_MIN_CONTEXT_COUNT = 60
_MAD_FLOOR = 1e-6
_Z_MIN = -8.0
_Z_MAX = 8.0
_Q1_COEF = 1.582
_Q2_COEF = 0.987


def _robust_mad(values):
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[~pd.isna(arr)]
    if arr.size == 0:
        return np.nan
    med = np.median(arr)
    mad = np.median(np.abs(arr - med))
    return float(mad)


def _fallback_context_and_neighbors(alpha, delta, nside):
    ra = np.mod(np.asarray(alpha, dtype=np.float64), 360.0)
    dec = np.asarray(delta, dtype=np.float64)

    n_lon = 4 * nside
    n_lat = 3 * nside

    lat_edges = np.degrees(np.arcsin(np.linspace(-1.0, 1.0, n_lat + 1)))
    lon_bin = np.floor((ra / 360.0) * n_lon).astype(np.int64)
    lon_bin = np.mod(lon_bin, n_lon)
    lat_bin = np.searchsorted(lat_edges, dec, side="right") - 1
    lat_bin = np.clip(lat_bin, 0, n_lat - 1)

    context = lat_bin * n_lon + lon_bin
    context = np.clip(context, 0, n_lon * n_lat - 1).astype(np.int64)

    unique_context = np.unique(context)
    neighbor_map = {}
    for ctx in unique_context:
        lat_idx, lon_idx = divmod(int(ctx), n_lon)
        neighbors = []
        for dlat in (-1, 0, 1):
            lat_n = lat_idx + dlat
            if 0 <= lat_n < n_lat:
                for dlon in (-1, 0, 1):
                    lon_n = (lon_idx + dlon) % n_lon
                    neighbors.append(lat_n * n_lon + lon_n)
        neighbor_map[int(ctx)] = np.array(sorted(set(neighbors)), dtype=np.int64)
    return context, neighbor_map


def _healpix_context_and_neighbors(alpha, delta, nside):
    import healpy as hp

    ra = np.mod(np.asarray(alpha, dtype=np.float64), 360.0)
    dec = np.asarray(delta, dtype=np.float64)
    theta = np.deg2rad(90.0 - np.clip(dec, -90.0, 90.0))
    phi = np.deg2rad(ra)

    context = hp.ang2pix(nside, theta, phi, nest=False).astype(np.int64)
    unique_ctx = np.unique(context)

    neighbor_map = {}
    for ctx in unique_ctx:
        ctx_i = int(ctx)
        neigh = hp.get_all_neighbours(nside, ctx_i, nest=False)
        neigh = np.asarray(neigh, dtype=np.int64)
        neigh = neigh[neigh >= 0]
        if neigh.size == 0:
            neighbor_map[ctx_i] = np.array((ctx_i,), dtype=np.int64)
        else:
            neighbor_map[ctx_i] = np.unique(np.concatenate((np.array((ctx_i,), dtype=np.int64), neigh)))
    return context, neighbor_map


def _context_and_neighbors(alpha, delta, nside):
    try:
        return _healpix_context_and_neighbors(alpha, delta, nside)
    except Exception:
        return _fallback_context_and_neighbors(alpha, delta, nside)


def add_local_reddening_free_locus_offsets(raw, deps, aux):
    del deps
    del aux

    u = raw["u"].to_numpy(dtype=np.float64)
    g = raw["g"].to_numpy(dtype=np.float64)
    r = raw["r"].to_numpy(dtype=np.float64)
    i = raw["i"].to_numpy(dtype=np.float64)
    z = raw["z"].to_numpy(dtype=np.float64)
    alpha = raw["alpha"].to_numpy(dtype=np.float64)
    delta = raw["delta"].to_numpy(dtype=np.float64)
    redshift = raw["redshift"].to_numpy(dtype=np.float64)

    c_ug = u - g
    c_gr = g - r
    c_ri = r - i
    c_iz = i - z
    q1 = c_gr - _Q1_COEF * c_ri
    q2 = c_ri - _Q2_COEF * c_iz

    context, neighbor_map = _context_and_neighbors(alpha, delta, _CONTEXT_NSIDE)
    context = context.astype(np.int64, copy=False)
    stratum = np.asarray(
        pd.cut(
            redshift,
            bins=_REDSHIFT_STRATA,
            labels=False,
            include_lowest=True,
            right=False,
        ),
        dtype=np.int64,
    )

    base = pd.DataFrame(
        {
            "context": context,
            "stratum": stratum,
            "q1": q1,
            "q2": q2,
            "c_ug": c_ug,
            "c_gr": c_gr,
            "c_ri": c_ri,
            "c_iz": c_iz,
        },
        index=raw.index,
    )
    feature_cols = ["q1", "q2", "c_ug", "c_gr", "c_ri", "c_iz"]

    ctx_strat_counts = base.groupby(["context", "stratum"], sort=False).size()
    ctx_all_counts = base.groupby("context", sort=False).size()

    ctx_strat_count_dict = ctx_strat_counts.to_dict()
    ctx_all_count_dict = ctx_all_counts.to_dict()

    n = len(base)
    n_ctx_strat = np.zeros(n, dtype=np.int64)
    n_ctx_all = np.zeros(n, dtype=np.int64)
    for row_idx, neigh_cells in enumerate(np.array([neighbor_map[int(c)] for c in context], dtype=object)):
        s = int(stratum[row_idx])
        total_strat = 0
        total_all = 0
        for neigh in neigh_cells:
            neigh_i = int(neigh)
            total_strat += int(ctx_strat_count_dict.get((neigh_i, s), 0))
            total_all += int(ctx_all_count_dict.get(neigh_i, 0))
        n_ctx_strat[row_idx] = total_strat
        n_ctx_all[row_idx] = total_all

    stratum_counts = np.bincount(stratum, minlength=_N_STRATA).astype(np.float64)
    total_rows = float(n)

    layer_ctx_strat = n_ctx_strat >= _MIN_CONTEXT_COUNT
    layer_ctx_unstrat = (~layer_ctx_strat) & (n_ctx_all >= _MIN_CONTEXT_COUNT)
    layer_global = (~layer_ctx_strat) & (~layer_ctx_unstrat)
    layer_global_strat = layer_global & (stratum_counts[stratum] >= _MIN_CONTEXT_COUNT)
    layer_global_full = layer_global & (~layer_global_strat)

    strat_med = base.groupby(["context", "stratum"], sort=False)[feature_cols].median()
    strat_mad = base.groupby(["context", "stratum"], sort=False)[feature_cols].agg(_robust_mad)
    ctx_med = base.groupby("context", sort=False)[feature_cols].median()
    ctx_mad = base.groupby("context", sort=False)[feature_cols].agg(_robust_mad)
    global_strat_med = base.groupby("stratum", sort=False)[feature_cols].median()
    global_strat_mad = base.groupby("stratum", sort=False)[feature_cols].agg(_robust_mad)
    global_all_med = base[feature_cols].median()
    global_all_mad = base[feature_cols].apply(_robust_mad)

    idx = pd.MultiIndex.from_arrays(
        [pd.Index(context, dtype=np.int64), pd.Index(stratum, dtype=np.int64)],
        names=("context", "stratum"),
    )

    out = pd.DataFrame(index=raw.index)
    for feat in feature_cols:
        x = base[feat].to_numpy(dtype=np.float64)

        med = np.full(n, float(global_all_med[feat]), dtype=np.float64)
        mad = np.full(n, float(global_all_mad[feat]), dtype=np.float64)

        med_strat = strat_med[feat].reindex(idx).to_numpy(dtype=np.float64)
        mad_strat = strat_mad[feat].reindex(idx).to_numpy(dtype=np.float64)
        med_ctx = ctx_med[feat].reindex(context).to_numpy(dtype=np.float64)
        mad_ctx = ctx_mad[feat].reindex(context).to_numpy(dtype=np.float64)
        med_global_strat = global_strat_med[feat].reindex(range(_N_STRATA)).to_numpy(dtype=np.float64)[stratum]
        mad_global_strat = global_strat_mad[feat].reindex(range(_N_STRATA)).to_numpy(dtype=np.float64)[stratum]

        miss = np.isnan(med_strat)
        if miss.any():
            med_strat[miss] = float(global_all_med[feat])
            mad_strat[miss] = float(global_all_mad[feat])

        miss = np.isnan(med_ctx)
        if miss.any():
            med_ctx[miss] = float(global_all_med[feat])
            mad_ctx[miss] = float(global_all_mad[feat])

        miss = np.isnan(med_global_strat)
        if miss.any():
            med_global_strat[miss] = float(global_all_med[feat])
            mad_global_strat[miss] = float(global_all_mad[feat])

        if layer_ctx_strat.any():
            loc = np.flatnonzero(layer_ctx_strat)
            med[loc] = med_strat[loc]
            mad[loc] = mad_strat[loc]

        if layer_ctx_unstrat.any():
            loc = np.flatnonzero(layer_ctx_unstrat)
            med[loc] = med_ctx[loc]
            mad[loc] = mad_ctx[loc]

        if layer_global_strat.any():
            loc = np.flatnonzero(layer_global_strat)
            med[loc] = med_global_strat[loc]
            mad[loc] = mad_global_strat[loc]

        mad = np.where(np.isfinite(mad) & (mad >= _MAD_FLOOR), mad, _MAD_FLOOR)
        z = (x - med) / mad
        out[f"z_{feat}"] = np.clip(z, _Z_MIN, _Z_MAX)

    out["z_locus_deviation"] = np.sqrt(out["z_q1"] ** 2 + out["z_q2"] ** 2)

    counts_df = ctx_strat_counts.reset_index(name="count")
    logdens_ctx_strat_med = np.full(_N_STRATA, np.log1p(total_rows), dtype=np.float64)
    logdens_ctx_strat_mad = np.full(_N_STRATA, _MAD_FLOOR, dtype=np.float64)
    for s in range(_N_STRATA):
        vals = counts_df.loc[counts_df["stratum"] == s, "count"].to_numpy(dtype=np.float64)
        if vals.size:
            lv = np.log1p(vals)
            med = np.median(lv)
            mad = np.median(np.abs(lv - med))
            logdens_ctx_strat_med[s] = med
            logdens_ctx_strat_mad[s] = mad if mad >= _MAD_FLOOR else _MAD_FLOOR
        else:
            logdens_ctx_strat_med[s] = np.log1p(stratum_counts[s])

    log_ctx_counts = np.log1p(ctx_all_counts.to_numpy(dtype=np.float64))
    logdens_ctx_med = np.median(log_ctx_counts)
    logdens_ctx_mad = np.median(np.abs(log_ctx_counts - logdens_ctx_med))
    if not np.isfinite(logdens_ctx_mad) or logdens_ctx_mad < _MAD_FLOOR:
        logdens_ctx_mad = _MAD_FLOOR
    logdens_ctx_med_by_strat = np.full(_N_STRATA, logdens_ctx_med, dtype=np.float64)
    logdens_ctx_mad_by_strat = np.full(_N_STRATA, logdens_ctx_mad, dtype=np.float64)

    logdens_global_strat_med = np.log1p(stratum_counts)
    logdens_global_strat_mad = np.full(_N_STRATA, _MAD_FLOOR, dtype=np.float64)

    logdens_global_all_med = np.log1p(total_rows)
    logdens_global_all_mad = _MAD_FLOOR

    log_nctx = np.zeros(n, dtype=np.float64)
    log_med = np.zeros(n, dtype=np.float64)
    log_mad = np.zeros(n, dtype=np.float64)

    if layer_ctx_strat.any():
        loc = np.flatnonzero(layer_ctx_strat)
        log_nctx[loc] = n_ctx_strat[loc].astype(np.float64)
        log_med[loc] = logdens_ctx_strat_med[stratum[loc]]
        log_mad[loc] = logdens_ctx_strat_mad[stratum[loc]]

    if layer_ctx_unstrat.any():
        loc = np.flatnonzero(layer_ctx_unstrat)
        log_nctx[loc] = n_ctx_all[loc].astype(np.float64)
        log_med[loc] = logdens_ctx_med_by_strat[stratum[loc]]
        log_mad[loc] = logdens_ctx_mad_by_strat[stratum[loc]]

    if layer_global_strat.any():
        loc = np.flatnonzero(layer_global_strat)
        log_nctx[loc] = stratum_counts[stratum[loc]]
        log_med[loc] = logdens_global_strat_med[stratum[loc]]
        log_mad[loc] = logdens_global_strat_mad[stratum[loc]]

    if layer_global_full.any():
        loc = np.flatnonzero(layer_global_full)
        log_nctx[loc] = total_rows
        log_med[loc] = logdens_global_all_med
        log_mad[loc] = logdens_global_all_mad

    z_logdens = (np.log1p(log_nctx) - log_med) / np.maximum(log_mad, _MAD_FLOOR)
    out["z_logdens"] = np.clip(z_logdens, _Z_MIN, _Z_MAX)

    out["context_from_stratified"] = layer_ctx_strat
    out["context_from_unstratified"] = layer_ctx_unstrat
    out["context_from_global_fallback"] = layer_global

    return out


FEATURE_GROUPS = [
    {
        "name": "local_reddening_free_locus_offsets",
        "fn": add_local_reddening_free_locus_offsets,
        "depends_on": [],
        "description": "Builds local reddening-insensitive color residuals by comparing object colors to context-and-redshift robust loci and emits normalized offsets, density offsets, and context fallback indicators.",
    }
]