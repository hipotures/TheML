import numpy as np
import pandas as pd
from itertools import product

_PRIMARY_BINS = (9, 9, 10)
_FALLBACK_BINS = 10
_MIN_PRIMARY = 40
_MIN_FALLBACK = 60

_SOURCE_GLOBAL = 5
_SOURCE_EXPANDED_FALLBACK = 4
_SOURCE_EXACT_FALLBACK = 3
_SOURCE_EXPANDED_PRIMARY = 2
_SOURCE_EXACT_PRIMARY = 1


def _safe_align(raw_index, values):
    if isinstance(values, pd.Series):
        if len(values) == len(raw_index) and not values.index.equals(raw_index):
            return pd.Series(values.to_numpy(), index=raw_index)
        return values.reindex(raw_index)
    if len(values) == len(raw_index):
        return pd.Series(np.asarray(values), index=raw_index)
    return pd.Series(np.asarray(values), index=raw_index).reindex(raw_index)


def _infer_train_mask(raw_index, aux):
    default = pd.Series(np.ones(len(raw_index), dtype=bool), index=raw_index)
    if aux is None or len(aux) == 0 or not isinstance(aux, pd.DataFrame):
        return default

    for key in ("is_train", "train_mask", "is_train_mask", "train"):
        if key not in aux.columns:
            continue
        candidate = _safe_align(raw_index, aux[key])
        if candidate.dropna().empty:
            continue
        if pd.api.types.is_bool_dtype(candidate):
            mask = candidate.fillna(False).astype(bool)
        elif pd.api.types.is_numeric_dtype(candidate):
            mask = (candidate.fillna(0) != 0)
        else:
            lowered = candidate.fillna("false").astype(str).str.lower().str.strip()
            mask = lowered.isin(("1", "true", "t", "yes", "y", "train", "training"))
        return mask.astype(bool)

    return default


def _infer_fold_series(raw_index, aux):
    if aux is None or len(aux) == 0 or not isinstance(aux, pd.DataFrame):
        return None

    for key in ("fold", "cv_fold", "fold_id", "fold_idx", "split"):
        if key not in aux.columns:
            continue
        candidate = _safe_align(raw_index, aux[key])
        if not candidate.dropna().empty:
            return candidate
    return None


def _clip_limits(values, low_q=0.001, high_q=0.999):
    arr = np.asarray(values, dtype=float)
    finite = np.isfinite(arr)
    if finite.sum() == 0:
        return 0.0, 1.0
    if finite.sum() < 3:
        lo = float(np.nanmin(arr[finite]))
        hi = float(np.nanmax(arr[finite]))
        if hi <= lo:
            hi = lo + 1e-6
        return lo, hi

    lo = float(np.quantile(arr[finite], low_q))
    hi = float(np.quantile(arr[finite], high_q))
    if not np.isfinite(lo) or not np.isfinite(hi):
        lo = float(np.nanmin(arr[finite]))
        hi = float(np.nanmax(arr[finite]))
    if hi <= lo:
        eps = 1e-6 * (abs(lo) + 1.0)
        lo -= eps
        hi += eps
    return lo, hi


def _fit_bins(values, n_bins):
    arr = np.asarray(values, dtype=float)
    finite = np.isfinite(arr)
    if finite.sum() < 3:
        if finite.sum() == 0:
            return tuple(np.linspace(0.0, 1.0, n_bins + 1))
        lo = float(np.nanmin(arr[finite]))
        hi = float(np.nanmax(arr[finite]))
        if hi <= lo:
            hi = lo + 1e-6
        return tuple(np.linspace(lo, hi, n_bins + 1))

    probs = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.quantile(arr[finite], probs).astype(float)
    edges[0] = float(np.nanmin(arr[finite]))
    edges[-1] = float(np.nanmax(arr[finite]))
    for i in range(1, len(edges)):
        if not np.isfinite(edges[i]) or edges[i] <= edges[i - 1]:
            edges[i] = edges[i - 1] + 1e-12 * (1.0 + abs(edges[i - 1]))
    for i in range(1, len(edges)):
        if edges[i] <= edges[i - 1]:
            edges[i] = edges[i - 1] + 1e-12 * (1.0 + abs(edges[i - 1]))
    return tuple(float(v) for v in edges)


def _assign_bins(values, edges):
    return (
        pd.cut(
            pd.Series(np.asarray(values, dtype=float)),
            bins=edges,
            labels=False,
            include_lowest=True,
            right=True,
        )
        .fillna(-1)
        .astype(np.int16)
        .to_numpy()
    )


def _build_lookup_table(train_df, key_cols, min_count):
    if train_df.empty:
        empty_index = pd.MultiIndex.from_arrays(
            [np.array([], dtype=np.int64) for _ in key_cols],
            names=list(key_cols),
        )
        return pd.DataFrame(columns=("z_med", "IQR", "MAD", "n"), index=empty_index)

    grouped = train_df.groupby(key_cols, observed=True)["redshift_clipped"].agg(
        z_med="median",
        q25=lambda s: s.quantile(0.25),
        q75=lambda s: s.quantile(0.75),
        MAD=lambda s: (s - s.median()).abs().median(),
        n="size",
    )
    grouped["IQR"] = grouped["q75"] - grouped["q25"]
    return grouped[["z_med", "IQR", "MAD", "n"]][grouped["n"] >= min_count]


def _neighbor_offsets(dim, radius):
    if radius == 1:
        return [off for off in product(range(-1, 2), repeat=dim) if any(v != 0 for v in off)]
    if radius == 2:
        return [off for off in product(range(-2, 3), repeat=dim) if max(abs(v) for v in off) == 2]
    return []


def _resolve_with_table(
    base_bins,
    table,
    min_count,
    unresolved,
    source,
    src_exact,
    src_expand,
    z_med,
    local_iqr,
    local_mad,
    local_n,
):
    if table is None or table.empty:
        return

    columns = table.columns
    z_pos = columns.get_loc("z_med")
    iqr_pos = columns.get_loc("IQR")
    mad_pos = columns.get_loc("MAD")
    n_pos = columns.get_loc("n")
    names = table.index.names
    dim = len(base_bins)

    base_arrays = [np.asarray(b, dtype=np.int64) for b in base_bins]

    exact_index = pd.MultiIndex.from_arrays(base_arrays, names=names)
    exact_lookup = table.reindex(exact_index).to_numpy()
    exact_ok = exact_lookup[:, n_pos] >= min_count
    exact_rows = np.flatnonzero(unresolved & exact_ok)
    if exact_rows.size > 0:
        z_med[exact_rows] = exact_lookup[exact_rows, z_pos]
        local_iqr[exact_rows] = exact_lookup[exact_rows, iqr_pos]
        local_mad[exact_rows] = exact_lookup[exact_rows, mad_pos]
        local_n[exact_rows] = exact_lookup[exact_rows, n_pos]
        source[exact_rows] = src_exact
        unresolved[exact_rows] = False

    if not unresolved.any():
        return

    for radius, src in ((1, src_expand), (2, src_expand)):
        for offset in _neighbor_offsets(dim, radius):
            unresolved_rows = np.flatnonzero(unresolved)
            if unresolved_rows.size == 0:
                break
            shifted = [base_arrays[d][unresolved_rows] + offset[d] for d in range(dim)]
            lookup_index = pd.MultiIndex.from_arrays(shifted, names=names)
            lookup = table.reindex(lookup_index).to_numpy()
            lookup_ok = lookup[:, n_pos] >= min_count
            if not lookup_ok.any():
                continue
            hit_rows = unresolved_rows[lookup_ok]
            hit_vals = lookup[lookup_ok]
            z_med[hit_rows] = hit_vals[:, z_pos]
            local_iqr[hit_rows] = hit_vals[:, iqr_pos]
            local_mad[hit_rows] = hit_vals[:, mad_pos]
            local_n[hit_rows] = hit_vals[:, n_pos]
            source[hit_rows] = src
            unresolved[hit_rows] = False
        if not unresolved.any():
            return


def _compute_features(raw, train_mask):
    index = raw.index
    n_rows = len(raw)
    train_mask = train_mask.astype(bool)
    if train_mask.sum() == 0:
        train_mask = pd.Series(np.ones(n_rows, dtype=bool), index=index)

    train_idx = train_mask.to_numpy()

    redshift = raw["redshift"].to_numpy(dtype=float)
    u = raw["u"].to_numpy(dtype=float)
    g = raw["g"].to_numpy(dtype=float)
    r = raw["r"].to_numpy(dtype=float)
    i = raw["i"].to_numpy(dtype=float)
    z = raw["z"].to_numpy(dtype=float)

    c_ug = u - g
    c_gr = g - r
    c_ri = r - i
    c_iz = i - z

    redshift_lo, redshift_hi = _clip_limits(redshift[train_idx])
    c_ug_lo, c_ug_hi = _clip_limits(c_ug[train_idx])
    c_gr_lo, c_gr_hi = _clip_limits(c_gr[train_idx])
    c_ri_lo, c_ri_hi = _clip_limits(c_ri[train_idx])
    c_iz_lo, c_iz_hi = _clip_limits(c_iz[train_idx])

    z_c = np.clip(redshift, redshift_lo, redshift_hi)
    c_ug_c = np.clip(c_ug, c_ug_lo, c_ug_hi)
    c_gr_c = np.clip(c_gr, c_gr_lo, c_gr_hi)
    c_ri_c = np.clip(c_ri, c_ri_lo, c_ri_hi)
    c_iz_c = np.clip(c_iz, c_iz_lo, c_iz_hi)

    edges_primary_c_gr = _fit_bins(c_gr_c[train_idx], _PRIMARY_BINS[0])
    edges_primary_c_ri = _fit_bins(c_ri_c[train_idx], _PRIMARY_BINS[1])
    edges_primary_r = _fit_bins(r[train_idx], _PRIMARY_BINS[2])

    edges_fallback_c_gr = _fit_bins(c_gr_c[train_idx], _FALLBACK_BINS)
    edges_fallback_c_ri = _fit_bins(c_ri_c[train_idx], _FALLBACK_BINS)
    edges_fallback_ug = _fit_bins(c_ug_c[train_idx], _FALLBACK_BINS)
    edges_fallback_iz = _fit_bins(c_iz_c[train_idx], _FALLBACK_BINS)

    b_gr_primary = _assign_bins(c_gr_c, edges_primary_c_gr)
    b_ri_primary = _assign_bins(c_ri_c, edges_primary_c_ri)
    b_r = _assign_bins(r, edges_primary_r)

    b_gr_fallback = _assign_bins(c_gr_c, edges_fallback_c_gr)
    b_ri_fallback = _assign_bins(c_ri_c, edges_fallback_c_ri)
    b_ug_fallback = _assign_bins(c_ug_c, edges_fallback_ug)
    b_iz_fallback = _assign_bins(c_iz_c, edges_fallback_iz)

    primary_df = pd.DataFrame(
        {
            "c_gr_bin": b_gr_primary[train_idx],
            "c_ri_bin": b_ri_primary[train_idx],
            "r_bin": b_r[train_idx],
            "redshift_clipped": z_c[train_idx],
        },
        index=index[train_idx],
    )
    fallback1_df = pd.DataFrame(
        {
            "c_gr_bin": b_gr_fallback[train_idx],
            "c_ri_bin": b_ri_fallback[train_idx],
            "redshift_clipped": z_c[train_idx],
        },
        index=index[train_idx],
    )
    fallback2_df = pd.DataFrame(
        {
            "c_ug_bin": b_ug_fallback[train_idx],
            "c_gr_bin": b_gr_fallback[train_idx],
            "redshift_clipped": z_c[train_idx],
        },
        index=index[train_idx],
    )
    fallback3_df = pd.DataFrame(
        {
            "c_ri_bin": b_ri_fallback[train_idx],
            "c_iz_bin": b_iz_fallback[train_idx],
            "redshift_clipped": z_c[train_idx],
        },
        index=index[train_idx],
    )

    primary_table = _build_lookup_table(primary_df, ["c_gr_bin", "c_ri_bin", "r_bin"], _MIN_PRIMARY)
    fallback1_table = _build_lookup_table(fallback1_df, ["c_gr_bin", "c_ri_bin"], _MIN_FALLBACK)
    fallback2_table = _build_lookup_table(fallback2_df, ["c_ug_bin", "c_gr_bin"], _MIN_FALLBACK)
    fallback3_table = _build_lookup_table(fallback3_df, ["c_ri_bin", "c_iz_bin"], _MIN_FALLBACK)

    global_train_z = z_c[train_idx]
    if global_train_z.size == 0:
        global_train_z = z_c

    global_n = float(global_train_z.size)
    global_med = float(np.nanmedian(global_train_z))
    global_q25 = float(np.nanquantile(global_train_z, 0.25))
    global_q75 = float(np.nanquantile(global_train_z, 0.75))
    global_iqr = float(global_q75 - global_q25)
    global_mad = float(np.nanmedian(np.abs(global_train_z - global_med)))

    z_med = np.full(n_rows, global_med, dtype=float)
    local_iqr = np.full(n_rows, global_iqr, dtype=float)
    local_mad = np.full(n_rows, global_mad, dtype=float)
    local_n = np.full(n_rows, global_n, dtype=float)
    source = np.full(n_rows, _SOURCE_GLOBAL, dtype=np.int16)
    unresolved = np.ones(n_rows, dtype=bool)

    _resolve_with_table(
        (b_gr_primary, b_ri_primary, b_r),
        primary_table,
        _MIN_PRIMARY,
        unresolved,
        source,
        _SOURCE_EXACT_PRIMARY,
        _SOURCE_EXPANDED_PRIMARY,
        z_med,
        local_iqr,
        local_mad,
        local_n,
    )
    _resolve_with_table(
        (b_gr_fallback, b_ri_fallback),
        fallback1_table,
        _MIN_FALLBACK,
        unresolved,
        source,
        _SOURCE_EXACT_FALLBACK,
        _SOURCE_EXPANDED_FALLBACK,
        z_med,
        local_iqr,
        local_mad,
        local_n,
    )
    _resolve_with_table(
        (b_ug_fallback, b_gr_fallback),
        fallback2_table,
        _MIN_FALLBACK,
        unresolved,
        source,
        _SOURCE_EXACT_FALLBACK,
        _SOURCE_EXPANDED_FALLBACK,
        z_med,
        local_iqr,
        local_mad,
        local_n,
    )
    _resolve_with_table(
        (b_ri_fallback, b_iz_fallback),
        fallback3_table,
        _MIN_FALLBACK,
        unresolved,
        source,
        _SOURCE_EXACT_FALLBACK,
        _SOURCE_EXPANDED_FALLBACK,
        z_med,
        local_iqr,
        local_mad,
        local_n,
    )

    residual = redshift - z_med
    residual_abs = np.abs(residual)
    residual_scaled = residual / np.maximum(local_iqr, 1e-3)
    local_log_count = np.log1p(local_n)
    ambiguity = local_iqr / (global_iqr + 1e-6)

    return pd.DataFrame(
        {
            "z_residual": residual,
            "z_residual_abs": residual_abs,
            "z_residual_scaled": residual_scaled,
            "local_log_count": local_log_count,
            "local_IQR": local_iqr,
            "local_MAD": local_mad,
            "local_ambiguity": ambiguity,
            "source_code": source,
        },
        index=index,
    )


def add_photoz_color_consistency(raw, deps, aux):
    _ = deps
    train_mask = _infer_train_mask(raw.index, aux)
    fold_series = _infer_fold_series(raw.index, aux)

    if (
        fold_series is None
        or fold_series.dropna().size <= 1
        or fold_series.dropna().nunique() <= 1
    ):
        return _compute_features(raw, train_mask)

    features = pd.DataFrame(index=raw.index)
    assigned = pd.Series(np.zeros(len(raw), dtype=bool), index=raw.index)

    for fold_value in fold_series.dropna().unique():
        fold_rows = fold_series == fold_value
        if not fold_rows.any():
            continue
        train_rows = train_mask & ~fold_rows
        if train_rows.sum() == 0:
            train_rows = train_mask
        fold_features = _compute_features(raw, train_rows)
        features.loc[fold_rows] = fold_features.loc[fold_rows]
        assigned.loc[fold_rows] = True

    remaining = ~assigned
    if remaining.any():
        full_features = _compute_features(raw, train_mask)
        features.loc[remaining] = full_features.loc[remaining]

    return features


FEATURE_GROUPS = [
    {
        "name": "photoz_color_consistency",
        "fn": add_photoz_color_consistency,
        "depends_on": [],
        "description": "Generate residual-to-local-color redshift consistency features from training-only quantile cells with fallback neighborhood expansion and global defaults.",
    }
]