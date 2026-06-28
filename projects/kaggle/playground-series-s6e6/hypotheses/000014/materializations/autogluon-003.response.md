import numpy as np
import pandas as pd


PRIMARY_BINS = (9, 9, 10)
FALLBACK_BINS = (10, 10)
PRIMARY_MIN_COUNT = 40
FALLBACK_MIN_COUNT = 60
N_FOLDS = 5
EPS = 1.0e-3


def _numeric_series(frame, name, default=0.0):
    if name in frame.columns:
        values = pd.to_numeric(frame[name], errors="coerce")
        med = values.replace([np.inf, -np.inf], np.nan).median()
        if not np.isfinite(med):
            med = default
        return values.replace([np.inf, -np.inf], np.nan).fillna(med).astype(float)
    return pd.Series(default, index=frame.index, dtype=float)


def _quantile_edges(values, n_bins):
    qs = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.nanquantile(np.asarray(values, dtype=float), qs)
    edges = np.unique(edges[np.isfinite(edges)])
    if edges.size < 2:
        center = float(np.nanmedian(values)) if np.isfinite(np.nanmedian(values)) else 0.0
        edges = np.array([center - 0.5, center + 0.5], dtype=float)
    edges[0] = -np.inf
    edges[-1] = np.inf
    return edges


def _clip_limits(values):
    arr = np.asarray(values, dtype=float)
    lo, hi = np.nanquantile(arr, (0.001, 0.999))
    if not np.isfinite(lo):
        lo = float(np.nanmin(arr)) if np.isfinite(np.nanmin(arr)) else -1.0
    if not np.isfinite(hi):
        hi = float(np.nanmax(arr)) if np.isfinite(np.nanmax(arr)) else 1.0
    if lo >= hi:
        lo -= 0.5
        hi += 0.5
    return float(lo), float(hi)


def _bin_values(values, edges):
    bins = np.searchsorted(edges[1:-1], np.asarray(values, dtype=float), side="right")
    return bins.astype(np.int16, copy=False)


def _weighted_quantile(values, weights, quantile):
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0.0)
    if not valid.any():
        return np.nan
    values = values[valid]
    weights = weights[valid]
    order = np.argsort(values)
    values = values[order]
    weights = weights[order]
    cutoff = quantile * weights.sum()
    return float(values[np.searchsorted(np.cumsum(weights), cutoff, side="left").clip(0, len(values) - 1)])


def _summarize_cell(group):
    z = group["redshift_work"].to_numpy(dtype=float)
    q10, q25, med, q75, q90 = np.nanquantile(z, (0.10, 0.25, 0.50, 0.75, 0.90))
    iqr = max(float(q75 - q25), EPS)
    return pd.Series(
        {
            "median": float(med),
            "q10": float(q10),
            "q25": float(q25),
            "q75": float(q75),
            "q90": float(q90),
            "iqr": iqr,
            "mad": float(np.nanmedian(np.abs(z - med))) if z.size else EPS,
            "n": int(z.size),
            "skew": float((q90 + q10 - 2.0 * med) / iqr),
        }
    )


def _build_table(frame, bin_cols, min_count):
    if frame.empty:
        return {}
    stats = frame.groupby(list(bin_cols), sort=False, observed=True).apply(_summarize_cell)
    stats = stats[stats["n"] >= min_count]
    table = {}
    for key, row in stats.iterrows():
        if not isinstance(key, tuple):
            key = (key,)
        table[tuple(int(x) for x in key)] = (
            float(row["median"]),
            float(row["q10"]),
            float(row["q25"]),
            float(row["q75"]),
            float(row["q90"]),
            max(float(row["iqr"]), EPS),
            max(float(row["mad"]), EPS),
            int(row["n"]),
            float(row["skew"]),
        )
    return table


def _aggregate_neighbors(table, key, radius):
    dims = len(key)
    ranges = [range(max(0, int(k) - radius), int(k) + radius + 1) for k in key]
    cells = []
    for candidate in np.array(np.meshgrid(*ranges)).T.reshape(-1, dims):
        tup = tuple(int(x) for x in candidate)
        if tup in table:
            cells.append(table[tup])
    if not cells:
        return None
    weights = np.array([cell[7] for cell in cells], dtype=float)
    total_n = int(weights.sum())
    med = _weighted_quantile([cell[0] for cell in cells], weights, 0.50)
    q10 = _weighted_quantile([cell[1] for cell in cells], weights, 0.10)
    q25 = _weighted_quantile([cell[2] for cell in cells], weights, 0.25)
    q75 = _weighted_quantile([cell[3] for cell in cells], weights, 0.75)
    q90 = _weighted_quantile([cell[4] for cell in cells], weights, 0.90)
    iqr = max(float(q75 - q25), EPS)
    mad = max(float(np.average([cell[6] for cell in cells], weights=weights)), EPS)
    skew = float((q90 + q10 - 2.0 * med) / iqr)
    return (float(med), float(q10), float(q25), float(q75), float(q90), iqr, mad, total_n, skew)


def _global_stats(z):
    arr = np.asarray(z, dtype=float)
    q10, q25, med, q75, q90 = np.nanquantile(arr, (0.10, 0.25, 0.50, 0.75, 0.90))
    iqr = max(float(q75 - q25), EPS)
    mad = max(float(np.nanmedian(np.abs(arr - med))), EPS)
    return (float(med), float(q10), float(q25), float(q75), float(q90), iqr, mad, int(arr.size), float((q90 + q10 - 2.0 * med) / iqr))


def _fit_reference(train_part):
    train = train_part.copy()
    z_median = train["redshift"].replace([np.inf, -np.inf], np.nan).median()
    if not np.isfinite(z_median):
        z_median = 0.0
    train["redshift_work"] = train["redshift"].replace([np.inf, -np.inf], np.nan).fillna(z_median)

    limits = {col: _clip_limits(train[col]) for col in ("c_ug", "c_gr", "c_ri", "c_iz", "r")}
    clipped = {}
    for col, (lo, hi) in limits.items():
        clipped[col] = train[col].clip(lo, hi)

    edges = {
        "c_gr_primary": _quantile_edges(clipped["c_gr"], PRIMARY_BINS[0]),
        "c_ri_primary": _quantile_edges(clipped["c_ri"], PRIMARY_BINS[1]),
        "r_primary": _quantile_edges(clipped["r"], PRIMARY_BINS[2]),
        "c_gr_fb": _quantile_edges(clipped["c_gr"], FALLBACK_BINS[0]),
        "c_ri_fb": _quantile_edges(clipped["c_ri"], FALLBACK_BINS[1]),
        "c_ug_fb": _quantile_edges(clipped["c_ug"], FALLBACK_BINS[0]),
        "c_iz_fb": _quantile_edges(clipped["c_iz"], FALLBACK_BINS[1]),
    }

    binned = train[["redshift_work"]].copy()
    binned["bp_c_gr"] = _bin_values(clipped["c_gr"], edges["c_gr_primary"])
    binned["bp_c_ri"] = _bin_values(clipped["c_ri"], edges["c_ri_primary"])
    binned["bp_r"] = _bin_values(clipped["r"], edges["r_primary"])
    binned["bf_c_gr"] = _bin_values(clipped["c_gr"], edges["c_gr_fb"])
    binned["bf_c_ri"] = _bin_values(clipped["c_ri"], edges["c_ri_fb"])
    binned["bf_c_ug"] = _bin_values(clipped["c_ug"], edges["c_ug_fb"])
    binned["bf_c_iz"] = _bin_values(clipped["c_iz"], edges["c_iz_fb"])

    return {
        "limits": limits,
        "edges": edges,
        "global": _global_stats(train["redshift_work"]),
        "primary": _build_table(binned, ("bp_c_gr", "bp_c_ri", "bp_r"), PRIMARY_MIN_COUNT),
        "fallbacks": (
            _build_table(binned, ("bf_c_gr", "bf_c_ri"), FALLBACK_MIN_COUNT),
            _build_table(binned, ("bf_c_ug", "bf_c_gr"), FALLBACK_MIN_COUNT),
            _build_table(binned, ("bf_c_ri", "bf_c_iz"), FALLBACK_MIN_COUNT),
        ),
    }


def _transform_reference(features, ref):
    n_rows = len(features)
    selected = np.empty((n_rows, 9), dtype=float)
    source = np.full(n_rows, 4.0, dtype=float)

    clipped = {}
    for col, (lo, hi) in ref["limits"].items():
        clipped[col] = features[col].clip(lo, hi)

    primary_keys = np.column_stack(
        [
            _bin_values(clipped["c_gr"], ref["edges"]["c_gr_primary"]),
            _bin_values(clipped["c_ri"], ref["edges"]["c_ri_primary"]),
            _bin_values(clipped["r"], ref["edges"]["r_primary"]),
        ]
    )
    fallback_keys = (
        np.column_stack([_bin_values(clipped["c_gr"], ref["edges"]["c_gr_fb"]), _bin_values(clipped["c_ri"], ref["edges"]["c_ri_fb"])]),
        np.column_stack([_bin_values(clipped["c_ug"], ref["edges"]["c_ug_fb"]), _bin_values(clipped["c_gr"], ref["edges"]["c_gr_fb"])]),
        np.column_stack([_bin_values(clipped["c_ri"], ref["edges"]["c_ri_fb"]), _bin_values(clipped["c_iz"], ref["edges"]["c_iz_fb"])]),
    )

    for pos in range(n_rows):
        key = tuple(int(x) for x in primary_keys[pos])
        cell = ref["primary"].get(key)
        if cell is not None:
            selected[pos] = cell
            source[pos] = 0.0
            continue

        for radius in (1, 2):
            cell = _aggregate_neighbors(ref["primary"], key, radius)
            if cell is not None:
                selected[pos] = cell
                source[pos] = 1.0
                break
        if source[pos] == 1.0:
            continue

        found = None
        found_source = 4.0
        for fb_idx, table in enumerate(ref["fallbacks"]):
            fb_key = tuple(int(x) for x in fallback_keys[fb_idx][pos])
            found = table.get(fb_key)
            if found is not None:
                found_source = 2.0
                break
        if found is None:
            for fb_idx, table in enumerate(ref["fallbacks"]):
                fb_key = tuple(int(x) for x in fallback_keys[fb_idx][pos])
                for radius in (1, 2):
                    found = _aggregate_neighbors(table, fb_key, radius)
                    if found is not None:
                        found_source = 3.0
                        break
                if found is not None:
                    break

        if found is None:
            found = ref["global"]
        selected[pos] = found
        source[pos] = found_source

    z = features["redshift"].to_numpy(dtype=float)
    z = np.where(np.isfinite(z), z, ref["global"][0])
    median = selected[:, 0]
    q10 = selected[:, 1]
    q90 = selected[:, 4]
    iqr = np.maximum(selected[:, 5], EPS)
    residual = z - median
    global_iqr = max(float(ref["global"][5]), EPS)

    return pd.DataFrame(
        {
            "redshift_local_residual": residual,
            "redshift_local_abs_residual": np.abs(residual),
            "redshift_local_residual_over_iqr": residual / iqr,
            "redshift_local_abs_residual_over_iqr": np.abs(residual) / iqr,
            "redshift_below_local_q10": z < q10,
            "redshift_above_local_q90": z > q90,
            "local_log_count": np.log1p(np.maximum(selected[:, 7], 0.0)),
            "local_iqr": iqr,
            "local_mad": np.maximum(selected[:, 6], EPS),
            "local_skew_proxy": selected[:, 8],
            "local_ambiguity": iqr / global_iqr,
            "source_code": source,
        },
        index=features.index,
    )


def _make_base_features(raw):
    u = _numeric_series(raw, "u")
    g = _numeric_series(raw, "g")
    r = _numeric_series(raw, "r")
    i = _numeric_series(raw, "i")
    zmag = _numeric_series(raw, "z")
    redshift = _numeric_series(raw, "redshift")
    return pd.DataFrame(
        {
            "c_ug": u - g,
            "c_gr": g - r,
            "c_ri": r - i,
            "c_iz": i - zmag,
            "r": r,
            "redshift": redshift,
        },
        index=raw.index,
    )


def _training_mask(raw, aux):
    if aux is not None and not aux.empty:
        for col in ("is_train", "train_mask"):
            if col in aux.columns and len(aux[col]) == len(raw):
                return aux[col].astype(bool).to_numpy()

    if "id" in raw.columns:
        ids = pd.to_numeric(raw["id"], errors="coerce")
        if ids.notna().all():
            ordered = ids.to_numpy(dtype=float)
            diffs = np.diff(ordered)
            breaks = np.flatnonzero(diffs != 1.0)
            if breaks.size:
                cut = int(breaks[0] + 1)
                mask = np.zeros(len(raw), dtype=bool)
                mask[:cut] = True
                return mask
            if ordered.min() == 0.0 and ordered.max() >= 1000.0:
                mask = np.ones(len(raw), dtype=bool)
                return mask

    return np.ones(len(raw), dtype=bool)


def add_photoz_color_consistency(raw, deps, aux):
    base = _make_base_features(raw)
    train_mask = _training_mask(raw, aux)
    if not train_mask.any():
        train_mask = np.ones(len(raw), dtype=bool)

    out_parts = []
    result = pd.DataFrame(index=raw.index)

    train_index = base.index[train_mask]
    test_index = base.index[~train_mask]

    if len(train_index) >= N_FOLDS * 2:
        ids = pd.to_numeric(raw.loc[train_index, "id"], errors="coerce") if "id" in raw.columns else pd.Series(np.arange(len(train_index)), index=train_index)
        fold_values = (pd.util.hash_pandas_object(ids, index=False).to_numpy(dtype=np.uint64) % np.uint64(N_FOLDS)).astype(int)
        for fold in range(N_FOLDS):
            hold_index = train_index[fold_values == fold]
            fit_index = train_index[fold_values != fold]
            if len(hold_index) == 0 or len(fit_index) == 0:
                continue
            ref = _fit_reference(base.loc[fit_index])
            out_parts.append(_transform_reference(base.loc[hold_index], ref))
    else:
        ref = _fit_reference(base.loc[train_index])
        out_parts.append(_transform_reference(base.loc[train_index], ref))

    if len(test_index):
        ref = _fit_reference(base.loc[train_index])
        out_parts.append(_transform_reference(base.loc[test_index], ref))

    if out_parts:
        result = pd.concat(out_parts, axis=0).reindex(raw.index)
    else:
        ref = _fit_reference(base)
        result = _transform_reference(base, ref).reindex(raw.index)

    return result


FEATURE_GROUPS = [
    {
        "name": "photoz_color_consistency",
        "fn": add_photoz_color_consistency,
        "depends_on": [],
        "description": "Robust local color-redshift residual features from quantile-binned ugriz color-magnitude neighborhoods.",
    }
]