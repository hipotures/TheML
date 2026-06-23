import numpy as np
import pandas as pd

REDSHIFT_QUANTILES = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
MIN_CELL_COUNT = 40
MIN_BIN_COUNT = 120
MAD_SCALE = 1.4826
MAD_EPSILON = 1e-8
CLIP_LIMIT = 8.0
MISSING_MAGNITUDE_THRESHOLD = -5000.0
AUX_REQUIRED_COLUMNS = ("u", "g", "r", "i", "z", "redshift")


def _to_float_series(values):
    arr = pd.to_numeric(values, errors="coerce")
    return arr.replace([np.inf, -np.inf], np.nan).astype("float64")


def _clean_magnitude(values):
    arr = _to_float_series(values)
    return arr.mask(arr < MISSING_MAGNITUDE_THRESHOLD, np.nan)


def _mad_value(values):
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.nan
    med = np.median(arr)
    mad = np.median(np.abs(arr - med)) * MAD_SCALE
    if mad <= MAD_EPSILON:
        return 0.0
    return float(mad)


def _assign_redshift_bins(redshift):
    redshift_float = _to_float_series(redshift)
    probs = (0.0,) + REDSHIFT_QUANTILES + (1.0,)
    finite = np.isfinite(redshift_float.to_numpy(dtype=float))
    if finite.any():
        try:
            bins, edges = pd.qcut(
                redshift_float,
                q=probs,
                labels=False,
                duplicates="drop",
                retbins=True,
            )
            edges = np.asarray(edges, dtype=float)
            edges = np.unique(edges[np.isfinite(edges)])
            if edges.size >= 2:
                bins = pd.Series(bins, index=redshift_float.index).fillna(-1).astype("int16")
                return bins, edges
        except Exception:
            pass

    if finite.any():
        finite_vals = redshift_float.loc[finite]
        lo = float(finite_vals.min())
        hi = float(finite_vals.max())
        if lo == hi:
            edges = np.array([lo - 0.5, hi + 0.5], dtype=float)
        else:
            edges = np.array([lo, hi], dtype=float)
    else:
        edges = np.array([0.0, 1.0], dtype=float)

    bins = pd.Series(np.zeros(len(redshift_float), dtype="int16"), index=redshift_float.index)
    return bins, edges


def _assign_bins_with_edges(redshift, edges):
    redshift_float = _to_float_series(redshift)
    edges = np.asarray(edges, dtype=float)
    edges = edges[np.isfinite(edges)]
    edges = np.unique(edges)
    if edges.size == 0:
        edges = np.array([0.0, 1.0], dtype=float)
    elif edges.size == 1:
        edges = np.array([edges[0] - 0.5, edges[0] + 0.5], dtype=float)
    bins = pd.cut(redshift_float, bins=edges, labels=False, include_lowest=True)
    if pd.api.types.is_categorical_dtype(bins.dtype):
        bins = bins.astype(float)
    return pd.Series(bins, index=redshift_float.index).fillna(-1).astype("int16")


def _compute_atmospheric_indices(frame):
    u = _clean_magnitude(frame["u"])
    g = _clean_magnitude(frame["g"])
    r = _clean_magnitude(frame["r"])
    i = _clean_magnitude(frame["i"])
    z = _clean_magnitude(frame["z"])

    ug = u - g
    gr = g - r
    ri = r - i
    iz = i - z

    p1 = 0.91 * ug + 0.415 * gr - 1.280
    v = 0.283 * ug - 0.354 * gr + 0.455 * ri + 0.766 * iz
    l = -0.436 * u + 1.129 * g - 0.119 * r - 0.574 * i + 0.1984
    return p1, v, l


def _fill_with_global(values):
    values = _to_float_series(values)
    finite = np.isfinite(values.to_numpy(dtype=float))
    if finite.any():
        fill = float(np.nanmedian(values.to_numpy(dtype=float)[finite]))
    else:
        fill = 0.0
    if not np.isfinite(fill):
        fill = 0.0
    return values.fillna(fill)


def _robust_zscore(
    values,
    redshift_bin,
    spectral_type,
    aux_values=None,
    aux_bin=None,
    min_cell_count=MIN_CELL_COUNT,
    min_bin_count=MIN_BIN_COUNT,
):
    values = _to_float_series(values)
    idx = values.index

    bins = pd.Series(redshift_bin, index=idx)
    bins = pd.to_numeric(bins, errors="coerce").fillna(-1).astype("int32")
    spectral = pd.Series(spectral_type, index=idx).astype("string").fillna("UNKNOWN").astype(str)

    valid_rows = np.isfinite(values.to_numpy(dtype=float)) & (bins.to_numpy(dtype=float) >= 0)
    base = pd.DataFrame({"value": values, "bin": bins, "spectral_type": spectral}, index=idx).loc[
        valid_rows
    ]

    if not base.empty:
        cell_stats = base.groupby(["bin", "spectral_type"], observed=True)["value"].agg(
            count="size",
            median="median",
            mad=_mad_value,
        )
    else:
        cell_stats = None

    pool = base.loc[:, ["value", "bin"]]

    if aux_values is not None and aux_bin is not None:
        aux_vals = _to_float_series(aux_values)
        aux_bins = pd.Series(aux_bin, index=aux_vals.index)
        aux_bins = pd.to_numeric(aux_bins, errors="coerce").fillna(-1).astype("int32")
        aux_pool = pd.DataFrame({"value": aux_vals, "bin": aux_bins})
        aux_pool = aux_pool.loc[
            np.isfinite(aux_pool["value"].to_numpy(dtype=float)) & (aux_pool["bin"].to_numpy(dtype=float) >= 0),
            ["value", "bin"],
        ]
        if not aux_pool.empty:
            pool = pd.concat([pool, aux_pool], ignore_index=True)

    if not pool.empty:
        bin_stats = pool.groupby("bin")["value"].agg(
            count="size",
            median="median",
            mad=_mad_value,
        )
        all_vals = pool["value"].to_numpy(dtype=float)
        global_median = float(np.nanmedian(all_vals))
        if not np.isfinite(global_median):
            global_median = 0.0
        global_mad = float(np.nanmedian(np.abs(all_vals - global_median)) * MAD_SCALE)
        if not np.isfinite(global_mad):
            global_mad = 0.0
    else:
        bin_stats = None
        global_median = 0.0
        global_mad = 0.0

    if global_mad <= MAD_EPSILON:
        global_mad = 0.0

    key = pd.Series(list(zip(bins.to_numpy(dtype=int), spectral.to_numpy(dtype=object))), index=idx)

    if cell_stats is not None:
        cell_count = key.map(cell_stats["count"].to_dict())
        cell_median = key.map(cell_stats["median"].to_dict())
        cell_mad = key.map(cell_stats["mad"].to_dict())
    else:
        cell_count = pd.Series(np.nan, index=idx)
        cell_median = pd.Series(np.nan, index=idx)
        cell_mad = pd.Series(np.nan, index=idx)

    bin_map = bin_stats["count"].to_dict() if bin_stats is not None else {}
    bin_median_map = bin_stats["median"].to_dict() if bin_stats is not None else {}
    bin_mad_map = bin_stats["mad"].to_dict() if bin_stats is not None else {}

    bin_count = pd.Series(bins.to_numpy(), index=idx).map(bin_map)
    bin_median = pd.Series(bins.to_numpy(), index=idx).map(bin_median_map)
    bin_mad = pd.Series(bins.to_numpy(), index=idx).map(bin_mad_map)

    median = cell_median.copy()
    mad = cell_mad.copy()
    use_cell = cell_count >= min_cell_count

    median = median.where(use_cell, bin_median)
    mad = mad.where(use_cell, bin_mad)

    need_global = (~use_cell) & ((bin_count < min_bin_count) | (bin_count.isna()))
    median = median.where(~need_global, global_median)
    mad = mad.where(~need_global, global_mad)

    median = median.fillna(global_median)
    mad = mad.fillna(0.0)
    mad = mad.where(mad > MAD_EPSILON, 0.0)

    values_np = values.to_numpy(dtype=float)
    median_np = median.to_numpy(dtype=float)
    mad_np = mad.to_numpy(dtype=float)

    z = np.zeros(len(idx), dtype=float)
    valid = np.isfinite(values_np) & np.isfinite(median_np) & (mad_np > MAD_EPSILON)
    if np.any(valid):
        z[valid] = (values_np[valid] - median_np[valid]) / mad_np[valid]

    z = np.clip(z, -CLIP_LIMIT, CLIP_LIMIT)
    return pd.Series(z, index=idx)


def add_segue_stellar_atmospheric_indices(raw, deps, aux):
    idx = raw.index
    raw_bins, raw_edges = _assign_redshift_bins(raw["redshift"])

    p1, v, l = _compute_atmospheric_indices(raw)
    p1 = _fill_with_global(p1)
    v = _fill_with_global(v)
    l = _fill_with_global(l)

    aux_bins = None
    aux_p1 = None
    aux_v = None
    aux_l = None
    if isinstance(aux, pd.DataFrame) and not aux.empty and all(
        col in aux.columns for col in AUX_REQUIRED_COLUMNS
    ):
        aux_bins = _assign_bins_with_edges(aux["redshift"], raw_edges)
        aux_p1, aux_v, aux_l = _compute_atmospheric_indices(aux)
        aux_p1 = _fill_with_global(aux_p1)
        aux_v = _fill_with_global(aux_v)
        aux_l = _fill_with_global(aux_l)

    spectral = (
        raw["spectral_type"]
        if "spectral_type" in raw.columns
        else pd.Series("UNKNOWN", index=idx)
    )

    z_p1 = _robust_zscore(p1, raw_bins, spectral, aux_values=aux_p1, aux_bin=aux_bins)
    z_v = _robust_zscore(v, raw_bins, spectral, aux_values=aux_v, aux_bin=aux_bins)
    z_l = _robust_zscore(l, raw_bins, spectral, aux_values=aux_l, aux_bin=aux_bins)

    p1_interval_violation = np.maximum(
        0.0, np.minimum(p1.to_numpy(dtype=float) + 0.25, -0.7 - p1.to_numpy(dtype=float))
    )
    l_low_tail = np.maximum(0.0, 0.07 - l.to_numpy(dtype=float))
    l_mid_tail = np.maximum(0.0, l.to_numpy(dtype=float) - 0.135)
    v_abs = np.abs(v.to_numpy(dtype=float))

    p1_v_l_outlier = (
        np.abs(z_p1.to_numpy(dtype=float))
        + np.abs(z_v.to_numpy(dtype=float))
        + np.abs(z_l.to_numpy(dtype=float))
    ) / 3.0
    v_abs_z_p1 = v_abs * z_p1.to_numpy(dtype=float)
    v_abs_z_l = v_abs * z_l.to_numpy(dtype=float)

    p1_v_l_outlier = np.clip(p1_v_l_outlier, 0.0, CLIP_LIMIT)
    v_abs_z_p1 = np.clip(v_abs_z_p1, -CLIP_LIMIT, CLIP_LIMIT)
    v_abs_z_l = np.clip(v_abs_z_l, -CLIP_LIMIT, CLIP_LIMIT)

    features = pd.DataFrame(
        {
            "p1": p1,
            "v": v,
            "l": l,
            "z_p1": z_p1,
            "z_v": z_v,
            "z_l": z_l,
            "p1_interval_violation": p1_interval_violation,
            "l_low_tail": l_low_tail,
            "l_mid_tail": l_mid_tail,
            "v_abs": v_abs,
            "p1_v_l_outlier": p1_v_l_outlier,
            "v_abs_z_p1": v_abs_z_p1,
            "v_abs_z_l": v_abs_z_l,
        },
        index=idx,
    )

    return features.replace([np.inf, -np.inf], np.nan).fillna(0.0)


FEATURE_GROUPS = [
    {
        "name": "segue_stellar_atmospheric_indices",
        "fn": add_segue_stellar_atmospheric_indices,
        "depends_on": [],
        "description": "Builds SDSS u,g,r,i,z atmospheric-color diagnostics and robust redshift/spectral-type standardized residual features with fallback binning and tail/intersection terms.",
    }
]
