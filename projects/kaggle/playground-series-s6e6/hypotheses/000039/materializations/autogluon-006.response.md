import numpy as np
import pandas as pd


REDSHIFT_BINS = (-np.inf, 0.02, 0.2, 0.6, 1.2, 2.4, 4.0, np.inf)
DELTA_BINS = (-np.inf, -5.0, 10.0, 25.0, 40.0, 55.0, np.inf)
COORDINATES = ("s", "w", "x", "y", "l")
LEVELS = (
    ("redshift_sky_tag", ("redshift_bin", "alpha_bin", "delta_bin", "tag_pair")),
    ("redshift_sky", ("redshift_bin", "alpha_bin", "delta_bin")),
    ("redshift_tag", ("redshift_bin", "tag_pair")),
    ("redshift", ("redshift_bin",)),
    ("tag", ("tag_pair",)),
    ("global", ()),
)


def _safe_series(raw, name, default=0.0):
    if name in raw.columns:
        return raw[name]
    return pd.Series(default, index=raw.index)


def _mad(values):
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.nan
    med = np.median(arr)
    return np.median(np.abs(arr - med))


def _group_stats(frame, coord, keys, valid_col):
    if not keys:
        valid = frame[valid_col].to_numpy(dtype=bool)
        vals = frame.loc[valid, coord].to_numpy(dtype=np.float64)
        return {
            "median": np.nanmedian(vals) if vals.size else 0.0,
            "mad": _mad(vals),
            "count": int(vals.size),
        }

    stats = (
        frame.loc[frame[valid_col], list(keys) + [coord]]
        .groupby(list(keys), sort=False, observed=True)[coord]
        .agg(median="median", mad=_mad, count="count")
    )
    return stats


def _lookup_stats(context, stats, keys):
    if not keys:
        n = len(context)
        return (
            np.full(n, float(stats["median"]), dtype=np.float64),
            np.full(n, float(stats["mad"]), dtype=np.float64),
            np.full(n, int(stats["count"]), dtype=np.int32),
        )

    lookup = context[list(keys)].merge(
        stats.reset_index(),
        on=list(keys),
        how="left",
        sort=False,
    )
    return (
        lookup["median"].to_numpy(dtype=np.float64),
        lookup["mad"].to_numpy(dtype=np.float64),
        lookup["count"].fillna(0).to_numpy(dtype=np.int32),
    )


def _resolve_coordinate_stats(frame, context, coord, valid_col, min_count):
    level_names = []
    medians = []
    mads = []
    counts = []

    for level_name, keys in LEVELS:
        stats = _group_stats(frame, coord, keys, valid_col)
        med, mad, cnt = _lookup_stats(context, stats, keys)
        level_names.append(level_name)
        medians.append(med)
        mads.append(mad)
        counts.append(cnt)

    selected_median = medians[-1].copy()
    selected_mad = mads[-1].copy()
    selected_count = counts[-1].copy()
    selected_level = np.full(len(frame), len(LEVELS) - 1, dtype=np.int16)

    for idx in range(len(LEVELS) - 2, -1, -1):
        ok = counts[idx] >= min_count
        selected_median[ok] = medians[idx][ok]
        selected_mad[ok] = mads[idx][ok]
        selected_count[ok] = counts[idx][ok]
        selected_level[ok] = idx

    selected_mad = np.where(np.isfinite(selected_mad), selected_mad, mads[-1])
    selected_median = np.where(np.isfinite(selected_median), selected_median, medians[-1])
    selected_count = np.where(selected_count > 0, selected_count, counts[-1])

    return selected_median, selected_mad, selected_count, selected_level


def add_local_principal_color_residuals(raw, deps, aux):
    index = raw.index
    u = _safe_series(raw, "u").astype("float64")
    g = _safe_series(raw, "g").astype("float64")
    r = _safe_series(raw, "r").astype("float64")
    i = _safe_series(raw, "i").astype("float64")
    z = _safe_series(raw, "z").astype("float64")
    alpha = _safe_series(raw, "alpha").astype("float64")
    delta = _safe_series(raw, "delta").astype("float64")
    redshift = _safe_series(raw, "redshift").astype("float64")

    frame = pd.DataFrame(index=index)
    frame["s"] = -0.249 * u + 0.794 * g - 0.555 * r + 0.234
    frame["w"] = -0.227 * g + 0.792 * r - 0.567 * i + 0.050
    frame["x"] = 0.707 * g - 0.707 * r - 0.983
    frame["y"] = -0.270 * r + 0.800 * i - 0.534 * z + 0.059
    frame["l"] = -0.436 * u + 1.129 * g - 0.119 * r - 0.574 * i + 0.1984

    gr = g - r
    l_valid = ((gr > 0.5) & (gr < 0.8)).to_numpy(dtype=bool)
    finite_base = np.isfinite(frame[["s", "w", "x", "y", "l"]]).all(axis=1).to_numpy(dtype=bool)

    for coord in COORDINATES:
        frame[coord + "_valid"] = finite_base
    frame["l_valid"] = l_valid & np.isfinite(frame["l"].to_numpy(dtype=np.float64))

    context = pd.DataFrame(index=index)
    context["redshift_bin"] = pd.cut(redshift, bins=list(REDSHIFT_BINS), labels=False, include_lowest=True).astype("Int64").astype(str)
    context["alpha_bin"] = np.floor((np.mod(alpha.to_numpy(dtype=np.float64), 360.0)) / 30.0).astype("int16").astype(str)
    context["delta_bin"] = pd.cut(delta, bins=list(DELTA_BINS), labels=False, include_lowest=True).astype("Int64").astype(str)

    spectral = _safe_series(raw, "spectral_type", "missing").astype(str)
    population = _safe_series(raw, "galaxy_population", "missing").astype(str)
    context["tag_pair"] = spectral + "|" + population

    for col in context.columns:
        frame[col] = context[col].to_numpy()

    out = pd.DataFrame(index=index)
    z_values = []

    for coord in COORDINATES:
        valid_col = coord + "_valid"
        min_count = 150 if coord == "l" else 300
        min_scale = 0.03 if coord == "l" else 0.02

        median, mad, count, level = _resolve_coordinate_stats(frame, context, coord, valid_col, min_count)
        scale = np.maximum(1.4826 * np.nan_to_num(mad, nan=0.0), min_scale)
        raw_values = frame[coord].to_numpy(dtype=np.float64)

        residual = raw_values - median
        score = np.clip(residual / scale, -10.0, 10.0)
        valid = frame[valid_col].to_numpy(dtype=bool)

        residual = np.where(valid, residual, 0.0)
        score = np.where(valid, score, 0.0)

        out[coord + "_residual"] = residual.astype("float32")
        out[coord + "_z"] = score.astype("float32")
        out[coord + "_abs_z"] = np.abs(score).astype("float32")
        out[coord + "_valid"] = valid.astype("int8")
        out[coord + "_context_count"] = count.astype("int32")
        out[coord + "_backoff_level"] = level.astype("int8")
        out[coord + "_scale"] = scale.astype("float32")
        z_values.append(score)

    z_matrix = np.vstack(z_values).T
    valid_matrix = out[[coord + "_valid" for coord in COORDINATES]].to_numpy(dtype=bool)
    abs_z = np.abs(z_matrix)
    masked_abs = np.where(valid_matrix, abs_z, np.nan)
    available = valid_matrix.sum(axis=1)

    out["max_abs_z"] = np.nanmax(masked_abs, axis=1).astype("float32")
    out["mean_abs_z"] = np.nanmean(masked_abs, axis=1).astype("float32")
    out["l2_z"] = np.sqrt(np.nanmean(np.where(valid_matrix, z_matrix * z_matrix, np.nan), axis=1) * available).astype("float32")
    out["tail_count_2"] = ((abs_z > 2.0) & valid_matrix).sum(axis=1).astype("int8")
    out["tail_count_35"] = ((abs_z > 3.5) & valid_matrix).sum(axis=1).astype("int8")
    out["tail_ratio_2"] = np.divide(out["tail_count_2"], np.maximum(available, 1)).astype("float32")
    out["positive_tail_count"] = ((z_matrix > 2.0) & valid_matrix).sum(axis=1).astype("int8")
    out["negative_tail_count"] = ((z_matrix < -2.0) & valid_matrix).sum(axis=1).astype("int8")
    out["available_color_count"] = available.astype("int8")
    out["in_locus"] = ((np.nanmax(masked_abs, axis=1) < 0.75) & (available >= 4)).astype("int8")

    sign_bits = (
        (z_matrix[:, 0] > 0).astype("int8")
        + 2 * (z_matrix[:, 1] > 0).astype("int8")
        + 4 * (z_matrix[:, 2] > 0).astype("int8")
        + 8 * (z_matrix[:, 3] > 0).astype("int8")
    )
    out["swxy_sign_code"] = sign_bits.astype("int8")
    out["l_color_valid"] = l_valid.astype("int8")

    return out


FEATURE_GROUPS = [
    {
        "name": "local_principal_color_residuals",
        "fn": add_local_principal_color_residuals,
        "depends_on": [],
        "description": "Robust locally normalized SDSS principal-color residuals with hierarchical context backoff.",
    }
]