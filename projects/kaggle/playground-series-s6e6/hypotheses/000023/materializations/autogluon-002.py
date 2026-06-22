import numpy as np
import pandas as pd


def _assign_redshift_bin(redshift, num_bins):
    clipped = redshift.clip(lower=0.0)
    if clipped.isna().any():
        clipped = clipped.fillna(0.0)

    clipped = clipped.astype("float64")
    clipped = clipped.reindex(redshift.index)
    if clipped.empty or clipped.nunique(dropna=True) <= 1:
        return pd.Series(0, index=clipped.index, name="redshift_bin", dtype="int64")

    try:
        bins = pd.qcut(clipped, q=num_bins, labels=False, duplicates="drop")
        if pd.api.types.is_categorical_dtype(bins):
            codes = bins.cat.codes
        else:
            codes = bins
        codes = pd.Series(codes, index=clipped.index, name="redshift_bin").fillna(0).astype("int64")
        return codes

    except ValueError:
        ranked = clipped.rank(method="first", pct=True)
        safe_bins = max(1, min(num_bins, max(1, len(clipped))))
        fallback = np.floor(ranked.to_numpy(dtype="float64") * safe_bins).astype("int64")
        fallback = np.clip(fallback, 0, safe_bins - 1)
        return pd.Series(fallback, index=clipped.index, name="redshift_bin", dtype="int64")


def add_aide_redshift_bin_color_residuals(raw, deps, aux):
    idx = raw.index
    redshift_bin = _assign_redshift_bin(raw["redshift"], 24)

    u = raw["u"].to_numpy(dtype="float64")
    g = raw["g"].to_numpy(dtype="float64")
    r = raw["r"].to_numpy(dtype="float64")
    i = raw["i"].to_numpy(dtype="float64")
    z = raw["z"].to_numpy(dtype="float64")

    color_df = pd.DataFrame(
        {
            "u_g": u - g,
            "g_r": g - r,
            "r_i": r - i,
            "i_z": i - z,
            "u_i": u - i,
            "u_r": u - r,
            "g_z": g - z,
            "r_z": r - z,
            "u_minus_2r_plus_i": u - 2.0 * r + i,
            "g_minus_2i_plus_z": g - 2.0 * i + z,
        },
        index=idx,
    )

    grouped = color_df.join(redshift_bin)
    means = grouped.groupby("redshift_bin", sort=False).transform("mean")
    stds = grouped.groupby("redshift_bin", sort=False).transform("std")

    deltas = color_df - means
    safe_stds = stds.replace(0.0, np.nan)
    zscores = (deltas / safe_stds).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    delta_values = deltas.to_numpy(dtype="float64")
    z_value = zscores.to_numpy(dtype="float64")

    aggregate = pd.DataFrame(
        {
            "color_residual_l2": np.sqrt(np.sum(delta_values * delta_values, axis=1)),
            "color_residual_abs_l1": np.sum(np.abs(delta_values), axis=1),
            "color_residual_mean": np.mean(delta_values, axis=1),
            "color_residual_std": np.std(delta_values, axis=1),
            "color_residual_max_abs_zscore": np.max(np.abs(z_value), axis=1),
        },
        index=idx,
    )

    return pd.concat(
        [
            redshift_bin.to_frame(),
            deltas.add_prefix("color_delta_"),
            zscores.add_prefix("color_zscore_"),
            aggregate,
        ],
        axis=1,
    )


FEATURE_GROUPS = [
    {
        "name": "aide_redshift_bin_color_residuals",
        "fn": add_aide_redshift_bin_color_residuals,
        "depends_on": [],
        "description": "Compute redshift-quantile color residuals and z-scored residual aggregates from photometric color features.",
    }
]