import numpy as np
import pandas as pd

BANDS = ("u", "g", "r", "i", "z")
BAND_WAVELENGTHS = (3551.0, 4686.0, 6165.0, 7481.0, 8931.0)
ANCHOR_WAVELENGTHS = (1500.0, 2200.0, 2900.0, 3650.0, 4500.0, 6200.0, 7600.0)


def _evaluate_anchor_values(x_sorted, y_sorted, anchor_x):
    n_rows = x_sorted.shape[0]
    anchor_x = np.asarray(anchor_x, dtype=float)
    n_anchors = anchor_x.size
    out = np.full((n_rows, n_anchors), np.nan, dtype=float)
    row_ids = np.arange(n_rows)
    x_min = x_sorted[:, 0]
    x_max = x_sorted[:, -1]
    n_points = x_sorted.shape[1]

    for idx, x_a in enumerate(anchor_x):
        inside = (x_a >= x_min) & (x_a <= x_max)
        if not np.any(inside):
            continue

        active_rows = row_ids[inside]
        left = np.sum(x_sorted[inside] <= x_a, axis=1) - 1
        idx_right = np.clip(left, 1, n_points - 1)

        x0 = x_sorted[active_rows, idx_right - 1]
        x1 = x_sorted[active_rows, idx_right]
        y0 = y_sorted[active_rows, idx_right - 1]
        y1 = y_sorted[active_rows, idx_right]

        denom = x1 - x0
        valid = (
            np.isfinite(x0)
            & np.isfinite(x1)
            & np.isfinite(y0)
            & np.isfinite(y1)
            & (np.abs(denom) > 0.0)
        )
        if not np.any(valid):
            continue

        rows_valid = active_rows[valid]
        out[rows_valid, idx] = y0[valid] + (y1[valid] - y0[valid]) * (x_a - x0[valid]) / denom[valid]

    return out


def _fit_linear(x_anchor, y_anchor, anchor_mask):
    n_rows = y_anchor.shape[0]
    x = np.asarray(x_anchor, dtype=float)
    x_mat = np.broadcast_to(x, y_anchor.shape)
    x2_mat = x_mat * x_mat

    valid = anchor_mask.astype(bool)

    counts = valid.sum(axis=1).astype(float)
    sx = np.sum(np.where(valid, x_mat, 0.0), axis=1)
    sy = np.sum(np.where(valid, y_anchor, 0.0), axis=1)
    sxx = np.sum(np.where(valid, x2_mat, 0.0), axis=1)
    sxy = np.sum(np.where(valid, x_mat * y_anchor, 0.0), axis=1)

    denom = counts * sxx - sx * sx
    good = (counts >= 2.0) & (np.abs(denom) > 1.0e-15)

    slope = np.full(n_rows, np.nan, dtype=float)
    intercept = np.full(n_rows, np.nan, dtype=float)

    slope[good] = (counts[good] * sxy[good] - sx[good] * sy[good]) / denom[good]
    intercept[good] = (sy[good] - slope[good] * sx[good]) / counts[good]

    pred = slope[:, None] * x + intercept[:, None]
    residual = np.where(valid, y_anchor - pred, np.nan)

    rmse = np.sqrt(np.nanmean(residual * residual, axis=1))
    mae = np.nanmean(np.abs(residual), axis=1)
    rmse[~good] = np.nan
    mae[~good] = np.nan

    return slope, intercept, rmse, mae, good


def add_restframe_anchor_sed_shape(raw, deps, aux):
    del deps
    del aux

    mags = raw.loc[:, BANDS].to_numpy(dtype=float)
    redshift = raw["redshift"].to_numpy(dtype=float)

    denom = 1.0 + redshift
    denom = np.where(denom > 1.0e-3, denom, 1.0e-3)

    wavelengths = np.array(BAND_WAVELENGTHS, dtype=float)
    x_band = np.log10(wavelengths[None, :] / denom[:, None])
    y_band = -0.4 * mags

    order = np.argsort(x_band, axis=1)
    x_sorted = np.take_along_axis(x_band, order, axis=1)
    y_sorted = np.take_along_axis(y_band, order, axis=1)

    anchors = np.array(ANCHOR_WAVELENGTHS, dtype=float)
    anchor_x = np.log10(anchors)
    anchor_y = _evaluate_anchor_values(x_sorted, y_sorted, anchor_x)

    anchor_mask = np.isfinite(anchor_y).astype("int8")
    n_anchor = anchor_mask.sum(axis=1).astype("int16")
    frac_anchor = n_anchor / float(anchor_y.shape[1])
    no_data_flag = (n_anchor < 2).astype("int8")

    seg_denom = np.diff(anchor_x)
    seg_names = (
        "1500_2200",
        "2200_2900",
        "2900_3650",
        "3650_4500",
        "4500_6200",
        "6200_7600",
    )
    seg_slopes = []
    seg_masks = []
    for i in range(6):
        valid_pair = (anchor_mask[:, i] == 1) & (anchor_mask[:, i + 1] == 1)
        slope = np.full_like(n_anchor, np.nan, dtype=float)
        slope[valid_pair] = (anchor_y[:, i + 1] - anchor_y[:, i]) / seg_denom[i]
        seg_slopes.append(slope)
        seg_masks.append(valid_pair.astype("int8"))

    seg_slopes = np.vstack(seg_slopes).T

    curvature_uv = np.full_like(seg_slopes[:, 0], np.nan, dtype=float)
    uv_ok = np.isfinite(seg_slopes[:, 1]) & np.isfinite(seg_slopes[:, 0])
    curvature_uv[uv_ok] = seg_slopes[uv_ok, 1] - seg_slopes[uv_ok, 0]

    curvature_opt = np.full_like(seg_slopes[:, 0], np.nan, dtype=float)
    opt_ok = np.isfinite(seg_slopes[:, 3]) & np.isfinite(seg_slopes[:, 2])
    curvature_opt[opt_ok] = seg_slopes[opt_ok, 3] - seg_slopes[opt_ok, 2]

    y_3650 = anchor_y[:, 3]
    y_2900 = anchor_y[:, 2]
    y_4500 = anchor_y[:, 4]
    x_2900, x_3650, x_4500 = anchor_x[2], anchor_x[3], anchor_x[4]

    y3650_from_2900_4500 = np.full_like(y_3650, np.nan, dtype=float)
    bracketing = (anchor_mask[:, 2] == 1) & (anchor_mask[:, 4] == 1)
    if np.any(bracketing):
        slope_bracket = (y_4500[bracketing] - y_2900[bracketing]) / (x_4500 - x_2900)
        y3650_from_2900_4500[bracketing] = y_2900[bracketing] + slope_bracket * (x_3650 - x_2900)

    b4000 = np.full_like(y_3650, np.nan, dtype=float)
    valid_b4000 = (anchor_mask[:, 3] == 1) & bracketing
    b4000[valid_b4000] = y_3650[valid_b4000] - y3650_from_2900_4500[valid_b4000]

    global_slope, global_intercept, global_rmse, global_mae, global_ok = _fit_linear(anchor_x, anchor_y, anchor_mask)
    global_pred = global_slope[:, None] * anchor_x + global_intercept[:, None]
    global_err = np.where(anchor_mask == 1, np.abs(anchor_y - global_pred), np.nan)

    stability_abs_err_mean = np.nanmean(global_err, axis=1)
    stability_abs_err_max = np.nanmax(global_err, axis=1)
    stability_abs_err_mean[~global_ok] = np.nan
    stability_abs_err_max[~global_ok] = np.nan

    blue_slope, blue_intercept, blue_rmse, _, blue_ok = _fit_linear(anchor_x[:4], anchor_y[:, :4], anchor_mask[:, :4])
    red_slope, red_intercept, red_rmse, _, red_ok = _fit_linear(anchor_x[4:], anchor_y[:, 4:], anchor_mask[:, 4:])

    break_sharpness = np.full(n_anchor.shape[0], np.nan, dtype=float)
    both_ok = blue_ok & red_ok
    if np.any(both_ok):
        y_blue3650 = blue_slope[both_ok] * x_3650 + blue_intercept[both_ok]
        y_red3650 = red_slope[both_ok] * x_3650 + red_intercept[both_ok]
        break_sharpness[both_ok] = y_blue3650 - y_red3650

    out = pd.DataFrame(index=raw.index)

    anchor_labels = ("1500", "2200", "2900", "3650", "4500", "6200", "7600")
    for j, label in enumerate(anchor_labels):
        out[f"anchor_{label}_y"] = anchor_y[:, j]
        out[f"anchor_{label}_mask"] = anchor_mask[:, j]

    out["n_anchor"] = n_anchor
    out["frac_anchor"] = frac_anchor
    out["no_data_flag"] = no_data_flag

    for j, seg_name in enumerate(seg_names):
        out[f"seg_{seg_name}_slope"] = seg_slopes[:, j]
        out[f"seg_{seg_name}_mask"] = seg_masks[j]

    out["curvature_uv"] = curvature_uv
    out["curvature_opt"] = curvature_opt
    out["b4000_residual"] = b4000
    out["global_slope"] = global_slope
    out["global_intercept"] = global_intercept
    out["global_rmse"] = global_rmse
    out["global_mae"] = global_mae
    out["blue_slope"] = blue_slope
    out["blue_intercept"] = blue_intercept
    out["blue_rmse"] = blue_rmse
    out["red_slope"] = red_slope
    out["red_intercept"] = red_intercept
    out["red_rmse"] = red_rmse
    out["break_sharpness"] = break_sharpness
    out["stability_abs_err_mean"] = stability_abs_err_mean
    out["stability_abs_err_max"] = stability_abs_err_max

    return out


FEATURE_GROUPS = [
    {
        "name": "restframe_anchor_sed_shape",
        "fn": add_restframe_anchor_sed_shape,
        "depends_on": [],
        "description": "Build rest-frame broadband-shape descriptors from ugriz by interpolating onto fixed wavelength anchors and deriving segment slopes, curvatures, and OLS profile metrics.",
    }
]