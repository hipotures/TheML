import numpy as np
import pandas as pd


FILTER_COLUMNS = ("u", "g", "r", "i", "z")
FILTER_WAVELENGTHS_ANGSTROM = (3551.0, 4686.0, 6165.0, 7481.0, 8931.0)
ANCHOR_WAVELENGTHS_ANGSTROM = (1500.0, 2200.0, 2900.0, 3650.0, 4500.0, 6200.0, 7600.0)
ANCHOR_NAMES = ("1500", "2200", "2900", "3650", "4500", "6200", "7600")


def add_restframe_anchor_sed_shape(raw, deps, aux):
    n_rows = len(raw)
    index = raw.index

    mags = raw.loc[:, FILTER_COLUMNS].to_numpy(dtype=float, copy=True)
    redshift = raw["redshift"].to_numpy(dtype=float, copy=True)
    scale = np.maximum(1.0 + redshift, 1.0e-3)

    filter_waves = np.asarray(FILTER_WAVELENGTHS_ANGSTROM, dtype=float)
    anchors = np.asarray(ANCHOR_WAVELENGTHS_ANGSTROM, dtype=float)
    log_filter_waves = np.log10(filter_waves)
    log_anchors = np.log10(anchors)

    x = log_filter_waves[None, :] - np.log10(scale)[:, None]
    y = -0.4 * mags

    x_min = x[:, 0]
    x_max = x[:, -1]
    y_anchor = np.full((n_rows, len(anchors)), np.nan, dtype=float)
    coverage = np.zeros((n_rows, len(anchors)), dtype=bool)

    for anchor_idx, log_anchor in enumerate(log_anchors):
        covered = (log_anchor >= x_min) & (log_anchor <= x_max)
        coverage[:, anchor_idx] = covered

        interval = np.searchsorted(log_filter_waves, log_anchor + np.log10(scale), side="right") - 1
        interval = np.clip(interval, 0, len(filter_waves) - 2)

        row_pos = np.arange(n_rows)
        x0 = x[row_pos, interval]
        x1 = x[row_pos, interval + 1]
        y0 = y[row_pos, interval]
        y1 = y[row_pos, interval + 1]
        weight = (log_anchor - x0) / (x1 - x0)

        vals = y0 + weight * (y1 - y0)
        y_anchor[covered, anchor_idx] = vals[covered]

    features = {}

    for anchor_idx, anchor_name in enumerate(ANCHOR_NAMES):
        features[f"anchor_{anchor_name}_logflux"] = y_anchor[:, anchor_idx]
        features[f"has_anchor_{anchor_name}"] = coverage[:, anchor_idx].astype(np.int8)

    n_anchor = coverage.sum(axis=1)
    features["n_anchor"] = n_anchor.astype(np.int16)
    features["frac_anchor"] = n_anchor.astype(float) / float(len(anchors))

    covered_waves = np.where(coverage, anchors[None, :], np.nan)
    features["min_covered_anchor_wave"] = np.where(n_anchor > 0, np.nanmin(covered_waves, axis=1), np.nan)
    features["max_covered_anchor_wave"] = np.where(n_anchor > 0, np.nanmax(covered_waves, axis=1), np.nan)

    median_anchor = np.nanmedian(y_anchor, axis=1)
    y_rel = y_anchor - median_anchor[:, None]
    y_rel[n_anchor == 0, :] = np.nan

    for anchor_idx, anchor_name in enumerate(ANCHOR_NAMES):
        features[f"anchor_{anchor_name}_yrel"] = y_rel[:, anchor_idx]

    slopes = np.full((n_rows, len(anchors) - 1), np.nan, dtype=float)
    segment_valid = coverage[:, :-1] & coverage[:, 1:]
    for seg_idx in range(len(anchors) - 1):
        denom = log_anchors[seg_idx + 1] - log_anchors[seg_idx]
        slopes[:, seg_idx] = (y_anchor[:, seg_idx + 1] - y_anchor[:, seg_idx]) / denom
        slopes[~segment_valid[:, seg_idx], seg_idx] = np.nan
        features[f"slope_{ANCHOR_NAMES[seg_idx]}_{ANCHOR_NAMES[seg_idx + 1]}"] = slopes[:, seg_idx]
        features[f"has_slope_{ANCHOR_NAMES[seg_idx]}_{ANCHOR_NAMES[seg_idx + 1]}"] = segment_valid[:, seg_idx].astype(np.int8)

    slope_changes = slopes[:, 1:] - slopes[:, :-1]
    slope_change_valid = segment_valid[:, 1:] & segment_valid[:, :-1]
    slope_changes[~slope_change_valid] = np.nan

    features["uv_curv"] = slope_changes[:, 0]
    features["near_break_curv"] = slope_changes[:, 2]
    features["red_curv"] = slope_changes[:, 4]
    features["mean_adjacent_slope_change"] = np.nanmean(slope_changes, axis=1)
    features["std_adjacent_slope_change"] = np.nanstd(slope_changes, axis=1)
    features["max_abs_adjacent_slope_change"] = np.nanmax(np.abs(slope_changes), axis=1)
    features["n_adjacent_slope_change"] = np.sum(slope_change_valid, axis=1).astype(np.int16)

    b3650 = np.full(n_rows, np.nan, dtype=float)
    b3650_valid = coverage[:, 2] & coverage[:, 3] & coverage[:, 4]
    pred3650 = y_anchor[:, 2] + (
        (log_anchors[3] - log_anchors[2]) / (log_anchors[4] - log_anchors[2])
    ) * (y_anchor[:, 4] - y_anchor[:, 2])
    b3650[b3650_valid] = y_anchor[b3650_valid, 3] - pred3650[b3650_valid]
    features["b3650_residual"] = b3650
    features["has_b3650_residual"] = b3650_valid.astype(np.int8)

    blue_side = coverage[:, :3]
    red_side = coverage[:, 4:]
    blue_count = blue_side.sum(axis=1)
    red_count = red_side.sum(axis=1)
    blue_mean = np.nanmean(np.where(blue_side, y_rel[:, :3], np.nan), axis=1)
    red_mean = np.nanmean(np.where(red_side, y_rel[:, 4:], np.nan), axis=1)
    broad_break_valid = (blue_count > 0) & (red_count > 0)
    broad_break = np.full(n_rows, np.nan, dtype=float)
    broad_break[broad_break_valid] = red_mean[broad_break_valid] - blue_mean[broad_break_valid]
    features["broad_break_contrast"] = broad_break
    features["has_broad_break_contrast"] = broad_break_valid.astype(np.int8)

    x_anchor = np.broadcast_to(log_anchors, y_anchor.shape)
    valid_float = coverage.astype(float)

    x_mean = np.divide(
        np.sum(np.where(coverage, x_anchor, 0.0), axis=1),
        n_anchor,
        out=np.full(n_rows, np.nan, dtype=float),
        where=n_anchor > 0,
    )
    y_mean = np.divide(
        np.sum(np.where(coverage, y_anchor, 0.0), axis=1),
        n_anchor,
        out=np.full(n_rows, np.nan, dtype=float),
        where=n_anchor > 0,
    )
    x_centered = np.where(coverage, x_anchor - x_mean[:, None], 0.0)
    y_centered = np.where(coverage, y_anchor - y_mean[:, None], 0.0)
    denom = np.sum(x_centered * x_centered * valid_float, axis=1)

    global_slope = np.divide(
        np.sum(x_centered * y_centered * valid_float, axis=1),
        denom,
        out=np.full(n_rows, np.nan, dtype=float),
        where=(n_anchor >= 2) & (denom > 0),
    )
    global_intercept = y_mean - global_slope * x_mean
    global_pred = global_slope[:, None] * x_anchor + global_intercept[:, None]
    global_resid = np.where(coverage, y_anchor - global_pred, np.nan)

    features["global_slope"] = global_slope
    features["global_rmse"] = np.sqrt(np.nanmean(global_resid * global_resid, axis=1))
    features["global_mae_resid"] = np.nanmean(np.abs(global_resid), axis=1)
    features["global_max_abs_resid"] = np.nanmax(np.abs(global_resid), axis=1)
    features["global_resid_3650"] = global_resid[:, 3]
    features["has_global_line"] = ((n_anchor >= 2) & (denom > 0)).astype(np.int8)

    side_specs = (
        ("blue", np.asarray((True, True, True, True, False, False, False), dtype=bool)),
        ("red", np.asarray((False, False, False, True, True, True, True), dtype=bool)),
    )
    side_predictions_3650 = {}

    for side_name, side_mask in side_specs:
        valid = coverage & side_mask[None, :]
        count = valid.sum(axis=1)
        valid_float_side = valid.astype(float)

        side_x_mean = np.divide(
            np.sum(np.where(valid, x_anchor, 0.0), axis=1),
            count,
            out=np.full(n_rows, np.nan, dtype=float),
            where=count > 0,
        )
        side_y_mean = np.divide(
            np.sum(np.where(valid, y_anchor, 0.0), axis=1),
            count,
            out=np.full(n_rows, np.nan, dtype=float),
            where=count > 0,
        )
        side_x_centered = np.where(valid, x_anchor - side_x_mean[:, None], 0.0)
        side_y_centered = np.where(valid, y_anchor - side_y_mean[:, None], 0.0)
        side_denom = np.sum(side_x_centered * side_x_centered * valid_float_side, axis=1)

        side_slope = np.divide(
            np.sum(side_x_centered * side_y_centered * valid_float_side, axis=1),
            side_denom,
            out=np.full(n_rows, np.nan, dtype=float),
            where=(count >= 2) & (side_denom > 0),
        )
        side_intercept = side_y_mean - side_slope * side_x_mean
        side_pred = side_slope[:, None] * x_anchor + side_intercept[:, None]
        side_resid = np.where(valid, y_anchor - side_pred, np.nan)

        features[f"{side_name}_slope"] = side_slope
        features[f"{side_name}_rmse"] = np.sqrt(np.nanmean(side_resid * side_resid, axis=1))
        features[f"n_{side_name}_line_anchor"] = count.astype(np.int16)
        features[f"has_{side_name}_line"] = ((count >= 2) & (side_denom > 0)).astype(np.int8)
        side_predictions_3650[side_name] = side_slope * log_anchors[3] + side_intercept

    both_side_lines = np.isfinite(side_predictions_3650["blue"]) & np.isfinite(side_predictions_3650["red"])
    slope_jump = features["red_slope"] - features["blue_slope"]
    break_sharpness = np.full(n_rows, np.nan, dtype=float)
    break_sharpness[both_side_lines] = (
        side_predictions_3650["red"][both_side_lines] - side_predictions_3650["blue"][both_side_lines]
    )

    features["slope_jump"] = slope_jump
    features["break_sharpness"] = break_sharpness
    features["has_two_side_lines"] = both_side_lines.astype(np.int8)

    return pd.DataFrame(features, index=index)


FEATURE_GROUPS = [
    {
        "name": "restframe_anchor_sed_shape",
        "fn": add_restframe_anchor_sed_shape,
        "depends_on": [],
        "description": "Rest-frame broadband SED slope, curvature, anchor coverage, and break-shape descriptors from ugriz magnitudes.",
    }
]