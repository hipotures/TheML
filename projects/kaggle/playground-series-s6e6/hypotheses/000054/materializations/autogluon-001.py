import numpy as np
import pandas as pd

_RESTFRAME_FILTER_WAVELENGTHS_AA = (3551.0, 4686.0, 6165.0, 7481.0, 8931.0)
_RESTFRAME_ANCHORS_AA = (1500.0, 2200.0, 2900.0, 3650.0, 4500.0, 6200.0, 7600.0)


def _interpolate_logflux_at_anchors(sorted_log_wavelengths, sorted_log_flux, anchor_log_wavelengths):
    n_rows = sorted_log_wavelengths.shape[0]
    n_anchors = anchor_log_wavelengths.shape[0]
    n_filters = sorted_log_wavelengths.shape[1]

    interpolated = np.full((n_rows, n_anchors), np.nan, dtype=np.float64)
    mask = np.zeros((n_rows, n_anchors), dtype=bool)
    row_idx = np.arange(n_rows, dtype=np.int64)

    for a_idx, anchor_log in enumerate(anchor_log_wavelengths):
        # how many points are <= anchor (insertion position for each row)
        insert_pos = np.sum(sorted_log_wavelengths <= anchor_log, axis=1)
        in_range = (insert_pos > 0) & (insert_pos < n_filters)

        left_idx = np.where(in_range, insert_pos - 1, 0)
        right_idx = np.where(in_range, insert_pos, 0)

        x_left = sorted_log_wavelengths[row_idx, left_idx]
        x_right = sorted_log_wavelengths[row_idx, right_idx]
        y_left = sorted_log_flux[row_idx, left_idx]
        y_right = sorted_log_flux[row_idx, right_idx]

        interpolated_col = np.full(n_rows, np.nan, dtype=np.float64)
        denom = x_right - x_left
        interpolated_col[in_range] = y_left[in_range] + (
            (anchor_log - x_left[in_range]) * (y_right[in_range] - y_left[in_range]) / denom[in_range]
        )

        interpolated[:, a_idx] = interpolated_col
        mask[:, a_idx] = in_range

    return interpolated, mask


def add_restframe_anchor_sed_shape(raw, deps, aux):
    index = raw.index
    n_rows = raw.shape[0]

    # 1) prepare inputs
    redshift = pd.to_numeric(raw["redshift"], errors="coerce").to_numpy(dtype=np.float64, copy=False)
    redshift_ok = np.isfinite(redshift)
    one_plus_z = np.where(redshift_ok & (redshift < -0.9), 0.1, 1.0 + redshift)
    one_plus_z = np.where(redshift_ok, one_plus_z, np.nan)

    # log10(flux) in each observed band: log10(10^{-0.4m}) = -0.4m
    mags = raw.loc[:, ["u", "g", "r", "i", "z"]].to_numpy(dtype=np.float64, copy=False)
    log_flux = -0.4 * mags

    filter_waves = np.array(_RESTFRAME_FILTER_WAVELENGTHS_AA, dtype=np.float64)
    anchor_waves = np.array(_RESTFRAME_ANCHORS_AA, dtype=np.float64)
    log_anchor_waves = np.log10(anchor_waves)
    anchor_x = np.broadcast_to(log_anchor_waves, (n_rows, log_anchor_waves.size))

    # 2) rest-frame wavelengths and sorted monotonic representation
    log_rest_wave = np.log10(filter_waves[None, :] / one_plus_z[:, None])
    sorted_idx = np.argsort(log_rest_wave, axis=1)
    sorted_log_wave = np.take_along_axis(log_rest_wave, sorted_idx, axis=1)
    sorted_log_flux = np.take_along_axis(log_flux, sorted_idx, axis=1)

    # 3) anchor interpolation + coverage masks
    anchor_log_flux, anchor_mask = _interpolate_logflux_at_anchors(
        sorted_log_wave, sorted_log_flux, log_anchor_waves
    )
    anchor_count = anchor_mask.sum(axis=1).astype(np.float64)

    # segment slopes in log-log space
    segment_defs = (
        ("1500_2200", 0, 1),
        ("2200_2900", 1, 2),
        ("2900_3650", 2, 3),
        ("3650_4500", 3, 4),
        ("4500_6200", 4, 5),
        ("6200_7600", 5, 6),
    )
    segment_slopes = {}
    for name, i0, i1 in segment_defs:
        s = np.full(n_rows, np.nan, dtype=np.float64)
        valid = np.isfinite(anchor_log_flux[:, i0]) & np.isfinite(anchor_log_flux[:, i1])
        dx = log_anchor_waves[i1] - log_anchor_waves[i0]
        s[valid] = (anchor_log_flux[valid, i1] - anchor_log_flux[valid, i0]) / dx
        segment_slopes[name] = s

    # curvature deltas
    d1 = segment_slopes["2200_2900"] - segment_slopes["2900_3650"]
    d2 = segment_slopes["3650_4500"] - segment_slopes["4500_6200"]

    # b4000 residual: anchored at 3650 from 2900-4500 line
    log3650 = log_anchor_waves[3]
    log2900 = log_anchor_waves[2]
    log4500 = log_anchor_waves[4]
    y2900 = anchor_log_flux[:, 2]
    y4500 = anchor_log_flux[:, 4]
    y3650 = anchor_log_flux[:, 3]
    has_2900_4500 = np.isfinite(y2900) & np.isfinite(y4500)
    pred3650 = np.full(n_rows, np.nan, dtype=np.float64)
    pred3650[has_2900_4500] = y2900[has_2900_4500] + (
        (log3650 - log2900) * (y4500[has_2900_4500] - y2900[has_2900_4500]) / (log4500 - log2900)
    )
    b4000 = np.full(n_rows, np.nan, dtype=np.float64)
    has_b4000 = np.isfinite(y3650) & np.isfinite(pred3650)
    b4000[has_b4000] = y3650[has_b4000] - pred3650[has_b4000]

    # global power-law in log-log space
    finite_anchor = np.isfinite(anchor_log_flux)
    cnt = finite_anchor.sum(axis=1).astype(np.float64)
    sx = np.sum(np.where(finite_anchor, anchor_x, 0.0), axis=1)
    sy = np.sum(np.where(finite_anchor, anchor_log_flux, 0.0), axis=1)
    sxx = np.sum(np.where(finite_anchor, anchor_x * anchor_x, 0.0), axis=1)
    sxy = np.sum(np.where(finite_anchor, anchor_x * anchor_log_flux, 0.0), axis=1)
    denom = cnt * sxx - sx * sx

    powerlaw_slope = np.full(n_rows, np.nan, dtype=np.float64)
    powerlaw_intercept = np.full(n_rows, np.nan, dtype=np.float64)
    can_fit_powerlaw = (cnt >= 2) & (np.abs(denom) > 0)
    powerlaw_slope[can_fit_powerlaw] = (
        (cnt[can_fit_powerlaw] * sxy[can_fit_powerlaw] - sx[can_fit_powerlaw] * sy[can_fit_powerlaw])
        / denom[can_fit_powerlaw]
    )
    powerlaw_intercept[can_fit_powerlaw] = (sy[can_fit_powerlaw] - powerlaw_slope[can_fit_powerlaw] * sx[can_fit_powerlaw]) / cnt[
        can_fit_powerlaw
    ]

    powerlaw_pred = powerlaw_intercept[:, None] + powerlaw_slope[:, None] * anchor_x
    powerlaw_resid = np.where(finite_anchor, anchor_log_flux - powerlaw_pred, np.nan)
    powerlaw_rms = np.full(n_rows, np.nan, dtype=np.float64)
    can_eval_powerlaw = cnt >= 2
    if np.any(can_eval_powerlaw):
        sq = np.where(np.isfinite(powerlaw_resid), powerlaw_resid * powerlaw_resid, 0.0)
        powerlaw_rms[can_eval_powerlaw] = np.sqrt(np.sum(sq[can_eval_powerlaw], axis=1) / cnt[can_eval_powerlaw])

    # constrained 2-slope piecewise fit with shared break at 3650
    y0 = y3650.copy()
    # fill 3650 from neighbor anchors if missing
    y0 = np.where(np.isfinite(y0), y0, pred3650)
    # fallback to power-law prediction at 3650 if still missing
    pred3650_from_pow = np.full(n_rows, np.nan, dtype=np.float64)
    pred3650_from_pow[can_fit_powerlaw] = powerlaw_intercept[can_fit_powerlaw] + powerlaw_slope[can_fit_powerlaw] * log3650
    y0 = np.where(np.isfinite(y0), y0, pred3650_from_pow)

    x0 = log3650

    # left side (1500-2900-3650), right side (3650-4500-6200-7600)
    left_x = anchor_x[:, :3]
    left_y = anchor_log_flux[:, :3]
    right_x = anchor_x[:, 4:]
    right_y = anchor_log_flux[:, 4:]

    left_fd = np.isfinite(left_y) & np.isfinite(y0)[:, None]
    right_fd = np.isfinite(right_y) & np.isfinite(y0)[:, None]

    left_dx = left_x - x0
    right_dx = right_x - x0

    left_num = np.sum(np.where(left_fd, left_dx * (left_y - y0[:, None]), 0.0), axis=1)
    left_den = np.sum(np.where(left_fd, left_dx * left_dx, 0.0), axis=1)
    right_num = np.sum(np.where(right_fd, right_dx * (right_y - y0[:, None]), 0.0), axis=1)
    right_den = np.sum(np.where(right_fd, right_dx * right_dx, 0.0), axis=1)

    left_cnt = np.sum(left_fd, axis=1).astype(np.float64)
    right_cnt = np.sum(right_fd, axis=1).astype(np.float64)

    piece_left_slope = np.full(n_rows, np.nan, dtype=np.float64)
    piece_right_slope = np.full(n_rows, np.nan, dtype=np.float64)

    left_ok = (left_cnt >= 1) & np.isfinite(y0) & (left_den != 0)
    right_ok = (right_cnt >= 1) & np.isfinite(y0) & (right_den != 0)

    piece_left_slope[left_ok] = left_num[left_ok] / left_den[left_ok]
    piece_right_slope[right_ok] = right_num[right_ok] / right_den[right_ok]

    # piecewise predictions with continuity at 3650 Å
    piece_pred = np.full_like(anchor_log_flux, np.nan, dtype=np.float64)
    piece_pred[:, 3] = y0

    if np.any(left_ok):
        piece_left_pred = y0[:, None] + piece_left_slope[:, None] * (left_x - x0)
        piece_pred[:, :3] = np.where(left_ok[:, None], piece_left_pred, np.nan)

    if np.any(right_ok):
        piece_right_pred = y0[:, None] + piece_right_slope[:, None] * (right_x - x0)
        piece_pred[:, 4:] = np.where(right_ok[:, None], piece_right_pred, np.nan)

    piece_resid = np.where(finite_anchor & np.isfinite(piece_pred), anchor_log_flux - piece_pred, np.nan)
    piece_cnt = np.sum(np.isfinite(piece_resid), axis=1).astype(np.float64)
    piece_sq = np.where(np.isfinite(piece_resid), piece_resid * piece_resid, 0.0)
    piecewise_rms = np.full(n_rows, np.nan, dtype=np.float64)
    piece_ok = piece_cnt >= 2
    if np.any(piece_ok):
        piecewise_rms[piece_ok] = np.sqrt(np.sum(piece_sq[piece_ok], axis=1) / piece_cnt[piece_ok])

    piecewise_sharpness = np.full(n_rows, np.nan, dtype=np.float64)
    both_sides = np.isfinite(piece_left_slope) & np.isfinite(piece_right_slope)
    piecewise_sharpness[both_sides] = piece_right_slope[both_sides] - piece_left_slope[both_sides]

    features = {}
    anchor_labels = ("1500", "2200", "2900", "3650", "4500", "6200", "7600")

    for idx, label in enumerate(anchor_labels):
        features[f"restframe_anchor_logf_{label}"] = anchor_log_flux[:, idx]
        features[f"restframe_anchor_logf_{label}_valid"] = anchor_mask[:, idx]

    for name, _, _ in segment_defs:
        if name == "1500_2200":
            features["restframe_slope_1500_2200"] = segment_slopes[name]
        elif name == "2200_2900":
            features["restframe_slope_2200_2900"] = segment_slopes[name]
        elif name == "2900_3650":
            features["restframe_slope_2900_3650"] = segment_slopes[name]
        elif name == "3650_4500":
            features["restframe_slope_3650_4500"] = segment_slopes[name]
        elif name == "4500_6200":
            features["restframe_slope_4500_6200"] = segment_slopes[name]
        elif name == "6200_7600":
            features["restframe_slope_6200_7600"] = segment_slopes[name]

    features["restframe_anchor_count"] = anchor_count
    features["restframe_left_anchor_count"] = left_cnt
    features["restframe_right_anchor_count"] = right_cnt

    features["restframe_curvature_d1_uv_bend"] = d1
    features["restframe_curvature_d2_opt_bend"] = d2
    features["restframe_b4000_residual"] = b4000

    features["restframe_powerlaw_slope"] = powerlaw_slope
    features["restframe_powerlaw_intercept"] = powerlaw_intercept
    features["restframe_powerlaw_rms_residual"] = powerlaw_rms
    features["restframe_powerlaw_anchor_count"] = cnt

    features["restframe_piecewise_left_slope"] = piece_left_slope
    features["restframe_piecewise_right_slope"] = piece_right_slope
    features["restframe_piecewise_sharpness"] = piecewise_sharpness
    features["restframe_piecewise_rms_residual"] = piecewise_rms
    features["restframe_piecewise_anchor_count"] = piece_cnt

    return pd.DataFrame(features, index=index)


FEATURE_GROUPS = [
    {
        "name": "restframe_anchor_sed_shape",
        "fn": add_restframe_anchor_sed_shape,
        "depends_on": [],
        "description": "Creates rest-frame log-log SED anchor fluxes at fixed wavelengths plus slope, curvature, break, and fit-mismatch diagnostics.",
    }
]