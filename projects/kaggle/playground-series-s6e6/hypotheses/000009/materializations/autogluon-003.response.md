import numpy as np
import pandas as pd


def add_canonical_locus_coordinates(raw, deps, aux):
    u = raw["u"].to_numpy(dtype="float64", copy=False)
    g = raw["g"].to_numpy(dtype="float64", copy=False)
    r = raw["r"].to_numpy(dtype="float64", copy=False)
    i = raw["i"].to_numpy(dtype="float64", copy=False)
    z = raw["z"].to_numpy(dtype="float64", copy=False)

    ug = u - g
    gr = g - r
    ri = r - i
    iz = i - z

    s = -0.249 * u + 0.794 * g - 0.555 * r + 0.234
    w = -0.227 * g + 0.792 * r - 0.567 * i + 0.050
    x = 0.707 * g - 0.707 * r - 0.988
    y = -0.270 * r + 0.800 * i - 0.534 * z + 0.054

    p1_s = 0.910 * u - 0.495 * g - 0.415 * r - 1.280
    p1_w = 0.928 * g - 0.556 * r - 0.372 * i - 0.425
    p1_x = ri
    p1_y = 0.895 * r - 0.448 * i - 0.447 * z - 0.600

    ns = np.clip(s / 0.031, -12.0, 12.0)
    nw = np.clip(w / 0.025, -12.0, 12.0)
    nx = np.clip(x / 0.042, -12.0, 12.0)
    ny = np.clip(y / 0.023, -12.0, 12.0)

    ms = (r <= 19.0) & (p1_s >= -0.2) & (p1_s <= 0.8)
    mw = (r <= 20.0) & (p1_w >= -0.2) & (p1_w <= 0.6)
    mx = (r <= 19.0) & (p1_x >= 0.8) & (p1_x <= 1.6)
    my = (r <= 19.5) & (p1_y >= 0.1) & (p1_y <= 1.2)

    normalized = np.column_stack((ns, nw, nx, ny))
    valid = np.column_stack((ms, mw, mx, my))
    active_count = valid.sum(axis=1).astype("int16")
    active_ratio = active_count.astype("float64") / 4.0

    effective_count = np.where(active_count > 0, active_count, 4).astype("float64")
    effective_valid = np.where((active_count > 0)[:, None], valid, True)
    effective = np.where(effective_valid, normalized, np.nan)
    abs_effective = np.abs(effective)

    min_abs = np.nanmin(abs_effective, axis=1)
    mean_abs = np.nanmean(abs_effective, axis=1)
    median_abs = np.nanmedian(abs_effective, axis=1)
    max_abs = np.nanmax(abs_effective, axis=1)
    std_abs = np.nanstd(abs_effective, axis=1)
    rms_abs = np.sqrt(np.nanmean(effective * effective, axis=1))
    signed_sum = np.nansum(effective, axis=1)
    signed_mean = signed_sum / effective_count

    count_abs_le_1 = np.nansum(abs_effective <= 1.0, axis=1).astype("int16")
    count_abs_le_2 = np.nansum(abs_effective <= 2.0, axis=1).astype("int16")
    count_abs_between_2_4 = np.nansum((abs_effective > 2.0) & (abs_effective <= 4.0), axis=1).astype("int16")
    count_abs_gt_4 = np.nansum(abs_effective > 4.0, axis=1).astype("int16")

    abs_all = np.abs(normalized)
    all_count_abs_le_2 = np.sum(abs_all <= 2.0, axis=1).astype("int16")
    all_count_abs_gt_4 = np.sum(abs_all > 4.0, axis=1).astype("int16")

    c_perp = ri - gr / 4.0 - 0.18
    c_perp_clipped = np.clip(c_perp, -2.0, 2.0)
    c_par = 0.7 * gr + 1.2 * (ri - 0.18)

    return pd.DataFrame(
        {
            "ug_color": ug,
            "gr_color": gr,
            "ri_color": ri,
            "iz_color": iz,
            "s_residual": s,
            "w_residual": w,
            "x_residual": x,
            "y_residual": y,
            "ns_clipped": ns,
            "nw_clipped": nw,
            "nx_clipped": nx,
            "ny_clipped": ny,
            "p1_s_coordinate": p1_s,
            "p1_w_coordinate": p1_w,
            "p1_x_coordinate": p1_x,
            "p1_y_coordinate": p1_y,
            "s_valid_mask": ms,
            "w_valid_mask": mw,
            "x_valid_mask": mx,
            "y_valid_mask": my,
            "active_count": active_count,
            "active_ratio": active_ratio,
            "effective_min_abs": min_abs,
            "effective_mean_abs": mean_abs,
            "effective_median_abs": median_abs,
            "effective_max_abs": max_abs,
            "effective_std_abs": std_abs,
            "effective_rms_abs": rms_abs,
            "effective_signed_sum": signed_sum,
            "effective_signed_mean": signed_mean,
            "effective_count_abs_le_1": count_abs_le_1,
            "effective_count_abs_le_2": count_abs_le_2,
            "effective_count_abs_between_2_4": count_abs_between_2_4,
            "effective_count_abs_gt_4": count_abs_gt_4,
            "effective_frac_abs_le_1": count_abs_le_1 / effective_count,
            "effective_frac_abs_le_2": count_abs_le_2 / effective_count,
            "effective_frac_abs_between_2_4": count_abs_between_2_4 / effective_count,
            "effective_frac_abs_gt_4": count_abs_gt_4 / effective_count,
            "all_axis_mean_abs": abs_all.mean(axis=1),
            "all_axis_max_abs": abs_all.max(axis=1),
            "all_axis_rms_abs": np.sqrt(np.mean(normalized * normalized, axis=1)),
            "all_axis_signed_sum": normalized.sum(axis=1),
            "all_axis_count_abs_le_2": all_count_abs_le_2,
            "all_axis_count_abs_gt_4": all_count_abs_gt_4,
            "c_perp": c_perp,
            "c_perp_abs": np.abs(c_perp),
            "c_perp_clipped_squared": c_perp_clipped * c_perp_clipped,
            "c_par": c_par,
            "c_perp_sign": np.sign(c_perp),
        },
        index=raw.index,
    )


FEATURE_GROUPS = [
    {
        "name": "canonical_locus_coordinates",
        "fn": add_canonical_locus_coordinates,
        "depends_on": [],
        "description": "Fixed SDSS locus residuals, validity-aware aggregate outlier scores, and red-galaxy track coordinates.",
    }
]