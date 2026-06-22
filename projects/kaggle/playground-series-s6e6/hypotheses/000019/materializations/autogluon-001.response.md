import numpy as np
import pandas as pd

_BANDS = ("u", "g", "r", "i", "z")
_WAVELENGTHS_A = (3543.0, 4770.0, 6231.0, 7625.0, 9134.0)
_CLIP_RANGE = (-1.0, 35.0)


def add_observed_sed_continuum_moments(raw, deps, aux):
    mags = raw.loc[:, _BANDS].to_numpy(dtype=float, copy=False)

    mags = np.where(np.isfinite(mags), mags, np.nan)
    mags = np.nan_to_num(mags, nan=0.0, posinf=0.0, neginf=0.0)
    mags = np.clip(mags, _CLIP_RANGE[0], _CLIP_RANGE[1])

    log_flux = -0.4 * mags
    log_flux = log_flux - np.nanmean(log_flux, axis=1, keepdims=True)

    wavelengths = np.array(_WAVELENGTHS_A, dtype=float)
    x = np.log(wavelengths)
    x = x - x.mean()
    x2 = x**2

    design = np.column_stack((x2, x, np.ones_like(x)))
    pinv_design = np.linalg.pinv(design)
    coeffs = (pinv_design @ log_flux.T).T
    quad_curvature = np.nan_to_num(coeffs[:, 0], nan=0.0, posinf=0.0, neginf=0.0)
    linear_slope = np.nan_to_num(coeffs[:, 1], nan=0.0, posinf=0.0, neginf=0.0)

    fit = (design @ coeffs.T).T
    resid = log_flux - fit
    resid = np.nan_to_num(resid, nan=0.0, posinf=0.0, neginf=0.0)

    residual_rms = np.sqrt(np.mean(resid * resid, axis=1))
    residual_max_abs = np.max(np.abs(resid), axis=1)
    residual_u = resid[:, 0]
    residual_r = resid[:, 2]
    residual_z = resid[:, 4]

    x_left = x[:3]
    x_right = x[2:]
    x_left_mean = x_left.mean()
    x_right_mean = x_right.mean()
    left_center = x_left - x_left_mean
    right_center = x_right - x_right_mean

    y_left = log_flux[:, :3]
    y_right = log_flux[:, 2:5]

    denom_left = np.sum(left_center * left_center)
    denom_right = np.sum(right_center * right_center)

    y_left_c = y_left - np.mean(y_left, axis=1, keepdims=True)
    y_right_c = y_right - np.mean(y_right, axis=1, keepdims=True)

    blue_slope = np.sum(y_left_c * left_center, axis=1) / denom_left
    red_slope = np.sum(y_right_c * right_center, axis=1) / denom_right

    blue_slope = np.nan_to_num(blue_slope, nan=0.0, posinf=0.0, neginf=0.0)
    red_slope = np.nan_to_num(red_slope, nan=0.0, posinf=0.0, neginf=0.0)
    red_minus_blue_slope_break = red_slope - blue_slope

    return pd.DataFrame(
        {
            "quadratic_curvature": quad_curvature,
            "linear_slope": linear_slope,
            "residual_rms": residual_rms,
            "residual_max_abs": residual_max_abs,
            "blue_side_slope_ugr": blue_slope,
            "red_side_slope_riz": red_slope,
            "red_minus_blue_slope_break": red_minus_blue_slope_break,
            "residual_r_center": residual_r,
            "residual_u_edge": residual_u,
            "residual_z_edge": residual_z,
        },
        index=raw.index,
    )


FEATURE_GROUPS = [
    {
        "name": "observed_sed_continuum_moments",
        "fn": add_observed_sed_continuum_moments,
        "depends_on": [],
        "description": "Fits a per-object quadratic continuum in log-flux vs log-wavelength across ugriz and emits slope/curvature, residual moments, side slopes, and edge-centered residual diagnostics.",
    }
]