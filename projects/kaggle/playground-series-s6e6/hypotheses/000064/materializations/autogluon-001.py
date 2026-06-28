import numpy as np
import pandas as pd


SDSS_BANDS = ("u", "g", "r", "i", "z")
SDSS_WAVELENGTHS_ANGSTROM = (3551.0, 4686.0, 6166.0, 7480.0, 8932.0)
BAND_ORDINALS = (0.0, 1.0, 2.0, 3.0, 4.0)

LOG_FLUX_SCALE = -0.9210340371976183
MIN_POSITIVE_REDSHIFT_DENOMINATOR = 0.05
MIN_REGION_WEIGHT = 1.0e-8
RESIDUAL_CLIP = 2.5

CORE_LEFT = 2550.0
CORE_PEAK = 3000.0
CORE_RIGHT = 3450.0

UV_FEII_LEFT = 2200.0
UV_FEII_PEAK = 2700.0
UV_FEII_RIGHT = 3050.0

BALMER_LEFT = 3000.0
BALMER_PEAK = 3600.0
BALMER_RIGHT = 4000.0

BLUE_CONT_LEFT = 1450.0
BLUE_CONT_PEAK = 1675.0
BLUE_CONT_RIGHT = 1900.0

RED_CONT_LEFT = 4200.0
RED_CONT_PEAK = 4850.0
RED_CONT_RIGHT = 5500.0


def _triangular_membership(x, left, peak, right):
    rising = (x - left) / (peak - left)
    falling = (right - x) / (right - peak)
    return np.clip(np.minimum(rising, falling), 0.0, 1.0)


def _weighted_mean(values, weights):
    denom = weights.sum(axis=1)
    numer = (values * weights).sum(axis=1)
    return np.divide(numer, denom, out=np.zeros(values.shape[0], dtype=float), where=denom > MIN_REGION_WEIGHT)


def _weighted_linear_residuals(x, y, weights):
    w_sum = weights.sum(axis=1)
    x_mean = _weighted_mean(x, weights)
    y_mean = _weighted_mean(y, weights)
    xc = x - x_mean[:, None]
    yc = y - y_mean[:, None]
    slope_num = (weights * xc * yc).sum(axis=1)
    slope_den = (weights * xc * xc).sum(axis=1)
    slope = np.divide(slope_num, slope_den, out=np.zeros(y.shape[0], dtype=float), where=slope_den > MIN_REGION_WEIGHT)
    intercept = y_mean - slope * x_mean
    fitted = intercept[:, None] + slope[:, None] * x
    residuals = np.clip(y - fitted, -RESIDUAL_CLIP, RESIDUAL_CLIP)
    return residuals, slope


def add_quasar_small_bump_bandpass_contrast(raw, deps, aux):
    index = raw.index
    n_rows = len(raw)

    obs_wavelengths = np.asarray(SDSS_WAVELENGTHS_ANGSTROM, dtype=float)
    band_ordinals = np.asarray(BAND_ORDINALS, dtype=float)

    redshift = pd.to_numeric(raw["redshift"], errors="coerce").to_numpy(dtype=float)
    redshift = np.nan_to_num(redshift, nan=0.0, posinf=7.0, neginf=0.0)
    z_denom = np.maximum(1.0 + redshift, MIN_POSITIVE_REDSHIFT_DENOMINATOR)

    rest_wavelengths = obs_wavelengths[None, :] / z_denom[:, None]
    log_rest_wavelengths = np.log(np.maximum(rest_wavelengths, 1.0))

    magnitudes = np.empty((n_rows, len(SDSS_BANDS)), dtype=float)
    for pos, band in enumerate(SDSS_BANDS):
        magnitudes[:, pos] = pd.to_numeric(raw[band], errors="coerce").to_numpy(dtype=float)

    finite_mag = np.isfinite(magnitudes)
    row_median_mag = np.nanmedian(np.where(finite_mag, magnitudes, np.nan), axis=1)
    row_median_mag = np.nan_to_num(row_median_mag, nan=0.0)
    magnitudes = np.where(finite_mag, magnitudes, row_median_mag[:, None])

    log_flux = LOG_FLUX_SCALE * magnitudes
    relative_log_flux = log_flux - log_flux.mean(axis=1)[:, None]

    core_w = _triangular_membership(rest_wavelengths, CORE_LEFT, CORE_PEAK, CORE_RIGHT)
    uv_feii_w = _triangular_membership(rest_wavelengths, UV_FEII_LEFT, UV_FEII_PEAK, UV_FEII_RIGHT)
    balmer_w = _triangular_membership(rest_wavelengths, BALMER_LEFT, BALMER_PEAK, BALMER_RIGHT)
    blue_cont_w = _triangular_membership(rest_wavelengths, BLUE_CONT_LEFT, BLUE_CONT_PEAK, BLUE_CONT_RIGHT)
    red_cont_w = _triangular_membership(rest_wavelengths, RED_CONT_LEFT, RED_CONT_PEAK, RED_CONT_RIGHT)

    bump_w = np.maximum.reduce((core_w, uv_feii_w, balmer_w))
    side_cont_w = np.maximum(blue_cont_w, red_cont_w)
    outside_bump_w = np.clip(1.0 - bump_w, 0.0, 1.0)

    outside_support = outside_bump_w.sum(axis=1)
    low_bump_fallback_w = 0.18 + 0.82 * outside_bump_w
    continuum_w = np.where(outside_support[:, None] >= 2.0, outside_bump_w, low_bump_fallback_w)

    residuals, continuum_slope = _weighted_linear_residuals(log_rest_wavelengths, relative_log_flux, continuum_w)

    core_coverage = core_w.sum(axis=1)
    uv_feii_coverage = uv_feii_w.sum(axis=1)
    balmer_coverage = balmer_w.sum(axis=1)
    blue_cont_coverage = blue_cont_w.sum(axis=1)
    red_cont_coverage = red_cont_w.sum(axis=1)
    bump_coverage = bump_w.sum(axis=1)
    side_cont_coverage = side_cont_w.sum(axis=1)

    core_contrast = _weighted_mean(residuals, core_w)
    uv_feii_contrast = _weighted_mean(residuals, uv_feii_w)
    balmer_contrast = _weighted_mean(residuals, balmer_w)
    bump_contrast = _weighted_mean(residuals, bump_w)
    blue_cont_contrast = _weighted_mean(residuals, blue_cont_w)
    red_cont_contrast = _weighted_mean(residuals, red_cont_w)

    uv_balmer_asymmetry = uv_feii_contrast - balmer_contrast
    side_continuum_balance = blue_cont_contrast - red_cont_contrast
    bump_vs_side_contrast = bump_contrast - _weighted_mean(residuals, side_cont_w)

    weighted_bump_center = _weighted_mean(rest_wavelengths, bump_w)
    weighted_core_center = _weighted_mean(rest_wavelengths, core_w)

    strongest_bump_band = np.argmax(bump_w, axis=1).astype(float)
    strongest_bump_weight = np.max(bump_w, axis=1)
    strongest_bump_band = np.where(strongest_bump_weight > MIN_REGION_WEIGHT, strongest_bump_band, -1.0)

    strongest_core_band = np.argmax(core_w, axis=1).astype(float)
    strongest_core_weight = np.max(core_w, axis=1)
    strongest_core_band = np.where(strongest_core_weight > MIN_REGION_WEIGHT, strongest_core_band, -1.0)

    bump_band_centroid = _weighted_mean(np.broadcast_to(band_ordinals, bump_w.shape), bump_w)
    core_band_centroid = _weighted_mean(np.broadcast_to(band_ordinals, core_w.shape), core_w)

    features = pd.DataFrame(
        {
            "small_bump_core_coverage": core_coverage,
            "small_bump_uv_feii_coverage": uv_feii_coverage,
            "small_bump_balmer_coverage": balmer_coverage,
            "small_bump_total_coverage": bump_coverage,
            "small_bump_blue_side_coverage": blue_cont_coverage,
            "small_bump_red_side_coverage": red_cont_coverage,
            "small_bump_side_continuum_coverage": side_cont_coverage,
            "small_bump_core_log_flux_excess": core_contrast,
            "small_bump_uv_feii_log_flux_excess": uv_feii_contrast,
            "small_bump_balmer_log_flux_excess": balmer_contrast,
            "small_bump_total_log_flux_excess": bump_contrast,
            "small_bump_uv_minus_balmer_excess": uv_balmer_asymmetry,
            "small_bump_excess_minus_side_continuum": bump_vs_side_contrast,
            "small_bump_blue_minus_red_side_residual": side_continuum_balance,
            "small_bump_continuum_log_slope": continuum_slope,
            "small_bump_weighted_rest_wavelength": weighted_bump_center,
            "small_bump_core_weighted_rest_wavelength": weighted_core_center,
            "small_bump_strongest_band_ordinal": strongest_bump_band,
            "small_bump_strongest_band_weight": strongest_bump_weight,
            "small_bump_core_strongest_band_ordinal": strongest_core_band,
            "small_bump_band_centroid_ordinal": bump_band_centroid,
            "small_bump_core_band_centroid_ordinal": core_band_centroid,
        },
        index=index,
    )

    return features


FEATURE_GROUPS = [
    {
        "name": "quasar_small_bump_bandpass_contrast",
        "fn": add_quasar_small_bump_bandpass_contrast,
        "depends_on": [],
        "description": "Measures rest-frame SDSS band coverage and log-flux residual contrast around the quasar small blue bump.",
    }
]