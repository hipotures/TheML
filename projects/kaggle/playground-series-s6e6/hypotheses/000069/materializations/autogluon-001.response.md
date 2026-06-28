import numpy as np
import pandas as pd


MAG_BANDS = ("u", "g", "r", "i", "z")
SCAN_ORDER_BANDS = ("r", "i", "u", "z", "g")
SDSS_EFFECTIVE_WAVELENGTHS = (3551.0, 4686.0, 6165.0, 7481.0, 8931.0)
EPSILON = 1.0e-8


def add_sdss_scan_order_incoherence(raw, deps, aux):
    bands = list(MAG_BANDS)
    scan_bands = list(SCAN_ORDER_BANDS)

    mags = raw.loc[:, bands].astype("float64").to_numpy(copy=True)
    finite_mags = np.where(np.isfinite(mags), mags, np.nan)

    med = np.nanmedian(finite_mags, axis=1, keepdims=True)
    centered_mag = finite_mags - med

    mag_abs = np.abs(centered_mag)
    mag_scale = np.nanmedian(mag_abs, axis=1, keepdims=True)
    mag_scale = np.where(np.isfinite(mag_scale) & (mag_scale > EPSILON), mag_scale, 1.0)
    centered_mag = np.clip(centered_mag, -12.0 * mag_scale, 12.0 * mag_scale)

    rel_log_flux = -0.4 * np.log(10.0) * centered_mag

    x = np.log(np.asarray(SDSS_EFFECTIVE_WAVELENGTHS, dtype="float64"))
    x = x - np.mean(x)
    design = np.column_stack((np.ones_like(x), x, x * x))

    try:
        beta = np.linalg.lstsq(design, rel_log_flux.T, rcond=None)[0]
        fitted = (design @ beta).T
        residuals = rel_log_flux - fitted
        if residuals.shape != rel_log_flux.shape or not np.all(np.isfinite(residuals)):
            residuals = rel_log_flux - np.nanmedian(rel_log_flux, axis=1, keepdims=True)
    except np.linalg.LinAlgError:
        residuals = rel_log_flux - np.nanmedian(rel_log_flux, axis=1, keepdims=True)

    residuals = np.where(np.isfinite(residuals), residuals, 0.0)
    resid_abs_sum = np.sum(np.abs(residuals), axis=1) + EPSILON

    band_pos = {band: pos for pos, band in enumerate(bands)}
    scan_idx = [band_pos[band] for band in scan_bands]
    scan_resid = residuals[:, scan_idx]

    scan_first = np.diff(scan_resid, axis=1)
    scan_second = np.diff(scan_first, axis=1)
    wavelength_first = np.diff(residuals, axis=1)

    scan_roughness = np.sum(np.abs(scan_first), axis=1)
    wavelength_roughness = np.sum(np.abs(wavelength_first), axis=1)

    max_scan_jump = np.max(np.abs(scan_first), axis=1)
    max_scan_second = np.max(np.abs(scan_second), axis=1)
    non_adjacent_jump = np.maximum(
        np.abs(scan_resid[:, 1] - scan_resid[:, 2]),
        np.abs(scan_resid[:, 3] - scan_resid[:, 4]),
    )

    early_minus_late = np.mean(scan_resid[:, :2], axis=1) - np.mean(scan_resid[:, -2:], axis=1)
    scan_range = np.max(scan_resid, axis=1) - np.min(scan_resid, axis=1)

    scan_centered = scan_resid - np.mean(scan_resid, axis=1, keepdims=True)
    autocov = np.sum(scan_centered[:, :-1] * scan_centered[:, 1:], axis=1)
    autovar = np.sum(scan_centered * scan_centered, axis=1) + EPSILON
    scan_autocorr = autocov / autovar

    roughness_ratio = scan_roughness / (wavelength_roughness + EPSILON)

    features = pd.DataFrame(
        {
            "scan_roughness_norm": np.clip(scan_roughness / resid_abs_sum, 0.0, 8.0),
            "wavelength_roughness_norm": np.clip(wavelength_roughness / resid_abs_sum, 0.0, 8.0),
            "scan_to_wavelength_roughness_ratio": np.clip(roughness_ratio, 0.0, 12.0),
            "max_scan_jump_norm": np.clip(max_scan_jump / resid_abs_sum, 0.0, 4.0),
            "max_scan_second_diff_norm": np.clip(max_scan_second / resid_abs_sum, 0.0, 6.0),
            "non_wavelength_adjacent_jump_norm": np.clip(non_adjacent_jump / resid_abs_sum, 0.0, 4.0),
            "early_minus_late_resid_norm": np.clip(early_minus_late / resid_abs_sum, -4.0, 4.0),
            "scan_resid_range_norm": np.clip(scan_range / resid_abs_sum, 0.0, 4.0),
            "scan_resid_autocorr": np.clip(scan_autocorr, -1.0, 1.0),
            "iu_jump_signed_norm": np.clip((scan_resid[:, 1] - scan_resid[:, 2]) / resid_abs_sum, -4.0, 4.0),
            "zg_jump_signed_norm": np.clip((scan_resid[:, 3] - scan_resid[:, 4]) / resid_abs_sum, -4.0, 4.0),
        },
        index=raw.index,
    )

    return features


FEATURE_GROUPS = [
    {
        "name": "sdss_scan_order_incoherence",
        "fn": add_sdss_scan_order_incoherence,
        "depends_on": [],
        "description": "Scan-order residual roughness features comparing SDSS temporal filter order against wavelength-order photometric smoothness.",
    }
]