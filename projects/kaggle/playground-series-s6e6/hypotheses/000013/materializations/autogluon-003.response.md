import numpy as np
import pandas as pd


BAND_NAMES = ("u", "g", "r", "i", "z")
BAND_WAVELENGTHS = (3551.0, 4686.0, 6166.0, 7480.0, 8932.0)
KERNEL_LOG_WIDTH = 0.08
MIN_OBS_WAVELENGTH = 3000.0
MAX_OBS_WAVELENGTH = 10000.0
KERNEL_TRUNCATION = 1.0e-4
MIN_FLUX_PROXY = 1.0e-12
RESIDUAL_CLIP_MIN = -5.0
RESIDUAL_CLIP_MAX = 5.0

QSO_LINE_TEMPLATE = (
    ("lya", 1215.24, 1.00),
    ("nv", 1240.81, 0.25),
    ("siv_oiv", 1399.80, 0.45),
    ("civ", 1549.48, 0.90),
    ("ciii", 1908.734, 0.65),
    ("mgii", 2799.117, 0.80),
    ("oii", 3727.092, 0.25),
    ("hbeta", 4862.68, 0.35),
    ("oiii", 5008.240, 0.35),
    ("halpha", 6564.614, 0.45),
)

GALAXY_LINE_TEMPLATE = (
    ("oii", 3727.092, 0.80),
    ("hdelta", 4102.89, 0.25),
    ("hgamma", 4341.68, 0.30),
    ("hbeta", 4862.68, 0.60),
    ("oiii_4960", 4960.295, 0.55),
    ("oiii_5008", 5008.240, 0.80),
    ("halpha", 6564.614, 1.00),
    ("nii", 6585.27, 0.40),
    ("sii_6718", 6718.29, 0.25),
    ("sii_6732", 6732.67, 0.25),
)


def _template_features(redshift, residuals, template, band_wavelengths, band_names, prefix):
    n_rows = redshift.shape[0]
    band_kernel_sum = np.zeros((n_rows, len(band_names)), dtype=np.float64)
    total = np.zeros(n_rows, dtype=np.float64)
    weighted_residual = np.zeros(n_rows, dtype=np.float64)
    positive_residual = np.zeros(n_rows, dtype=np.float64)
    negative_residual = np.zeros(n_rows, dtype=np.float64)

    weight_total = 0.0
    for _, _, weight in template:
        weight_total += weight

    valid_redshift = redshift >= 0.0
    positive_part = np.maximum(residuals, 0.0)
    negative_part = np.minimum(residuals, 0.0)

    for _, rest_wavelength, raw_weight in template:
        weight = raw_weight / weight_total
        observed = rest_wavelength * (1.0 + redshift)
        valid_observed = valid_redshift & (observed >= MIN_OBS_WAVELENGTH) & (observed <= MAX_OBS_WAVELENGTH)

        if not np.any(valid_observed):
            continue

        log_ratio = np.log(observed[:, None] / band_wavelengths[None, :])
        kernel = np.exp(-0.5 * (log_ratio / KERNEL_LOG_WIDTH) ** 2)
        kernel[kernel < KERNEL_TRUNCATION] = 0.0
        kernel[~valid_observed, :] = 0.0
        weighted_kernel = weight * kernel

        band_kernel_sum += weighted_kernel
        total += weighted_kernel.sum(axis=1)
        weighted_residual += (weighted_kernel * residuals).sum(axis=1)
        positive_residual += (weighted_kernel * positive_part).sum(axis=1)
        negative_residual += (weighted_kernel * negative_part).sum(axis=1)

    peak_strength = band_kernel_sum.max(axis=1)
    strongest_idx = band_kernel_sum.argmax(axis=1)
    strongest_band = np.asarray(band_names, dtype=object)[strongest_idx]
    strongest_band = strongest_band.astype(object)
    strongest_band[peak_strength <= 0.0] = "outside"

    return {
        prefix + "_total_resonance": total,
        prefix + "_residual_resonance": weighted_residual,
        prefix + "_positive_resonance": positive_residual,
        prefix + "_negative_resonance": negative_residual,
        prefix + "_peak_band_strength": peak_strength,
        prefix + "_strongest_band": strongest_band,
    }


def add_emission_line_bandpass_resonance(raw, deps, aux):
    band_names = tuple(BAND_NAMES)
    band_wavelengths = np.asarray(BAND_WAVELENGTHS, dtype=np.float64)

    magnitudes = raw.loc[:, band_names].to_numpy(dtype=np.float64, copy=True)
    flux_proxy = np.maximum(np.power(10.0, -0.4 * magnitudes), MIN_FLUX_PROXY)
    log_flux = np.log(flux_proxy)
    centered_log_flux = log_flux - np.median(log_flux, axis=1, keepdims=True)

    log_band_wavelengths = np.log(band_wavelengths)

    residuals = np.empty_like(centered_log_flux)

    gr_slope = (
        (centered_log_flux[:, 2] - centered_log_flux[:, 1])
        / (log_band_wavelengths[2] - log_band_wavelengths[1])
    )
    ri_slope = (
        (centered_log_flux[:, 3] - centered_log_flux[:, 2])
        / (log_band_wavelengths[3] - log_band_wavelengths[2])
    )

    continuum_u = centered_log_flux[:, 1] + gr_slope * (
        log_band_wavelengths[0] - log_band_wavelengths[1]
    )
    continuum_g = centered_log_flux[:, 0] + (
        (centered_log_flux[:, 2] - centered_log_flux[:, 0])
        * (log_band_wavelengths[1] - log_band_wavelengths[0])
        / (log_band_wavelengths[2] - log_band_wavelengths[0])
    )
    continuum_r = centered_log_flux[:, 1] + (
        (centered_log_flux[:, 3] - centered_log_flux[:, 1])
        * (log_band_wavelengths[2] - log_band_wavelengths[1])
        / (log_band_wavelengths[3] - log_band_wavelengths[1])
    )
    continuum_i = centered_log_flux[:, 2] + (
        (centered_log_flux[:, 4] - centered_log_flux[:, 2])
        * (log_band_wavelengths[3] - log_band_wavelengths[2])
        / (log_band_wavelengths[4] - log_band_wavelengths[2])
    )
    continuum_z = centered_log_flux[:, 2] + ri_slope * (
        log_band_wavelengths[4] - log_band_wavelengths[2]
    )

    residuals[:, 0] = centered_log_flux[:, 0] - continuum_u
    residuals[:, 1] = centered_log_flux[:, 1] - continuum_g
    residuals[:, 2] = centered_log_flux[:, 2] - continuum_r
    residuals[:, 3] = centered_log_flux[:, 3] - continuum_i
    residuals[:, 4] = centered_log_flux[:, 4] - continuum_z
    residuals = np.clip(residuals, RESIDUAL_CLIP_MIN, RESIDUAL_CLIP_MAX)

    redshift = raw["redshift"].to_numpy(dtype=np.float64, copy=True)

    qso = _template_features(
        redshift,
        residuals,
        QSO_LINE_TEMPLATE,
        band_wavelengths,
        band_names,
        "qso",
    )
    gal = _template_features(
        redshift,
        residuals,
        GALAXY_LINE_TEMPLATE,
        band_wavelengths,
        band_names,
        "galaxy",
    )

    data = {}
    data.update(qso)
    data.update(gal)
    data["qso_minus_galaxy_total_resonance"] = (
        qso["qso_total_resonance"] - gal["galaxy_total_resonance"]
    )
    data["qso_minus_galaxy_residual_resonance"] = (
        qso["qso_residual_resonance"] - gal["galaxy_residual_resonance"]
    )
    data["qso_minus_galaxy_positive_resonance"] = (
        qso["qso_positive_resonance"] - gal["galaxy_positive_resonance"]
    )
    data["qso_minus_galaxy_peak_band_strength"] = (
        qso["qso_peak_band_strength"] - gal["galaxy_peak_band_strength"]
    )

    return pd.DataFrame(data, index=raw.index)


FEATURE_GROUPS = [
    {
        "name": "emission_line_bandpass_resonance",
        "fn": add_emission_line_bandpass_resonance,
        "depends_on": [],
        "description": "Redshifted emission-line resonance summaries against SDSS ugriz passbands and local continuum residuals.",
    }
]