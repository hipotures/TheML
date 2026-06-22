import numpy as np
import pandas as pd

SDSS_BANDS = ("u", "g", "r", "i", "z")
SDSS_BAND_WAVELENGTH_A = (3551.0, 4686.0, 6166.0, 7480.0, 8932.0)

QUASAR_EMISSION_LINES = (
    (1215.67, 1.00),  # Ly-alpha
    (1549.06, 0.75),  # C IV
    (1640.42, 0.22),  # He II
    (1908.73, 0.45),  # C III]
    (2798.75, 0.60),  # Mg II
    (3426.00, 0.32),  # [Ne V]
    (3727.09, 0.12),  # [O II] (weak for broad-line quasars but keeps coverage)
    (4861.33, 0.30),  # H-beta
    (5006.84, 0.22),  # [O III]
    (6562.80, 0.18),  # H-alpha
)

GALAXY_EMISSION_LINES = (
    (3727.09, 1.00),   # [O II]
    (4101.73, 0.35),   # H-delta
    (4340.47, 0.40),   # H-gamma
    (4861.33, 0.90),   # H-beta
    (4958.91, 0.35),   # [O III]
    (5006.84, 1.00),   # [O III]
    (6562.80, 0.95),   # H-alpha
    (6583.45, 0.70),   # [N II]
    (6716.44, 0.40),   # [S II]
    (6730.82, 0.30),   # [S II]
)

RES_COEFF_SCALE = 0.08
OBS_LAMBDA_MIN_A = 3000.0
OBS_LAMBDA_MAX_A = 10000.0
FLUX_EPS = 1e-12


def _line_resonance(redshift, line_defs, band_wavelengths):
    z = np.asarray(redshift, dtype=float)
    band_wavelengths = np.asarray(band_wavelengths, dtype=float)
    n_rows = z.shape[0]
    per_band = np.zeros((n_rows, band_wavelengths.size), dtype=float)
    per_row_total = np.zeros(n_rows, dtype=float)

    if n_rows == 0:
        return per_band, per_row_total

    for rest_lambda, line_weight in line_defs:
        obs_lambda = np.asarray(rest_lambda, dtype=float) * (1.0 + z)
        valid = (
            np.isfinite(obs_lambda)
            & np.isfinite(z)
            & (z >= 0.0)
            & (obs_lambda >= OBS_LAMBDA_MIN_A)
            & (obs_lambda <= OBS_LAMBDA_MAX_A)
        )
        ratio = obs_lambda[:, None] / band_wavelengths[None, :]
        kernel = np.exp(-0.5 * (np.log(ratio) / RES_COEFF_SCALE) ** 2)
        kernel[~valid, :] = 0.0
        weighted = kernel * float(line_weight)
        per_band += weighted
        per_row_total += weighted.sum(axis=1)

    return per_band, per_row_total


def _band_excess(log_flux, band_wavelengths):
    lf = np.asarray(log_flux, dtype=float)
    waves = np.asarray(band_wavelengths, dtype=float)
    left_right = (
        (1, 2),  # u edge uses g,r
        (0, 2),  # g
        (1, 3),  # r
        (2, 4),  # i
        (2, 3),  # z edge uses r,i
    )
    excess = np.zeros_like(lf)

    for j, (li, ri) in enumerate(left_right):
        wl_target = waves[j]
        wl_left = waves[li]
        wl_right = waves[ri]
        slope = (lf[:, ri] - lf[:, li]) / (wl_right - wl_left)
        continuum = lf[:, li] + slope * (wl_target - wl_left)
        excess[:, j] = lf[:, j] - continuum

    return excess


def add_emission_line_bandpass_resonance(raw, deps, aux):
    band_waves = np.asarray(SDSS_BAND_WAVELENGTH_A, dtype=float)
    band_cols = list(SDSS_BANDS)

    z = np.asarray(raw["redshift"].to_numpy(dtype=float))
    mags = raw[band_cols].to_numpy(dtype=float)

    flux = np.power(10.0, -0.4 * mags)
    row_median = np.nanmedian(flux, axis=1)
    safe_row_median = np.where(np.isfinite(row_median) & (row_median > 0.0), row_median, 1.0)
    norm_flux = flux / safe_row_median[:, None]
    log_flux = np.log(np.maximum(norm_flux, FLUX_EPS))

    galaxy_band_res, galaxy_row_res = _line_resonance(z, GALAXY_EMISSION_LINES, band_waves)
    quasar_band_res, quasar_row_res = _line_resonance(z, QUASAR_EMISSION_LINES, band_waves)

    total_band_res = galaxy_band_res + quasar_band_res
    band_signal = total_band_res.sum(axis=1)

    max_band_value = np.max(total_band_res, axis=1)
    has_signal = band_signal > 0.0
    max_band_idx = np.full(z.shape[0], -1, dtype=np.int16)
    max_band_idx[has_signal] = np.argmax(total_band_res[has_signal], axis=1)

    strongest_band_labels = np.array(SDSS_BANDS, dtype=object)
    nearest_resonant_band = np.full(z.shape[0], "outside", dtype=object)
    nearest_resonant_band[has_signal] = strongest_band_labels[max_band_idx[has_signal]]

    excess = _band_excess(log_flux, band_waves)
    weighted_excess_numer = np.sum(total_band_res * np.clip(excess, 0.0, np.inf), axis=1)
    weighted_signed_numer = np.sum(total_band_res * excess, axis=1)

    denom = np.where(band_signal > 0.0, band_signal, np.nan)
    weighted_excess = np.divide(weighted_excess_numer, denom, out=np.zeros_like(weighted_excess_numer, dtype=float), where=band_signal > 0.0)
    weighted_signed_excess = np.divide(
        weighted_signed_numer,
        denom,
        out=np.zeros_like(weighted_signed_numer, dtype=float),
        where=band_signal > 0.0,
    )

    features = pd.DataFrame(
        {
            "emission_resonance_quasar_total": quasar_row_res,
            "emission_resonance_galaxy_total": galaxy_row_res,
            "emission_resonance_q_minus_g": quasar_row_res - galaxy_row_res,
            "emission_resonance_max_single_band": max_band_value,
            "emission_resonance_nearest_band_index": max_band_idx,
            "emission_resonance_strongest_band": nearest_resonant_band,
            "emission_resonance_total_band_sum": band_signal,
            "emission_resonance_flux_excess": weighted_excess,
            "emission_resonance_signed_excess": weighted_signed_excess,
        },
        index=raw.index,
    )
    return features


FEATURE_GROUPS = [
    {
        "name": "emission_line_bandpass_resonance",
        "fn": add_emission_line_bandpass_resonance,
        "depends_on": [],
        "description": "Compute redshift-dependent SDSS ugriz emission-line resonance features from line-to-band kernel overlap and continuum-compared band-excess residuals.",
    }
]