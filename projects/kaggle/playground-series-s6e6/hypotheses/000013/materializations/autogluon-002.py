import numpy as np
import pandas as pd

BANDS = ("u", "g", "r", "i", "z")
BAND_CENTERS_AA = (3551.0, 4686.0, 6166.0, 7480.0, 8931.0)
LAMBDA_MIN_AA = 3000.0
LAMBDA_MAX_AA = 10000.0
LOG_LAMBDA_SIGMA = 0.06
KERNEL_EPS = 1e-4
FLUX_FLOOR = 1e-12

# (line_name, rest_wavelength_AA, weight)
QSO_LINES = (
    ("Ly_alpha", 1215.24, 1.0),
    ("Nv", 1240.81, 1.0),
    ("SiIV_OIV", 1399.8, 1.0),
    ("CIV", 1549.48, 1.0),
    ("CIII", 1908.734, 1.0),
    ("MgII", 2799.117, 1.0),
    ("OII_3727", 3727.092, 1.0),
    ("Hb", 4862.68, 1.0),
    ("OIII_5008", 5008.240, 1.0),
    ("Ha", 6564.614, 1.0),
)

GAL_LINES = (
    ("OII_3727", 3727.092, 1.0),
    ("Hd", 4102.89, 1.0),
    ("Hg", 4341.68, 1.0),
    ("Hb", 4862.68, 1.0),
    ("OIII_4960", 4960.295, 1.0),
    ("OIII_5008", 5008.240, 1.0),
    ("Ha", 6564.614, 1.0),
    ("NII_6585", 6585.27, 1.0),
    ("SII_6718", 6718.29, 1.0),
    ("SII_6733", 6732.67, 1.0),
)


def _line_band_weights(redshift, line_defs, band_centers):
    """Compute Σ_l w_l k_{b,l} per band for all rows."""
    redshift = np.asarray(redshift, dtype=np.float64)
    n_rows = redshift.shape[0]
    band_centers = np.asarray(band_centers, dtype=np.float64)

    totals = np.zeros((n_rows, band_centers.shape[0]), dtype=np.float64)
    valid_redshift = np.isfinite(redshift) & (redshift >= 0.0)

    for _, lam0, weight in line_defs:
        lam_obs = lam0 * (1.0 + redshift)
        active = valid_redshift & (lam_obs >= LAMBDA_MIN_AA) & (lam_obs <= LAMBDA_MAX_AA)
        if not np.any(active):
            continue

        rr = np.log(lam_obs[active, None] / band_centers[None, :])
        k = np.exp(-0.5 * (rr / LOG_LAMBDA_SIGMA) ** 2)
        if KERNEL_EPS > 0:
            k = np.where(k >= KERNEL_EPS, k, 0.0)
        totals[active] += weight * k

    return totals


def _row_centered_log_flux_and_residuals(raw):
    mags = raw.loc[:, BANDS].to_numpy(dtype=np.float64)

    # y_b = log(f_b), f_b = max(10^{-0.4 m_b}, 1e-12)
    y = np.log(np.maximum(np.power(10.0, -0.4 * mags), FLUX_FLOOR))
    y_center = y - np.median(y, axis=1, keepdims=True)

    # local continuum residuals on y'_b via linear interpolation in log λ
    band_log = np.log(np.asarray(BAND_CENTERS_AA, dtype=np.float64))
    lu, lg, lr, li, lz = band_log

    y_u = y_center[:, 0]
    y_g = y_center[:, 1]
    y_r = y_center[:, 2]
    y_i = y_center[:, 3]
    y_z = y_center[:, 4]

    yhat_u = y_g + (y_r - y_g) * ((lu - lg) / (lr - lg))      # u uses g and r
    yhat_g = y_u + (y_r - y_u) * ((lg - lu) / (lr - lu))      # g uses u and r
    yhat_r = y_g + (y_i - y_g) * ((lr - lg) / (li - lg))      # r uses g and i
    yhat_i = y_r + (y_z - y_r) * ((li - lr) / (lz - lr))      # i uses r and z
    yhat_z = y_i + (y_r - y_i) * ((lz - li) / (li - lr))      # z uses i and r

    residuals = np.column_stack(
        (
            y_u - yhat_u,
            y_g - yhat_g,
            y_r - yhat_r,
            y_i - yhat_i,
            y_z - yhat_z,
        )
    )
    return y_center, residuals


def _peak_band_features(class_band_sum):
    """Return peak band idx, peak band name, and 4-way peak bin (with outside bin)."""
    n_rows = class_band_sum.shape[0]
    total_per_row = np.sum(class_band_sum, axis=1)

    has_signal = total_per_row > 0.0
    peak_idx = np.full(n_rows, -1, dtype=np.int16)
    if np.any(has_signal):
        peak_idx[has_signal] = np.argmax(class_band_sum[has_signal], axis=1)

    band_names = np.array(BANDS, dtype=object)
    peak_band = np.full(n_rows, "outside", dtype=object)
    if np.any(has_signal):
        idx = np.flatnonzero(has_signal)
        peak_band[idx] = band_names[peak_idx[idx]]

    # Four-way bin: u, g, r, i/z  (outside is its own bin)
    peak_bin4_code = np.full(n_rows, -1, dtype=np.int8)
    peak_bin4_band = np.full(n_rows, "outside", dtype=object)
    if np.any(has_signal):
        idx = np.flatnonzero(has_signal)
        compressed = np.minimum(peak_idx[idx], 3)
        peak_bin4_code[idx] = compressed
        peak_bin4_band_labels = ("u", "g", "r", "i_or_z", "outside")
        peak_bin4_band[idx] = np.array(peak_bin4_band_labels[:-1], dtype=object)[compressed]

    return peak_idx, peak_band, peak_bin4_code, peak_bin4_band


def add_emission_line_bandpass_resonance(raw, deps, aux):
    """
    Deterministic astrophysical resonance features from redshifted emission-line overlap
    with SDSS passbands and local broadband continuum residual structure.
    """
    redshift = raw["redshift"].to_numpy(dtype=np.float64)

    _, residuals = _row_centered_log_flux_and_residuals(raw)

    band_sum_qso = _line_band_weights(redshift, QSO_LINES, BAND_CENTERS_AA)
    band_sum_gal = _line_band_weights(redshift, GAL_LINES, BAND_CENTERS_AA)

    # Class summaries
    T_qso = band_sum_qso.sum(axis=1)
    T_gal = band_sum_gal.sum(axis=1)
    E_qso = np.sum(residuals * band_sum_qso, axis=1)
    E_gal = np.sum(residuals * band_sum_gal, axis=1)
    S_qso = np.max(band_sum_qso, axis=1)
    S_gal = np.max(band_sum_gal, axis=1)

    # Margins
    M_t = T_qso - T_gal
    M_e = E_qso - E_gal
    M_s = S_qso - S_gal

    # Peak-band binning
    qso_peak_idx, qso_peak_band, qso_peak_bin4_code, qso_peak_bin4_band = _peak_band_features(
        band_sum_qso
    )
    gal_peak_idx, gal_peak_band, gal_peak_bin4_code, gal_peak_bin4_band = _peak_band_features(
        band_sum_gal
    )

    features = pd.DataFrame(index=raw.index)
    features["qso_total_resonance"] = T_qso
    features["gal_total_resonance"] = T_gal
    features["qso_residual_resonance"] = E_qso
    features["gal_residual_resonance"] = E_gal
    features["qso_peak_strength"] = S_qso
    features["gal_peak_strength"] = S_gal
    features["margin_total_resonance"] = M_t
    features["margin_residual_resonance"] = M_e
    features["margin_peak_resonance"] = M_s

    features["qso_peak_band_idx"] = qso_peak_idx
    features["gal_peak_band_idx"] = gal_peak_idx
    features["qso_peak_band"] = qso_peak_band
    features["gal_peak_band"] = gal_peak_band
    features["qso_peak_band_bin4_idx"] = qso_peak_bin4_code
    features["gal_peak_band_bin4_idx"] = gal_peak_bin4_code
    features["qso_peak_band_bin4"] = qso_peak_bin4_band
    features["gal_peak_band_bin4"] = gal_peak_bin4_band

    return features


FEATURE_GROUPS = [
    {
        "name": "emission_line_bandpass_resonance",
        "fn": add_emission_line_bandpass_resonance,
        "depends_on": [],
        "description": "Build redshifted line-passband resonance and local continuum-residual summary features for qso-vs-galaxy class diagnostics.",
    },
]