import numpy as np
import pandas as pd


BAND_COLUMNS = ("u", "g", "r", "i", "z")
WAVELENGTH_NM = (355.1, 468.6, 616.5, 748.1, 893.1)
ANCHOR_WAVELENGTH_NM = (400.0, 550.0, 700.0)
FLUX_SHARE_ALPHA = 1e-12
EPS = 1e-12


def add_flux_allocation_entropy(raw, deps, aux):
    index = raw.index
    mags = raw.loc[:, BAND_COLUMNS].to_numpy(dtype=np.float64, copy=True)

    finite_mags = np.isfinite(mags).all(axis=1)
    safe_mags = np.where(np.isfinite(mags), mags, 0.0)

    with np.errstate(over="ignore", under="ignore", invalid="ignore", divide="ignore"):
        a = -0.4 * np.log(10.0) * safe_mags
        a = a - np.max(a, axis=1, keepdims=True)
        q = np.exp(a)
        denom = np.sum(q, axis=1, keepdims=True) + len(BAND_COLUMNS) * FLUX_SHARE_ALPHA
        p = (q + FLUX_SHARE_ALPHA) / denom

    valid_p = (
        finite_mags
        & np.isfinite(p).all(axis=1)
        & (np.abs(np.sum(p, axis=1) - 1.0) <= 1e-6)
        & (p >= 0.0).all(axis=1)
    )
    invalid_flux_share = ~valid_p
    p[invalid_flux_share, :] = 1.0 / len(BAND_COLUMNS)

    p_u = p[:, 0]
    p_g = p[:, 1]
    p_r = p[:, 2]
    p_i = p[:, 3]
    p_z = p[:, 4]

    with np.errstate(over="ignore", under="ignore", invalid="ignore", divide="ignore"):
        entropy = -np.sum(p * np.log(np.maximum(p, EPS)), axis=1)
        entropy_norm = entropy / np.log(float(len(BAND_COLUMNS)))
        simpson_concentration = np.sum(p * p, axis=1)
        gini_impurity = 1.0 - simpson_concentration
        effective_band_count = 1.0 / (simpson_concentration + EPS)

        p_sorted = np.sort(p, axis=1)[:, ::-1]
        p_max = p_sorted[:, 0]
        p_second = p_sorted[:, 1]
        p_min = p_sorted[:, -1]
        p_spread = p_max - p_min
        top_two_ratio = p_max / (p_second + EPS)
        top_two_gap = p_max - p_second

        blue_share = p_u + p_g
        central_share = p_r
        red_share = p_i + p_z
        blue_minus_red = blue_share - red_share
        blue_fraction = blue_share / (blue_share + red_share + EPS)

        x = np.log(np.asarray(WAVELENGTH_NM, dtype=np.float64))
        x_min = np.min(x)
        x_max = np.max(x)
        x_range = x_max - x_min

        mu = np.sum(p * x.reshape(1, -1), axis=1)
        centered = x.reshape(1, -1) - mu.reshape(-1, 1)
        variance = np.sum(p * centered * centered, axis=1)
        variance = np.maximum(variance, 0.0)
        sd = np.sqrt(variance)
        skew = np.sum(p * centered ** 3, axis=1) / (sd ** 3 + EPS)
        excess_kurtosis = np.sum(p * centered ** 4, axis=1) / (sd ** 4 + EPS) - 3.0
        mu_norm = (mu - x_min) / (x_range + EPS)
        variance_norm = variance / (x_range * x_range + EPS)

        redshift = raw["redshift"].to_numpy(dtype=np.float64, copy=True)
        safe_redshift = np.where(np.isfinite(redshift), redshift, 0.0)
        zc = np.clip(safe_redshift, -0.95, 20.0)
        redshift_shift = np.log1p(zc)

        x_rf = x.reshape(1, -1) - redshift_shift.reshape(-1, 1)
        mu_rf = np.sum(p * x_rf, axis=1)
        mu_rf_norm = (mu_rf - x_min) / (x_range + EPS)
        delta_mu = mu - mu_rf

        anchor_logs = np.log(np.asarray(ANCHOR_WAVELENGTH_NM, dtype=np.float64))
        d_anchor_400 = np.sum(p * np.abs(x_rf - anchor_logs[0]), axis=1)
        d_anchor_550 = np.sum(p * np.abs(x_rf - anchor_logs[1]), axis=1)
        d_anchor_700 = np.sum(p * np.abs(x_rf - anchor_logs[2]), axis=1)

    top_two_ratio = np.clip(top_two_ratio, 0.0, 1e6)
    skew = np.clip(skew, -8.0, 8.0)
    excess_kurtosis = np.clip(excess_kurtosis, -8.0, 8.0)

    features = pd.DataFrame(
        {
            "invalid_flux_share": invalid_flux_share.astype(np.int8),
            "p_u": p_u,
            "p_g": p_g,
            "p_r": p_r,
            "p_i": p_i,
            "p_z": p_z,
            "entropy": entropy,
            "entropy_norm": entropy_norm,
            "simpson_concentration": simpson_concentration,
            "gini_impurity": gini_impurity,
            "effective_band_count": effective_band_count,
            "p_max": p_max,
            "p_min": p_min,
            "p_spread": p_spread,
            "top_two_ratio": top_two_ratio,
            "top_two_gap": top_two_gap,
            "blue_share": blue_share,
            "central_share": central_share,
            "red_share": red_share,
            "blue_minus_red": blue_minus_red,
            "blue_fraction": blue_fraction,
            "p_u_minus_p_i": p_u - p_i,
            "p_g_minus_p_r": p_g - p_r,
            "p_r_minus_p_z": p_r - p_z,
            "p_z_minus_p_g": p_z - p_g,
            "log_wavelength_centroid": mu,
            "log_wavelength_variance": variance,
            "log_wavelength_sd": sd,
            "log_wavelength_skew": skew,
            "log_wavelength_excess_kurtosis": excess_kurtosis,
            "log_wavelength_centroid_norm": mu_norm,
            "log_wavelength_variance_norm": variance_norm,
            "rest_log_wavelength_centroid": mu_rf,
            "rest_log_wavelength_centroid_norm": mu_rf_norm,
            "delta_log_wavelength_centroid": delta_mu,
            "rest_anchor_distance_400nm": d_anchor_400,
            "rest_anchor_distance_550nm": d_anchor_550,
            "rest_anchor_distance_700nm": d_anchor_700,
        },
        index=index,
    )

    return features.replace([np.inf, -np.inf], 0.0).fillna(0.0)


FEATURE_GROUPS = [
    {
        "name": "flux_allocation_entropy",
        "fn": add_flux_allocation_entropy,
        "depends_on": [],
        "description": "Stable normalized ugriz flux-share entropy, concentration, allocation, wavelength-location, and redshift-aware placement features.",
    }
]