import numpy as np
import pandas as pd


def add_aide_broadband_flux_ratios(raw, deps, aux):
    idx = raw.index

    ratio_pairs = (
        ("u", "g"),
        ("g", "r"),
        ("r", "i"),
        ("i", "z"),
        ("u", "i"),
        ("u", "r"),
        ("g", "z"),
        ("r", "z"),
    )
    bands = ("u", "g", "r", "i", "z")
    eps = 1e-6
    log_flux_scale = -0.4 * np.log(10.0)
    exp_clip_minmax = (-30.0, 30.0)

    features = {}

    for num_band, den_band in ratio_pairs:
        num = raw[num_band].astype(float)
        den = raw[den_band].astype(float)
        features[f"aide_ratio_{num_band}_over_{den_band}"] = num / (den + eps)
        features[f"aide_abs_color_gap_{num_band}_{den_band}"] = (num - den).abs()

    for band in bands:
        mag = raw[band].astype(float)
        log_flux = log_flux_scale * mag
        features[f"aide_log_flux_{band}"] = log_flux
        features[f"aide_flux_clip_exp_{band}"] = np.exp(np.clip(log_flux, exp_clip_minmax[0], exp_clip_minmax[1]))

    redshift = raw["redshift"].astype(float)
    z_nonneg = redshift.clip(lower=0.0)
    features["aide_redshift_nonneg"] = z_nonneg
    features["aide_log1p_redshift_nonneg"] = np.log1p(z_nonneg)
    features["aide_redshift_sq_nonneg"] = z_nonneg ** 2
    features["aide_redshift_cu_nonneg"] = z_nonneg ** 3

    u = raw["u"].astype(float)
    g = raw["g"].astype(float)
    r = raw["r"].astype(float)
    i = raw["i"].astype(float)
    z = raw["z"].astype(float)

    features["aide_curvature_u_2r_i"] = u - 2.0 * r + i
    features["aide_curvature_g_2i_z"] = g - 2.0 * i + z

    return pd.DataFrame(features, index=idx)


FEATURE_GROUPS = [
    {
        "name": "aide_broadband_flux_ratios",
        "fn": add_aide_broadband_flux_ratios,
        "depends_on": [],
        "description": "Creates safe broadband magnitude ratios, absolute color gaps, pseudo-flux encodings, redshift low-order transforms, and curvature terms.",
    }
]