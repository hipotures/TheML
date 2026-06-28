import numpy as np
import pandas as pd


EPS = 0.000001
LOG_FLUX_K = -0.9210340371976183
LOG_FLUX_MIN = -30.0
LOG_FLUX_MAX = 10.0
OUTPUT_MIN = -1000000.0
OUTPUT_MAX = 1000000.0
BANDS = ("u", "g", "r", "i", "z")
BAND_PAIRS = (
    ("u", "g"),
    ("g", "r"),
    ("r", "i"),
    ("i", "z"),
    ("u", "r"),
    ("u", "i"),
    ("u", "z"),
    ("g", "z"),
    ("r", "z"),
)


def add_aide_broadband_flux_ratios(raw, deps, aux):
    index = raw.index
    features = pd.DataFrame(index=index)

    mags = {}
    flux = {}

    for band in BANDS:
        mag = pd.to_numeric(raw[band], errors="coerce").astype("float64")
        log_flux = (LOG_FLUX_K * mag).clip(LOG_FLUX_MIN, LOG_FLUX_MAX)
        flux_proxy = np.exp(log_flux)

        mags[band] = mag
        flux[band] = pd.Series(flux_proxy, index=index)

        features[f"{band}_log_flux_proxy"] = log_flux
        features[f"{band}_flux_proxy"] = flux[band]

    for left, right in BAND_PAIRS:
        left_mag = mags[left]
        right_mag = mags[right]
        right_abs = right_mag.abs()
        safe_right_abs = right_abs.where(right_abs >= EPS, EPS)
        signed_denominator = np.sign(right_mag).replace(0.0, 1.0) * safe_right_abs

        features[f"{left}_{right}_abs_gap"] = (left_mag - right_mag).abs()
        features[f"{left}_{right}_signed_norm_gap"] = (left_mag - right_mag) / (right_abs + EPS)
        features[f"{left}_{right}_safe_mag_ratio"] = left_mag / signed_denominator
        features[f"{left}_{right}_flux_ratio"] = flux[left] / (flux[right] + EPS)

    color_ug = mags["u"] - mags["g"]
    color_gr = mags["g"] - mags["r"]
    color_ri = mags["r"] - mags["i"]
    color_iz = mags["i"] - mags["z"]
    color_gz = mags["g"] - mags["z"]

    features["curvature_u_2r_i"] = mags["u"] - 2.0 * mags["r"] + mags["i"]
    features["curvature_g_2i_z"] = mags["g"] - 2.0 * mags["i"] + mags["z"]
    features["color_curvature_ug_gr"] = color_ug - color_gr
    features["color_curvature_gr_ri"] = color_gr - color_ri
    features["color_curvature_ri_iz"] = color_ri - color_iz

    redshift = pd.to_numeric(raw["redshift"], errors="coerce").astype("float64")
    zr = redshift.clip(lower=0.0)
    zr_clipped = zr.clip(0.0, 8.0)
    log1p_zr = np.log1p(zr_clipped)
    zr2 = zr_clipped * zr_clipped
    zr3 = zr2 * zr_clipped

    features["zr_nonnegative"] = zr
    features["zr_log1p_clipped"] = log1p_zr
    features["zr2_clipped"] = zr2
    features["zr3_clipped"] = zr3

    for band in BANDS:
        features[f"log1p_zr_x_{band}_flux_proxy"] = log1p_zr * flux[band]

    features["zr_x_color_ug"] = zr_clipped * color_ug
    features["zr_x_color_gr"] = zr_clipped * color_gr
    features["zr2_x_color_ri"] = zr2 * color_ri
    features["zr2_x_color_gz"] = zr2 * color_gz

    features = features.replace([np.inf, -np.inf], np.nan)
    features = features.fillna(0.0)
    features = features.clip(OUTPUT_MIN, OUTPUT_MAX)

    return features


FEATURE_GROUPS = [
    {
        "name": "aide_broadband_flux_ratios",
        "fn": add_aide_broadband_flux_ratios,
        "depends_on": [],
        "description": "Deterministic ugriz magnitude, flux-ratio, curvature, and redshift-interaction photometric features.",
    }
]