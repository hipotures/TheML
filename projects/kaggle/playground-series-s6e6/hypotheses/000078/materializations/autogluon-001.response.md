import numpy as np
import pandas as pd


APO_LATITUDE_DEG = 32.78036
SDSS_REFERENCE_AIRMASS = 1.3
AIRMASS_MIN = 1.0
AIRMASS_MAX = 1.6
VECTOR_EPSILON = 1e-9
BANDS = ("u", "g", "r", "i", "z")
EXTINCTION_COEFFICIENTS = (0.48, 0.18, 0.10, 0.07, 0.05)


def add_apo_airmass_chromaticity(raw, deps, aux):
    index = raw.index

    delta = pd.to_numeric(raw["delta"], errors="coerce").to_numpy(dtype=float)
    delta_clipped = np.clip(delta, -90.0, 90.0)

    zenith_angle = np.abs(delta_clipped - APO_LATITUDE_DEG)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        airmass_raw = 1.0 / np.cos(np.deg2rad(zenith_angle))

    airmass = np.where(np.isfinite(airmass_raw), airmass_raw, SDSS_REFERENCE_AIRMASS)
    airmass = np.clip(airmass, AIRMASS_MIN, AIRMASS_MAX)
    delta_airmass = airmass - SDSS_REFERENCE_AIRMASS

    coeffs = np.asarray(EXTINCTION_COEFFICIENTS, dtype=float)
    band_offsets = delta_airmass[:, None] * coeffs[None, :]

    mags = raw.loc[:, BANDS].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    observed_colors = mags[:, :-1] - mags[:, 1:]

    extinction_color_vector = coeffs[:-1] - coeffs[1:]
    extinction_norm = float(np.sqrt(np.sum(extinction_color_vector * extinction_color_vector)))
    unit_extinction_vector = extinction_color_vector / (extinction_norm + VECTOR_EPSILON)

    projection = observed_colors @ unit_extinction_vector
    projected_colors = projection[:, None] * unit_extinction_vector[None, :]
    residual_colors = observed_colors - projected_colors
    residual_norm = np.sqrt(np.sum(residual_colors * residual_colors, axis=1))

    features = pd.DataFrame(index=index)
    features["apo_airmass"] = airmass
    features["apo_airmass_delta_ref"] = delta_airmass
    features["apo_zenith_angle_deg"] = zenith_angle

    for band_position, band in enumerate(BANDS):
        features[f"apo_extinction_offset_{band}"] = band_offsets[:, band_position]

    color_pairs = (("u", "g"), ("g", "r"), ("r", "i"), ("i", "z"))
    predicted_color_shifts = band_offsets[:, :-1] - band_offsets[:, 1:]
    for color_position, (left_band, right_band) in enumerate(color_pairs):
        features[f"apo_predicted_color_shift_{left_band}_{right_band}"] = predicted_color_shifts[:, color_position]

    features["apo_extinction_color_projection"] = projection
    features["apo_extinction_color_residual_norm"] = residual_norm
    features["apo_blue_airmass_ug"] = observed_colors[:, 0] * delta_airmass
    features["apo_blue_airmass_gr"] = observed_colors[:, 1] * delta_airmass

    features["apo_airmass_bin_low"] = airmass < 1.15
    features["apo_airmass_bin_reference"] = (airmass >= 1.15) & (airmass < 1.30)
    features["apo_airmass_bin_high"] = (airmass >= 1.30) & (airmass < 1.45)
    features["apo_airmass_bin_extreme"] = airmass >= 1.45

    return features


FEATURE_GROUPS = [
    {
        "name": "apo_airmass_chromaticity",
        "fn": add_apo_airmass_chromaticity,
        "depends_on": [],
        "description": "APO latitude airmass and SDSS ugriz chromatic extinction geometry features.",
    }
]