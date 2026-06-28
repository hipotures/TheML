import numpy as np
import pandas as pd


EPSILON = 1e-12
FLUX_MIN = 1e-12
FLUX_MAX = 1e6
LUPTON_Q = 10.0
LUPTON_SCALE = 30.0


def add_lupton_rgb_chromaticity(raw, deps, aux):
    u_mag = pd.to_numeric(raw["u"], errors="coerce").to_numpy(dtype=float, copy=True)
    g_mag = pd.to_numeric(raw["g"], errors="coerce").to_numpy(dtype=float, copy=True)
    r_mag = pd.to_numeric(raw["r"], errors="coerce").to_numpy(dtype=float, copy=True)
    i_mag = pd.to_numeric(raw["i"], errors="coerce").to_numpy(dtype=float, copy=True)
    z_mag = pd.to_numeric(raw["z"], errors="coerce").to_numpy(dtype=float, copy=True)

    u_flux = np.clip(np.nan_to_num(10.0 ** ((22.5 - u_mag) / 2.5), nan=FLUX_MIN, posinf=FLUX_MAX, neginf=FLUX_MIN), FLUX_MIN, FLUX_MAX)
    g_flux = np.clip(np.nan_to_num(10.0 ** ((22.5 - g_mag) / 2.5), nan=FLUX_MIN, posinf=FLUX_MAX, neginf=FLUX_MIN), FLUX_MIN, FLUX_MAX)
    r_flux = np.clip(np.nan_to_num(10.0 ** ((22.5 - r_mag) / 2.5), nan=FLUX_MIN, posinf=FLUX_MAX, neginf=FLUX_MIN), FLUX_MIN, FLUX_MAX)
    i_flux = np.clip(np.nan_to_num(10.0 ** ((22.5 - i_mag) / 2.5), nan=FLUX_MIN, posinf=FLUX_MAX, neginf=FLUX_MIN), FLUX_MIN, FLUX_MAX)
    z_flux = np.clip(np.nan_to_num(10.0 ** ((22.5 - z_mag) / 2.5), nan=FLUX_MIN, posinf=FLUX_MAX, neginf=FLUX_MIN), FLUX_MIN, FLUX_MAX)

    blue = 0.30 * u_flux + 0.70 * g_flux
    green = 0.55 * r_flux + 0.45 * g_flux
    red = 0.65 * i_flux + 0.35 * z_flux

    total = red + green + blue
    valid_total = total > EPSILON
    safe_total = np.where(valid_total, total, 1.0)
    intensity = total / 3.0
    safe_intensity = intensity + EPSILON

    red_frac = np.where(valid_total, red / safe_total, 1.0 / 3.0)
    green_frac = np.where(valid_total, green / safe_total, 1.0 / 3.0)
    blue_frac = np.where(valid_total, blue / safe_total, 1.0 / 3.0)

    compressed_red = np.arcsinh(LUPTON_Q * red / safe_intensity) / np.arcsinh(LUPTON_SCALE)
    compressed_green = np.arcsinh(LUPTON_Q * green / safe_intensity) / np.arcsinh(LUPTON_SCALE)
    compressed_blue = np.arcsinh(LUPTON_Q * blue / safe_intensity) / np.arcsinh(LUPTON_SCALE)

    compressed_total = compressed_red + compressed_green + compressed_blue
    valid_compressed_total = compressed_total > EPSILON
    safe_compressed_total = np.where(valid_compressed_total, compressed_total, 1.0)

    compressed_red_frac = np.where(valid_compressed_total, compressed_red / safe_compressed_total, 1.0 / 3.0)
    compressed_green_frac = np.where(valid_compressed_total, compressed_green / safe_compressed_total, 1.0 / 3.0)
    compressed_blue_frac = np.where(valid_compressed_total, compressed_blue / safe_compressed_total, 1.0 / 3.0)

    hue_angle = np.arctan2(np.sqrt(3.0) * (green - blue), (2.0 * red) - green - blue)
    max_frac = np.maximum(np.maximum(red_frac, green_frac), blue_frac)
    min_frac = np.minimum(np.minimum(red_frac, green_frac), blue_frac)
    saturation = max_frac - min_frac

    whitepoint_distance = np.sqrt(
        ((red_frac - (1.0 / 3.0)) ** 2)
        + ((green_frac - (1.0 / 3.0)) ** 2)
        + ((blue_frac - (1.0 / 3.0)) ** 2)
    )

    entropy_terms = np.stack([red_frac, green_frac, blue_frac], axis=0)
    chromatic_entropy = -np.sum(entropy_terms * np.log(np.clip(entropy_terms, EPSILON, 1.0)), axis=0) / np.log(3.0)

    dominant_index = np.argmax(np.stack([red, green, blue], axis=0), axis=0)

    return pd.DataFrame(
        {
            "red_fraction": red_frac,
            "green_fraction": green_frac,
            "blue_fraction": blue_frac,
            "compressed_red": compressed_red,
            "compressed_green": compressed_green,
            "compressed_blue": compressed_blue,
            "compressed_red_fraction": compressed_red_frac,
            "compressed_green_fraction": compressed_green_frac,
            "compressed_blue_fraction": compressed_blue_frac,
            "hue_angle": hue_angle,
            "saturation": saturation,
            "whitepoint_distance": whitepoint_distance,
            "chromatic_entropy": chromatic_entropy,
            "red_minus_blue": red_frac - blue_frac,
            "blue_minus_green": blue_frac - green_frac,
            "red_minus_green": red_frac - green_frac,
            "log10_total_flux": np.log10(total + EPSILON),
            "dominant_red": dominant_index == 0,
            "dominant_green": dominant_index == 1,
            "dominant_blue": dominant_index == 2,
        },
        index=raw.index,
    )


FEATURE_GROUPS = [
    {
        "name": "lupton_rgb_chromaticity",
        "fn": add_lupton_rgb_chromaticity,
        "depends_on": [],
        "description": "Encodes SDSS ugriz magnitudes as Lupton-inspired RGB chromaticity, compressed color fractions, hue, saturation, entropy, and dominant-channel indicators.",
    }
]