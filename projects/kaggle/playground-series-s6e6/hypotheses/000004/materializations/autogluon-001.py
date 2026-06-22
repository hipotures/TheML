import numpy as np
import pandas as pd


def add_distance_corrected_luminosity_signature(raw, deps, aux):
    band_cols = ("u", "g", "r", "i", "z")

    redshift = raw["redshift"].to_numpy(dtype=float)
    z_pos = np.maximum(redshift, 0.0)

    c = 299792.458
    H0 = 70.0
    q0 = -0.55

    d_mpc = (c / H0) * z_pos * (1.0 + 0.5 * (1.0 - q0) * z_pos)
    d_mpc = np.maximum(d_mpc, 1e-4)

    distance_term = 5.0 * np.log10(d_mpc) + 25.0

    pseudo_abs = np.column_stack(
        [
            raw[band].to_numpy(dtype=float) - distance_term
            for band in band_cols
        ]
    )

    pseudo_abs_median = np.median(pseudo_abs, axis=1)
    pseudo_abs_min = np.min(pseudo_abs, axis=1)
    pseudo_abs_max = np.max(pseudo_abs, axis=1)
    pseudo_abs_range = pseudo_abs_max - pseudo_abs_min

    features = {
        "lum_abs_median_5band": pseudo_abs_median,
        "lum_abs_min_5band": pseudo_abs_min,
        "lum_abs_max_5band": pseudo_abs_max,
        "lum_abs_range_5band": pseudo_abs_range,
        "lum_abs_regime_le_m24": pseudo_abs_median <= -24,
        "lum_abs_regime_m24_to_m20": (pseudo_abs_median > -24) & (pseudo_abs_median <= -20),
        "lum_abs_regime_m20_to_m16": (pseudo_abs_median > -20) & (pseudo_abs_median <= -16),
        "lum_abs_regime_m16_to_m8": (pseudo_abs_median > -16) & (pseudo_abs_median <= -8),
        "lum_abs_regime_gt_m8": pseudo_abs_median > -8,
    }

    return pd.DataFrame(features, index=raw.index)


FEATURE_GROUPS = [
    {
        "name": "distance_corrected_luminosity_signature",
        "fn": add_distance_corrected_luminosity_signature,
        "depends_on": [],
        "description": "Compute redshift-adjusted pseudo-absolute magnitudes and compact summaries that encode intrinsic luminosity regimes for broadband features.",
    }
]