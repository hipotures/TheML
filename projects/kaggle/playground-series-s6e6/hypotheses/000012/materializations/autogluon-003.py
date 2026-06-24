import numpy as np
import pandas as pd

FEATURE_SCALE_COLOR = (0.35, 0.35, 0.50, 0.50, 0.35, 0.35, 0.35, 0.35)
FEATURE_SCALE_MAG = (3.0, 3.0, 2.0, 2.0, 1.7, 1.7, 1.4, 1.4, 1.4, 1.4)
UG_CLIP = (-0.60, 3.20)


def _to_float_array(series):
    return pd.to_numeric(series, errors="coerce").to_numpy(dtype="float64")


def _sanitize_and_clip(margins):
    margins = np.where(np.isfinite(margins), margins, 0.0)
    return np.clip(margins, -10.0, 10.0)


def _row_logsumexp(values):
    row_max = np.max(values, axis=1, keepdims=True)
    stable = np.exp(values - row_max)
    return np.squeeze(row_max, axis=1) + np.log(np.sum(stable, axis=1))


def add_faint_blue_galaxy_wedge_margins(raw, deps, aux):
    u = _to_float_array(raw["u"])
    g = _to_float_array(raw["g"])
    r = _to_float_array(raw["r"])
    i = _to_float_array(raw["i"])
    z = _to_float_array(raw["z"])

    ug = u - g
    gr = g - r
    ri = r - i
    iz = i - z

    ug_clipped = np.clip(ug, UG_CLIP[0], UG_CLIP[1])

    color_margins = np.column_stack(
        [
            gr - (0.40 + 0.60 * ug_clipped),
            (1.70 - 0.10 * ug_clipped) - gr,
            ug + 0.50,
            3.00 - ug,
            ri + 0.50,
            1.80 - ri,
            iz + 1.00,
            1.50 - iz,
        ]
    )
    color_margins = _sanitize_and_clip(color_margins)
    color_norm = color_margins / np.asarray(FEATURE_SCALE_COLOR, dtype="float64")

    mag_margins = np.column_stack(
        [
            u - 18.0,
            24.0 - u,
            g - 18.0,
            21.5 - g,
            r - 17.8,
            19.5 - r,
            i - 16.5,
            20.5 - i,
            z - 16.0,
            20.0 - z,
        ]
    )
    mag_margins = _sanitize_and_clip(mag_margins)
    mag_norm = mag_margins / np.asarray(FEATURE_SCALE_MAG, dtype="float64")

    color_depth_min = np.min(color_norm, axis=1)
    color_depth_q20 = np.percentile(color_norm, 20, axis=1)
    color_depth_soft = -_row_logsumexp(-8.0 * color_norm) / 8.0
    color_viol = np.sum(color_norm < 0, axis=1).astype(np.int64)

    mag_depth_min = np.min(mag_norm, axis=1)
    mag_depth_q20 = np.percentile(mag_norm, 20, axis=1)
    mag_depth_soft = -_row_logsumexp(-8.0 * mag_norm) / 8.0
    mag_viol = np.sum(mag_norm < 0, axis=1).astype(np.int64)

    sample_intensity = np.exp(0.1411 * (gr - (0.40 + 0.60 * ug)))
    sample_intensity = np.where(np.isfinite(sample_intensity), sample_intensity, 0.0)
    sample_intensity = np.clip(sample_intensity, 0.0, 10.0)

    color_only_score = 0.60 * color_depth_soft - 0.25 * color_viol
    full_score = (
        0.60 * color_depth_soft
        + 0.40 * mag_depth_soft
        - 0.25 * color_viol
        - 0.10 * mag_viol
        + sample_intensity
    )

    return pd.DataFrame(
        {
            "color_depth_min": color_depth_min,
            "color_depth_q20": color_depth_q20,
            "color_depth_soft": color_depth_soft,
            "color_viol": color_viol,
            "mag_depth_min": mag_depth_min,
            "mag_depth_q20": mag_depth_q20,
            "mag_depth_soft": mag_depth_soft,
            "mag_viol": mag_viol,
            "color_only_score": color_only_score,
            "sample_intensity": sample_intensity,
            "wedge_score": full_score,
        },
        index=raw.index,
    )


FEATURE_GROUPS = [
    {
        "name": "faint_blue_galaxy_wedge_margins",
        "fn": add_faint_blue_galaxy_wedge_margins,
        "depends_on": [],
        "description": "Builds soft and bounded SDSS-style wedge and magnitude-depth features with violation and smooth-membership scores for faint-blue galaxy selection.",
    }
]