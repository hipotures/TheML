import numpy as np
import pandas as pd


COLOR_MARGIN_SCALES = (0.35, 0.35, 0.50, 0.50, 0.35, 0.35, 0.35, 0.35)
MAG_MARGIN_SCALES = (3.0, 3.0, 2.0, 2.0, 1.7, 1.7, 1.4, 1.4, 1.4, 1.4)


def add_faint_blue_galaxy_wedge_margins(raw, deps, aux):
    index = raw.index

    u = pd.to_numeric(raw["u"], errors="coerce").astype("float64")
    g = pd.to_numeric(raw["g"], errors="coerce").astype("float64")
    r = pd.to_numeric(raw["r"], errors="coerce").astype("float64")
    i = pd.to_numeric(raw["i"], errors="coerce").astype("float64")
    z = pd.to_numeric(raw["z"], errors="coerce").astype("float64")

    ug = u - g
    gr = g - r
    ri = r - i
    iz = i - z

    ug_line = ug.clip(lower=-0.60, upper=3.20)
    gr_lower_line = 0.40 + 0.60 * ug_line
    gr_upper_line = 1.70 - 0.10 * ug_line

    color_raw = pd.DataFrame(
        {
            "gr_lower_margin": gr - gr_lower_line,
            "gr_upper_margin": gr_upper_line - gr,
            "ug_lower_margin": ug + 0.50,
            "ug_upper_margin": 3.00 - ug,
            "ri_lower_margin": ri + 0.50,
            "ri_upper_margin": 1.80 - ri,
            "iz_lower_margin": iz + 1.00,
            "iz_upper_margin": 1.50 - iz,
        },
        index=index,
    ).replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(lower=-10.0, upper=10.0)

    mag_raw = pd.DataFrame(
        {
            "u_faint_margin": u - 18.0,
            "u_bright_margin": 24.0 - u,
            "g_faint_margin": g - 18.0,
            "g_bright_margin": 21.5 - g,
            "r_faint_margin": r - 17.8,
            "r_bright_margin": 19.5 - r,
            "i_faint_margin": i - 16.5,
            "i_bright_margin": 20.5 - i,
            "z_faint_margin": z - 16.0,
            "z_bright_margin": 20.0 - z,
        },
        index=index,
    ).replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(lower=-10.0, upper=10.0)

    color_scale = np.asarray(COLOR_MARGIN_SCALES, dtype="float64")
    mag_scale = np.asarray(MAG_MARGIN_SCALES, dtype="float64")

    color_norm = color_raw.to_numpy(dtype="float64", copy=True) / color_scale
    mag_norm = mag_raw.to_numpy(dtype="float64", copy=True) / mag_scale

    color_positive = np.maximum(color_norm, 0.0)
    mag_positive = np.maximum(mag_norm, 0.0)
    color_negative = np.maximum(-color_norm, 0.0)
    mag_negative = np.maximum(-mag_norm, 0.0)

    color_softmin = -np.log(np.exp(-8.0 * color_norm).sum(axis=1)) / 8.0
    mag_softmin = -np.log(np.exp(-8.0 * mag_norm).sum(axis=1)) / 8.0

    color_violation_count = (color_norm < 0.0).sum(axis=1).astype("int16")
    mag_violation_count = (mag_norm < 0.0).sum(axis=1).astype("int16")
    color_violation_depth = color_negative.sum(axis=1)
    mag_violation_depth = mag_negative.sum(axis=1)

    combined_wedge_depth = 0.65 * color_softmin + 0.35 * mag_softmin
    combined_wedge_penalty = (
        combined_wedge_depth
        - 0.20 * color_violation_count
        - 0.08 * mag_violation_count
        - 0.15 * color_violation_depth
        - 0.05 * mag_violation_depth
    )

    gr_lower_excess = gr - gr_lower_line
    sloped_intensity = np.exp(0.1411 * gr_lower_excess.replace([np.inf, -np.inf], np.nan).fillna(0.0))
    sloped_intensity = np.clip(sloped_intensity, 0.0, 10.0)

    features = pd.DataFrame(index=index)

    for col in color_raw.columns:
        features[col] = color_raw[col].astype("float32")
    for col in mag_raw.columns:
        features[col] = mag_raw[col].astype("float32")

    features["color_min_norm_margin"] = color_norm.min(axis=1).astype("float32")
    features["color_p20_norm_margin"] = np.percentile(color_norm, 20, axis=1).astype("float32")
    features["color_softmin_norm_margin"] = color_softmin.astype("float32")
    features["color_mean_positive_margin"] = color_positive.mean(axis=1).astype("float32")
    features["color_violation_count"] = color_violation_count
    features["color_violation_depth"] = color_violation_depth.astype("float32")

    features["mag_min_norm_margin"] = mag_norm.min(axis=1).astype("float32")
    features["mag_p20_norm_margin"] = np.percentile(mag_norm, 20, axis=1).astype("float32")
    features["mag_softmin_norm_margin"] = mag_softmin.astype("float32")
    features["mag_mean_positive_margin"] = mag_positive.mean(axis=1).astype("float32")
    features["mag_violation_count"] = mag_violation_count
    features["mag_violation_depth"] = mag_violation_depth.astype("float32")

    features["combined_wedge_depth"] = combined_wedge_depth.astype("float32")
    features["combined_wedge_penalty"] = combined_wedge_penalty.astype("float32")
    features["all_inside_color"] = color_violation_count == 0
    features["all_inside_magnitude"] = mag_violation_count == 0
    features["all_inside_full"] = (color_violation_count == 0) & (mag_violation_count == 0)
    features["sloped_boundary_intensity"] = sloped_intensity.astype("float32")
    features["log1p_sloped_boundary_intensity"] = np.log1p(sloped_intensity).astype("float32")

    return features


FEATURE_GROUPS = [
    {
        "name": "faint_blue_galaxy_wedge_margins",
        "fn": add_faint_blue_galaxy_wedge_margins,
        "depends_on": [],
        "description": "Signed and normalized margins for a calibrated faint blue galaxy color-magnitude selection wedge.",
    }
]