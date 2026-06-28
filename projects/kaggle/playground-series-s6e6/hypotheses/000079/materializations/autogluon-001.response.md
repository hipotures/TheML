import numpy as np
import pandas as pd


RR_BOX_BOUNDS = {
    "ug": (0.99, 1.28),
    "gr": (-0.11, 0.31),
    "ri": (-0.13, 0.20),
    "iz": (-0.19, 0.23),
}

ROTATED_COMPLETE_BOUNDS = {
    "dug": (-0.05, 0.35),
    "dgr": (0.06, 0.55),
}

ROTATED_RESTRICTED_BOUNDS = {
    "dug": (0.15, 0.35),
    "dgr": (0.23, 0.55),
}

SPECTRAL_COMPATIBILITY = {
    "A/F": 1.0,
    "O/B": 0.4,
    "G/K": 0.1,
    "M": 0.0,
}


def _numeric(raw, column):
    values = pd.to_numeric(raw[column], errors="coerce").astype("float64")
    return values.replace([np.inf, -np.inf], np.nan)


def _clip_or_worst(values, lower=-10.0, upper=10.0, worst=-10.0):
    clipped = values.clip(lower=lower, upper=upper)
    return clipped.where(np.isfinite(clipped), worst)


def _positive_violation(values, low, high, half_width):
    lower_violation = ((low - values) / half_width).clip(lower=0.0, upper=10.0)
    upper_violation = ((values - high) / half_width).clip(lower=0.0, upper=10.0)
    lower_violation = lower_violation.where(np.isfinite(lower_violation), 10.0)
    upper_violation = upper_violation.where(np.isfinite(upper_violation), 10.0)
    return lower_violation, upper_violation


def _interval_features(values, low, high):
    half_width = (high - low) / 2.0
    center = (low + high) / 2.0

    lower_margin = _clip_or_worst((values - low) / half_width)
    upper_margin = _clip_or_worst((high - values) / half_width)
    lower_violation, upper_violation = _positive_violation(values, low, high, half_width)
    center_distance = _clip_or_worst((values - center) / half_width, worst=10.0)

    return lower_margin, upper_margin, lower_violation, upper_violation, center_distance


def add_rr_lyrae_instability_strip_margins(raw, deps, aux):
    u = _numeric(raw, "u")
    g = _numeric(raw, "g")
    r = _numeric(raw, "r")
    i = _numeric(raw, "i")
    z = _numeric(raw, "z")
    redshift = _numeric(raw, "redshift")

    ug = u - g
    gr = g - r
    ri = r - i
    iz = i - z

    features = pd.DataFrame(index=raw.index)
    colors = {
        "ug": ug,
        "gr": gr,
        "ri": ri,
        "iz": iz,
    }

    box_margins = []
    box_violations = []
    center_distances = []

    for name, values in colors.items():
        low, high = RR_BOX_BOUNDS[name]
        lower_margin, upper_margin, lower_violation, upper_violation, center_distance = _interval_features(
            values, low, high
        )

        features[f"{name}_color"] = values.where(np.isfinite(values), np.nan)
        features[f"{name}_rr_lower_margin"] = lower_margin
        features[f"{name}_rr_upper_margin"] = upper_margin
        features[f"{name}_rr_lower_violation"] = lower_violation
        features[f"{name}_rr_upper_violation"] = upper_violation
        features[f"{name}_rr_center_distance"] = center_distance

        box_margins.extend([lower_margin, upper_margin])
        box_violations.extend([lower_violation, upper_violation])
        center_distances.append(center_distance)

    margin_frame = pd.concat(box_margins, axis=1)
    violation_frame = pd.concat(box_violations, axis=1)
    center_distance_frame = pd.concat(center_distances, axis=1)

    features["rr_box_min_margin"] = margin_frame.min(axis=1)
    features["rr_box_mean_violation"] = violation_frame.mean(axis=1)
    features["rr_box_max_violation"] = violation_frame.max(axis=1)
    features["rr_box_center_distance_sq"] = (center_distance_frame ** 2).sum(axis=1)
    features["rr_box_inside"] = (features["rr_box_max_violation"] <= 0.0).astype("int8")
    features["rr_box_score"] = 1.0 / (1.0 + features["rr_box_mean_violation"] + features["rr_box_max_violation"])

    dug = ug + 0.67 * gr - 1.07
    dgr = 0.45 * ug - gr - 0.12

    features["rotated_dug"] = dug.where(np.isfinite(dug), np.nan)
    features["rotated_dgr"] = dgr.where(np.isfinite(dgr), np.nan)

    rotated_scores = {}

    for strip_name, bounds in (
        ("complete", ROTATED_COMPLETE_BOUNDS),
        ("restricted", ROTATED_RESTRICTED_BOUNDS),
    ):
        strip_margins = []
        strip_violations = []

        for axis_name, values in (("dug", dug), ("dgr", dgr)):
            low, high = bounds[axis_name]
            lower_margin, upper_margin, lower_violation, upper_violation, center_distance = _interval_features(
                values, low, high
            )

            features[f"{strip_name}_{axis_name}_lower_margin"] = lower_margin
            features[f"{strip_name}_{axis_name}_upper_margin"] = upper_margin
            features[f"{strip_name}_{axis_name}_lower_violation"] = lower_violation
            features[f"{strip_name}_{axis_name}_upper_violation"] = upper_violation
            features[f"{strip_name}_{axis_name}_center_distance"] = center_distance

            strip_margins.extend([lower_margin, upper_margin])
            strip_violations.extend([lower_violation, upper_violation])

        strip_margin_frame = pd.concat(strip_margins, axis=1)
        strip_violation_frame = pd.concat(strip_violations, axis=1)

        features[f"{strip_name}_strip_min_margin"] = strip_margin_frame.min(axis=1)
        features[f"{strip_name}_strip_mean_violation"] = strip_violation_frame.mean(axis=1)
        features[f"{strip_name}_strip_max_violation"] = strip_violation_frame.max(axis=1)
        features[f"{strip_name}_strip_inside"] = (features[f"{strip_name}_strip_max_violation"] <= 0.0).astype("int8")

        score = 1.0 / (
            1.0
            + features[f"{strip_name}_strip_mean_violation"]
            + features[f"{strip_name}_strip_max_violation"]
        )
        features[f"{strip_name}_strip_score"] = score
        rotated_scores[strip_name] = score

    brightness_gate = ((r >= 14.0) & (r <= 20.0)).astype("float64")
    u_calibrated_gate = (u < 21.1).astype("float64")
    r_calibrated_gate = (r < 19.7).astype("float64")
    finite_photometry_gate = np.isfinite(u) & np.isfinite(g) & np.isfinite(r) & np.isfinite(i) & np.isfinite(z)

    spectral = raw["spectral_type"].astype("string")
    spectral_compatibility = spectral.map(SPECTRAL_COMPATIBILITY).astype("float64").fillna(0.0)

    near_zero_redshift_support = 1.0 / (1.0 + (redshift.abs() / 0.004) ** 2)
    near_zero_redshift_support = near_zero_redshift_support.where(np.isfinite(near_zero_redshift_support), 0.0)

    calibrated_gate = (
        brightness_gate
        * u_calibrated_gate
        * r_calibrated_gate
        * finite_photometry_gate.astype("float64")
    )

    features["rr_brightness_gate_14_20_r"] = brightness_gate
    features["rr_u_single_epoch_gate"] = u_calibrated_gate
    features["rr_r_single_epoch_gate"] = r_calibrated_gate
    features["rr_calibrated_regime_gate"] = calibrated_gate
    features["rr_spectral_compatibility"] = spectral_compatibility
    features["rr_near_zero_redshift_support"] = near_zero_redshift_support

    support_gate = calibrated_gate * spectral_compatibility * near_zero_redshift_support

    features["rr_box_calibrated_score"] = features["rr_box_score"] * calibrated_gate
    features["rr_box_stellar_support"] = features["rr_box_score"] * support_gate

    for strip_name, score in rotated_scores.items():
        features[f"{strip_name}_strip_calibrated_score"] = score * calibrated_gate
        features[f"{strip_name}_strip_stellar_support"] = score * support_gate

    features["rr_combined_color_score"] = (
        features["rr_box_score"]
        * rotated_scores["complete"]
        * rotated_scores["restricted"]
    ) ** (1.0 / 3.0)
    features["rr_combined_stellar_support"] = features["rr_combined_color_score"] * support_gate

    return features


FEATURE_GROUPS = [
    {
        "name": "rr_lyrae_instability_strip_margins",
        "fn": add_rr_lyrae_instability_strip_margins,
        "depends_on": [],
        "description": "RR Lyrae instability-strip color margins with brightness, spectral-type, and near-zero-redshift consistency gates.",
    }
]