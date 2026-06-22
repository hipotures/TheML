import numpy as np
import pandas as pd

POINT_DEPTH_MARGINS = (
    ("r", 17.77, "r_minus_17_77"),
    ("i", 15.0, "i_minus_15_0"),
    ("i", 19.1, "i_minus_19_1"),
    ("i", 20.2, "i_minus_20_2"),
    ("i", 20.4, "i_minus_20_4"),
    ("i", 21.3, "i_minus_21_3"),
    ("i", 22.45, "i_minus_22_45"),
    ("g", 22.0, "g_minus_22_0"),
    ("r", 21.85, "r_minus_21_85"),
    ("r", 19.0, "r_minus_19_0"),
    ("r", 19.5, "r_minus_19_5"),
)

INTERVAL_DEPTH_MARGINS = (
    ("i", 15.0, 19.1, "i_interval_15_19_1", "i_interval_15_19_1_signed_margin"),
    ("i", 15.0, 20.2, "i_interval_15_20_2", "i_interval_15_20_2_signed_margin"),
    ("i", 17.75, 22.45, "i_interval_17_75_22_45", "i_interval_17_75_22_45_signed_margin"),
    ("r", 17.77, 19.0, "r_interval_17_77_19_0", "r_interval_17_77_19_0_signed_margin"),
    ("r", 19.0, 19.5, "r_interval_19_0_19_5", "r_interval_19_0_19_5_signed_margin"),
)

QUASAR_DEPTH_BOUNDARIES = (
    ("g", 22.0),
    ("r", 21.85),
    ("i", 19.1),
    ("i", 20.2),
    ("i", 20.4),
    ("i", 21.3),
    ("i", 22.45),
)

GALAXY_DEPTH_BOUNDARIES = (
    ("r", 17.77),
    ("r", 19.0),
    ("r", 19.5),
)


def _to_float_array(values):
    return pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)


def _interval_membership_and_signed_margin(values, lower, upper):
    values = _to_float_array(values)
    valid = np.isfinite(values)

    indicator = np.zeros(len(values), dtype=np.int8)
    signed_margin = np.zeros(len(values), dtype=float)

    if not np.any(valid):
        return indicator, signed_margin

    between = valid & (values >= lower) & (values <= upper)
    below = valid & (values < lower)
    above = valid & (values > upper)

    indicator[between] = 1

    # Signed margins:
    # inside interval: negative distance to nearest bound (closer -> closer to zero),
    # below interval: negative margin to lower edge,
    # above interval: positive margin to upper edge.
    signed_margin[below] = values[below] - lower
    signed_margin[above] = values[above] - upper
    signed_margin[between] = -np.minimum(values[between] - lower, upper - values[between])

    return indicator, signed_margin


def _nearest_signed_boundary_distance(raw, boundaries):
    n = len(raw)
    min_abs = np.full(n, np.inf, dtype=float)
    nearest_signed = np.zeros(n, dtype=float)

    for column, threshold in boundaries:
        values = _to_float_array(raw[column])
        valid = np.isfinite(values)
        if not np.any(valid):
            continue

        signed = values - float(threshold)
        abs_signed = np.abs(signed)
        better = valid & (abs_signed < min_abs)

        min_abs[better] = abs_signed[better]
        nearest_signed[better] = signed[better]

    nearest_signed[~np.isfinite(min_abs)] = 0.0
    return nearest_signed


def add_survey_depth_limit_margins(raw, deps, aux):
    index = raw.index
    features = {}

    for column, threshold, feature_name in POINT_DEPTH_MARGINS:
        values = _to_float_array(raw[column])
        out = np.zeros(len(values), dtype=float)
        valid = np.isfinite(values)
        out[valid] = values[valid] - threshold
        features[feature_name] = pd.Series(out, index=index)

    for column, lower, upper, indicator_name, margin_name in INTERVAL_DEPTH_MARGINS:
        values = raw[column]
        indicator, signed_margin = _interval_membership_and_signed_margin(values, lower, upper)
        features[indicator_name] = pd.Series(indicator, index=index)
        features[margin_name] = pd.Series(signed_margin, index=index)

    quasar_nearest = _nearest_signed_boundary_distance(raw, QUASAR_DEPTH_BOUNDARIES)
    galaxy_nearest = _nearest_signed_boundary_distance(raw, GALAXY_DEPTH_BOUNDARIES)

    features["min_signed_distance_to_quasar_depth_boundary"] = pd.Series(quasar_nearest, index=index)
    features["min_signed_distance_to_galaxy_depth_boundary"] = pd.Series(galaxy_nearest, index=index)
    features["quasar_minus_galaxy_depth_profile"] = pd.Series(quasar_nearest - galaxy_nearest, index=index)

    return pd.DataFrame(features, index=index)


FEATURE_GROUPS = [
    {
        "name": "survey_depth_limit_margins",
        "fn": add_survey_depth_limit_margins,
        "depends_on": [],
        "description": "Encode SDSS depth-regime structure with signed magnitude margins, interval membership, and compact quasar-vs-galaxy depth proximity profiles.",
    },
]