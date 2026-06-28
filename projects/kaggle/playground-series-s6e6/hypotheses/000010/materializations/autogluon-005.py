import numpy as np
import pandas as pd

BAND_NAMES = ("u", "g", "r", "i", "z")
OBS_WAVELENGTHS_ANGSTROM = (3543.0, 4770.0, 6231.0, 7625.0, 9134.0)
LANDMARKS_ANGSTROM = (912.0, 1216.0, 2500.0, 3646.0, 4000.0, 7000.0)
LANDMARK_NAMES = ("lya_limit_912", "lya_1216", "uv_2500", "balmer_3646", "break_4000", "optical_7000")
REST_BIN_NAMES = (
    "below_912",
    "912_1216",
    "1216_2500",
    "2500_3646",
    "3646_4000",
    "4000_7000",
    "above_7000",
)
REST_BIN_EDGES = (912.0, 1216.0, 2500.0, 3646.0, 4000.0, 7000.0)


def add_rest_frame_filter_landmarks(raw, deps, aux):
    z_safe = np.maximum(raw["redshift"].to_numpy(dtype=float), 0.0)
    denom = 1.0 + z_safe

    obs_waves = np.asarray(OBS_WAVELENGTHS_ANGSTROM, dtype=float)
    rest_lambdas = obs_waves.reshape(1, -1) / denom.reshape(-1, 1)
    log_rest_lambdas = np.log(rest_lambdas)

    mags = raw.loc[:, BAND_NAMES].to_numpy(dtype=float)

    span_min = rest_lambdas[:, 0]
    span_max = rest_lambdas[:, -1]
    span_width = span_max - span_min

    features = pd.DataFrame(index=raw.index)

    features["neg_redshift_flag"] = (raw["redshift"].to_numpy(dtype=float) < 0.0).astype(np.int8)
    for idx, band in enumerate(BAND_NAMES):
        features[f"{band}_rest_lambda"] = rest_lambdas[:, idx]

    features["span_min"] = span_min
    features["span_max"] = span_max
    features["span_width"] = span_width
    features["log_span_width"] = np.log1p(span_width)

    for idx in range(len(BAND_NAMES) - 1):
        left = BAND_NAMES[idx]
        right = BAND_NAMES[idx + 1]
        center = np.exp((log_rest_lambdas[:, idx] + log_rest_lambdas[:, idx + 1]) / 2.0)
        features[f"{left}_{right}_interval_rest_center"] = center

    bin_ids = np.searchsorted(np.asarray(REST_BIN_EDGES, dtype=float), rest_lambdas, side="right")
    bin_counts = np.zeros((len(raw), len(REST_BIN_NAMES)), dtype=float)
    row_idx = np.arange(len(raw))
    for band_idx in range(len(BAND_NAMES)):
        bin_counts[row_idx, bin_ids[:, band_idx]] += 1.0

    bin_fracs = bin_counts / float(len(BAND_NAMES))
    for idx, bin_name in enumerate(REST_BIN_NAMES):
        features[f"{bin_name}_count"] = bin_counts[:, idx]
        features[f"{bin_name}_frac"] = bin_fracs[:, idx]

    entropy_terms = np.zeros_like(bin_fracs)
    positive_frac_mask = bin_fracs > 0.0
    entropy_terms[positive_frac_mask] = bin_fracs[positive_frac_mask] * np.log(bin_fracs[positive_frac_mask])
    features["rest_bin_count_entropy"] = -np.sum(entropy_terms, axis=1)

    interval_left_logs = log_rest_lambdas[:, :-1]
    interval_right_logs = log_rest_lambdas[:, 1:]
    interval_center_logs = (interval_left_logs + interval_right_logs) / 2.0

    for landmark_name, landmark in zip(LANDMARK_NAMES, LANDMARKS_ANGSTROM):
        log_landmark = np.log(float(landmark))
        in_span = (span_min <= landmark) & (landmark <= span_max)
        side = np.where(landmark < span_min, -1, np.where(landmark > span_max, 1, 0)).astype(np.int8)
        edge_distance = np.where(landmark < span_min, span_min - landmark, np.where(landmark > span_max, landmark - span_max, 0.0))

        log_ratios = log_rest_lambdas - log_landmark
        abs_log_ratios = np.abs(log_ratios)
        nearest_idx = np.argmin(abs_log_ratios, axis=1)
        nearest_mag = mags[row_idx, nearest_idx]
        nearest_lambda = rest_lambdas[row_idx, nearest_idx]
        signed_log_ratio = log_ratios[row_idx, nearest_idx]

        bracket_left_idx = np.sum(rest_lambdas <= landmark, axis=1) - 1
        bracket_left_idx = np.clip(bracket_left_idx, 0, len(BAND_NAMES) - 2)
        bracket_right_idx = bracket_left_idx + 1

        bracket_left_mag = mags[row_idx, bracket_left_idx]
        bracket_right_mag = mags[row_idx, bracket_right_idx]
        bracket_left_log_lambda = log_rest_lambdas[row_idx, bracket_left_idx]
        bracket_right_log_lambda = log_rest_lambdas[row_idx, bracket_right_idx]
        bracket_den = bracket_right_log_lambda - bracket_left_log_lambda

        frac_position_inside = (log_landmark - bracket_left_log_lambda) / bracket_den
        interp_mag_inside = bracket_left_mag + frac_position_inside * (bracket_right_mag - bracket_left_mag)
        curvature_inside = interp_mag_inside - ((bracket_left_mag + bracket_right_mag) / 2.0)

        frac_position = np.where(in_span, frac_position_inside, 0.0)
        interp_mag = np.where(in_span, interp_mag_inside, 0.0)
        curvature = np.where(in_span, curvature_inside, 0.0)

        prefix = f"{landmark_name}"
        features[f"{prefix}_in_span"] = in_span.astype(np.int8)
        features[f"{prefix}_side"] = side
        features[f"{prefix}_outside_log_distance"] = np.log1p(edge_distance)
        features[f"{prefix}_nearest_band_idx"] = nearest_idx.astype(np.int8)
        features[f"{prefix}_nearest_band_mag"] = nearest_mag
        features[f"{prefix}_nearest_band_rest_lambda"] = nearest_lambda
        features[f"{prefix}_signed_log_ratio"] = signed_log_ratio
        features[f"{prefix}_abs_log_ratio"] = np.abs(signed_log_ratio)
        features[f"{prefix}_bracket_left_idx"] = bracket_left_idx.astype(np.int8)
        features[f"{prefix}_bracket_right_idx"] = bracket_right_idx.astype(np.int8)
        features[f"{prefix}_bracket_left_mag"] = bracket_left_mag
        features[f"{prefix}_bracket_right_mag"] = bracket_right_mag
        features[f"{prefix}_bracket_color"] = bracket_left_mag - bracket_right_mag
        features[f"{prefix}_fractional_position"] = frac_position
        features[f"{prefix}_interpolated_mag_at_landmark"] = interp_mag
        features[f"{prefix}_curvature_proxy"] = curvature

        for interval_idx in range(len(BAND_NAMES) - 1):
            left = BAND_NAMES[interval_idx]
            right = BAND_NAMES[interval_idx + 1]
            interval_in = (
                (rest_lambdas[:, interval_idx] <= landmark)
                & (landmark <= rest_lambdas[:, interval_idx + 1])
            )
            features[f"{left}_{right}_{prefix}_center_signed_log_distance"] = interval_center_logs[:, interval_idx] - log_landmark
            features[f"{left}_{right}_{prefix}_inside_interval"] = interval_in.astype(np.int8)

    return features


FEATURE_GROUPS = [
    {
        "name": "rest_frame_filter_landmarks",
        "fn": add_rest_frame_filter_landmarks,
        "depends_on": [],
        "description": "Rest-frame passband coverage and spectral landmark alignment features from ugriz photometry and redshift.",
    }
]