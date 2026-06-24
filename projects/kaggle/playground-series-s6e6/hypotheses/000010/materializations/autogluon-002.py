import numpy as np
import pandas as pd

BASE_WAVELENGTHS = (3543.0, 4770.0, 6231.0, 7625.0, 9134.0)
REST_BIN_EDGES = (912.0, 1216.0, 2500.0, 3646.0, 4000.0, 7000.0)
LANDMARKS = (912.0, 1216.0, 2500.0, 3646.0, 4000.0, 7000.0)
MAG_BANDS = ("u", "g", "r", "i", "z")


def add_rest_frame_filter_landmarks(raw, deps, aux):
    redshift = raw["redshift"].to_numpy(dtype=float)
    z_safe = np.maximum(redshift, 0.0)
    z_neg_flag = redshift < 0.0

    mags = raw[list(MAG_BANDS)].to_numpy(dtype=float)
    obs_waves = np.array(BASE_WAVELENGTHS, dtype=float)
    rest_waves = obs_waves / (1.0 + z_safe[:, None])

    n_rows = raw.shape[0]
    row_idx = np.arange(n_rows)

    span_min = np.min(rest_waves, axis=1)
    span_max = np.max(rest_waves, axis=1)
    span_width = span_max - span_min

    bin_edges = np.array(REST_BIN_EDGES, dtype=float)
    bin_idx = np.digitize(rest_waves, bin_edges, right=False)
    counts = np.zeros((n_rows, bin_edges.size + 1), dtype=np.int16)
    np.add.at(counts, (np.repeat(row_idx, rest_waves.shape[1]), bin_idx.ravel()), 1)

    probs = counts.astype(float) / float(obs_waves.shape[0])
    has_count = counts > 0
    entropy = -np.sum(np.where(has_count, probs * np.log(probs), 0.0), axis=1)
    nonzero_bins = has_count.sum(axis=1).astype(float)
    rest_bin_entropy = np.divide(
        entropy,
        np.log(nonzero_bins),
        out=np.zeros_like(entropy),
        where=nonzero_bins > 1.0,
    )

    features = {
        "z_safe": z_safe,
        "z_neg_flag": z_neg_flag.astype(np.int8),
        "rest_wave_u": rest_waves[:, 0],
        "rest_wave_g": rest_waves[:, 1],
        "rest_wave_r": rest_waves[:, 2],
        "rest_wave_i": rest_waves[:, 3],
        "rest_wave_z": rest_waves[:, 4],
        "span_min": span_min,
        "span_max": span_max,
        "span_width": span_width,
        "n_lt_912": counts[:, 0],
        "n_912_1216": counts[:, 1],
        "n_1216_2500": counts[:, 2],
        "n_2500_3646": counts[:, 3],
        "n_3646_4000": counts[:, 4],
        "n_4000_7000": counts[:, 5],
        "n_gt_7000": counts[:, 6],
        "total_bands": np.full(n_rows, 5, dtype=np.int8),
        "rest_bin_entropy": rest_bin_entropy,
    }

    for landmark in LANDMARKS:
        name = f"{int(landmark)}"
        nearest_idx = np.argmin(np.abs(rest_waves - landmark), axis=1)
        selected_wave = rest_waves[row_idx, nearest_idx]
        selected_mag = mags[row_idx, nearest_idx]
        d = selected_wave - landmark

        in_span = (span_min <= landmark) & (span_max >= landmark)
        below_span = landmark < span_min
        above_span = landmark > span_max

        has_left = nearest_idx > 0
        has_right = nearest_idx < (mags.shape[1] - 1)

        left_wave = np.where(has_left, rest_waves[row_idx, nearest_idx - 1], np.nan)
        right_wave = np.where(has_right, rest_waves[row_idx, nearest_idx + 1], np.nan)
        left_mag = np.where(has_left, mags[row_idx, nearest_idx - 1], np.nan)
        right_mag = np.where(has_right, mags[row_idx, nearest_idx + 1], np.nan)

        interior_straddle = has_left & has_right & (left_wave <= landmark) & (right_wave >= landmark)

        left_delta = np.full(n_rows, np.nan, dtype=float)
        right_delta = np.full(n_rows, np.nan, dtype=float)

        left_delta[interior_straddle] = selected_mag[interior_straddle] - left_mag[interior_straddle]
        right_delta[interior_straddle] = right_mag[interior_straddle] - selected_mag[interior_straddle]

        idx_last = nearest_idx == (mags.shape[1] - 1)
        idx_first = nearest_idx == 0
        left_delta[idx_last] = selected_mag[idx_last] - left_mag[idx_last]
        right_delta[idx_first] = right_mag[idx_first] - selected_mag[idx_first]

        residual = np.zeros(n_rows, dtype=float)

        upper_ok = (landmark > selected_wave) & (nearest_idx < (mags.shape[1] - 1))
        upper_mask = upper_ok & (landmark < right_wave)
        if np.any(upper_mask):
            pred_upper = selected_mag + (right_mag - selected_mag) * (landmark - selected_wave) / (right_wave - selected_wave)
            residual[upper_mask] = selected_mag[upper_mask] - pred_upper[upper_mask]

        lower_ok = (landmark < selected_wave) & (nearest_idx > 0)
        lower_mask = lower_ok & (landmark > left_wave)
        if np.any(lower_mask):
            pred_lower = left_mag + (selected_mag - left_mag) * (landmark - left_wave) / (selected_wave - left_wave)
            residual[lower_mask] = selected_mag[lower_mask] - pred_lower[lower_mask]

        residual = np.where(in_span, residual, 0.0)

        outside_distance = np.where(
            in_span,
            0.0,
            np.log1p(np.abs(landmark - np.where(below_span, span_min, span_max))),
        )
        outside_side = np.where(in_span, 0, np.where(below_span, 0, 1)).astype(np.int8)
        outside_endpoint_band = np.where(in_span, 0, np.where(below_span, 1, 5)).astype(np.int8)

        features[f"landmark_{name}_in_span"] = in_span.astype(np.int8)
        features[f"landmark_{name}_below_span"] = below_span.astype(np.int8)
        features[f"landmark_{name}_above_span"] = above_span.astype(np.int8)
        features[f"landmark_{name}_nearest_band_id"] = nearest_idx + 1
        features[f"landmark_{name}_signed_log_dist"] = np.sign(d) * np.log1p(np.abs(d))
        features[f"landmark_{name}_abs_log_dist"] = np.log1p(np.abs(d))
        features[f"landmark_{name}_nearest_band_mag"] = selected_mag
        features[f"landmark_{name}_left_delta"] = left_delta
        features[f"landmark_{name}_right_delta"] = right_delta
        features[f"landmark_{name}_residual"] = residual
        features[f"landmark_{name}_outside_distance"] = outside_distance
        features[f"landmark_{name}_outside_side"] = outside_side
        features[f"landmark_{name}_outside_endpoint_band"] = outside_endpoint_band

    return pd.DataFrame(features, index=raw.index)


FEATURE_GROUPS = [
    {
        "name": "rest_frame_filter_landmarks",
        "fn": add_rest_frame_filter_landmarks,
        "depends_on": [],
        "description": "Construct rest-frame band-location occupancy and landmark-alignment descriptors from ugriz magnitudes and redshift-aware wavelength placement.",
    },
]