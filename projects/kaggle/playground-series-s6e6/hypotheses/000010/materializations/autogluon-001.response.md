import numpy as np
import pandas as pd

_BAND_COLUMNS = ("u", "g", "r", "i", "z")
_BAND_WAVELENGTHS_A = (3543.0, 4770.0, 6231.0, 7625.0, 9134.0)
_LANDMARKS = (("1216", 1216.0), ("2500", 2500.0), ("4000", 4000.0))


def add_rest_frame_filter_landmarks(raw, deps, aux):
    band_wl = np.asarray(_BAND_WAVELENGTHS_A, dtype=float)
    redshift = raw["redshift"].to_numpy(dtype=float)
    z_safe = np.maximum(redshift, 0.0)

    rest_wl = band_wl[None, :] / (1.0 + z_safe)[:, None]
    mags = raw.loc[:, list(_BAND_COLUMNS)].to_numpy(dtype=float)
    n_rows = raw.shape[0]
    row_index = np.arange(n_rows)

    features = {}
    features["neg_redshift_flag"] = (redshift < 0.0).astype(np.int8)

    features["rest_band_count_lt_1216"] = (rest_wl < 1216.0).sum(axis=1).astype(np.int16)
    features["rest_band_count_lt_2500"] = (rest_wl < 2500.0).sum(axis=1).astype(np.int16)
    features["rest_band_count_lt_4000"] = (rest_wl < 4000.0).sum(axis=1).astype(np.int16)
    features["rest_band_count_4000_7000"] = ((rest_wl >= 4000.0) & (rest_wl <= 7000.0)).sum(axis=1).astype(np.int16)
    features["rest_band_count_gt_7000"] = (rest_wl > 7000.0).sum(axis=1).astype(np.int16)

    num_bands = len(_BAND_COLUMNS)

    for landmark_name, landmark in _LANDMARKS:
        delta = rest_wl - landmark
        signed_log_dist = np.sign(delta) * np.log1p(np.abs(delta))
        for i, band in enumerate(_BAND_COLUMNS):
            features[f"landmark_{landmark_name}_signed_logdist_{band}"] = signed_log_dist[:, i]

        lte = (rest_wl <= landmark).sum(axis=1)
        nearest_left = np.where(
            lte <= 0,
            0,
            np.where(lte >= num_bands, num_bands - 1, lte - 1),
        )
        nearest_right = np.where(
            lte <= 0,
            0,
            np.where(lte >= num_bands, num_bands - 1, lte),
        )

        left_dist = np.abs(rest_wl[row_index, nearest_left] - landmark)
        right_dist = np.abs(rest_wl[row_index, nearest_right] - landmark)
        nearest_idx = np.where(left_dist <= right_dist, nearest_left, nearest_right).astype(np.int8)

        features[f"landmark_{landmark_name}_nearest_band_idx"] = nearest_idx
        features[f"landmark_{landmark_name}_nearest_mag"] = mags[row_index, nearest_idx]

        inside = (landmark >= rest_wl[:, 0]) & (landmark <= rest_wl[:, -1])
        features[f"landmark_{landmark_name}_inside_span"] = inside.astype(np.int8)

        lower = (rest_wl < landmark).sum(axis=1) - 1
        upper = lower + 1
        not_on_landmark_band = ~(rest_wl == landmark).any(axis=1)
        has_adjacent = (
            (lower >= 0)
            & (upper <= (num_bands - 1))
            & (landmark > rest_wl[:, 0])
            & (landmark < rest_wl[:, -1])
            & not_on_landmark_band
        )

        contrast = np.full(n_rows, np.nan, dtype=float)
        if has_adjacent.any():
            contrast[has_adjacent] = (
                mags[row_index[has_adjacent], upper[has_adjacent]]
                - mags[row_index[has_adjacent], lower[has_adjacent]]
            )
        features[f"landmark_{landmark_name}_adjacent_color_contrast"] = contrast

    return pd.DataFrame(features, index=raw.index)


FEATURE_GROUPS = [
    {
        "name": "rest_frame_filter_landmarks",
        "fn": add_rest_frame_filter_landmarks,
        "depends_on": [],
        "description": "Create redshift-aware rest-frame filter alignment features by comparing ugriz rest wavelengths to astrophysical landmarks.",
    }
]