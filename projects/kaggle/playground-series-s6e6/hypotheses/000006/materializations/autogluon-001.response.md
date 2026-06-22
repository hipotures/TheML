import numpy as np
import pandas as pd


def add_bandpass_break_localization(raw, deps, aux):
    band_cols = ["u", "g", "r", "i", "z"]
    magnitudes = raw[band_cols].to_numpy(dtype=float)
    lflux = -0.4 * magnitudes

    d_ug = lflux[:, 0] - lflux[:, 1]
    d_gr = lflux[:, 1] - lflux[:, 2]
    d_ri = lflux[:, 2] - lflux[:, 3]
    d_iz = lflux[:, 3] - lflux[:, 4]

    diffs = np.column_stack([d_ug, d_gr, d_ri, d_iz])
    abs_diffs = np.abs(diffs)

    dominant_idx = np.argmax(abs_diffs, axis=1)
    rows = np.arange(len(raw))

    d_k = diffs[rows, dominant_idx]
    abs_d_k = np.abs(d_k)

    blue_side_means = np.empty((len(raw), 4), dtype=float)
    blue_side_means[:, 0] = 0.0
    blue_side_means[:, 1] = d_ug
    blue_side_means[:, 2] = (d_ug + d_gr) / 2.0
    blue_side_means[:, 3] = (d_ug + d_gr + d_ri) / 3.0
    blue_mean = blue_side_means[rows, dominant_idx]

    red_side_means = np.empty((len(raw), 4), dtype=float)
    red_side_means[:, 0] = (d_gr + d_ri + d_iz) / 3.0
    red_side_means[:, 1] = (d_ri + d_iz) / 2.0
    red_side_means[:, 2] = d_iz
    red_side_means[:, 3] = 0.0
    red_mean = red_side_means[rows, dominant_idx]

    local_contrast = d_k - (blue_mean + red_mean) / 2.0
    sharpness_share = abs_d_k / (np.sum(abs_diffs, axis=1) + 1e-8)

    turnover = np.zeros(len(raw), dtype=np.uint8)
    has_both_sides = (dominant_idx > 0) & (dominant_idx < 3)
    left_diff = diffs[rows[has_both_sides], dominant_idx[has_both_sides] - 1]
    right_diff = diffs[rows[has_both_sides], dominant_idx[has_both_sides]]
    turnover[has_both_sides] = (np.sign(left_diff) * np.sign(right_diff) < 0).astype(np.uint8)

    features = pd.DataFrame(
        {
            "bandpass_break_localization_dominant_index": dominant_idx.astype(np.int8),
            "bandpass_break_localization_dominant_diff": d_k,
            "bandpass_break_localization_dominant_abs_diff": abs_d_k,
            "bandpass_break_localization_blue_mean_diff": blue_mean,
            "bandpass_break_localization_red_mean_diff": red_mean,
            "bandpass_break_localization_local_contrast": local_contrast,
            "bandpass_break_localization_sharpness_share": sharpness_share,
            "bandpass_break_localization_turnover_across_break": turnover,
            "bandpass_break_localization_break_ug": (dominant_idx == 0).astype(np.uint8),
            "bandpass_break_localization_break_gr": (dominant_idx == 1).astype(np.uint8),
            "bandpass_break_localization_break_ri": (dominant_idx == 2).astype(np.uint8),
            "bandpass_break_localization_break_iz": (dominant_idx == 3).astype(np.uint8),
        },
        index=raw.index,
    )

    return features


FEATURE_GROUPS = [
    {
        "name": "bandpass_break_localization",
        "fn": add_bandpass_break_localization,
        "depends_on": [],
        "description": "Localizes the dominant ugriz broadband break and derives its amplitude, position, side-mean context, sharpness, and turnover behavior.",
    }
]