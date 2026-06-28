import numpy as np
import pandas as pd


BANDS = ("u", "g", "r", "i", "z")
ADJACENT_COLOR_PAIRS = (("u", "g"), ("g", "r"), ("r", "i"), ("i", "z"))
ALL_COLOR_PAIRS = (
    ("u", "g"),
    ("u", "r"),
    ("u", "i"),
    ("u", "z"),
    ("g", "r"),
    ("g", "i"),
    ("g", "z"),
    ("r", "i"),
    ("r", "z"),
    ("i", "z"),
)
PHOTOMETRIC_BAND_NOISE_MAG = 0.03
MARGIN_CLIP = 30.0


def add_noise_scale_color_decisiveness(raw, deps, aux):
    sigma_color = np.sqrt(2.0) * PHOTOMETRIC_BAND_NOISE_MAG
    features = {}

    adjacent_margins = []
    all_margins = []

    for left, right in ALL_COLOR_PAIRS:
        color = raw[left].to_numpy(dtype="float64", copy=False) - raw[right].to_numpy(dtype="float64", copy=False)
        margin = np.clip(color / sigma_color, -MARGIN_CLIP, MARGIN_CLIP)
        all_margins.append(margin)
        features[f"margin_{left}_minus_{right}_sigma"] = margin.astype("float32", copy=False)

    for left, right in ADJACENT_COLOR_PAIRS:
        color = raw[left].to_numpy(dtype="float64", copy=False) - raw[right].to_numpy(dtype="float64", copy=False)
        margin = np.clip(color / sigma_color, -MARGIN_CLIP, MARGIN_CLIP)
        adjacent_margins.append(margin)

    adjacent = np.column_stack(adjacent_margins)
    all_pairs = np.column_stack(all_margins)

    abs_adjacent = np.abs(adjacent)
    abs_all = np.abs(all_pairs)

    for prefix, margins, abs_margins in (
        ("adjacent", adjacent, abs_adjacent),
        ("all_pair", all_pairs, abs_all),
    ):
        features[f"{prefix}_ambiguous_count_abs_le_1"] = (abs_margins <= 1.0).sum(axis=1).astype("int8")
        features[f"{prefix}_moderate_positive_count_1_to_3"] = ((margins > 1.0) & (margins <= 3.0)).sum(axis=1).astype("int8")
        features[f"{prefix}_moderate_negative_count_1_to_3"] = ((margins < -1.0) & (margins >= -3.0)).sum(axis=1).astype("int8")
        features[f"{prefix}_strong_positive_count_gt_3"] = (margins > 3.0).sum(axis=1).astype("int8")
        features[f"{prefix}_strong_negative_count_gt_3"] = (margins < -3.0).sum(axis=1).astype("int8")

        features[f"{prefix}_abs_margin_min"] = abs_margins.min(axis=1).astype("float32", copy=False)
        features[f"{prefix}_abs_margin_median"] = np.median(abs_margins, axis=1).astype("float32", copy=False)
        features[f"{prefix}_abs_margin_mean"] = abs_margins.mean(axis=1).astype("float32", copy=False)
        features[f"{prefix}_abs_margin_max"] = abs_margins.max(axis=1).astype("float32", copy=False)

        confidence = np.tanh(abs_margins / 3.0)
        features[f"{prefix}_confidence_mean"] = confidence.mean(axis=1).astype("float32", copy=False)
        features[f"{prefix}_confidence_min"] = confidence.min(axis=1).astype("float32", copy=False)
        features[f"{prefix}_confidence_max"] = confidence.max(axis=1).astype("float32", copy=False)

    robust_signs = np.sign(adjacent)
    robust_signs[abs_adjacent <= 1.0] = 0.0
    features["adjacent_robust_nonzero_sign_count"] = (robust_signs != 0.0).sum(axis=1).astype("int8")
    features["adjacent_robust_sign_change_count"] = (
        (robust_signs[:, 1:] * robust_signs[:, :-1]) < 0.0
    ).sum(axis=1).astype("int8")

    return pd.DataFrame(features, index=raw.index)


FEATURE_GROUPS = [
    {
        "name": "noise_scale_color_decisiveness",
        "fn": add_noise_scale_color_decisiveness,
        "depends_on": [],
        "description": "Noise-normalized ugriz color margins and decisiveness summaries using a fixed SDSS-like photometric precision scale.",
    }
]