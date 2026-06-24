import numpy as np
import pandas as pd

EPSILON = 1e-8
_SOFT_INDEX = (1.0, 2.0, 3.0, 4.0)


def _low_med_high_bins(values, index):
    ranks = pd.Series(values, index=index).rank(pct=True, method="average")
    bins = pd.Series("high", index=index, dtype="object")
    bins[ranks <= (1.0 / 3.0)] = "low"
    bins[(ranks > (1.0 / 3.0)) & (ranks <= (2.0 / 3.0))] = "med"
    return bins


def add_bandpass_break_localization(raw, deps, aux):
    u = raw["u"].to_numpy(dtype=np.float64)
    g = raw["g"].to_numpy(dtype=np.float64)
    r = raw["r"].to_numpy(dtype=np.float64)
    i = raw["i"].to_numpy(dtype=np.float64)
    z = raw["z"].to_numpy(dtype=np.float64)

    d_ug = u - g
    d_gr = g - r
    d_ri = r - i
    d_iz = i - z

    d_stack = np.column_stack((d_ug, d_gr, d_ri, d_iz))
    a_stack = np.abs(d_stack)

    a_ug = a_stack[:, 0]
    a_gr = a_stack[:, 1]
    a_ri = a_stack[:, 2]
    a_iz = a_stack[:, 3]

    A = np.sum(a_stack, axis=1)
    near_flat = A < 1e-6
    denom = A + EPSILON

    p_stack = a_stack / denom[:, None]
    entropy = -np.sum(p_stack * np.log(np.maximum(p_stack, EPSILON)), axis=1)
    position_soft = np.sum(p_stack * np.asarray(_SOFT_INDEX), axis=1)

    dominant = np.argmax(a_stack, axis=1)
    row = np.arange(raw.shape[0])

    d_k = d_stack[row, dominant]
    s_k = np.sign(d_k)
    sharpness = np.abs(d_k) / denom

    cum_a = np.cumsum(a_stack, axis=1)

    blue_sum = np.zeros_like(A, dtype=np.float64)
    has_blue = dominant > 0
    blue_sum[has_blue] = cum_a[row[has_blue], dominant[has_blue] - 1]
    blue_count = dominant.astype(np.float64)
    blue_mean = np.where(has_blue, blue_sum / blue_count, 0.0)

    red_sum = A - cum_a[row, dominant]
    red_count = (3.0 - dominant.astype(np.float64))
    red_mean = np.where(red_count > 0, red_sum / red_count, 0.0)

    continuity = d_k - (blue_mean + red_mean) / 2.0
    asymmetry = blue_mean - red_mean

    left_sign = np.zeros_like(s_k)
    right_sign = np.zeros_like(s_k)

    has_left = dominant > 0
    has_right = dominant < 3

    left_sign[has_left] = np.sign(d_stack[row[has_left], dominant[has_left] - 1])
    right_sign[has_right] = np.sign(d_stack[row[has_right], dominant[has_right] + 1])

    turnover = (has_left & (left_sign != s_k)).astype(np.int8) + (has_right & (right_sign != s_k)).astype(np.int8)

    p_stack = np.where(near_flat[:, None], 0.0, p_stack)
    a_stack = np.where(near_flat[:, None], 0.0, a_stack)
    d_stack = np.where(near_flat[:, None], 0.0, d_stack)
    entropy = np.where(near_flat, 0.0, entropy)
    sharpness = np.where(near_flat, 0.0, sharpness)
    continuity = np.where(near_flat, 0.0, continuity)
    asymmetry = np.where(near_flat, 0.0, asymmetry)
    turnover = np.where(near_flat, 0, turnover)
    d_k = np.where(near_flat, 0.0, d_k)
    s_k = np.where(near_flat, 0.0, s_k)
    blue_mean = np.where(near_flat, 0.0, blue_mean)
    red_mean = np.where(near_flat, 0.0, red_mean)

    break_present = (~near_flat).astype(np.int8)
    break_is_flat = near_flat.astype(np.int8)
    break_position = np.where(near_flat, -1, dominant + 1).astype(np.int64)
    break_position_soft = np.where(near_flat, -1.0, position_soft)
    break_jump = d_k
    break_sign = s_k.astype(np.int8)

    sharpness_bin = _low_med_high_bins(sharpness, raw.index).astype("object")
    entropy_bin = _low_med_high_bins(entropy, raw.index).astype("object")
    sharpness_bin[near_flat] = "low"
    entropy_bin[near_flat] = "low"

    new_features = pd.DataFrame(
        {
            "break_present": break_present,
            "break_is_flat": break_is_flat,
            "break_position": break_position,
            "break_position_soft": break_position_soft,
            "break_jump": break_jump,
            "break_sign": break_sign,
            "break_strength": np.abs(break_jump),
            "break_sharpness": sharpness,
            "break_entropy": entropy,
            "break_blue_mean": blue_mean,
            "break_red_mean": red_mean,
            "break_asymmetry": asymmetry,
            "break_continuity": continuity,
            "break_turnover_count": turnover,
            "break_d_ug": d_stack[:, 0],
            "break_d_gr": d_stack[:, 1],
            "break_d_ri": d_stack[:, 2],
            "break_d_iz": d_stack[:, 3],
            "break_a_ug": a_stack[:, 0],
            "break_a_gr": a_stack[:, 1],
            "break_a_ri": a_stack[:, 2],
            "break_a_iz": a_stack[:, 3],
            "break_p_ug": p_stack[:, 0],
            "break_p_gr": p_stack[:, 1],
            "break_p_ri": p_stack[:, 2],
            "break_p_iz": p_stack[:, 3],
            "break_sharpness_bin": sharpness_bin,
            "break_entropy_bin": entropy_bin,
        },
        index=raw.index,
    )
    return new_features


FEATURE_GROUPS = [
    {
        "name": "bandpass_break_localization",
        "fn": add_bandpass_break_localization,
        "depends_on": [],
        "description": "Builds dominant-band break localization and confidence-aware soft-assignment descriptors from ugriz color discontinuities.",
    }
]