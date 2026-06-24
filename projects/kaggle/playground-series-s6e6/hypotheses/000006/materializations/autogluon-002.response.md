import numpy as np
import pandas as pd


EPS = 1e-8
FLAT_THRESHOLD = 1e-6
INDEX_WEIGHT = np.array([1.0, 2.0, 3.0, 4.0], dtype=float)


def _frequency_bins_3(values: pd.Series) -> pd.Series:
    """
    Equal-mass low/medium/high bins via rank percentiles.
    Uses only training/test covariate values passed in `values`.
    """
    ranks = values.rank(method="average", pct=True)
    bins = np.where(ranks <= 1.0 / 3.0, "low", np.where(ranks <= 2.0 / 3.0, "medium", "high"))
    return pd.Series(bins, index=values.index, dtype="object")


def add_bandpass_break_localization(raw, deps, aux):
    # u,g,r,i,z bands only
    lu = -0.4 * raw["u"].astype(float)
    lg = -0.4 * raw["g"].astype(float)
    lr = -0.4 * raw["r"].astype(float)
    li = -0.4 * raw["i"].astype(float)
    lz = -0.4 * raw["z"].astype(float)

    d_ug = lu - lg
    d_gr = lg - lr
    d_ri = lr - li
    d_iz = li - lz

    a_ug = d_ug.abs()
    a_gr = d_gr.abs()
    a_ri = d_ri.abs()
    a_iz = d_iz.abs()

    A = a_ug + a_gr + a_ri + a_iz
    flat_mask = A < FLAT_THRESHOLD

    a_matrix = np.column_stack(
        [a_ug.to_numpy(), a_gr.to_numpy(), a_ri.to_numpy(), a_iz.to_numpy()]
    )
    d_matrix = np.column_stack(
        [d_ug.to_numpy(), d_gr.to_numpy(), d_ri.to_numpy(), d_iz.to_numpy()]
    )

    k_idx = np.argmax(a_matrix, axis=1)
    k = k_idx + 1  # 1..4 in ug,gr,ri,iz order (tie broken by smallest j)
    p_matrix = a_matrix / (A.to_numpy()[:, None] + EPS)
    p_matrix = np.where(np.isfinite(p_matrix), p_matrix, 0.0)
    entropy = -(p_matrix * np.log(np.maximum(p_matrix, EPS))).sum(axis=1)
    k_soft = (p_matrix * INDEX_WEIGHT[None, :]).sum(axis=1)

    d_k = d_matrix[np.arange(raw.shape[0]), k_idx]
    s_k = np.sign(d_k)

    a_ug_arr = a_ug.to_numpy()
    a_gr_arr = a_gr.to_numpy()
    a_ri_arr = a_ri.to_numpy()
    a_iz_arr = a_iz.to_numpy()
    d_ug_arr = d_ug.to_numpy()
    d_gr_arr = d_gr.to_numpy()
    d_ri_arr = d_ri.to_numpy()
    d_iz_arr = d_iz.to_numpy()

    blue_mean = np.zeros(raw.shape[0], dtype=float)
    blue_mean = np.where(k == 2, a_ug_arr, blue_mean)
    blue_mean = np.where(k == 3, (a_ug_arr + a_gr_arr) / 2.0, blue_mean)
    blue_mean = np.where(k == 4, (a_ug_arr + a_gr_arr + a_ri_arr) / 3.0, blue_mean)

    red_mean = np.zeros(raw.shape[0], dtype=float)
    red_mean = np.where(k == 1, (a_gr_arr + a_ri_arr + a_iz_arr) / 3.0, red_mean)
    red_mean = np.where(k == 2, (a_ri_arr + a_iz_arr) / 2.0, red_mean)
    red_mean = np.where(k == 3, a_iz_arr, red_mean)

    continuity = d_k - (blue_mean + red_mean) / 2.0
    asymmetry = blue_mean - red_mean
    sharpness = np.abs(d_k) / (A.to_numpy() + EPS)
    sign_prev = np.where(k == 2, d_ug_arr, np.where(k == 3, d_gr_arr, np.where(k == 4, d_ri_arr, 0.0)))
    sign_next = np.where(k == 1, d_gr_arr, np.where(k == 2, d_ri_arr, np.where(k == 3, d_iz_arr, 0.0)))
    turnover_count = ((k > 1) & (np.sign(sign_prev) != s_k)).astype(np.int16) + (
        (k < 4) & (np.sign(sign_next) != s_k)
    ).astype(np.int16)

    sharpness_bin = _frequency_bins_3(pd.Series(sharpness, index=raw.index))
    entropy_bin = _frequency_bins_3(pd.Series(entropy, index=raw.index))

    break_present = (~flat_mask).astype(np.int8)
    break_position = np.where(flat_mask, -1, k).astype(np.int16)
    break_soft_position = np.where(flat_mask, 0.0, k_soft)
    break_is_flat = flat_mask.astype(np.int8)
    break_dominant_delta = np.where(flat_mask, 0.0, d_k)
    break_dominant_abs = np.where(flat_mask, 0.0, np.abs(d_k))
    break_dominant_sign = np.where(
        flat_mask, 0, np.where(s_k > 0, 1, np.where(s_k < 0, -1, 0))
    ).astype(np.int8)
    break_continuity = np.where(flat_mask, 0.0, continuity)
    break_asymmetry = np.where(flat_mask, 0.0, asymmetry)
    break_sharpness = np.where(flat_mask, 0.0, sharpness)
    break_entropy = np.where(flat_mask, 0.0, entropy)
    break_blue_mean = np.where(flat_mask, 0.0, blue_mean)
    break_red_mean = np.where(flat_mask, 0.0, red_mean)
    break_turnover_count = np.where(flat_mask, 0, turnover_count)
    break_soft_ug = np.where(flat_mask, 0.0, p_matrix[:, 0])
    break_soft_gr = np.where(flat_mask, 0.0, p_matrix[:, 1])
    break_soft_ri = np.where(flat_mask, 0.0, p_matrix[:, 2])
    break_soft_iz = np.where(flat_mask, 0.0, p_matrix[:, 3])

    return pd.DataFrame(
        {
            "break_present": break_present,
            "break_position": break_position,
            "break_is_flat": break_is_flat,
            "break_soft_position": break_soft_position,
            "break_dominant_delta": break_dominant_delta,
            "break_dominant_abs": break_dominant_abs,
            "break_dominant_sign": break_dominant_sign,
            "break_continuity": break_continuity,
            "break_asymmetry": break_asymmetry,
            "break_sharpness": break_sharpness,
            "break_entropy": break_entropy,
            "break_turnover_count": break_turnover_count,
            "break_blue_mean": break_blue_mean,
            "break_red_mean": break_red_mean,
            "break_weight_ug": break_soft_ug,
            "break_weight_gr": break_soft_gr,
            "break_weight_ri": break_soft_ri,
            "break_weight_iz": break_soft_iz,
            "break_sharpness_confidence": sharpness_bin,
            "break_entropy_confidence": entropy_bin,
        },
        index=raw.index,
    )


FEATURE_GROUPS = [
    {
        "name": "bandpass_break_localization",
        "fn": add_bandpass_break_localization,
        "depends_on": [],
        "description": "Localizes the strongest broadband break across ugriz and summarizes its strength, position certainty, continuity, asymmetry, and turnover behavior.",
    }
]