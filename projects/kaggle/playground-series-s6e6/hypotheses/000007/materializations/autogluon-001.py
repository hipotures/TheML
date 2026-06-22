import numpy as np
import pandas as pd


def _fill_zero_signs(sign_paths: np.ndarray) -> np.ndarray:
    filled = sign_paths.copy()
    for i in range(filled.shape[0]):
        row = filled[i]
        nonzero_idx = np.flatnonzero(row)
        if nonzero_idx.size == 0:
            continue
        for j in range(row.size):
            if row[j] == 0:
                nearest = nonzero_idx[np.argmin(np.abs(nonzero_idx - j))]
                row[j] = row[nearest]
        filled[i] = row
    return filled


def add_band_order_topology(raw, deps, aux):
    bands = ["u", "g", "r", "i", "z"]
    mags = raw[bands].astype(float)

    ranks_df = mags.rank(axis=1, method="first", ascending=True).astype(np.int8)
    ranks = ranks_df.to_numpy(dtype=np.int8)

    band_to_idx = {"u": 0, "g": 1, "r": 2, "i": 3, "z": 4}
    brightest_idx = ranks_df.idxmin(axis=1).map(band_to_idx).astype(np.int8)
    faintest_idx = ranks_df.idxmax(axis=1).map(band_to_idx).astype(np.int8)

    upper_mask = np.triu(np.ones((5, 5), dtype=bool), k=1)
    inv_total = (ranks[:, :, None] > ranks[:, None, :])[:, upper_mask].sum(axis=1).astype(np.int8)
    inv_vs_increasing = inv_total.copy()
    inv_vs_decreasing = (10 - inv_total).astype(np.int8)

    eps = 0.02
    d1 = mags["g"].to_numpy() - mags["u"].to_numpy()
    d2 = mags["r"].to_numpy() - mags["g"].to_numpy()
    d3 = mags["i"].to_numpy() - mags["r"].to_numpy()
    d4 = mags["z"].to_numpy() - mags["i"].to_numpy()
    diffs = np.column_stack((d1, d2, d3, d4))

    signs = np.zeros_like(diffs, dtype=np.int8)
    signs[diffs > eps] = 1
    signs[diffs < -eps] = -1
    signs = _fill_zero_signs(signs)

    sign_changes_mask = signs[:, :-1] != signs[:, 1:]
    sign_change_count = sign_changes_mask.sum(axis=1).astype(np.int8)
    sign_monotone = (sign_change_count == 0).astype(np.int8)
    sign_one_change = (sign_change_count == 1).astype(np.int8)
    sign_first_change_pos = np.where(
        sign_changes_mask.any(axis=1),
        sign_changes_mask.argmax(axis=1).astype(np.int8) + 1,
        0,
    ).astype(np.int8)

    flat = (mags.max(axis=1) - mags.min(axis=1)) <= eps
    flat_np = flat.to_numpy()

    inv_total = np.where(flat_np, 0, inv_total)
    inv_vs_increasing = np.where(flat_np, 0, inv_vs_increasing)
    inv_vs_decreasing = np.where(flat_np, 0, inv_vs_decreasing)
    sign_change_count = np.where(flat_np, 0, sign_change_count)
    sign_monotone = np.where(flat_np, 1, sign_monotone)
    sign_one_change = np.where(flat_np, 0, sign_one_change)
    sign_first_change_pos = np.where(flat_np, 0, sign_first_change_pos)

    sign_monotone = sign_monotone.astype(np.int8)
    sign_one_change = sign_one_change.astype(np.int8)
    sign_change_count = sign_change_count.astype(np.int8)
    sign_first_change_pos = sign_first_change_pos.astype(np.int8)
    inv_total = inv_total.astype(np.int8)
    inv_vs_increasing = inv_vs_increasing.astype(np.int8)
    inv_vs_decreasing = inv_vs_decreasing.astype(np.int8)

    flat_topology = flat.astype(np.int8)
    bright_interior = brightest_idx.isin([1, 2, 3]).astype(np.int8)
    bright_edge = brightest_idx.isin([0, 4]).astype(np.int8)
    faint_interior = faintest_idx.isin([1, 2, 3]).astype(np.int8)
    faint_edge = faintest_idx.isin([0, 4]).astype(np.int8)

    return pd.DataFrame(
        {
            "rank_u": ranks_df["u"],
            "rank_g": ranks_df["g"],
            "rank_r": ranks_df["r"],
            "rank_i": ranks_df["i"],
            "rank_z": ranks_df["z"],
            "brightest_band_wl_idx": brightest_idx,
            "faintest_band_wl_idx": faintest_idx,
            "flat_topology": flat_topology,
            "inversion_count": inv_total,
            "inversion_count_vs_increasing": inv_vs_increasing,
            "inversion_count_vs_decreasing": inv_vs_decreasing,
            "sign_d1": pd.Series(signs[:, 0], index=raw.index, dtype=np.int8),
            "sign_d2": pd.Series(signs[:, 1], index=raw.index, dtype=np.int8),
            "sign_d3": pd.Series(signs[:, 2], index=raw.index, dtype=np.int8),
            "sign_d4": pd.Series(signs[:, 3], index=raw.index, dtype=np.int8),
            "sign_monotone": sign_monotone,
            "sign_change_count": sign_change_count,
            "sign_exactly_one_change": sign_one_change,
            "sign_first_change_pos": sign_first_change_pos,
            "brightest_band_interior": bright_interior,
            "brightest_band_edge": bright_edge,
            "faintest_band_interior": faint_interior,
            "faintest_band_edge": faint_edge,
        },
        index=raw.index,
    )


FEATURE_GROUPS = [
    {
        "name": "band_order_topology",
        "fn": add_band_order_topology,
        "depends_on": [],
        "description": "Encodes ugriz brightness-order topology via per-row band ranks, inversion statistics, monotone sign-path transitions, and edge/interior bright/faint band positions.",
    }
]