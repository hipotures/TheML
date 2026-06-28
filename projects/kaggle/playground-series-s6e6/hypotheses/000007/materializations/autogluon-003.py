import numpy as np
import pandas as pd

BAND_ORDER = ("u", "g", "r", "i", "z")
BAND_COUNT = 5
PAIR_COUNT = 10
ADJACENT_COUNT = 4
TOPOLOGY_TOLERANCE = 0.02


def _fill_zero_signs(signs):
    filled = signs.copy()
    n_rows, n_cols = filled.shape

    for row_idx in range(n_rows):
        row = filled[row_idx]
        zero_positions = np.flatnonzero(row == 0)

        if zero_positions.size == 0:
            continue

        pos = 0
        while pos < n_cols:
            if row[pos] != 0:
                pos += 1
                continue

            start = pos
            while pos < n_cols and row[pos] == 0:
                pos += 1
            end = pos

            left = row[start - 1] if start > 0 else 0
            right = row[end] if end < n_cols else 0

            if left != 0 and right != 0 and left == right:
                row[start:end] = left
            elif left != 0 and right == 0:
                row[start:end] = left
            elif left == 0 and right != 0:
                row[start:end] = right

    return filled


def add_band_order_topology(raw, deps, aux):
    index = raw.index
    tau = TOPOLOGY_TOLERANCE

    mags = raw.loc[:, BAND_ORDER].to_numpy(dtype=float, copy=True)
    n_rows = mags.shape[0]

    delta = np.nanmax(mags, axis=1) - np.nanmin(mags, axis=1)
    flat = delta <= (2.0 * tau)

    tied_pairs = np.zeros(n_rows, dtype=np.int16)
    non_tied_pairs = np.zeros(n_rows, dtype=np.int16)
    inversion_forward = np.zeros(n_rows, dtype=np.float64)
    inversion_reverse = np.zeros(n_rows, dtype=np.float64)

    for left in range(BAND_COUNT - 1):
        left_values = mags[:, left]
        for right in range(left + 1, BAND_COUNT):
            diff = left_values - mags[:, right]
            tied = np.abs(diff) <= tau
            tied_pairs += tied.astype(np.int16)
            non_tied = ~tied
            non_tied_pairs += non_tied.astype(np.int16)

            forward_inv = non_tied & (diff > tau)
            reverse_inv = non_tied & (diff < -tau)
            inversion_forward += forward_inv.astype(np.float64)
            inversion_reverse += reverse_inv.astype(np.float64)

    denom = np.where(non_tied_pairs > 0, non_tied_pairs, 1)
    inversion_forward_norm = inversion_forward / denom
    inversion_reverse_norm = inversion_reverse / denom

    sort_order = np.argsort(mags, axis=1, kind="stable")
    sorted_mags = np.take_along_axis(mags, sort_order, axis=1)
    sorted_gaps = np.diff(sorted_mags, axis=1)
    new_block_after = sorted_gaps > tau

    rank_blocks_sorted = np.zeros((n_rows, BAND_COUNT), dtype=np.int16)
    rank_blocks_sorted[:, 1:] = np.cumsum(new_block_after, axis=1) + 1

    rank_block = np.empty((n_rows, BAND_COUNT), dtype=np.int16)
    row_ids = np.arange(n_rows)[:, None]
    rank_block[row_ids, sort_order] = rank_blocks_sorted

    num_rank_blocks = np.max(rank_block, axis=1) + 1

    max_tie_block_size = np.ones(n_rows, dtype=np.int16)
    for block_id in range(BAND_COUNT):
        block_size = np.sum(rank_block == block_id, axis=1).astype(np.int16)
        max_tie_block_size = np.maximum(max_tie_block_size, block_size)

    brightest_block_position = rank_block[:, 0]
    for band_idx in range(1, BAND_COUNT):
        brighter = mags[:, band_idx] < (mags[np.arange(n_rows), np.argmin(mags, axis=1)] - tau)
        brightest_block_position = np.where(brighter, rank_block[:, band_idx], brightest_block_position)

    faintest_block_position = rank_block[:, 0]
    for band_idx in range(1, BAND_COUNT):
        fainter = mags[:, band_idx] > (mags[np.arange(n_rows), np.argmax(mags, axis=1)] + tau)
        faintest_block_position = np.where(fainter, rank_block[:, band_idx], faintest_block_position)

    brightest_block_position = np.min(rank_block, axis=1)
    faintest_block_position = np.max(rank_block, axis=1)

    brightest_bands = rank_block == brightest_block_position[:, None]
    faintest_bands = rank_block == faintest_block_position[:, None]

    brightest_is_edge = brightest_bands[:, 0] | brightest_bands[:, 4]
    faintest_is_edge = faintest_bands[:, 0] | faintest_bands[:, 4]
    brightest_is_interior = brightest_bands[:, 1] | brightest_bands[:, 2] | brightest_bands[:, 3]
    faintest_is_interior = faintest_bands[:, 1] | faintest_bands[:, 2] | faintest_bands[:, 3]

    slopes = np.diff(mags, axis=1)
    raw_signs = np.zeros((n_rows, ADJACENT_COUNT), dtype=np.int8)
    raw_signs[slopes > tau] = 1
    raw_signs[slopes < -tau] = -1

    filled_signs = _fill_zero_signs(raw_signs)
    ambiguous_slope_count = np.sum(filled_signs == 0, axis=1).astype(np.int16)

    sign_changes = np.zeros(n_rows, dtype=np.int16)
    first_change = np.zeros(n_rows, dtype=np.int16)
    second_change = np.zeros(n_rows, dtype=np.int16)

    for boundary in range(ADJACENT_COUNT - 1):
        left = filled_signs[:, boundary]
        right = filled_signs[:, boundary + 1]
        changed = (left != 0) & (right != 0) & (left != right)
        next_count = sign_changes + changed.astype(np.int16)
        boundary_position = boundary + 1

        first_change = np.where((sign_changes == 0) & changed, boundary_position, first_change)
        second_change = np.where((sign_changes == 1) & changed, boundary_position, second_change)
        sign_changes = next_count

    nonzero_count = np.sum(filled_signs != 0, axis=1)
    monotone = ((ambiguous_slope_count == 0) & (sign_changes == 0) & (nonzero_count > 0)).astype(np.int8)
    one_turn = (sign_changes == 1).astype(np.int8)

    turn_type = np.full(n_rows, "none", dtype=object)
    for boundary in range(ADJACENT_COUNT - 1):
        boundary_position = boundary + 1
        at_first = first_change == boundary_position
        left = filled_signs[:, boundary]
        right = filled_signs[:, boundary + 1]
        turn_type = np.where(at_first & (left < 0) & (right > 0), "peak_like", turn_type)
        turn_type = np.where(at_first & (left > 0) & (right < 0), "valley_like", turn_type)

    topology_bucket = np.full(n_rows, "ambiguous", dtype=object)
    topology_bucket = np.where(monotone == 1, "strict_monotone", topology_bucket)
    topology_bucket = np.where(sign_changes == 1, "single_turn", topology_bucket)
    topology_bucket = np.where(sign_changes > 1, "multi_turn", topology_bucket)
    topology_bucket = np.where(flat, "flat", topology_bucket)

    output = pd.DataFrame(
        {
            "flat_topology": flat.astype(np.int8),
            "brightness_delta": delta,
            "brightest_block_position": brightest_block_position.astype(np.int16),
            "faintest_block_position": faintest_block_position.astype(np.int16),
            "brightest_is_edge": brightest_is_edge.astype(np.int8),
            "faintest_is_edge": faintest_is_edge.astype(np.int8),
            "brightest_is_interior": brightest_is_interior.astype(np.int8),
            "faintest_is_interior": faintest_is_interior.astype(np.int8),
            "num_rank_blocks": num_rank_blocks.astype(np.int16),
            "max_tie_block_size": max_tie_block_size.astype(np.int16),
            "tie_fraction": tied_pairs.astype(np.float64) / PAIR_COUNT,
            "inversion_forward_norm": inversion_forward_norm,
            "inversion_reverse_norm": inversion_reverse_norm,
            "monotone": monotone,
            "sign_changes": sign_changes,
            "one_turn": one_turn,
            "first_change": first_change,
            "second_change": second_change,
            "ambiguous_slope_count": ambiguous_slope_count,
            "single_turn_type": turn_type,
            "topology_bucket": topology_bucket,
        },
        index=index,
    )

    neutral_mask = flat.to_numpy() if hasattr(flat, "to_numpy") else flat
    neutral_columns = [
        "brightest_block_position",
        "faintest_block_position",
        "brightest_is_edge",
        "faintest_is_edge",
        "brightest_is_interior",
        "faintest_is_interior",
        "num_rank_blocks",
        "max_tie_block_size",
        "inversion_forward_norm",
        "inversion_reverse_norm",
        "sign_changes",
        "one_turn",
        "first_change",
        "second_change",
        "ambiguous_slope_count",
    ]
    output.loc[neutral_mask, neutral_columns] = 0
    output.loc[neutral_mask, "tie_fraction"] = 1.0
    output.loc[neutral_mask, "monotone"] = 1
    output.loc[neutral_mask, "single_turn_type"] = "none"
    output.loc[neutral_mask, "topology_bucket"] = "flat"

    return output


FEATURE_GROUPS = [
    {
        "name": "band_order_topology",
        "fn": add_band_order_topology,
        "depends_on": [],
        "description": "Encodes tolerance-aware ugriz brightness ordering, tie structure, inversions, and adjacent slope topology.",
    }
]