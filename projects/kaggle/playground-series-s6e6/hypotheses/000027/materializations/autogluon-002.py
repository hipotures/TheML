import numpy as np
import pandas as pd

_BLOCK_SIZES = (1000, 100000)
_MOD_BASES = (2, 11, 97)


def add_aide_id_sequence_scan_context(raw, deps, aux):
    n_rows = len(raw)
    row_index = raw.index

    if "id" in raw.columns:
        id_values = pd.to_numeric(raw["id"], errors="coerce")
        is_missing_id = id_values.isna().astype("bool")
        sequence_values = id_values.astype(float).to_numpy(copy=True)

        if is_missing_id.any():
            fallback_row_order = np.arange(n_rows, dtype=float)
            missing_positions = is_missing_id.to_numpy()
            sequence_values[missing_positions] = fallback_row_order[missing_positions]
    else:
        is_missing_id = pd.Series(True, index=row_index, dtype="bool")
        sequence_values = np.arange(n_rows, dtype=float)

    sort_order = np.argsort(sequence_values, kind="mergesort")

    rank_values = np.empty(n_rows, dtype=float)
    rank_values[sort_order] = np.arange(1.0, float(n_rows) + 1.0)

    rank_norm_values = (rank_values - 1.0) / max(n_rows - 1, 1)
    floor_id = np.floor(sequence_values).astype(np.int64)

    prev_gap_values = np.full(n_rows, np.nan, dtype=float)
    next_gap_values = np.full(n_rows, np.nan, dtype=float)
    if n_rows > 1:
        sorted_sequence = sequence_values[sort_order]
        prev_gap_values[sort_order[1:]] = np.diff(sorted_sequence)
        next_gap_values[sort_order[:-1]] = np.diff(sorted_sequence)

    block_size_1k, block_size_100k = _BLOCK_SIZES
    mod_2, mod_11, mod_97 = _MOD_BASES

    features = {
        "aide_id_sequence_missing_id": is_missing_id.astype(bool),
        "aide_id_sequence_rank": rank_values,
        "aide_id_sequence_rank_norm": rank_norm_values,
        "aide_id_sequence_block_1k": floor_id // block_size_1k,
        "aide_id_sequence_block_100k": floor_id // block_size_100k,
        "aide_id_sequence_mod_2": floor_id % mod_2,
        "aide_id_sequence_mod_11": floor_id % mod_11,
        "aide_id_sequence_mod_97": floor_id % mod_97,
        "aide_id_sequence_is_even": (floor_id % 2) == 0,
        "aide_id_sequence_log1p_id": np.log1p(np.clip(sequence_values, a_min=0.0, a_max=None)),
        "aide_id_sequence_prev_gap": prev_gap_values,
        "aide_id_sequence_next_gap": next_gap_values,
    }

    return pd.DataFrame(
        {name: pd.Series(values, index=row_index) for name, values in features.items()},
        index=row_index,
    )


FEATURE_GROUPS = [
    {
        "name": "aide_id_sequence_scan_context",
        "fn": add_aide_id_sequence_scan_context,
        "depends_on": [],
        "description": "Extracts deterministic id-order context features including rank, normalized rank, coarse bins, parity, modulo residues, and neighborhood id gaps.",
    }
]