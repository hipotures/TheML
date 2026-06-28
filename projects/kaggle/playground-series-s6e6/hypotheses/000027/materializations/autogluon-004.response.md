import numpy as np
import pandas as pd


_ID_MODULI = (2, 11, 97)
_BLOCK_SIZES = (1000, 10000, 100000)


def _as_bool_split(series, positive_values):
    lowered = series.astype(str).str.lower()
    return lowered.isin(positive_values).to_numpy(dtype=bool)


def _infer_train_mask(raw, aux, id_values, id_valid):
    n_rows = len(raw)

    if isinstance(aux, pd.DataFrame) and len(aux) == n_rows:
        for col in ("is_train", "_is_train", "train", "in_train"):
            if col in aux.columns:
                return aux[col].fillna(False).astype(bool).to_numpy()

        for col in ("is_test", "_is_test", "test", "in_test"):
            if col in aux.columns:
                return ~aux[col].fillna(False).astype(bool).to_numpy()

        for col in ("split", "dataset", "table", "source", "_split", "_dataset"):
            if col in aux.columns:
                values = aux[col]
                train_mask = _as_bool_split(values, {"train", "training", "tr"})
                test_mask = _as_bool_split(values, {"test", "testing", "te", "holdout", "submission"})
                if train_mask.any() and test_mask.any():
                    return train_mask

    valid_ids = id_values[id_valid]
    if valid_ids.size == 0:
        return np.ones(n_rows, dtype=bool)

    min_id = float(np.nanmin(valid_ids))
    max_id = float(np.nanmax(valid_ids))

    if min_id <= 0.0 and max_id >= 800000.0:
        return id_valid & (id_values <= 577346.0)

    return np.ones(n_rows, dtype=bool)


def add_aide_id_sequence_scan_context(raw, deps, aux):
    index = raw.index
    n_rows = len(raw)

    if "id" in raw.columns:
        id_numeric = pd.to_numeric(raw["id"], errors="coerce")
    else:
        id_numeric = pd.Series(np.nan, index=index, dtype="float64")

    id_arr = id_numeric.to_numpy(dtype="float64", copy=True)
    id_valid = np.isfinite(id_arr)
    id_missing = ~id_valid

    train_mask = _infer_train_mask(raw, aux, id_arr, id_valid)
    if train_mask.shape[0] != n_rows:
        train_mask = np.ones(n_rows, dtype=bool)

    train_valid_mask = train_mask & id_valid
    if train_valid_mask.any():
        train_ids = id_arr[train_valid_mask]
    elif id_valid.any():
        train_ids = id_arr[id_valid]
    else:
        train_ids = np.array([0.0], dtype="float64")

    train_min = float(np.nanmin(train_ids))
    train_max = float(np.nanmax(train_ids))
    train_span = max(1.0, train_max - train_min)

    id_filled = np.where(id_valid, id_arr, train_min)
    id_offset = id_filled - train_min
    id_rel_train = np.clip(id_offset / train_span, -0.25, 2.0)

    log_id = np.log1p(np.maximum(0.0, id_filled))
    train_log = log_id[train_valid_mask] if train_valid_mask.any() else log_id[np.isfinite(log_id)]
    if train_log.size == 0:
        log_mean = 0.0
        log_std = 1.0
    else:
        log_mean = float(np.mean(train_log))
        log_std = max(float(np.std(train_log)), 1e-6)
    log_id_z = (log_id - log_mean) / log_std

    id_rank_global = pd.Series(id_filled, index=index).rank(method="first", ascending=True).to_numpy(dtype="float64") - 1.0
    id_rank_global_norm = id_rank_global / max(1.0, float(n_rows - 1))

    id_rank_within_table = np.zeros(n_rows, dtype="float64")
    id_rank_within_table_norm = np.zeros(n_rows, dtype="float64")
    for mask_value in (False, True):
        mask = train_mask == mask_value
        count = int(mask.sum())
        if count > 0:
            ranks = pd.Series(id_filled[mask]).rank(method="first", ascending=True).to_numpy(dtype="float64") - 1.0
            id_rank_within_table[mask] = ranks
            id_rank_within_table_norm[mask] = ranks / max(1.0, float(count - 1))

    order = np.argsort(id_filled, kind="mergesort")
    sorted_ids = id_filled[order]

    sorted_prev_gap = np.zeros(n_rows, dtype="float64")
    sorted_next_gap = np.zeros(n_rows, dtype="float64")
    if n_rows > 1:
        diffs = np.diff(sorted_ids)
        sorted_prev_gap[1:] = diffs
        sorted_next_gap[:-1] = diffs

    train_order_mask = train_mask[order]
    positive_train_gaps = np.concatenate(
        [
            sorted_prev_gap[(sorted_prev_gap > 0.0) & train_order_mask],
            sorted_next_gap[(sorted_next_gap > 0.0) & train_order_mask],
        ]
    )
    if positive_train_gaps.size > 0:
        gap_cap = max(1.0, float(np.percentile(positive_train_gaps, 99.0)))
    else:
        positive_gaps = np.concatenate([sorted_prev_gap[sorted_prev_gap > 0.0], sorted_next_gap[sorted_next_gap > 0.0]])
        gap_cap = max(1.0, float(np.percentile(positive_gaps, 99.0))) if positive_gaps.size > 0 else 1.0

    sorted_prev_gap = np.clip(sorted_prev_gap, 0.0, gap_cap)
    sorted_next_gap = np.clip(sorted_next_gap, 0.0, gap_cap)

    prev_gap = np.zeros(n_rows, dtype="float64")
    next_gap = np.zeros(n_rows, dtype="float64")
    prev_gap[order] = sorted_prev_gap
    next_gap[order] = sorted_next_gap

    is_first_global = np.zeros(n_rows, dtype=bool)
    is_last_global = np.zeros(n_rows, dtype=bool)
    if n_rows > 0:
        is_first_global[order[0]] = True
        is_last_global[order[-1]] = True

    non_boundary_prev = ~is_first_global
    non_boundary_next = ~is_last_global
    prev_gap_eq_1 = non_boundary_prev & np.isclose(prev_gap, 1.0)
    next_gap_eq_1 = non_boundary_next & np.isclose(next_gap, 1.0)
    has_noncontiguous_neighbor = (non_boundary_prev & ~prev_gap_eq_1) | (non_boundary_next & ~next_gap_eq_1)

    features = pd.DataFrame(index=index)
    features["id_valid"] = id_valid
    features["id_missing"] = id_missing
    features["id_value"] = id_filled
    features["id_offset"] = id_offset
    features["id_rel_train"] = id_rel_train
    features["id_rank_global"] = id_rank_global
    features["id_rank_global_norm"] = id_rank_global_norm
    features["id_rank_within_table"] = id_rank_within_table
    features["id_rank_within_table_norm"] = id_rank_within_table_norm
    features["table_is_test_like"] = ~train_mask

    for block_size in _BLOCK_SIZES:
        block = np.floor(id_offset / float(block_size))
        features[f"block_{block_size // 1000}k"] = block
        if block_size in (1000, 10000):
            features[f"within_block_{block_size // 1000}k_norm"] = np.mod(id_offset, float(block_size)) / float(block_size)

    for modulus in _ID_MODULI:
        residue = np.mod(id_filled, float(modulus))
        angle = 2.0 * np.pi * residue / float(modulus)
        features[f"id_mod_{modulus}_norm"] = residue / float(modulus)
        features[f"id_mod_{modulus}_sin"] = np.sin(angle)
        features[f"id_mod_{modulus}_cos"] = np.cos(angle)

    features["id_even"] = np.mod(id_filled, 2.0) == 0.0
    features["prev_gap"] = prev_gap
    features["next_gap"] = next_gap
    features["min_neighbor_gap"] = np.minimum(prev_gap, next_gap)
    features["max_neighbor_gap"] = np.maximum(prev_gap, next_gap)
    features["is_first_global"] = is_first_global
    features["is_last_global"] = is_last_global
    features["prev_gap_eq_1"] = prev_gap_eq_1
    features["next_gap_eq_1"] = next_gap_eq_1
    features["has_noncontiguous_neighbor"] = has_noncontiguous_neighbor
    features["log_id"] = log_id
    features["log_id_z"] = log_id_z

    return features


FEATURE_GROUPS = [
    {
        "name": "aide_id_sequence_scan_context",
        "fn": add_aide_id_sequence_scan_context,
        "depends_on": [],
        "description": "Deterministic id sequence, block, periodic, and neighborhood context features for preprocessing.",
    }
]