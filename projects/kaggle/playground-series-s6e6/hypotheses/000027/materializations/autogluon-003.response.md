import numpy as np
import pandas as pd

_ID_RESIDUES = (2, 11, 97)
_TWO_PI = 2.0 * np.pi


def add_aide_id_sequence_scan_context(raw, deps, aux):
    idx = raw.index
    _ = (deps, aux)

    ids_raw = pd.to_numeric(raw["id"], errors="coerce")
    id_valid = ids_raw.notna().astype("int8")
    id_missing = (1 - id_valid).astype("int8")
    ids = ids_raw.fillna(0).astype("int64")

    n_rows = len(raw)
    valid_ids = ids[ids_raw.notna()]
    if len(valid_ids) == 0:
        id_min = 0
        id_max = 0
        id_span = 1
    else:
        id_min = int(valid_ids.min())
        id_max = int(valid_ids.max())
        span = int(id_max - id_min)
        id_span = 1 if span <= 0 else span

    ordered_idx = ids.sort_values(kind="mergesort").index
    id_rank = pd.Series(0, index=idx, dtype="int64")
    if n_rows:
        id_rank.loc[ordered_idx] = np.arange(n_rows, dtype=np.int64)

    if n_rows <= 1:
        id_rank_norm = pd.Series(0.0, index=idx, dtype="float32")
    else:
        id_rank_norm = (id_rank.astype("float64") / float(n_rows - 1)).clip(0.0, 1.0).astype("float32")

    id_rel = ((ids.astype("float64") - float(id_min)) / float(id_span)).clip(0.0, 1.0).astype("float32")
    id_rel = id_rel.where(ids_raw.notna(), 0.0)

    log_id = np.log1p(np.maximum(ids.astype("float64"), 0.0))
    log_id_mean = float(log_id.mean())
    log_id_std = float(log_id.std(ddof=0))
    if not np.isfinite(log_id_std) or log_id_std <= 0.0:
        log_id_std = 1.0
    log_id_z = ((log_id - log_id_std * 0.0 - log_id_mean) / log_id_std).astype("float32")
    log_id_z = log_id_z.where(ids_raw.notna(), 0.0)

    id_from_min = (ids - id_min)
    id_from_min = id_from_min.where(id_from_min >= 0, 0).astype("int64")
    block_1k = np.floor_divide(id_from_min, 1000).astype("int64")
    block_100k = np.floor_divide(id_from_min, 100000).astype("int64")
    within_1k = (id_from_min % 1000).astype("float32") / 1000.0

    ids_ordered = ids.loc[ordered_idx].astype("int64")
    prev_id = ids_ordered.shift(1)
    next_id = ids_ordered.shift(-1)

    prev_gap_ordered = (ids_ordered - prev_id).fillna(0.0)
    next_gap_ordered = (next_id - ids_ordered).fillna(0.0)

    prev_gap = pd.Series(0.0, index=idx, dtype="float32")
    next_gap = pd.Series(0.0, index=idx, dtype="float32")
    if n_rows:
        prev_gap.loc[ordered_idx] = prev_gap_ordered.to_numpy(dtype="float64")
        next_gap.loc[ordered_idx] = next_gap_ordered.to_numpy(dtype="float64")

    all_gaps = pd.concat([prev_gap_ordered, next_gap_ordered], axis=0).replace([np.inf, -np.inf], np.nan).dropna()
    if len(all_gaps) > 0:
        cap = float(np.nanpercentile(all_gaps.to_numpy(), 99.0))
        if not np.isfinite(cap) or cap <= 0.0:
            cap = 0.0
    else:
        cap = 0.0

    if cap > 0.0:
        prev_gap = prev_gap.clip(0.0, cap)
        next_gap = next_gap.clip(0.0, cap)

    is_first = pd.Series(0, index=idx, dtype="int8")
    is_last = pd.Series(0, index=idx, dtype="int8")
    if n_rows:
        is_first.loc[ordered_idx[0]] = 1
        is_last.loc[ordered_idx[-1]] = 1

    noncontiguous = ((prev_gap > 1.0) | (next_gap > 1.0)).astype("int8")
    adjacent_step = ((prev_gap == 1.0) & (next_gap == 1.0)).astype("int8")
    id_even = (ids % 2).astype("int8")

    features = {
        "id_valid": id_valid,
        "id_missing": id_missing,
        "id_rank": id_rank.astype("int64"),
        "id_rank_norm": id_rank_norm,
        "id_min": pd.Series(id_min, index=idx, dtype="int64"),
        "id_max": pd.Series(id_max, index=idx, dtype="int64"),
        "id_span": pd.Series(id_span, index=idx, dtype="int64"),
        "id_rel": id_rel,
        "log_id": log_id.astype("float32"),
        "log_id_z": log_id_z,
        "block_1k": block_1k,
        "block_100k": block_100k,
        "within_1k": within_1k,
        "prev_gap": prev_gap,
        "next_gap": next_gap,
        "is_first": is_first,
        "is_last": is_last,
        "noncontiguous": noncontiguous,
        "adjacent_step": adjacent_step,
        "id_even": id_even,
    }

    for modulus in _ID_RESIDUES:
        residue = (ids % modulus).astype("float64")
        phase = _TWO_PI * (residue / float(modulus))
        features[f"id_mod_{modulus}_sin"] = np.sin(phase)
        features[f"id_mod_{modulus}_cos"] = np.cos(phase)

    return pd.DataFrame(features, index=idx)


FEATURE_GROUPS = [
    {
        "name": "aide_id_sequence_scan_context",
        "fn": add_aide_id_sequence_scan_context,
        "depends_on": [],
        "description": "Encodes integer id order, span-normalized position, block and periodic signatures, and local id-order continuity signals to capture sequence-structure artifacts.",
    }
]