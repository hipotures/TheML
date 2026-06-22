from __future__ import annotations

import numpy as np
import pandas as pd


def _id_values(raw: pd.DataFrame) -> pd.Series:
    if "id" not in raw:
        return pd.Series(np.arange(len(raw), dtype="float64"), index=raw.index)
    return pd.to_numeric(raw["id"], errors="coerce").astype("float64")


def aide_id_sequence_scan_context(raw: pd.DataFrame, deps: dict[str, pd.DataFrame] | None = None, aux: pd.DataFrame | None = None) -> pd.DataFrame:
    out = pd.DataFrame(index=raw.index)
    ids = _id_values(raw)
    fallback = pd.Series(np.arange(len(raw), dtype="float64"), index=raw.index)
    filled = ids.fillna(fallback)

    rank = filled.rank(method="first")
    denom = max(len(raw) - 1, 1)
    out["aide_id_is_missing"] = ids.isna().astype("float64")
    out["aide_id_rank"] = rank
    out["aide_id_rank_norm"] = (rank - 1.0) / denom
    out["aide_id_block_1k"] = np.floor(filled / 1_000.0)
    out["aide_id_block_100k"] = np.floor(filled / 100_000.0)
    out["aide_id_mod_2"] = np.mod(filled, 2.0)
    out["aide_id_mod_11"] = np.mod(filled, 11.0)
    out["aide_id_mod_97"] = np.mod(filled, 97.0)
    out["aide_id_even"] = (np.mod(filled, 2.0) == 0).astype("float64")
    out["aide_id_log1p"] = np.log1p(filled.clip(lower=0))

    ordered = pd.DataFrame({"id": filled, "position": np.arange(len(raw))}, index=raw.index).sort_values("id", kind="mergesort")
    prev_gap = ordered["id"].diff()
    next_gap = ordered["id"].shift(-1) - ordered["id"]
    ordered["aide_id_gap_prev"] = prev_gap
    ordered["aide_id_gap_next"] = next_gap
    restored = ordered.sort_values("position", kind="mergesort")
    out["aide_id_gap_prev"] = restored["aide_id_gap_prev"].to_numpy()
    out["aide_id_gap_next"] = restored["aide_id_gap_next"].to_numpy()
    return out.replace([np.inf, -np.inf], np.nan)


FEATURE_GROUPS = [
    {
        "name": "aide_id_sequence_scan_context",
        "fn": aide_id_sequence_scan_context,
        "depends_on": [],
        "description": "AIDE id rank, block, modulo, parity, log, and neighbor-gap sequence features.",
    }
]
