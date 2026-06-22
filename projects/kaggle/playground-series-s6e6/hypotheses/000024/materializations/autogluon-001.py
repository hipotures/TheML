from __future__ import annotations

import numpy as np
import pandas as pd


NUMERIC_COLS = ("u", "g", "r", "i", "z", "redshift")
CAT_COLS = ("spectral_type", "galaxy_population")


def _num(raw: pd.DataFrame, col: str) -> pd.Series:
    if col not in raw:
        return pd.Series(np.nan, index=raw.index, dtype="float64")
    return pd.to_numeric(raw[col], errors="coerce").astype("float64")


def _cat(raw: pd.DataFrame, col: str) -> pd.Series:
    if col not in raw:
        return pd.Series("__missing__", index=raw.index, dtype="object")
    return raw[col].astype("string").fillna("__missing__").astype("object")


def _qbin(values: pd.Series, q: int = 12) -> pd.Series:
    ranked = values.rank(method="first")
    try:
        return pd.qcut(ranked, q=q, labels=False, duplicates="drop")
    except ValueError:
        return pd.Series(0, index=values.index, dtype="float64")


def aide_catalog_rank_frequency_context(raw: pd.DataFrame, deps: dict[str, pd.DataFrame] | None = None, aux: pd.DataFrame | None = None) -> pd.DataFrame:
    out = pd.DataFrame(index=raw.index)
    cats = {col: _cat(raw, col) for col in CAT_COLS}

    for col, values in cats.items():
        freq = values.map(values.value_counts(normalize=True)).astype("float64")
        out[f"aide_{col}_freq"] = freq

    cross = cats["spectral_type"].astype(str) + "__" + cats["galaxy_population"].astype(str)
    out["aide_spectral_type_x_pop_freq"] = cross.map(cross.value_counts(normalize=True)).astype("float64")

    rank_cols = {col: _num(raw, col) for col in NUMERIC_COLS}
    rank_cols["redshift_log1p"] = np.log1p(rank_cols["redshift"].clip(lower=0))
    rank_values = []
    for col, values in rank_cols.items():
        rank = values.rank(pct=True)
        qbin = pd.Series(_qbin(values), index=raw.index)
        out[f"aide_{col}_rank_pct"] = rank
        out[f"aide_{col}_qbin"] = qbin.astype("float64")
        out[f"aide_{col}_qbin_freq"] = qbin.map(qbin.value_counts(normalize=True)).astype("float64")
        rank_values.append(rank)

        for cat_col, cat_values in cats.items():
            grouped_rank = values.groupby(cat_values).rank(pct=True)
            out[f"aide_{col}_rank_within_{cat_col}"] = grouped_rank

    rank_frame = pd.concat(rank_values, axis=1)
    out["aide_numeric_rank_mean"] = rank_frame.mean(axis=1)
    out["aide_numeric_rank_std"] = rank_frame.std(axis=1)
    return out.replace([np.inf, -np.inf], np.nan)


FEATURE_GROUPS = [
    {
        "name": "aide_catalog_rank_frequency_context",
        "fn": aide_catalog_rank_frequency_context,
        "depends_on": [],
        "description": "AIDE category frequency, cross-frequency, global rank, qbin, and within-category rank context.",
    }
]
