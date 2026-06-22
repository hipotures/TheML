from __future__ import annotations

import numpy as np
import pandas as pd

try:
    from sklearn.preprocessing import SplineTransformer
except Exception:  # pragma: no cover - sklearn is expected in the modeling env.
    SplineTransformer = None


MAGS = ("u", "g", "r", "i", "z")


def _num(frame: pd.DataFrame, col: str, index: pd.Index | None = None) -> pd.Series:
    if col not in frame:
        idx = frame.index if index is None else index
        return pd.Series(np.nan, index=idx, dtype="float64")
    return pd.to_numeric(frame[col], errors="coerce").astype("float64")


def _base_features(frame: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=frame.index)
    for col in ("alpha", "delta", "u", "g", "r", "i", "z", "redshift"):
        values = _num(frame, col)
        if col == "redshift":
            values = values.clip(lower=0)
        out[col] = values
    out["u_g"] = out["u"] - out["g"]
    out["g_r"] = out["g"] - out["r"]
    out["r_i"] = out["r"] - out["i"]
    out["i_z"] = out["i"] - out["z"]
    out["u_r"] = out["u"] - out["r"]
    out["g_i"] = out["g"] - out["i"]
    out["r_z"] = out["r"] - out["z"]
    out["u_z"] = out["u"] - out["z"]
    return out.replace([np.inf, -np.inf], np.nan)


def _spline(values: pd.Series, fit_values: pd.Series, n_knots: int, periodic: bool) -> np.ndarray:
    filled_fit = fit_values.dropna().to_numpy(dtype="float64").reshape(-1, 1)
    filled_values = values.fillna(fit_values.median()).to_numpy(dtype="float64").reshape(-1, 1)
    if SplineTransformer is None or len(filled_fit) < n_knots:
        centered = filled_values - np.nanmean(filled_fit) if len(filled_fit) else filled_values
        return np.hstack([filled_values, centered])
    transformer = SplineTransformer(
        n_knots=n_knots,
        degree=3,
        extrapolation="periodic" if periodic else "constant",
        include_bias=False,
    )
    transformer.fit(filled_fit)
    return transformer.transform(filled_values)


def aide_smooth_spline_sed_interactions(raw: pd.DataFrame, deps: dict[str, pd.DataFrame] | None = None, aux: pd.DataFrame | None = None) -> pd.DataFrame:
    base = _base_features(raw)
    aux_base = _base_features(aux) if aux is not None and not aux.empty else pd.DataFrame(columns=base.columns)
    out = pd.DataFrame(index=raw.index)

    specs = {
        "alpha": (7, True),
        "delta": (5, False),
        "u": (5, False),
        "g": (5, False),
        "r": (5, False),
        "i": (5, False),
        "z": (5, False),
        "redshift": (5, False),
        "u_g": (5, False),
        "g_r": (5, False),
        "r_i": (5, False),
        "i_z": (5, False),
        "u_r": (5, False),
        "g_i": (5, False),
        "r_z": (5, False),
        "u_z": (5, False),
    }
    spline_blocks: dict[str, np.ndarray] = {}
    for col, (n_knots, periodic) in specs.items():
        fit_values = pd.concat([base[col], aux_base[col]], axis=0) if col in aux_base else base[col]
        block = _spline(base[col], fit_values, n_knots=n_knots, periodic=periodic)
        spline_blocks[col] = block
        for idx in range(min(block.shape[1], 6)):
            out[f"aide_spline_{col}_{idx}"] = block[:, idx]

    for left, right in (("redshift", "u_g"), ("redshift", "g_r"), ("redshift", "r_i"), ("g_r", "r_i")):
        left_block = spline_blocks[left][:, :2]
        right_block = spline_blocks[right][:, :2]
        for i in range(left_block.shape[1]):
            for j in range(right_block.shape[1]):
                out[f"aide_tx_{left}_{i}_x_{right}_{j}"] = left_block[:, i] * right_block[:, j]

    return out.replace([np.inf, -np.inf], np.nan)


FEATURE_GROUPS = [
    {
        "name": "aide_smooth_spline_sed_interactions",
        "fn": aide_smooth_spline_sed_interactions,
        "depends_on": [],
        "description": "AIDE spline basis features and compact tensor interactions.",
    }
]
