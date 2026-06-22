from __future__ import annotations

import numpy as np
import pandas as pd


AUX_COLS = ("alpha", "delta", "u", "g", "r", "i", "z", "redshift")


def _num(frame: pd.DataFrame, col: str, index: pd.Index | None = None) -> pd.Series:
    if col not in frame:
        idx = frame.index if index is None else index
        return pd.Series(np.nan, index=idx, dtype="float64")
    return pd.to_numeric(frame[col], errors="coerce").astype("float64")


def _clean_aux(aux: pd.DataFrame | None) -> pd.DataFrame:
    if aux is None or aux.empty:
        return pd.DataFrame(columns=AUX_COLS)
    cleaned = pd.DataFrame(index=aux.index)
    for col in AUX_COLS:
        values = _num(aux, col)
        if col in {"u", "g", "r", "i", "z"}:
            values = values.mask(values <= -9000)
        if col == "redshift":
            values = values.clip(lower=0)
        cleaned[col] = values
    return cleaned.dropna(how="all")


def aide_aux_reference_distribution_distance(raw: pd.DataFrame, deps: dict[str, pd.DataFrame] | None = None, aux: pd.DataFrame | None = None) -> pd.DataFrame:
    out = pd.DataFrame(index=raw.index)
    aux_ref = _clean_aux(aux)
    raw_values = pd.DataFrame(index=raw.index)
    for col in AUX_COLS:
        values = _num(raw, col)
        if col == "redshift":
            values = values.clip(lower=0)
        raw_values[col] = values

    if aux_ref.empty:
        for col in AUX_COLS:
            out[f"aide_aux_{col}_z"] = 0.0
            out[f"aide_aux_{col}_med_delta_abs"] = 0.0
            out[f"aide_aux_{col}_cdf"] = 0.5
        out["aide_aux_joint_l2"] = 0.0
        out["aide_aux_ref_mahalanobis2"] = 0.0
        out["aide_aux_ref_mahalanobis"] = 0.0
        out["aide_aux_ref_mahalanobis_inlier"] = 1.0
        return out

    joint = pd.Series(0.0, index=raw.index)
    for col in AUX_COLS:
        ref = aux_ref[col].dropna()
        if ref.empty:
            out[f"aide_aux_{col}_z"] = 0.0
            out[f"aide_aux_{col}_med_delta_abs"] = 0.0
            out[f"aide_aux_{col}_cdf"] = 0.5
            continue
        median = ref.median()
        mad = (ref - median).abs().median()
        scale = 1.4826 * mad if mad and np.isfinite(mad) else ref.std(ddof=0)
        if not scale or not np.isfinite(scale):
            scale = 1.0
        z_score = (raw_values[col] - median) / scale
        out[f"aide_aux_{col}_z"] = z_score
        out[f"aide_aux_{col}_med_delta_abs"] = (raw_values[col] - median).abs()
        ranks = np.searchsorted(np.sort(ref.to_numpy()), raw_values[col].to_numpy(), side="right") / max(len(ref), 1)
        out[f"aide_aux_{col}_cdf"] = ranks
        joint = joint + z_score.fillna(0.0) ** 2

    out["aide_aux_joint_l2"] = np.sqrt(joint)

    matrix = aux_ref[list(AUX_COLS)].dropna()
    if len(matrix) <= len(AUX_COLS):
        out["aide_aux_ref_mahalanobis2"] = out["aide_aux_joint_l2"] ** 2
    else:
        center = matrix.median(axis=0)
        spread = (matrix - center).abs().median(axis=0).replace(0, np.nan)
        standardized_ref = ((matrix - center) / spread.fillna(1.0)).fillna(0.0)
        standardized_raw = ((raw_values[list(AUX_COLS)] - center) / spread.fillna(1.0)).fillna(0.0)
        cov = np.cov(standardized_ref.to_numpy(), rowvar=False)
        inv_cov = np.linalg.pinv(cov)
        vals = standardized_raw.to_numpy()
        out["aide_aux_ref_mahalanobis2"] = np.einsum("ij,jk,ik->i", vals, inv_cov, vals)

    out["aide_aux_ref_mahalanobis"] = np.sqrt(out["aide_aux_ref_mahalanobis2"].clip(lower=0))
    threshold = np.nanpercentile(out["aide_aux_ref_mahalanobis"], 95)
    out["aide_aux_ref_mahalanobis_inlier"] = (out["aide_aux_ref_mahalanobis"] <= threshold).astype("float64")
    return out.replace([np.inf, -np.inf], np.nan)


FEATURE_GROUPS = [
    {
        "name": "aide_aux_reference_distribution_distance",
        "fn": aide_aux_reference_distribution_distance,
        "depends_on": [],
        "description": "AIDE robust auxiliary-reference z, CDF, joint distance, and Mahalanobis features.",
    }
]
