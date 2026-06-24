import numpy as np
import pandas as pd

_BANDS = ("u", "g", "r", "i")
_BAND_CLIP_QUANTILES = (0.005, 0.995)
_BAND_CLIP_FALLBACK = (-30.0, 35.0)
_POCKET_C1_LOW = 0.85
_POCKET_C1_HIGH = 1.35
_POCKET_C3_OFFSET = 0.2
_POCKET_R_LIMIT = 20.5
_C1_BIN_PROBS = (1.0 / 3.0, 2.0 / 3.0)
_SOFT_GAP_BIN_PROBS = (0.33, 0.67)
_CORE_SYMMETRIC_QUANTILE = 0.5


def _to_float(series):
    values = pd.to_numeric(series, errors="coerce").astype("float64")
    return values.replace([np.inf, -np.inf], np.nan)


def _clip_band(series):
    values = _to_float(series)
    finite = values.dropna()
    lower, upper = _BAND_CLIP_FALLBACK

    if not finite.empty:
        q_low = finite.quantile(_BAND_CLIP_QUANTILES[0])
        q_high = finite.quantile(_BAND_CLIP_QUANTILES[1])
        if (
            pd.notna(q_low)
            and pd.notna(q_high)
            and np.isfinite(q_low)
            and np.isfinite(q_high)
            and q_low < q_high
        ):
            lower = float(q_low)
            upper = float(q_high)

    return values.clip(lower=lower, upper=upper)


def _quantile_tertile_labels(values, q1, q2, labels, fallback_sign=True):
    index = values.index
    arr = _to_float(values).to_numpy()
    result = pd.Series(labels[1], index=index, dtype="object")

    finite = np.isfinite(arr)
    if finite.any():
        finite_values = pd.Series(arr[finite])
        low = finite_values.quantile(q1)
        high = finite_values.quantile(q2)

        if (
            pd.notna(low)
            and pd.notna(high)
            and np.isfinite(low)
            and np.isfinite(high)
            and low < high
        ):
            result.loc[arr <= low] = labels[0]
            result.loc[arr >= high] = labels[2]
            return result

        if fallback_sign:
            result.loc[arr < 0.0] = labels[0]
            result.loc[arr > 0.0] = labels[2]

    result.loc[np.isposinf(arr)] = labels[2]
    result.loc[np.isneginf(arr)] = labels[0]
    return result


def _q_core_label(values, quantile, labels):
    index = values.index
    arr = _to_float(values).to_numpy()
    result = pd.Series(labels[1], index=index, dtype="object")

    finite = np.isfinite(arr)
    if finite.any():
        finite_abs = pd.Series(np.abs(arr[finite]))
        threshold = finite_abs.quantile(quantile)

        if pd.notna(threshold) and np.isfinite(threshold) and threshold > 0:
            result.loc[arr < -threshold] = labels[0]
            result.loc[arr > threshold] = labels[2]
            return result

        result.loc[arr < 0.0] = labels[0]
        result.loc[arr > 0.0] = labels[2]
        return result

    result.loc[np.isposinf(arr)] = labels[2]
    result.loc[np.isneginf(arr)] = labels[0]
    return result


def add_eboss_ptf_c1_c3_geometry(raw, deps, aux):
    u = _clip_band(raw["u"])
    g = _clip_band(raw["g"])
    r = _clip_band(raw["r"])
    i = _clip_band(raw["i"])
    r_for_flag = _to_float(raw["r"])

    c_ug = u - g
    c_gr = g - r
    c_ri = r - i

    c1 = 0.95 * c_ug + 0.31 * c_gr + 0.11 * c_ri
    c3 = -0.39 * c_ug + 0.79 * c_gr + 0.47 * c_ri

    d1 = (1.4 - 0.55 * c1) - c3
    d2 = (0.3 - 0.1 * c1) - c3

    S1 = d1.clip(lower=0.0)
    S2 = d2.clip(lower=0.0)
    core = np.minimum(d1, d2)
    V1 = (-d1).clip(lower=0.0)
    V2 = (-d2).clip(lower=0.0)
    soft_gap = np.maximum(V1, V2)
    normalized_gap = soft_gap.div(1.0 + c1.abs())

    pocket_c1 = pd.Series(
        np.where(
            c1 < _POCKET_C1_LOW,
            _POCKET_C1_LOW - c1,
            np.where(c1 > _POCKET_C1_HIGH, c1 - _POCKET_C1_HIGH, 0.0),
        ),
        index=raw.index,
        dtype="float64",
    )
    pocket_c3 = (-c3 - _POCKET_C3_OFFSET).clip(lower=0.0)
    pocket_flag = (
        (r_for_flag < _POCKET_R_LIMIT)
        & (pocket_c1 <= 1e-12)
        & (pocket_c3 <= 1e-12)
    ).astype("int8")
    pocket_distance = np.maximum(pocket_c1, pocket_c3)

    q_core = _q_core_label(core, _CORE_SYMMETRIC_QUANTILE, ("below", "near", "above"))
    c1_bin = _quantile_tertile_labels(
        c1,
        _C1_BIN_PROBS[0],
        _C1_BIN_PROBS[1],
        ("below", "near", "above"),
        fallback_sign=True,
    )
    soft_gap_bin = _quantile_tertile_labels(
        soft_gap,
        _SOFT_GAP_BIN_PROBS[0],
        _SOFT_GAP_BIN_PROBS[1],
        ("below", "near", "above"),
        fallback_sign=False,
    )

    return pd.DataFrame(
        {
            "c_ug": c_ug,
            "c_gr": c_gr,
            "c_ri": c_ri,
            "c1": c1,
            "c3": c3,
            "d1": d1,
            "d2": d2,
            "S1": S1,
            "S2": S2,
            "core": core,
            "V1": V1,
            "V2": V2,
            "soft_gap": soft_gap,
            "normalized_gap": normalized_gap,
            "pocket_c1": pocket_c1,
            "pocket_c3": pocket_c3,
            "pocket_distance": pocket_distance,
            "pocket_flag": pocket_flag,
            "q_core": q_core,
            "c1_bin": c1_bin,
            "soft_gap_bin": soft_gap_bin,
        },
        index=raw.index,
    )


FEATURE_GROUPS = [
    {
        "name": "eboss_ptf_c1_c3_geometry",
        "fn": add_eboss_ptf_c1_c3_geometry,
        "depends_on": [],
        "description": "Build robust c1/c3 color-plane wedge geometry features with boundary-separation signals, bright-pocket geometry constraints, and quantile-derived companion bins.",
    }
]