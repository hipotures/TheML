import numpy as np
import pandas as pd


TRAIN_ID_MAX = 577346
COLOR_CLIP_LOW_Q = 0.001
COLOR_CLIP_HIGH_Q = 0.999
FALLBACK_COLOR_LOW = -10.0
FALLBACK_COLOR_HIGH = 10.0


def _train_mask(raw):
    if "id" not in raw.columns:
        return pd.Series(True, index=raw.index)
    ids = pd.to_numeric(raw["id"], errors="coerce")
    mask = ids.le(TRAIN_ID_MAX)
    if bool(mask.any()):
        return mask.fillna(False)
    return pd.Series(True, index=raw.index)


def _finite_train_values(values, train_mask):
    train_values = values.loc[train_mask]
    return train_values[np.isfinite(train_values)]


def _safe_quantile(values, q, fallback):
    finite = values[np.isfinite(values)]
    if finite.empty:
        return fallback
    out = float(finite.quantile(q))
    if np.isfinite(out):
        return out
    return fallback


def _color_series(raw, left, right, train_mask):
    color = pd.to_numeric(raw[left], errors="coerce") - pd.to_numeric(raw[right], errors="coerce")
    finite_train = _finite_train_values(color, train_mask)

    if finite_train.empty:
        median = 0.0
        low = FALLBACK_COLOR_LOW
        high = FALLBACK_COLOR_HIGH
    else:
        median = float(finite_train.median())
        if not np.isfinite(median):
            median = 0.0
        low = _safe_quantile(finite_train, COLOR_CLIP_LOW_Q, FALLBACK_COLOR_LOW)
        high = _safe_quantile(finite_train, COLOR_CLIP_HIGH_Q, FALLBACK_COLOR_HIGH)
        if not np.isfinite(low) or not np.isfinite(high) or low >= high:
            low = FALLBACK_COLOR_LOW
            high = FALLBACK_COLOR_HIGH

    color = color.where(np.isfinite(color), median)
    return color.clip(lower=low, upper=high).astype(float)


def _tertile_bin(values, train_mask):
    train_values = _finite_train_values(values, train_mask)
    q1 = _safe_quantile(train_values, 1.0 / 3.0, 0.0)
    q2 = _safe_quantile(train_values, 2.0 / 3.0, 0.0)
    if not np.isfinite(q1) or not np.isfinite(q2) or q1 >= q2:
        q1 = _safe_quantile(train_values, 0.5, 0.0)
        q2 = q1

    arr = values.to_numpy(dtype=float, copy=False)
    bins = np.where(arr <= q1, "low", np.where(arr <= q2, "mid", "high"))
    return pd.Series(bins, index=values.index, dtype="object")


def add_eboss_ptf_c1_c3_geometry(raw, deps, aux):
    train_mask = _train_mask(raw)

    c_ug = _color_series(raw, "u", "g", train_mask)
    c_gr = _color_series(raw, "g", "r", train_mask)
    c_ri = _color_series(raw, "r", "i", train_mask)

    c1 = 0.95 * c_ug + 0.31 * c_gr + 0.11 * c_ri
    c3 = -0.39 * c_ug + 0.79 * c_gr + 0.47 * c_ri

    b1 = 1.4 - 0.55 * c1
    b2 = 0.3 - 0.1 * c1
    m1 = b1 - c3
    m2 = b2 - c3

    core_margin = np.minimum(m1, m2)
    abs_core_margin = np.abs(core_margin)
    inside_margin_1 = np.maximum(m1, 0.0)
    inside_margin_2 = np.maximum(m2, 0.0)
    violation_1 = np.maximum(-m1, 0.0)
    violation_2 = np.maximum(-m2, 0.0)
    max_violation = np.maximum(violation_1, violation_2)
    normalized_violation = max_violation / (1.0 + np.abs(c1))

    pocket_c1_gap = np.maximum.reduce((0.85 - c1, np.zeros(len(raw)), c1 - 1.35))
    pocket_c3_gap = np.maximum(-0.2 - c3, 0.0)
    pocket_distance = np.maximum(pocket_c1_gap, pocket_c3_gap)

    r_mag = pd.to_numeric(raw["r"], errors="coerce")
    bright_depth = np.maximum(20.5 - r_mag.fillna(20.5), 0.0)
    pocket_flag = (
        r_mag.lt(20.5).fillna(False)
        & pd.Series(pocket_c1_gap == 0.0, index=raw.index)
        & pd.Series(pocket_c3_gap == 0.0, index=raw.index)
    )

    core_margin_series = pd.Series(core_margin, index=raw.index, dtype=float)
    q45 = _safe_quantile(core_margin_series.loc[train_mask], 0.45, 0.0)
    q55 = _safe_quantile(core_margin_series.loc[train_mask], 0.55, 0.0)
    near_low = min(q45, 0.0)
    near_high = max(q55, 0.0)
    if not np.isfinite(near_low) or not np.isfinite(near_high) or near_low > near_high:
        near_low = 0.0
        near_high = 0.0

    core_arr = core_margin_series.to_numpy(dtype=float, copy=False)
    core_margin_bin = np.where(
        core_arr < near_low,
        "negative",
        np.where(core_arr <= near_high, "near_zero", "positive"),
    )

    c1_series = pd.Series(c1, index=raw.index, dtype=float)
    max_violation_series = pd.Series(max_violation, index=raw.index, dtype=float)

    return pd.DataFrame(
        {
            "c1": c1_series,
            "c3": pd.Series(c3, index=raw.index, dtype=float),
            "core_margin": core_margin_series,
            "abs_core_margin": pd.Series(abs_core_margin, index=raw.index, dtype=float),
            "inside_margin_1": pd.Series(inside_margin_1, index=raw.index, dtype=float),
            "inside_margin_2": pd.Series(inside_margin_2, index=raw.index, dtype=float),
            "violation_1": pd.Series(violation_1, index=raw.index, dtype=float),
            "violation_2": pd.Series(violation_2, index=raw.index, dtype=float),
            "max_violation": max_violation_series,
            "normalized_violation": pd.Series(normalized_violation, index=raw.index, dtype=float),
            "pocket_c1_gap": pd.Series(pocket_c1_gap, index=raw.index, dtype=float),
            "pocket_c3_gap": pd.Series(pocket_c3_gap, index=raw.index, dtype=float),
            "pocket_distance": pd.Series(pocket_distance, index=raw.index, dtype=float),
            "pocket_flag": pocket_flag.astype("int8"),
            "bright_scaled_pocket_distance": pd.Series(
                pocket_distance / (1.0 + bright_depth.to_numpy(dtype=float, copy=False)),
                index=raw.index,
                dtype=float,
            ),
            "core_margin_bin": pd.Series(core_margin_bin, index=raw.index, dtype="object"),
            "c1_tertile": _tertile_bin(c1_series, train_mask),
            "max_violation_bin": _tertile_bin(max_violation_series, train_mask),
        },
        index=raw.index,
    )


FEATURE_GROUPS = [
    {
        "name": "eboss_ptf_c1_c3_geometry",
        "fn": add_eboss_ptf_c1_c3_geometry,
        "depends_on": [],
        "description": "Robust SDSS-style c1/c3 color-plane wedge margins and bright-pocket proximity features.",
    }
]