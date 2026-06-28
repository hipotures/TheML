import numpy as np
import pandas as pd


TRAIN_ID_MAX = 577346
EPS_SCALE = 0.01
CLIP_LOW_Q = 0.005
CLIP_HIGH_Q = 0.995
PENALTY_CAP = 20.0
INLIER_THRESHOLD = 0.05
BRANCH_BOUNDARY_SCALE_MULTIPLIER = 0.25


def _as_float_series(raw, name):
    return pd.to_numeric(raw[name], errors="coerce").astype("float64")


def _training_mask(raw):
    if "id" not in raw.columns:
        return pd.Series(True, index=raw.index)
    ids = pd.to_numeric(raw["id"], errors="coerce")
    mask = ids <= TRAIN_ID_MAX
    if mask.any():
        return mask
    return pd.Series(True, index=raw.index)


def _robust_stats(values, mask):
    train_values = values.loc[mask]
    median = train_values.median()
    mad = (train_values - median).abs().median()
    scale = max(float(mad), EPS_SCALE) if np.isfinite(mad) else EPS_SCALE
    low = train_values.quantile(CLIP_LOW_Q)
    high = train_values.quantile(CLIP_HIGH_Q)
    if not np.isfinite(low):
        low = values.min()
    if not np.isfinite(high):
        high = values.max()
    if not np.isfinite(low):
        low = -np.inf
    if not np.isfinite(high):
        high = np.inf
    return scale, float(low), float(high)


def _clip(values, stats):
    _, low, high = stats
    return values.clip(lower=low, upper=high)


def _interval_penalty(values, lower, upper, stats):
    scale, _, _ = stats
    clipped = _clip(values, stats)
    penalty = np.maximum(lower - clipped, 0.0) + np.maximum(clipped - upper, 0.0)
    return penalty / scale


def _upper_penalty(values, upper, stats):
    scale, _, _ = stats
    clipped = _clip(values, stats)
    return np.maximum(clipped - upper, 0.0) / scale


def _lower_penalty(values, lower, stats):
    scale, _, _ = stats
    clipped = _clip(values, stats)
    return np.maximum(lower - clipped, 0.0) / scale


def _capped(values):
    return np.asarray(values, dtype="float64").clip(0.0, PENALTY_CAP)


def add_segue_stellar_subtype_margins(raw, deps, aux):
    idx = raw.index
    train_mask = _training_mask(raw)

    u = _as_float_series(raw, "u")
    g = _as_float_series(raw, "g")
    r = _as_float_series(raw, "r")
    i = _as_float_series(raw, "i")
    z = _as_float_series(raw, "z")
    redshift = _as_float_series(raw, "redshift") if "redshift" in raw.columns else pd.Series(0.0, index=idx)

    ug = u - g
    gr = g - r
    ri = r - i
    iz = i - z
    gi = g - i
    l_color = -0.436 * u + 1.129 * g - 0.119 * r - 0.574 * i + 0.1984
    v_color = 0.283 * ug - 0.354 * gr + 0.455 * ri + 0.766 * iz
    wd_line = ug + 2.0 * gr
    m_subdwarf_line = ri - 0.787 * gr + 0.356
    mswd_lower_line = gr + ri - 1.13
    mswd_upper_line = gr - 0.95 * ri - 0.50

    stats = {
        "r": _robust_stats(r, train_mask),
        "ug": _robust_stats(ug, train_mask),
        "gr": _robust_stats(gr, train_mask),
        "ri": _robust_stats(ri, train_mask),
        "iz": _robust_stats(iz, train_mask),
        "gi": _robust_stats(gi, train_mask),
        "l_color": _robust_stats(l_color, train_mask),
        "v_color": _robust_stats(v_color, train_mask),
        "wd_line": _robust_stats(wd_line, train_mask),
        "m_subdwarf_line": _robust_stats(m_subdwarf_line, train_mask),
        "mswd_lower_line": _robust_stats(mswd_lower_line, train_mask),
        "mswd_upper_line": _robust_stats(mswd_upper_line, train_mask),
    }

    white_dwarf = (
        _interval_penalty(gr, -1.0, -0.2, stats["gr"])
        + _interval_penalty(ug, -1.0, 0.7, stats["ug"])
        + _upper_penalty(wd_line, -0.1, stats["wd_line"])
    ) / 3.0

    cool_wd_blue_branch = (
        _interval_penalty(ug, -0.4, 1.0, stats["ug"])
        + _upper_penalty(gi, 0.6, stats["gi"]) * BRANCH_BOUNDARY_SCALE_MULTIPLIER
    ) / 2.0
    cool_wd_red_branch = (
        _interval_penalty(ug, -0.2, 1.4, stats["ug"])
        + _lower_penalty(gi, 0.6, stats["gi"]) * BRANCH_BOUNDARY_SCALE_MULTIPLIER
    ) / 2.0
    cool_white_dwarf = (
        _interval_penalty(r, 14.5, 20.5, stats["r"])
        + _interval_penalty(gi, -2.0, 1.7, stats["gi"])
        + np.minimum(cool_wd_blue_branch, cool_wd_red_branch)
    ) / 3.0

    a_bhb = (
        _interval_penalty(ug, 0.8, 1.5, stats["ug"])
        + _interval_penalty(gr, -0.5, 0.2, stats["gr"])
        + _interval_penalty(v_color, -0.15, 0.15, stats["v_color"])
    ) / 3.0

    k_giants = (
        _interval_penalty(gr, 0.35, 0.8, stats["gr"])
        + _interval_penalty(ri, 0.15, 0.6, stats["ri"])
        + _lower_penalty(l_color, 0.07, stats["l_color"])
    ) / 3.0

    low_metallicity = (
        _interval_penalty(gr, -0.5, 0.75, stats["gr"])
        + _interval_penalty(ug, 0.6, 3.0, stats["ug"])
        + _lower_penalty(l_color, 0.135, stats["l_color"])
    ) / 3.0

    m_subdwarf = (
        _upper_penalty(ri, 0.9, stats["ri"])
        + _upper_penalty(m_subdwarf_line, 0.0, stats["m_subdwarf_line"])
        + _interval_penalty(gi, 1.8, 2.4, stats["gi"])
    ) / 3.0

    mswd_blue_iz_branch = (
        _interval_penalty(iz, -0.5, 1.0, stats["iz"])
        + _upper_penalty(ri, 1.0, stats["ri"]) * BRANCH_BOUNDARY_SCALE_MULTIPLIER
    ) / 2.0
    mswd_red_iz_branch = (
        _interval_penalty(iz, -0.2, 0.7, stats["iz"])
        + _lower_penalty(ri, 1.0, stats["ri"]) * BRANCH_BOUNDARY_SCALE_MULTIPLIER
    ) / 2.0
    main_sequence_white_dwarf_pairs = (
        _upper_penalty(ug, 2.25, stats["ug"])
        + _interval_penalty(gr, -0.2, 1.2, stats["gr"])
        + _interval_penalty(ri, 0.5, 2.0, stats["ri"])
        + _lower_penalty(mswd_lower_line, 0.0, stats["mswd_lower_line"])
        + _upper_penalty(mswd_upper_line, 0.0, stats["mswd_upper_line"])
        + np.minimum(mswd_blue_iz_branch, mswd_red_iz_branch)
    ) / 6.0

    prototype_matrix = np.column_stack(
        [
            _capped(white_dwarf),
            _capped(cool_white_dwarf),
            _capped(a_bhb),
            _capped(k_giants),
            _capped(low_metallicity),
            _capped(m_subdwarf),
            _capped(main_sequence_white_dwarf_pairs),
        ]
    )
    sorted_margins = np.sort(prototype_matrix, axis=1)
    best = sorted_margins[:, 0]
    second = sorted_margins[:, 1]
    mean_top3 = sorted_margins[:, :3].mean(axis=1)

    out = pd.DataFrame(index=idx)
    out["white_dwarf_penalty"] = prototype_matrix[:, 0]
    out["cool_white_dwarf_penalty"] = prototype_matrix[:, 1]
    out["a_bhb_penalty"] = prototype_matrix[:, 2]
    out["k_giants_penalty"] = prototype_matrix[:, 3]
    out["low_metallicity_penalty"] = prototype_matrix[:, 4]
    out["m_subdwarf_penalty"] = prototype_matrix[:, 5]
    out["main_sequence_white_dwarf_pairs_penalty"] = prototype_matrix[:, 6]
    out["star_margin_best"] = best
    out["star_margin_second"] = second
    out["star_margin_gap"] = second - best
    out["star_margin_mean_top3"] = mean_top3
    out["star_inlier_count"] = (prototype_matrix <= INLIER_THRESHOLD).sum(axis=1).astype("int16")
    out["star_confidence"] = np.exp(-best)
    out["star_margin_best_redshift_log_interaction"] = best * np.log1p(np.maximum(redshift.to_numpy(dtype="float64"), 0.0))
    return out


FEATURE_GROUPS = [
    {
        "name": "segue_stellar_subtype_margins",
        "fn": add_segue_stellar_subtype_margins,
        "depends_on": [],
        "description": "Robust continuous margins to canonical SDSS-like stellar subtype color envelopes.",
    }
]