import numpy as np
import pandas as pd

_FEATURE_COLOR_COLUMNS = ("u", "g", "r", "i", "z")
_COLOR_CLIP_MIN = -10.0
_COLOR_CLIP_MAX = 10.0

_C1_WEIGHTS = (0.95, 0.31, 0.11)
_C3_WEIGHTS = (-0.39, 0.79, 0.47)

_BOUNDARY1_INTERCEPT = 1.4
_BOUNDARY1_SLOPE = 0.55
_BOUNDARY2_INTERCEPT = 0.3
_BOUNDARY2_SLOPE = 0.1

_POCKET_R_MAX = 20.5
_POCKET_C1_MIN = 0.85
_POCKET_C1_MAX = 1.35
_POCKET_C3_MAX = -0.2

_CORE_BIN_LOWER_DEFAULT = -0.12
_CORE_BIN_UPPER_DEFAULT = 0.12
_CORE_BIN_NEAR_MIN = 0.06


def _compute_eboss_geometry(frame):
    u = frame["u"].astype("float64")
    g = frame["g"].astype("float64")
    r = frame["r"].astype("float64")
    i = frame["i"].astype("float64")
    z = frame["z"].astype("float64")

    color_ug = (u - g).clip(_COLOR_CLIP_MIN, _COLOR_CLIP_MAX)
    color_gr = (g - r).clip(_COLOR_CLIP_MIN, _COLOR_CLIP_MAX)
    color_ri = (r - i).clip(_COLOR_CLIP_MIN, _COLOR_CLIP_MAX)
    color_iz = (i - z).clip(_COLOR_CLIP_MIN, _COLOR_CLIP_MAX)

    c1 = (
        _C1_WEIGHTS[0] * color_ug
        + _C1_WEIGHTS[1] * color_gr
        + _C1_WEIGHTS[2] * color_ri
    )
    c3 = (
        _C3_WEIGHTS[0] * color_ug
        + _C3_WEIGHTS[1] * color_gr
        + _C3_WEIGHTS[2] * color_ri
    )

    d1 = (_BOUNDARY1_INTERCEPT - _BOUNDARY1_SLOPE * c1) - c3
    d2 = (_BOUNDARY2_INTERCEPT - _BOUNDARY2_SLOPE * c1) - c3

    core_score = pd.Series(
        np.minimum(d1.to_numpy(), d2.to_numpy()), index=frame.index
    )
    violation_upper = (-d1).clip(lower=0.0)
    violation_lower = (-d2).clip(lower=0.0)
    core_abs = core_score.abs()

    pocket_flag = (
        (r < _POCKET_R_MAX)
        & (c1 >= _POCKET_C1_MIN)
        & (c1 <= _POCKET_C1_MAX)
        & (c3 > _POCKET_C3_MAX)
    ).astype("int8")

    pocket_dist_c1 = pd.concat(
        {
            "left": (_POCKET_C1_MIN - c1).clip(lower=0.0),
            "right": (c1 - _POCKET_C1_MAX).clip(lower=0.0),
            "center": pd.Series(0.0, index=frame.index),
        },
        axis=1,
    ).max(axis=1)

    pocket_dist_c3 = (-c3 - _POCKET_C3_MAX).clip(lower=0.0)

    return {
        "color_ug": color_ug,
        "color_gr": color_gr,
        "color_ri": color_ri,
        "color_iz": color_iz,
        "c1_rot": c1,
        "c3_rot": c3,
        "boundary_d1": d1,
        "boundary_d2": d2,
        "core_score": core_score,
        "violation_upper": violation_upper,
        "violation_lower": violation_lower,
        "core_score_abs": core_abs,
        "pocket_flag": pocket_flag,
        "pocket_dist_c1": pocket_dist_c1,
        "pocket_dist_c3": pocket_dist_c3,
    }


def _build_core_bins(core_score):
    non_na = core_score.dropna()

    lower = _CORE_BIN_LOWER_DEFAULT
    upper = _CORE_BIN_UPPER_DEFAULT

    if not non_na.empty:
        neg = non_na[non_na < 0.0]
        pos = non_na[non_na >= 0.0]

        if len(neg) > 0:
            neg_q = float(neg.quantile(0.66))
            if np.isfinite(neg_q):
                lower = neg_q

        if len(pos) > 0:
            pos_q = float(pos.quantile(0.34))
            if np.isfinite(pos_q):
                upper = pos_q

    if not np.isfinite(lower) or lower >= 0:
        lower = _CORE_BIN_LOWER_DEFAULT
    if not np.isfinite(upper) or upper <= 0:
        upper = _CORE_BIN_UPPER_DEFAULT
    if not lower < upper:
        lower = _CORE_BIN_LOWER_DEFAULT
        upper = _CORE_BIN_UPPER_DEFAULT

    quantile_bin = pd.cut(
        core_score,
        bins=[-np.inf, lower, upper, np.inf],
        labels=("core_below", "core_near", "core_above"),
        include_lowest=True,
    )

    near_half = max(_CORE_BIN_NEAR_MIN, min(0.5, 0.5 * (upper - lower)))
    signed_bin = pd.Series(
        np.where(
            core_score < -near_half,
            "below",
            np.where(core_score > near_half, "above", "near"),
        ),
        index=core_score.index,
        dtype="string",
    )

    return quantile_bin, signed_bin


def add_eboss_ptf_c1_c3_geometry(raw, deps, aux):
    geometry = _compute_eboss_geometry(raw)
    core_score = geometry["core_score"]

    core_quantile_bin, core_signed_bin = _build_core_bins(core_score)

    features = pd.DataFrame(
        {
            "color_ug": geometry["color_ug"],
            "color_gr": geometry["color_gr"],
            "color_ri": geometry["color_ri"],
            "color_iz": geometry["color_iz"],
            "c1_rot": geometry["c1_rot"],
            "c3_rot": geometry["c3_rot"],
            "boundary_d1": geometry["boundary_d1"],
            "boundary_d2": geometry["boundary_d2"],
            "core_score": core_score,
            "violation_upper": geometry["violation_upper"],
            "violation_lower": geometry["violation_lower"],
            "core_score_abs": geometry["core_score_abs"],
            "pocket_flag": geometry["pocket_flag"],
            "pocket_dist_c1": geometry["pocket_dist_c1"],
            "pocket_dist_c3": geometry["pocket_dist_c3"],
            "core_quantile_bin": core_quantile_bin,
            "core_signed_bin": core_signed_bin,
        },
        index=raw.index,
    )

    if isinstance(aux, pd.DataFrame) and not aux.empty:
        if all(col in aux.columns for col in _FEATURE_COLOR_COLUMNS):
            aux_geometry = _compute_eboss_geometry(aux)
            aux_core = aux_geometry["core_score"]
            aux_center = float(aux_core.median())
            aux_scale = float(aux_core.quantile(0.75) - aux_core.quantile(0.25))
            if not np.isfinite(aux_scale) or aux_scale == 0.0:
                aux_scale = 1.0
            features["core_score_aux_z"] = (core_score - aux_center) / aux_scale

    return features


FEATURE_GROUPS = [
    {
        "name": "eboss_ptf_c1_c3_geometry",
        "fn": add_eboss_ptf_c1_c3_geometry,
        "depends_on": [],
        "description": "Build eBOSS-style c1/c3 rotated color-geometry features with boundary distances, rejection-pocket metrics, and robust binning.",
    }
]