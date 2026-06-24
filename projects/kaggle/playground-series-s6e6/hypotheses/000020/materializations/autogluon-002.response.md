import numpy as np
import pandas as pd


_QSO_THRESHOLDS = (
    ("i", (15.0, 19.1, 20.2, 20.4, 21.3, 22.45)),
    ("g", (22.0,)),
    ("r", (21.85,)),
)
_GALAXY_R_THRESHOLDS = (17.77, 19.2, 19.5)
_I_INTERVALS = (
    (15.0, 19.1),
    (15.0, 20.2),
    (20.2, 21.3),
    (21.3, 22.45),
)
_R_INTERVALS = (
    (17.77, 19.2),
    (19.2, 19.5),
)


def _threshold_token(value):
    return str(value).replace(".", "_")


def _as_float_array(values):
    return pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)


def _describe_regime(margin_matrix, band_names):
    abs_matrix = np.abs(np.where(np.isnan(margin_matrix), np.inf, margin_matrix))
    min_abs = np.min(abs_matrix, axis=1)
    min_abs = np.where(np.isfinite(min_abs), min_abs, 0.0)

    nearest_idx = np.argmin(abs_matrix, axis=1)
    valid = np.isfinite(min_abs)

    nearest_band = np.array(["none"] * len(margin_matrix), dtype=object)
    nearest_sign = np.zeros(len(margin_matrix), dtype=np.int8)

    if np.any(valid):
        name_lookup = np.array(band_names, dtype=object)
        nearest_band[valid] = name_lookup[nearest_idx[valid]]
        nearest_margin = margin_matrix[np.arange(len(margin_matrix)), nearest_idx]
        nearest_sign[valid] = np.sign(nearest_margin[valid]).astype(np.int8)

    return min_abs, nearest_band, nearest_sign


def add_survey_depth_limit_margins(raw, deps, aux):
    del deps, aux

    i = _as_float_array(raw["i"])
    g = _as_float_array(raw["g"])
    r = _as_float_array(raw["r"])

    i_finite = np.isfinite(i)
    g_finite = np.isfinite(g)
    r_finite = np.isfinite(r)

    features = {}

    for threshold in _GALAXY_R_THRESHOLDS:
        label = _threshold_token(threshold)
        margin = r - float(threshold)
        features[f"margin_r_{label}"] = np.where(r_finite, margin, 0.0)

    for threshold in _QSO_THRESHOLDS[0][1]:
        label = _threshold_token(threshold)
        margin = i - float(threshold)
        features[f"margin_i_{label}"] = np.where(i_finite, margin, 0.0)

    for threshold in _QSO_THRESHOLDS[1][1]:
        label = _threshold_token(threshold)
        margin = g - float(threshold)
        features[f"margin_g_{label}"] = np.where(g_finite, margin, 0.0)

    for threshold in _QSO_THRESHOLDS[2][1]:
        label = _threshold_token(threshold)
        margin = r - float(threshold)
        features[f"margin_r_qso_{label}"] = np.where(r_finite, margin, 0.0)

    n = len(raw)

    for lo, hi in _I_INTERVALS:
        lo_f = float(lo)
        hi_f = float(hi)
        token_lo = _threshold_token(lo_f)
        token_hi = _threshold_token(hi_f)
        in_interval = (i_finite & (i >= lo_f) & (i <= hi_f)).astype(np.int8)

        dist = np.zeros(n, dtype=float)
        below = i_finite & (i < lo_f)
        above = i_finite & (i > hi_f)
        dist[below] = i[below] - lo_f
        dist[above] = i[above] - hi_f

        features[f"in_interval_i_{token_lo}_{token_hi}"] = in_interval
        features[f"dist_to_interval_i_{token_lo}_{token_hi}"] = np.where(i_finite, dist, 0.0)

    for lo, hi in __R_INTERVALS:
        lo_f = float(lo)
        hi_f = float(hi)
        token_lo = _threshold_token(lo_f)
        token_hi = _threshold_token(hi_f)
        in_interval = (r_finite & (r >= lo_f) & (r <= hi_f)).astype(np.int8)

        dist = np.zeros(n, dtype=float)
        below = r_finite & (r < lo_f)
        above = r_finite & (r > hi_f)
        dist[below] = r[below] - lo_f
        dist[above] = r[above] - hi_f

        features[f"in_interval_r_{token_lo}_{token_hi}"] = in_interval
        features[f"dist_to_interval_r_{token_lo}_{token_hi}"] = np.where(r_finite, dist, 0.0)

    qso_margins = []
    qso_names = []
    for band, thresholds in _QSO_THRESHOLDS:
        if band == "i":
            arr = i
            finite = i_finite
            band_prefix = "i"
        elif band == "g":
            arr = g
            finite = g_finite
            band_prefix = "g"
        else:
            arr = r
            finite = r_finite
            band_prefix = "r"

        for threshold in thresholds:
            token = _threshold_token(threshold)
            qso_margins.append(np.where(finite, arr - float(threshold), np.nan))
            qso_names.append(f"{band_prefix}_{token}")

    d_qso, nearest_qso_band, nearest_qso_sign = _describe_regime(np.column_stack(qso_margins), qso_names)
    features["d_qso"] = d_qso
    features["nearest_qso_band"] = nearest_qso_band
    features["nearest_qso_margin_sign"] = nearest_qso_sign

    gal_margins = []
    gal_names = []
    for threshold in _GALAXY_R_THRESHOLDS:
        token = _threshold_token(threshold)
        gal_margins.append(np.where(r_finite, r - float(threshold), np.nan))
        gal_names.append(f"r_{token}")

    d_gal, nearest_gal_band, nearest_gal_sign = _describe_regime(np.column_stack(gal_margins), gal_names)
    features["d_gal"] = d_gal
    features["nearest_gal_band"] = nearest_gal_band
    features["nearest_gal_margin_sign"] = nearest_gal_sign

    features["boundary_gap"] = d_gal - d_qso

    return pd.DataFrame(features, index=raw.index)


FEATURE_GROUPS = [
    {
        "name": "survey_depth_limit_margins",
        "fn": add_survey_depth_limit_margins,
        "depends_on": [],
        "description": "Builds signed SDSS targeting-boundary margin, interval-membership, and boundary-proximity descriptors in ugriz space for selection-depth geometry.",
    }
]