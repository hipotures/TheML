import numpy as np
import pandas as pd


GALAXY_R_THRESHOLDS = (17.77, 19.2, 19.5)
QSO_THRESHOLDS = (
    ("i", 15.0),
    ("i", 19.1),
    ("i", 20.2),
    ("i", 20.4),
    ("i", 21.3),
    ("i", 22.45),
    ("g", 22.0),
    ("r", 21.85),
)
INTERVALS = (
    ("i", 15.0, 19.1),
    ("i", 15.0, 20.2),
    ("i", 20.2, 21.3),
    ("i", 21.3, 22.45),
    ("r", 17.77, 19.2),
    ("r", 19.2, 19.5),
    ("r", 17.77, 19.5),
)


def _label(value):
    return str(value).replace(".", "_").replace("-", "neg_")


def _numeric_series(raw, column):
    if column in raw.columns:
        return pd.to_numeric(raw[column], errors="coerce")
    return pd.Series(np.nan, index=raw.index, dtype="float64")


def _valid_mask(values):
    return np.isfinite(values.to_numpy(dtype="float64", copy=False))


def _signed_margin(values, valid, threshold):
    arr = values.to_numpy(dtype="float64", copy=False) - threshold
    return np.where(valid, arr, 0.0)


def _interval_signed_margin(values, valid, lower, upper):
    arr = values.to_numpy(dtype="float64", copy=False)
    margin = np.where(arr < lower, arr - lower, np.where(arr > upper, arr - upper, 0.0))
    return np.where(valid, margin, 0.0)


def add_survey_depth_limit_margins(raw, deps, aux):
    index = raw.index
    features = {}

    g = _numeric_series(raw, "g")
    r = _numeric_series(raw, "r")
    i = _numeric_series(raw, "i")
    band_values = {"g": g, "r": r, "i": i}
    band_valid = {band: _valid_mask(values) for band, values in band_values.items()}

    for threshold in GALAXY_R_THRESHOLDS:
        threshold_name = _label(threshold)
        features[f"signed_margin_r_{threshold_name}"] = _signed_margin(
            r, band_valid["r"], threshold
        )

    for band, threshold in QSO_THRESHOLDS:
        threshold_name = _label(threshold)
        features[f"signed_margin_{band}_{threshold_name}"] = _signed_margin(
            band_values[band], band_valid[band], threshold
        )

    for band, lower, upper in INTERVALS:
        lower_name = _label(lower)
        upper_name = _label(upper)
        values = band_values[band]
        valid = band_valid[band]
        arr = values.to_numpy(dtype="float64", copy=False)

        inside = valid & (arr >= lower) & (arr <= upper)
        signed = _interval_signed_margin(values, valid, lower, upper)

        features[f"inside_{band}_{lower_name}_{upper_name}"] = inside.astype("int8")
        features[f"signed_interval_margin_{band}_{lower_name}_{upper_name}"] = signed
        features[f"abs_interval_margin_{band}_{lower_name}_{upper_name}"] = np.abs(signed)

    qso_abs_margins = []
    qso_signed_margins = []
    qso_names = []
    for band, threshold in QSO_THRESHOLDS:
        name = f"{band}_{_label(threshold)}"
        signed = _signed_margin(band_values[band], band_valid[band], threshold)
        abs_margin = np.where(band_valid[band], np.abs(signed), np.inf)
        qso_names.append(name)
        qso_signed_margins.append(signed)
        qso_abs_margins.append(abs_margin)

    qso_abs = np.column_stack(qso_abs_margins)
    qso_signed = np.column_stack(qso_signed_margins)
    qso_nearest_idx = np.argmin(qso_abs, axis=1)
    qso_has_valid = np.isfinite(qso_abs).any(axis=1)
    row_idx = np.arange(len(index))

    nearest_qso_abs = np.where(qso_has_valid, qso_abs[row_idx, qso_nearest_idx], 0.0)
    nearest_qso_signed = np.where(
        qso_has_valid, qso_signed[row_idx, qso_nearest_idx], 0.0
    )

    features["nearest_qso_abs_margin"] = nearest_qso_abs
    features["nearest_qso_signed_margin"] = nearest_qso_signed

    for position, name in enumerate(qso_names):
        features[f"nearest_qso_is_{name}"] = (
            qso_has_valid & (qso_nearest_idx == position)
        ).astype("int8")

    gal_abs_margins = []
    gal_signed_margins = []
    gal_names = []
    for threshold in GALAXY_R_THRESHOLDS:
        name = f"r_{_label(threshold)}"
        signed = _signed_margin(r, band_valid["r"], threshold)
        abs_margin = np.where(band_valid["r"], np.abs(signed), np.inf)
        gal_names.append(name)
        gal_signed_margins.append(signed)
        gal_abs_margins.append(abs_margin)

    gal_abs = np.column_stack(gal_abs_margins)
    gal_signed = np.column_stack(gal_signed_margins)
    gal_nearest_idx = np.argmin(gal_abs, axis=1)
    gal_has_valid = np.isfinite(gal_abs).any(axis=1)

    nearest_gal_abs = np.where(gal_has_valid, gal_abs[row_idx, gal_nearest_idx], 0.0)
    nearest_gal_signed = np.where(
        gal_has_valid, gal_signed[row_idx, gal_nearest_idx], 0.0
    )

    features["nearest_gal_abs_margin"] = nearest_gal_abs
    features["nearest_gal_signed_margin"] = nearest_gal_signed
    features["qso_minus_gal_boundary_distance"] = nearest_qso_abs - nearest_gal_abs

    for position, name in enumerate(gal_names):
        features[f"nearest_gal_is_{name}"] = (
            gal_has_valid & (gal_nearest_idx == position)
        ).astype("int8")

    return pd.DataFrame(features, index=index)


FEATURE_GROUPS = [
    {
        "name": "survey_depth_limit_margins",
        "fn": add_survey_depth_limit_margins,
        "depends_on": [],
        "description": "Encodes g, r, and i magnitude proximity to SDSS-style galaxy and quasar depth boundaries.",
    }
]