import numpy as np
import pandas as pd

BANDS = ("u", "g", "r", "i", "z")
H0 = 70.0
C_KM_S = 299792.458
Q0 = -0.55
REDSHIFT_MIN = 1.0e-4
REDSHIFT_MAX = 7.5
DL_MIN_MPC = 1.0e-4
DL_MAX_MPC = 1.0e8
MEDIAN_REGIME_LABELS = (
    "le_neg27",
    "neg27_to_neg23",
    "neg23_to_neg19",
    "neg19_to_neg14",
    "gt_neg14",
)


def add_distance_corrected_luminosity_signature(raw, deps, aux):
    redshift = pd.to_numeric(raw["redshift"], errors="coerce").fillna(0.0)
    z_eff = redshift.clip(lower=REDSHIFT_MIN, upper=REDSHIFT_MAX)

    d_l_mpc = (C_KM_S / H0) * z_eff * (1.0 + 0.5 * (1.0 - Q0) * z_eff)
    d_l_mpc = d_l_mpc.clip(lower=DL_MIN_MPC, upper=DL_MAX_MPC)
    distance_modulus = 5.0 * np.log10(d_l_mpc) + 25.0

    pseudo_abs = pd.DataFrame(index=raw.index)
    for band in BANDS:
        mag = pd.to_numeric(raw[band], errors="coerce")
        pseudo_abs[band] = mag - distance_modulus

    features = pd.DataFrame(index=raw.index)
    features["pseudo_abs_mag_available_band_count"] = pseudo_abs.notna().sum(axis=1).astype("int8")

    features["pseudo_abs_mag_median"] = pseudo_abs.median(axis=1, skipna=True)
    features["pseudo_abs_mag_mean"] = pseudo_abs.mean(axis=1, skipna=True)
    features["pseudo_abs_mag_min"] = pseudo_abs.min(axis=1, skipna=True)
    features["pseudo_abs_mag_max"] = pseudo_abs.max(axis=1, skipna=True)
    features["pseudo_abs_mag_p25"] = pseudo_abs.quantile(0.25, axis=1, interpolation="linear")
    features["pseudo_abs_mag_p75"] = pseudo_abs.quantile(0.75, axis=1, interpolation="linear")
    features["pseudo_abs_mag_iqr"] = features["pseudo_abs_mag_p75"] - features["pseudo_abs_mag_p25"]
    features["pseudo_abs_mag_span"] = features["pseudo_abs_mag_max"] - features["pseudo_abs_mag_min"]

    has_any_band = features["pseudo_abs_mag_available_band_count"] > 0
    brightest = pseudo_abs.idxmin(axis=1, skipna=True).where(has_any_band, "missing")
    faintest = pseudo_abs.idxmax(axis=1, skipna=True).where(has_any_band, "missing")
    features["pseudo_abs_mag_brightest_band"] = brightest.astype("string")
    features["pseudo_abs_mag_faintest_band"] = faintest.astype("string")

    median_mag = features["pseudo_abs_mag_median"]
    regime = pd.Series("missing", index=raw.index, dtype="string")
    regime = regime.mask(median_mag <= -27.0, MEDIAN_REGIME_LABELS[0])
    regime = regime.mask((median_mag > -27.0) & (median_mag <= -23.0), MEDIAN_REGIME_LABELS[1])
    regime = regime.mask((median_mag > -23.0) & (median_mag <= -19.0), MEDIAN_REGIME_LABELS[2])
    regime = regime.mask((median_mag > -19.0) & (median_mag <= -14.0), MEDIAN_REGIME_LABELS[3])
    regime = regime.mask(median_mag > -14.0, MEDIAN_REGIME_LABELS[4])
    features["pseudo_abs_mag_median_regime"] = regime

    return features


FEATURE_GROUPS = [
    {
        "name": "distance_corrected_luminosity_signature",
        "fn": add_distance_corrected_luminosity_signature,
        "depends_on": [],
        "description": "Redshift-calibrated pseudo-absolute magnitude summaries and categorical luminosity-regime indicators.",
    }
]