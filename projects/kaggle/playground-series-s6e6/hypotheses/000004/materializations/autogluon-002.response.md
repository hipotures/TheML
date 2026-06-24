import numpy as np
import pandas as pd

_H0 = 70.0
_LIGHT_SPEED_KMS = 299792.458
_Q0 = -0.55
_Z_SAFE_MIN = 1.0e-4
_D_L_MIN = 1.0e-4
_D_L_MAX = 1.0e8
_PHOTOMETRIC_BANDS = ("u", "g", "r", "i", "z")
_REGIME_EDGES = (-27.0, -23.0, -19.0, -14.0)
_REGIME_LABELS = ("<=-27", "(-27,-23]", "(-23,-19]", "(-19,-14]", ">-14")


def _compute_luminosity_distance_mpc(redshift_series):
    redshift = redshift_series.fillna(0.0).astype(float)
    z_safe = redshift.clip(lower=_Z_SAFE_MIN)
    distance = (( _LIGHT_SPEED_KMS / _H0)
                * (1.0 + z_safe)
                * (z_safe + 0.5 * (1.0 - _Q0) * (z_safe ** 2)))
    return distance.clip(lower=_D_L_MIN, upper=_D_L_MAX)


def _luminosity_regime_from_median(median_abs_mag):
    regime = pd.Series(np.nan, index=median_abs_mag.index, dtype=object)
    regime.loc[median_abs_mag <= _REGIME_EDGES[0]] = _REGIME_LABELS[0]
    regime.loc[(median_abs_mag > _REGIME_EDGES[0]) & (median_abs_mag <= _REGIME_EDGES[1])] = _REGIME_LABELS[1]
    regime.loc[(median_abs_mag > _REGIME_EDGES[1]) & (median_abs_mag <= _REGIME_EDGES[2])] = _REGIME_LABELS[2]
    regime.loc[(median_abs_mag > _REGIME_EDGES[2]) & (median_abs_mag <= _REGIME_EDGES[3])] = _REGIME_LABELS[3]
    regime.loc[median_abs_mag > _REGIME_EDGES[3]] = _REGIME_LABELS[4]
    return regime


def add_distance_corrected_luminosity_signature(raw, deps, aux):
    # No dependency or auxiliary usage by design for this group; keep references explicit.
    _ = deps, aux

    present_redshift = raw["redshift"] if "redshift" in raw.columns else pd.Series(0.0, index=raw.index)
    luminosity_distance_mpc = _compute_luminosity_distance_mpc(present_redshift.astype(float))
    distance_modulus = 5.0 * np.log10(luminosity_distance_mpc) + 25.0

    available_bands = [band for band in _PHOTOMETRIC_BANDS if band in raw.columns]
    if available_bands:
        magnitudes = raw.loc[:, available_bands].apply(pd.to_numeric, errors="coerce")
        abs_magnitudes = magnitudes.sub(distance_modulus, axis=0)
        band_count = abs_magnitudes.notna().sum(axis=1).astype(int)
        median_abs = abs_magnitudes.median(axis=1, skipna=True)
        min_abs = abs_magnitudes.min(axis=1, skipna=True)
        max_abs = abs_magnitudes.max(axis=1, skipna=True)
        q25 = abs_magnitudes.quantile(0.25, axis=1)
        q75 = abs_magnitudes.quantile(0.75, axis=1)
    else:
        band_count = pd.Series(0, index=raw.index, dtype=int)
        median_abs = pd.Series(np.nan, index=raw.index, dtype=float)
        min_abs = pd.Series(np.nan, index=raw.index, dtype=float)
        max_abs = pd.Series(np.nan, index=raw.index, dtype=float)
        q25 = pd.Series(np.nan, index=raw.index, dtype=float)
        q75 = pd.Series(np.nan, index=raw.index, dtype=float)

    iqr = q75 - q25
    span = max_abs - min_abs
    median_regime = _luminosity_regime_from_median(median_abs)
    fallback_used = band_count < len(_PHOTOMETRIC_BANDS)

    return pd.DataFrame(
        {
            "pseudo_abs_mag_median": median_abs,
            "pseudo_abs_mag_min": min_abs,
            "pseudo_abs_mag_max": max_abs,
            "pseudo_abs_mag_p25": q25,
            "pseudo_abs_mag_p75": q75,
            "pseudo_abs_mag_iqr": iqr,
            "pseudo_abs_mag_span": span,
            "pseudo_abs_mag_band_count": band_count,
            "pseudo_abs_mag_regime": median_regime,
            "pseudo_abs_mag_fallback": fallback_used,
        },
        index=raw.index,
    )


FEATURE_GROUPS = [
    {
        "name": "distance_corrected_luminosity_signature",
        "fn": add_distance_corrected_luminosity_signature,
        "depends_on": [],
        "description": "Compute redshift-calibrated pseudo-absolute magnitudes per band and add robust summary/range descriptors with physically motivated median regimes.",
    }
]