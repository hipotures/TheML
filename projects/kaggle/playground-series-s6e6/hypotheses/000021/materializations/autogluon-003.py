import numpy as np
import pandas as pd

_EPS = 1e-6
_LM_SCALE = -0.9210340371976183
_BANDS = ("u", "g", "r", "i", "z")
_COLOR_PAIRS = (("u", "g"), ("g", "r"), ("r", "i"), ("i", "z"), ("u", "r"), ("u", "i"), ("u", "z"), ("g", "z"), ("r", "z"))
_ZR_MAX = 1_000_000.0


def _to_float(series):
    return pd.to_numeric(series, errors="coerce").astype("float64")


def _finite(series):
    return series.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _dependency_columns(deps):
    cols = set()
    if deps:
        for dep_df in deps.values():
            if isinstance(dep_df, pd.DataFrame):
                cols.update(dep_df.columns)
    return cols


def add_aide_broadband_flux_ratios(raw, deps, aux):
    idx = raw.index
    existing = _dependency_columns(deps)
    features = {}
    lf = {}

    # Bandwise magnitude -> linearized magnitude and pseudo-flux features.
    for band in _BANDS:
        m = _to_float(raw[band])
        lm = (_LM_SCALE * m).clip(lower=-20.0, upper=10.0)
        lf_band = np.exp(lm)
        features[f"lm_{band}"] = _finite(lm)
        features[f"lf_{band}"] = _finite(lf_band)
        lf[band] = _finite(lf_band)

    # Pairwise first-order color geometry features.
    for band_a, band_b in _COLOR_PAIRS:
        a = _to_float(raw[band_a])
        b = _to_float(raw[band_b])

        color_gap = (a - b).abs()
        denom = b.abs().clip(lower=_EPS)
        ratio_norm = (a - b) / denom
        safe_mag_ratio = a / denom

        gap_name = f"color_gap_{band_a}_{band_b}"
        ratio_name = f"ratio_norm_{band_a}_{band_b}"
        safe_name = f"safe_mag_ratio_{band_a}_{band_b}"

        if gap_name not in existing:
            features[gap_name] = _finite(color_gap)
        if ratio_name not in existing:
            features[ratio_name] = _finite(ratio_norm)
        if safe_name not in existing:
            features[safe_name] = _finite(safe_mag_ratio)

    # Curvature terms.
    u = _to_float(raw["u"])
    g = _to_float(raw["g"])
    r = _to_float(raw["r"])
    i = _to_float(raw["i"])
    z = _to_float(raw["z"])

    features["c1"] = _finite(u - 2.0 * r + i)
    features["c2"] = _finite(g - 2.0 * i + z)

    # Redshift polynomial and controlled interactions.
    redshift = _to_float(raw["redshift"])
    zr = redshift.clip(lower=0.0)
    z1 = np.log1p(zr.clip(upper=_ZR_MAX))
    z2 = np.square(zr)
    z3 = np.power(zr, 3.0)

    features["z1"] = _finite(z1)
    features["z2"] = _finite(z2)
    features["z3"] = _finite(z3)

    features["z1_x_lf_u"] = _finite(z1 * lf["u"])
    features["z1_x_lf_g"] = _finite(z1 * lf["g"])
    features["z2_x_ri"] = _finite(z2 * (r - i))
    features["z3_x_gz"] = _finite(z3 * (g - z))

    return pd.DataFrame(features, index=idx)


FEATURE_GROUPS = [
    {
        "name": "aide_broadband_flux_ratios",
        "fn": add_aide_broadband_flux_ratios,
        "depends_on": [],
        "description": "Build clipped flux-domain, pairwise color, curvature, and redshift-interaction features from u,g,r,i,z magnitudes and redshift.",
    }
]