import numpy as np
import pandas as pd

_REST_BREAK_WAVELENGTH_A = 4000.0

_G_LEFT = 3551.0
_G_RIGHT = 4686.0
_GR_LEFT = 4686.0
_GR_RIGHT = 6165.0
_RI_LEFT = 6165.0
_RI_RIGHT = 7481.0
_IZ_LEFT = 7481.0
_IZ_RIGHT = 8931.0

_G_WIDTH = 1135.0
_GR_WIDTH = 1479.0
_RI_WIDTH = 1316.0
_IZ_WIDTH = 1450.0


def add_redshifted_4000a_break_curvature(raw, deps, aux):
    redshift = raw["redshift"].to_numpy(dtype=float)
    lambda_break = _REST_BREAK_WAVELENGTH_A * (1.0 + redshift)

    u = raw["u"].to_numpy(dtype=float)
    g = raw["g"].to_numpy(dtype=float)
    r = raw["r"].to_numpy(dtype=float)
    i = raw["i"].to_numpy(dtype=float)
    z = raw["z"].to_numpy(dtype=float)

    c1 = u - g
    c2 = g - r
    c3 = r - i
    c4 = i - z

    regime_G = (lambda_break >= _G_LEFT) & (lambda_break < _G_RIGHT)
    regime_GR = (lambda_break >= _GR_LEFT) & (lambda_break < _GR_RIGHT)
    regime_RI = (lambda_break >= _RI_LEFT) & (lambda_break < _RI_RIGHT)
    regime_IZ = (lambda_break >= _IZ_LEFT) & (lambda_break < _IZ_RIGHT)
    regime_out_of_band = ~(regime_G | regime_GR | regime_RI | regime_IZ)

    p = np.zeros_like(lambda_break, dtype=float)
    w = np.zeros_like(lambda_break, dtype=float)

    if regime_G.any():
        p_g = (lambda_break[regime_G] - _G_LEFT) / _G_WIDTH
        p[regime_G] = np.clip(p_g, 0.0, 1.0)
        w[regime_G] = 4.0 * p[regime_G] * (1.0 - p[regime_G])

    if regime_GR.any():
        p_gr = (lambda_break[regime_GR] - _GR_LEFT) / _GR_WIDTH
        p[regime_GR] = np.clip(p_gr, 0.0, 1.0)
        w[regime_GR] = 4.0 * p[regime_GR] * (1.0 - p[regime_GR])

    if regime_RI.any():
        p_ri = (lambda_break[regime_RI] - _RI_LEFT) / _RI_WIDTH
        p[regime_RI] = np.clip(p_ri, 0.0, 1.0)
        w[regime_RI] = 4.0 * p[regime_RI] * (1.0 - p[regime_RI])

    if regime_IZ.any():
        p_iz = (lambda_break[regime_IZ] - _IZ_LEFT) / _IZ_WIDTH
        p[regime_IZ] = np.clip(p_iz, 0.0, 1.0)
        w[regime_IZ] = 4.0 * p[regime_IZ] * (1.0 - p[regime_IZ])

    f_G = regime_G.astype(float)
    f_GR = regime_GR.astype(float)
    f_RI = regime_RI.astype(float)
    f_IZ = regime_IZ.astype(float)
    f_OOB = regime_out_of_band.astype(float)

    jump_G = c1 - c2
    asym_G = (c1 - c2) - (c2 - c3)

    jump_GR = c2 - 0.5 * (c1 + c3)
    asym_GR = (c2 - c1) - (c3 - c2)

    jump_RI = c3 - 0.5 * (c2 + c4)
    asym_RI = (c3 - c2) - (c4 - c3)

    jump_IZ = c4 - c3
    asym_IZ = (c4 - c3) - (c3 - c2)

    features = pd.DataFrame(
        {
            "weighted_jump_G": w * jump_G * f_G,
            "weighted_asym_G": w * asym_G * f_G,
            "weighted_jump_GR": w * jump_GR * f_GR,
            "weighted_asym_GR": w * asym_GR * f_GR,
            "weighted_jump_RI": w * jump_RI * f_RI,
            "weighted_asym_RI": w * asym_RI * f_RI,
            "weighted_jump_IZ": w * jump_IZ * f_IZ,
            "weighted_asym_IZ": w * asym_IZ * f_IZ,
            "p": p,
            "w": w,
            "regime_G": f_G,
            "regime_GR": f_GR,
            "regime_RI": f_RI,
            "regime_IZ": f_IZ,
            "regime_out_of_band": f_OOB,
        },
        index=raw.index,
    )
    return features


FEATURE_GROUPS = [
    {
        "name": "redshifted_4000a_break_curvature",
        "fn": add_redshifted_4000a_break_curvature,
        "depends_on": [],
        "description": "Encodes redshift-localized 4000 Å break geometry through weighted curvature/jump features and regime flags across ugriz bands.",
    }
]