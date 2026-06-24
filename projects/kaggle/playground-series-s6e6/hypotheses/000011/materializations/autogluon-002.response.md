import pandas as pd

_G_PERP_OFFSET = 0.177
_G_PAR_SHIFT = 0.177
_CUT_I_R_BIAS = 13.116
_CUT_I_CPAR_SCALE = 0.3
_CUT_I_R_MIN = 19.2
_CUT_I_CPERP_TOL = 0.2
_CUT_II_CPERP_BIAS = 0.449
_CUT_II_GMR_SCALE = 6.0
_CUT_II_RMI_TERM = 0.25
_CUT_II_GMR_OFFSET = 1.296
_CUT_II_R_MIN = 19.5
_MARGIN_CLIP_BOUNDS = (-20.0, 20.0)


def add_lrg_target_cut_margins(raw, deps, aux):
    g = raw["g"].astype("float64")
    r = raw["r"].astype("float64")
    i = raw["i"].astype("float64")

    gmr = g - r
    rmi = r - i
    c_perp = rmi - gmr / 4.0 - _G_PERP_OFFSET
    c_par = 0.7 * gmr + 1.2 * (rmi - _G_PAR_SHIFT)

    vI1 = r - (_CUT_I_R_BIAS + c_par / _CUT_I_CPAR_SCALE)
    vI2 = r - _CUT_I_R_MIN
    vI3 = c_perp.abs() - _CUT_I_CPERP_TOL
    score_cutI_raw = pd.concat([vI1, vI2, vI3], axis=1).max(axis=1)
    score_cutI = score_cutI_raw.clip(*_MARGIN_CLIP_BOUNDS)

    vII1 = (_CUT_II_CPERP_BIAS - gmr / _CUT_II_GMR_SCALE) - c_perp
    vII2 = (_CUT_II_GMR_OFFSET + _CUT_II_RMI_TERM * rmi) - gmr
    vII3 = r - _CUT_II_R_MIN
    score_cutII_raw = pd.concat([vII1, vII2, vII3], axis=1).max(axis=1)
    score_cutII = score_cutII_raw.clip(*_MARGIN_CLIP_BOUNDS)

    score_lrg_any = pd.concat([score_cutI_raw, score_cutII_raw], axis=1).min(axis=1).clip(*_MARGIN_CLIP_BOUNDS)

    iI = (score_cutI_raw <= 0).astype("int8")
    iII = (score_cutII_raw <= 0).astype("int8")

    return pd.DataFrame(
        {
            "gmr": gmr,
            "rmi": rmi,
            "c_perp": c_perp,
            "c_par": c_par,
            "score_cutI": score_cutI,
            "score_cutII": score_cutII,
            "score_lrg_any": score_lrg_any,
            "iI": iI,
            "iII": iII,
        },
        index=raw.index,
    )


FEATURE_GROUPS = [
    {
        "name": "lrg_target_cut_margins",
        "fn": add_lrg_target_cut_margins,
        "depends_on": [],
        "description": "Project magnitudes into LRG color coordinates and emit signed cut-margin features for DR2/DR3 Cut I and Cut II membership geometry.",
    }
]