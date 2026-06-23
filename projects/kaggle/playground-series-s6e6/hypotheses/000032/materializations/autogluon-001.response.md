import pandas as pd

LAMBDA_U = 3562.0
LAMBDA_G = 4686.0
LAMBDA_R = 6165.0
LAMBDA_I = 7481.0
LAMBDA_Z = 8931.0
REDSHIFT_OUT_OF_BAND_MIN = 0.0
REDSHIFT_OUT_OF_BAND_MAX = 1.23


def add_redshifted_4000a_break_curvature(raw, deps, aux):
    idx = raw.index

    redshift = pd.to_numeric(raw["redshift"], errors="coerce")
    u = pd.to_numeric(raw["u"], errors="coerce")
    g = pd.to_numeric(raw["g"], errors="coerce")
    r = pd.to_numeric(raw["r"], errors="coerce")
    i = pd.to_numeric(raw["i"], errors="coerce")
    z = pd.to_numeric(raw["z"], errors="coerce")

    lambda_break = 4000.0 * (1.0 + redshift)

    regime_g = (lambda_break >= LAMBDA_U) & (lambda_break < LAMBDA_G)
    regime_gr = (lambda_break >= LAMBDA_G) & (lambda_break < LAMBDA_R)
    regime_ri = (lambda_break >= LAMBDA_R) & (lambda_break < LAMBDA_I)
    regime_iz = (lambda_break >= LAMBDA_I) & (lambda_break < LAMBDA_Z)

    out_of_band = (redshift < REDSHIFT_OUT_OF_BAND_MIN) | (redshift >= REDSHIFT_OUT_OF_BAND_MAX)
    active_g = regime_g & (~out_of_band)
    active_gr = regime_gr & (~out_of_band)
    active_ri = regime_ri & (~out_of_band)
    active_iz = regime_iz & (~out_of_band)

    p_g = ((lambda_break - LAMBDA_U) / (LAMBDA_G - LAMBDA_U)).clip(0.0, 1.0)
    p_gr = ((lambda_break - LAMBDA_G) / (LAMBDA_R - LAMBDA_G)).clip(0.0, 1.0)
    p_ri = ((lambda_break - LAMBDA_R) / (LAMBDA_I - LAMBDA_R)).clip(0.0, 1.0)
    p_iz = ((lambda_break - LAMBDA_I) / (LAMBDA_Z - LAMBDA_I)).clip(0.0, 1.0)

    w_g = 4.0 * p_g * (1.0 - p_g)
    w_gr = 4.0 * p_gr * (1.0 - p_gr)
    w_ri = 4.0 * p_ri * (1.0 - p_ri)
    w_iz = 4.0 * p_iz * (1.0 - p_iz)

    p_g = p_g.where(active_g, 0.0)
    p_gr = p_gr.where(active_gr, 0.0)
    p_ri = p_ri.where(active_ri, 0.0)
    p_iz = p_iz.where(active_iz, 0.0)

    c1 = u - g
    c2 = g - r
    c3 = r - i
    c4 = i - z

    break_g = (c1) - 0.5 * (c2 + c1)
    asym_g = (c1 - c2) - (c2 - c3)

    break_gr = c2 - 0.5 * (c1 + c3)
    asym_gr = (c2 - c1) - (c3 - c2)

    break_ri = c3 - 0.5 * (c2 + c4)
    asym_ri = (c3 - c2) - (c4 - c3)

    break_iz = c4 - c3
    asym_iz = c4 - c3

    weighted_break_g = w_g * break_g * active_g.astype("float64")
    weighted_asym_g = w_g * asym_g * active_g.astype("float64")

    weighted_break_gr = w_gr * break_gr * active_gr.astype("float64")
    weighted_asym_gr = w_gr * asym_gr * active_gr.astype("float64")

    weighted_break_ri = w_ri * break_ri * active_ri.astype("float64")
    weighted_asym_ri = w_ri * asym_ri * active_ri.astype("float64")

    weighted_break_iz = w_iz * break_iz * active_iz.astype("float64")
    weighted_asym_iz = w_iz * asym_iz * active_iz.astype("float64")

    new_features = pd.DataFrame(
        {
            "out_of_band": out_of_band.astype("int8"),
            "regime_g": active_g.astype("int8"),
            "regime_gr": active_gr.astype("int8"),
            "regime_ri": active_ri.astype("int8"),
            "regime_iz": active_iz.astype("int8"),
            "p_g": p_g,
            "p_gr": p_gr,
            "p_ri": p_ri,
            "p_iz": p_iz,
            "break_excess_g": weighted_break_g,
            "break_asym_g": weighted_asym_g,
            "break_excess_gr": weighted_break_gr,
            "break_asym_gr": weighted_asym_gr,
            "break_excess_ri": weighted_break_ri,
            "break_asym_ri": weighted_asym_ri,
            "break_excess_iz": weighted_break_iz,
            "break_asym_iz": weighted_asym_iz,
        },
        index=idx,
    )
    return new_features


FEATURE_GROUPS = [
    {
        "name": "redshifted_4000a_break_curvature",
        "fn": add_redshifted_4000a_break_curvature,
        "depends_on": [],
        "description": "Projects the 4000A break into ugriz bands by redshift and creates regime-gated, edge-weighted curvature and asymmetry features plus regime position/flags.",
    },
]