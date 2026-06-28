import numpy as np
import pandas as pd


SDSS_EFFECTIVE_WAVELENGTHS_ANGSTROM = (3551.0, 4686.0, 6165.0, 7481.0, 8931.0)


def add_redshifted_4000a_break_curvature(raw, deps, aux):
    lambda_u, lambda_g, lambda_r, lambda_i, lambda_z = SDSS_EFFECTIVE_WAVELENGTHS_ANGSTROM

    redshift = raw["redshift"].astype("float64")
    u_mag = raw["u"].astype("float64")
    g_mag = raw["g"].astype("float64")
    r_mag = raw["r"].astype("float64")
    i_mag = raw["i"].astype("float64")
    z_mag = raw["z"].astype("float64")

    lambda_break = 4000.0 * (1.0 + redshift)

    c_ug = u_mag - g_mag
    c_gr = g_mag - r_mag
    c_ri = r_mag - i_mag
    c_iz = i_mag - z_mag

    in_ug = (lambda_break >= lambda_u) & (lambda_break < lambda_g)
    in_gr = (lambda_break >= lambda_g) & (lambda_break < lambda_r)
    in_ri = (lambda_break >= lambda_r) & (lambda_break < lambda_i)
    in_iz = (lambda_break >= lambda_i) & (lambda_break < lambda_z)
    out_of_band = ~(in_ug | in_gr | in_ri | in_iz)

    p = pd.Series(0.0, index=raw.index, dtype="float64")
    p_ug = ((lambda_break - lambda_u) / (lambda_g - lambda_u)).clip(0.0, 1.0)
    p_gr = ((lambda_break - lambda_g) / (lambda_r - lambda_g)).clip(0.0, 1.0)
    p_ri = ((lambda_break - lambda_r) / (lambda_i - lambda_r)).clip(0.0, 1.0)
    p_iz = ((lambda_break - lambda_i) / (lambda_z - lambda_i)).clip(0.0, 1.0)

    p = p.mask(in_ug, p_ug)
    p = p.mask(in_gr, p_gr)
    p = p.mask(in_ri, p_ri)
    p = p.mask(in_iz, p_iz)

    w = 4.0 * p * (1.0 - p)

    jump_ug = c_ug - c_gr
    jump_gr = c_gr - 0.5 * (c_ug + c_ri)
    jump_ri = c_ri - 0.5 * (c_gr + c_iz)
    jump_iz = c_iz - c_ri

    curv_ug = c_ug - 2.0 * c_gr + c_ri
    curv_gr = c_ug - 2.0 * c_gr + c_ri
    curv_ri = c_gr - 2.0 * c_ri + c_iz
    curv_iz = c_gr - 2.0 * c_ri + c_iz

    return pd.DataFrame(
        {
            "break_position_p": p,
            "break_centrality_w": w,
            "in_ug": in_ug.astype("int8"),
            "in_gr": in_gr.astype("int8"),
            "in_ri": in_ri.astype("int8"),
            "in_iz": in_iz.astype("int8"),
            "out_of_band": out_of_band.astype("int8"),
            "ug_weighted_jump": in_ug.astype("float64") * w * jump_ug,
            "gr_weighted_jump": in_gr.astype("float64") * w * jump_gr,
            "ri_weighted_jump": in_ri.astype("float64") * w * jump_ri,
            "iz_weighted_jump": in_iz.astype("float64") * w * jump_iz,
            "ug_weighted_curvature": in_ug.astype("float64") * w * curv_ug,
            "gr_weighted_curvature": in_gr.astype("float64") * w * curv_gr,
            "ri_weighted_curvature": in_ri.astype("float64") * w * curv_ri,
            "iz_weighted_curvature": in_iz.astype("float64") * w * curv_iz,
        },
        index=raw.index,
    )


FEATURE_GROUPS = [
    {
        "name": "redshifted_4000a_break_curvature",
        "fn": add_redshifted_4000a_break_curvature,
        "depends_on": [],
        "description": "Encodes redshift-localized ugriz curvature around the observed 4000 Angstrom break.",
    }
]