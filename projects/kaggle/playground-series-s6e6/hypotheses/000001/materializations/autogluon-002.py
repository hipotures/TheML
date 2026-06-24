import numpy as np
import pandas as pd


def _finite_two(a, b):
    return np.isfinite(a) & np.isfinite(b)


def _finite_three(a, b, c):
    return np.isfinite(a) & np.isfinite(b) & np.isfinite(c)


def add_broadband_color_shape(raw, deps, aux):
    u = pd.to_numeric(raw["u"], errors="coerce")
    g = pd.to_numeric(raw["g"], errors="coerce")
    r = pd.to_numeric(raw["r"], errors="coerce")
    i = pd.to_numeric(raw["i"], errors="coerce")
    z = pd.to_numeric(raw["z"], errors="coerce")

    lambda_u, lambda_g, lambda_r, lambda_i, lambda_z = 3562.0, 4686.0, 6165.0, 7481.0, 8931.0

    ln_u = np.log(lambda_u)
    ln_g = np.log(lambda_g)
    ln_r = np.log(lambda_r)
    ln_i = np.log(lambda_i)
    ln_z = np.log(lambda_z)

    du = ln_g - ln_u
    dr = ln_r - ln_g
    di = ln_i - ln_r
    dz = ln_z - ln_i

    # Color descriptors
    color_ug = (u - g).where(_finite_two(u, g))
    color_gr = (g - r).where(_finite_two(g, r))
    color_ri = (r - i).where(_finite_two(r, i))
    color_iz = (i - z).where(_finite_two(i, z))
    color_ur = (u - r).where(_finite_two(u, r))
    color_gi = (g - i).where(_finite_two(g, i))
    color_gz = (g - z).where(_finite_two(g, z))
    color_uz = (u - z).where(_finite_two(u, z))

    # Wavelength-normalized adjacent slopes
    slope_ug = (u - g).where(_finite_two(u, g)) / du
    slope_gr = (g - r).where(_finite_two(g, r)) / dr
    slope_ri = (r - i).where(_finite_two(r, i)) / di
    slope_iz = (i - z).where(_finite_two(i, z)) / dz

    # Curvatures
    curvature_ugr = (u - 2.0 * g + r).where(_finite_three(u, g, r))
    curvature_gri = (g - 2.0 * r + i).where(_finite_three(g, r, i))
    curvature_riz = (r - 2.0 * i + z).where(_finite_three(r, i, z))
    curvature_uz = (u - 2.0 * r + z).where(_finite_three(u, r, z))

    # Slope-scale curvatures (double-span scaled)
    k_ugr = (slope_gr - slope_ug) / (0.5 * (ln_r - ln_u))
    k_ugr = k_ugr.where(_finite_three(u, g, r))

    k_gri = (slope_ri - slope_gr) / (0.5 * (ln_i - ln_g))
    k_gri = k_gri.where(_finite_three(g, r, i))

    k_riz = (slope_iz - slope_ri) / (0.5 * (ln_z - ln_r))
    k_riz = k_riz.where(_finite_three(r, i, z))

    return pd.DataFrame(
        {
            "color_ug": color_ug,
            "color_gr": color_gr,
            "color_ri": color_ri,
            "color_iz": color_iz,
            "color_ur": color_ur,
            "color_gi": color_gi,
            "color_gz": color_gz,
            "color_uz": color_uz,
            "slope_ug": slope_ug,
            "slope_gr": slope_gr,
            "slope_ri": slope_ri,
            "slope_iz": slope_iz,
            "curvature_ugr": curvature_ugr,
            "curvature_gri": curvature_gri,
            "curvature_riz": curvature_riz,
            "curvature_uz": curvature_uz,
            "k_ugr": k_ugr,
            "k_gri": k_gri,
            "k_riz": k_riz,
        },
        index=raw.index,
    )


FEATURE_GROUPS = [
    {
        "name": "broadband_color_shape",
        "fn": add_broadband_color_shape,
        "depends_on": [],
        "description": "Builds ugriz-based color, wavelength-normalized slope, and curvature descriptors to model optical SED shape independent of absolute flux level.",
    },
]