import numpy as np
import pandas as pd

_BANDS = ("u", "g", "r", "i", "z")
_ZEFF_FLOOR = 1e-4
_M_MIN = -35.0
_M_MAX = 10.0
_LUM_DIST_GRID_STEPS = 4000
_LUM_DIST_MAX_Z = 10.0
_LIGHT_SPEED_KM_S = 299792.458
_H0_KM_S_MPC = 70.0
_OMEGA_M = 0.30
_OMEGA_L = 0.70


def _luminosity_distance_mpc(redshift):
    z = np.asarray(redshift, dtype=np.float64)
    z = z.reshape(-1)
    z_eff = np.maximum(z, _ZEFF_FLOOR)

    z_grid = np.linspace(0.0, _LUM_DIST_MAX_Z, _LUM_DIST_GRID_STEPS + 1)
    e_inv = 1.0 / np.sqrt(_OMEGA_M * (1.0 + z_grid) ** 3 + _OMEGA_L)

    dz = z_grid[1] - z_grid[0]
    integral_grid = np.zeros_like(z_grid)
    integral_grid[1:] = np.cumsum((e_inv[:-1] + e_inv[1:]) * (0.5 * dz))

    z_lookup = np.minimum(z_eff, _LUM_DIST_MAX_Z)
    idx = np.searchsorted(z_grid, z_lookup, side="right") - 1
    idx = np.clip(idx, 0, _LUM_DIST_GRID_STEPS - 1)

    z0 = z_grid[idx]
    e0 = e_inv[idx]
    z1 = z_grid[idx + 1]
    e1 = e_inv[idx + 1]
    frac = (z_lookup - z0) / (z1 - z0)
    e_at_z = e0 + (e1 - e0) * frac

    integral = integral_grid[idx] + 0.5 * (e0 + e_at_z) * (z_lookup - z0)

    over_mask = z_eff > _LUM_DIST_MAX_Z
    if np.any(over_mask):
        z_tail = z_eff[over_mask]
        tail = (2.0 / np.sqrt(_OMEGA_M)) * (
            (1.0 + _LUM_DIST_MAX_Z) ** -0.5 - (1.0 + z_tail) ** -0.5
        )
        integral = integral.copy()
        integral[over_mask] = integral[over_mask] + tail

    d_c_mpc = (_LIGHT_SPEED_KM_S / _H0_KM_S_MPC) * integral
    d_l_mpc = (1.0 + z_eff) * d_c_mpc
    return d_l_mpc, z_eff


def add_redshift_luminosity_plausibility(raw, deps, aux):
    redshift = raw["redshift"].to_numpy(dtype=np.float64)
    d_l_mpc, z_eff = _luminosity_distance_mpc(redshift)

    mags_obs = raw.loc[:, list(_BANDS)].to_numpy(dtype=np.float64)
    distance_modulus = (5.0 * np.log10(d_l_mpc.reshape(-1, 1)) + 25.0)
    abs_mag = mags_obs - distance_modulus
    abs_mag = np.clip(abs_mag, _M_MIN, _M_MAX)

    m_u = abs_mag[:, 0]
    m_g = abs_mag[:, 1]
    m_r = abs_mag[:, 2]
    m_i = abs_mag[:, 3]
    m_z = abs_mag[:, 4]

    m_mean = np.nanmean(abs_mag, axis=1)
    m_median = np.nanmedian(abs_mag, axis=1)
    m_iqr = np.nanpercentile(abs_mag, 75, axis=1) - np.nanpercentile(abs_mag, 25, axis=1)
    m_sd = np.nanstd(abs_mag, axis=1)
    m_min = np.nanmin(abs_mag, axis=1)
    m_max = np.nanmax(abs_mag, axis=1)

    m_rg = m_r - m_g
    m_ri = m_r - m_i
    m_spread = m_max - m_min

    delta_star = m_r + 10.0
    delta_midgal = m_r + 18.0
    delta_brightgal = m_r + 23.0
    delta_qso = m_r + 27.0

    features = pd.DataFrame(
        {
            "z_nonpos": redshift <= 0.0,
            "z_low": redshift < 0.01,
            "z_hi": redshift >= 3.0,
            "z_eff": z_eff,
            "M_u": m_u,
            "M_g": m_g,
            "M_r": m_r,
            "M_i": m_i,
            "M_z": m_z,
            "M_mean": m_mean,
            "M_median": m_median,
            "M_iqr": m_iqr,
            "M_sd": m_sd,
            "M_min": m_min,
            "M_max": m_max,
            "M_spread": m_spread,
            "M_rg": m_rg,
            "M_ri": m_ri,
            "delta_star": delta_star,
            "delta_midgal": delta_midgal,
            "delta_brightgal": delta_brightgal,
            "delta_qso": delta_qso,
            "Mr_le_minus27": m_r <= -27.0,
            "Mr_between_minus27_and_minus23": (m_r > -27.0) & (m_r <= -23.0),
            "Mr_between_minus23_and_minus18": (m_r > -23.0) & (m_r <= -18.0),
            "Mr_between_minus18_and_minus10": (m_r > -18.0) & (m_r <= -10.0),
            "Mr_gt_minus10": m_r > -10.0,
            "Mr_x_z_eff": m_r * z_eff,
            "Mri_x_z_eff": m_ri * z_eff,
        },
        index=raw.index,
    )

    return features


FEATURE_GROUPS = [
    {
        "name": "redshift_luminosity_plausibility",
        "fn": add_redshift_luminosity_plausibility,
        "depends_on": [],
        "description": "Compute redshift-normalized absolute ugriz magnitudes and redshift-conditioned magnitude-regime features.",
    }
]