import numpy as np
import pandas as pd

_COSMO_H0 = 70.0
_COSMO_OMEGA_M = 0.3
_COSMO_OMEGA_L = 0.7
_COSMO_C_KM_S = 299792.458
_COSMO_Z_MIN = 1e-4
_COSMO_GRID_POINTS = 2048
_COSMO_DL_MIN_MPC = 1.0e-4
_COSMO_DL_MAX_MPC = 1.0e7
_COSMO_ABS_MAG_MIN = -90.0
_COSMO_ABS_MAG_MAX = 80.0
_BANDS = ("u", "g", "r", "i", "z")
_THRESHOLDS_M = (-10.0, -18.0, -23.0, -27.0)


def _luminosity_distance_mpc(redshift_for_distance):
    redshift_for_distance = np.asarray(redshift_for_distance, dtype=np.float64)
    if redshift_for_distance.size == 0:
        return redshift_for_distance.copy()

    z_max = float(np.max(redshift_for_distance))
    if not np.isfinite(z_max):
        z_max = float(_COSMO_Z_MIN)

    # Build a fixed one-dimensional cosmology integral grid and re-use it for all rows.
    z_grid = np.linspace(0.0, z_max, num=_COSMO_GRID_POINTS)
    hz = np.sqrt(_COSMO_OMEGA_M * np.power(1.0 + z_grid, 3.0) + _COSMO_OMEGA_L)
    inv_hz = 1.0 / hz

    if _COSMO_GRID_POINTS > 1:
        dz = z_grid[1] - z_grid[0]
        seg = (inv_hz[:-1] + inv_hz[1:]) * 0.5 * dz
        comoving = np.empty_like(z_grid)
        comoving[0] = 0.0
        np.cumsum(seg, out=comoving[1:])
    else:
        comoving = np.zeros_like(z_grid)

    # Mpc = (c/H0) * (integral 0..z dz/E(z)); DL = (1+z) * D_C
    dl = (1.0 + redshift_for_distance) * np.interp(redshift_for_distance, z_grid, comoving * (_COSMO_C_KM_S / _COSMO_H0))
    dl = np.clip(dl, _COSMO_DL_MIN_MPC, _COSMO_DL_MAX_MPC)
    return dl


def add_redshift_luminosity_plausibility(raw, deps, aux):
    del deps, aux  # explicit dependency/aux declaration-only no-op
    z = raw["redshift"].to_numpy(dtype=np.float64, copy=False)
    z_eff = np.maximum(z, _COSMO_Z_MIN)

    dist_mpc = _luminosity_distance_mpc(z_eff)
    dist_mpc = np.clip(dist_mpc, _COSMO_DL_MIN_MPC, _COSMO_DL_MAX_MPC)
    log10_dl = np.log10(dist_mpc)

    abs_mag = {}
    for band in _BANDS:
        m_obs = raw[band].to_numpy(dtype=np.float64, copy=False)
        m_abs = m_obs - 5.0 * log10_dl - 25.0
        abs_mag[band] = np.clip(m_abs, _COSMO_ABS_MAG_MIN, _COSMO_ABS_MAG_MAX)

    abs_matrix = np.column_stack((abs_mag["u"], abs_mag["g"], abs_mag["r"], abs_mag["i"], abs_mag["z"]))
    abs_center_ri = 0.5 * (abs_mag["r"] + abs_mag["i"])
    abs_brightest = np.min(abs_matrix, axis=1)
    abs_faintest = np.max(abs_matrix, axis=1)
    abs_spread = abs_faintest - abs_brightest
    abs_mean = np.mean(abs_matrix, axis=1)

    thresh_stellar, thresh_galaxy, thresh_qso, thresh_extreme = _THRESHOLDS_M

    features = {
        "redshift_non_positive": z <= 0.0,
        "redshift_very_low": z < 0.01,
        "redshift_for_distance": z_eff,
        "luminosity_distance_mpc": dist_mpc,
        "abs_mag_u": abs_mag["u"],
        "abs_mag_g": abs_mag["g"],
        "abs_mag_r": abs_mag["r"],
        "abs_mag_i": abs_mag["i"],
        "abs_mag_z": abs_mag["z"],
        "abs_mag_ri_center": abs_center_ri,
        "abs_mag_brightest_band": abs_brightest,
        "abs_mag_faintest_band": abs_faintest,
        "abs_mag_band_spread": abs_spread,
        "abs_mag_band_mean": abs_mean,
        "abs_mag_margin_to_stellar_scale": abs_center_ri - thresh_stellar,
        "abs_mag_margin_to_galaxy_scale": abs_center_ri - thresh_galaxy,
        "abs_mag_margin_to_quasar_scale": abs_center_ri - thresh_qso,
        "abs_mag_margin_to_extreme_quasar_scale": abs_center_ri - thresh_extreme,
        "abs_mag_r_margin_to_stellar_scale": abs_mag["r"] - thresh_stellar,
        "abs_mag_r_margin_to_galaxy_scale": abs_mag["r"] - thresh_galaxy,
        "abs_mag_r_margin_to_quasar_scale": abs_mag["r"] - thresh_qso,
        "abs_mag_r_margin_to_extreme_quasar_scale": abs_mag["r"] - thresh_extreme,
        "abs_band_spread_uirz": (abs_mag["u"] - abs_mag["r"]) + (abs_mag["r"] - abs_mag["i"]) + (abs_mag["i"] - abs_mag["z"]),
    }

    return pd.DataFrame(features, index=raw.index)


FEATURE_GROUPS = [
    {
        "name": "redshift_luminosity_plausibility",
        "fn": add_redshift_luminosity_plausibility,
        "depends_on": [],
        "description": "Convert apparent ugri z magnitudes into redshift-scaled pseudo absolute magnitudes with robust distance-distance proxies and regime-distance summaries.",
    }
]