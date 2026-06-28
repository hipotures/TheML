import numpy as np
import pandas as pd


_C_LIGHT_KM_S = 299792.458
_H0_KM_S_MPC = 70.0
_OMEGA_M = 0.3
_OMEGA_L = 0.7
_SOLAR_ABS_MAG_R = 4.65
_SOLAR_ABS_MAG_I = 4.58
_MIN_Z_EFF = 0.0001
_LOW_Z_GALAXY_LIMIT = 0.003
_HIGH_Z_GALAXY_LIMIT = 1.2
_SDSS_UR_RED_BLUE_SPLIT = 2.22
_LOW_LOG_MASS = 7.0
_HIGH_LOG_MASS = 12.5


def add_galaxy_mass_to_light_sequence(raw, deps, aux):
    index = raw.index

    u_mag = pd.to_numeric(raw["u"], errors="coerce").to_numpy(dtype=float)
    g_mag = pd.to_numeric(raw["g"], errors="coerce").to_numpy(dtype=float)
    r_mag = pd.to_numeric(raw["r"], errors="coerce").to_numpy(dtype=float)
    i_mag = pd.to_numeric(raw["i"], errors="coerce").to_numpy(dtype=float)
    redshift = pd.to_numeric(raw["redshift"], errors="coerce").to_numpy(dtype=float)

    z_eff = np.maximum(np.nan_to_num(redshift, nan=0.0), _MIN_Z_EFF)
    z_for_distance = np.clip(z_eff, _MIN_Z_EFF, 8.0)

    grid = np.linspace(0.0, 8.0, 4097)
    inv_e = 1.0 / np.sqrt(_OMEGA_M * np.power(1.0 + grid, 3.0) + _OMEGA_L)
    dz = grid[1] - grid[0]
    cumulative_integral = np.concatenate(
        ([0.0], np.cumsum((inv_e[:-1] + inv_e[1:]) * 0.5 * dz))
    )
    comoving_integral = np.interp(z_for_distance, grid, cumulative_integral)

    luminosity_distance_mpc = (_C_LIGHT_KM_S / _H0_KM_S_MPC) * (1.0 + z_for_distance) * comoving_integral
    luminosity_distance_mpc = np.clip(luminosity_distance_mpc, 1.0e-6, 1.0e8)
    distance_modulus = 5.0 * np.log10(luminosity_distance_mpc) + 25.0

    abs_r = np.clip(r_mag - distance_modulus, -35.0, 25.0)
    abs_i = np.clip(i_mag - distance_modulus, -35.0, 25.0)

    gr = np.clip(g_mag - r_mag, -1.5, 3.5)
    gi = np.clip(g_mag - i_mag, -2.0, 4.5)
    ur = np.clip(u_mag - r_mag, -2.5, 6.0)

    log_ml_r = np.clip(-0.306 + 1.097 * gr, -2.5, 2.5)
    log_ml_i = np.clip(-0.152 + 0.518 * gi, -2.5, 2.5)
    log_lum_r = np.clip(-0.4 * (abs_r - _SOLAR_ABS_MAG_R), -4.0, 14.5)
    log_lum_i = np.clip(-0.4 * (abs_i - _SOLAR_ABS_MAG_I), -4.0, 14.5)

    log_mass_r = np.clip(log_lum_r + log_ml_r, 0.0, 17.0)
    log_mass_i = np.clip(log_lum_i + log_ml_i, 0.0, 17.0)
    log_mass_center = 0.5 * (log_mass_r + log_mass_i)
    log_mass_disagreement = np.abs(log_mass_r - log_mass_i)

    low_mass_deficit = np.maximum(0.0, _LOW_LOG_MASS - log_mass_center)
    high_mass_excess = np.maximum(0.0, log_mass_center - _HIGH_LOG_MASS)
    mass_plausibility_penalty = low_mass_deficit + high_mass_excess + 0.5 * log_mass_disagreement

    ur_red_sequence_margin = ur - _SDSS_UR_RED_BLUE_SPLIT
    implied_red_sequence = ur >= _SDSS_UR_RED_BLUE_SPLIT

    galaxy_population = raw["galaxy_population"].astype("string")
    declared_red_sequence = galaxy_population.eq("Red_Sequence").to_numpy(dtype=bool)
    declared_blue_cloud = galaxy_population.eq("Blue_Cloud").to_numpy(dtype=bool)
    population_color_agrees = (
        (implied_red_sequence & declared_red_sequence)
        | (~implied_red_sequence & declared_blue_cloud)
    )

    redshift_nonpositive = redshift <= 0.0
    redshift_too_local = redshift < _LOW_Z_GALAXY_LIMIT
    redshift_extrapolated_high = redshift > _HIGH_Z_GALAXY_LIMIT

    local_under_mass_signal = redshift_too_local.astype(float) * low_mass_deficit
    high_z_over_mass_signal = redshift_extrapolated_high.astype(float) * high_mass_excess
    galaxy_interpretation_stress = (
        mass_plausibility_penalty
        + redshift_nonpositive.astype(float) * 2.0
        + redshift_too_local.astype(float) * 1.0
        + redshift_extrapolated_high.astype(float) * 0.75
        + (~population_color_agrees).astype(float) * 0.5
    )

    return pd.DataFrame(
        {
            "z_eff": z_eff,
            "redshift_nonpositive": redshift_nonpositive,
            "redshift_too_local_for_galaxy": redshift_too_local,
            "redshift_high_extrapolated_for_galaxy": redshift_extrapolated_high,
            "distance_modulus_flat_lcdm": distance_modulus,
            "pseudo_abs_r": abs_r,
            "pseudo_abs_i": abs_i,
            "color_gr_clipped": gr,
            "color_gi_clipped": gi,
            "color_ur_clipped": ur,
            "log_ml_r_color_proxy": log_ml_r,
            "log_ml_i_color_proxy": log_ml_i,
            "pseudo_log_stellar_mass_r": log_mass_r,
            "pseudo_log_stellar_mass_i": log_mass_i,
            "pseudo_log_stellar_mass_center": log_mass_center,
            "mass_proxy_r_i_disagreement": log_mass_disagreement,
            "low_mass_deficit_below_7": low_mass_deficit,
            "high_mass_excess_above_12_5": high_mass_excess,
            "ur_red_sequence_margin": ur_red_sequence_margin,
            "implied_red_sequence_from_ur": implied_red_sequence,
            "population_color_agrees": population_color_agrees,
            "local_under_mass_signal": local_under_mass_signal,
            "high_z_over_mass_signal": high_z_over_mass_signal,
            "galaxy_interpretation_stress": galaxy_interpretation_stress,
        },
        index=index,
    )


FEATURE_GROUPS = [
    {
        "name": "galaxy_mass_to_light_sequence",
        "fn": add_galaxy_mass_to_light_sequence,
        "depends_on": [],
        "description": "Galaxy-interpretation mass-to-light and red-blue sequence plausibility features from optical colors and redshift-scaled luminosity.",
    }
]