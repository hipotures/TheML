import numpy as np
import pandas as pd


H0_KM_S_MPC = 70.0
OMEGA_M = 0.3
OMEGA_L = 0.7
C_KM_S = 299792.458
SDSS_R_BAND_SEEING_ARCSEC = 1.32
KPC_PER_MPC = 1000.0
ARCSEC_PER_RADIAN = 206264.80624709636
MIN_COSMO_REDSHIFT = 0.005
REDSHIFT_CLIP_MIN = 0.0001
REDSHIFT_CLIP_MAX = 7.0
DISTANCE_GRID_SIZE = 4097
SURFACE_BRIGHTNESS_INNER = 23.0
SURFACE_BRIGHTNESS_OUTER = 24.5


def add_seeing_limited_resolvability_proxy(raw, deps, aux):
    redshift = pd.to_numeric(raw["redshift"], errors="coerce").astype(float)
    r_mag = pd.to_numeric(raw["r"], errors="coerce").astype(float)
    population = raw["galaxy_population"].astype("string")

    z_raw = redshift.to_numpy(dtype=float, copy=True)
    r_values = r_mag.to_numpy(dtype=float, copy=True)

    finite_z = np.isfinite(z_raw)
    negative_redshift = finite_z & (z_raw < 0.0)
    local_redshift = finite_z & (z_raw <= MIN_COSMO_REDSHIFT)
    valid_cosmo = finite_z & (z_raw > MIN_COSMO_REDSHIFT)

    z_clipped = np.clip(np.where(finite_z, z_raw, REDSHIFT_CLIP_MIN), REDSHIFT_CLIP_MIN, REDSHIFT_CLIP_MAX)

    z_grid = np.linspace(0.0, REDSHIFT_CLIP_MAX, DISTANCE_GRID_SIZE)
    inv_e = 1.0 / np.sqrt(OMEGA_M * np.power(1.0 + z_grid, 3.0) + OMEGA_L)
    dz = np.diff(z_grid)
    cumulative_integral = np.empty_like(z_grid)
    cumulative_integral[0] = 0.0
    cumulative_integral[1:] = np.cumsum(0.5 * (inv_e[:-1] + inv_e[1:]) * dz)

    comoving_mpc = (C_KM_S / H0_KM_S_MPC) * np.interp(z_clipped, z_grid, cumulative_integral)
    luminosity_distance_mpc = (1.0 + z_clipped) * comoving_mpc
    angular_diameter_distance_mpc = comoving_mpc / (1.0 + z_clipped)

    distance_modulus = 5.0 * np.log10(np.maximum(luminosity_distance_mpc, 1.0e-12)) + 25.0
    pseudo_abs_r = r_values - distance_modulus

    pop_values = population.fillna("").to_numpy(dtype=str, copy=True)
    is_red_sequence = pop_values == "Red_Sequence"
    is_blue_cloud = pop_values == "Blue_Cloud"

    brightness_scale = np.power(10.0, -0.4 * (pseudo_abs_r + 21.0))
    red_size_kpc = 4.8 * np.power(np.maximum(brightness_scale, 1.0e-8), 0.62)
    blue_size_kpc = 3.6 * np.power(np.maximum(brightness_scale, 1.0e-8), 0.38)
    default_size_kpc = 4.0 * np.power(np.maximum(brightness_scale, 1.0e-8), 0.50)

    physical_half_light_kpc = np.where(is_red_sequence, red_size_kpc, np.where(is_blue_cloud, blue_size_kpc, default_size_kpc))
    physical_half_light_kpc = np.clip(physical_half_light_kpc, 0.3, 30.0)

    angular_half_light_arcsec = (
        physical_half_light_kpc
        / np.maximum(angular_diameter_distance_mpc * KPC_PER_MPC, 1.0e-12)
        * ARCSEC_PER_RADIAN
    )

    size_to_seeing = angular_half_light_arcsec / SDSS_R_BAND_SEEING_ARCSEC
    log_size_to_seeing = np.log1p(np.maximum(size_to_seeing, 0.0))

    soft_resolved_0p5 = 1.0 / (1.0 + np.exp(-4.0 * (size_to_seeing - 0.5)))
    soft_resolved_1p0 = 1.0 / (1.0 + np.exp(-4.0 * (size_to_seeing - 1.0)))
    soft_resolved_2p0 = 1.0 / (1.0 + np.exp(-3.0 * (size_to_seeing - 2.0)))

    surface_brightness_half_light = r_values + 2.5 * np.log10(
        np.maximum(2.0 * np.pi * np.square(angular_half_light_arcsec), 1.0e-12)
    )
    sb_margin_23 = SURFACE_BRIGHTNESS_INNER - surface_brightness_half_light
    sb_margin_24p5 = SURFACE_BRIGHTNESS_OUTER - surface_brightness_half_light

    invalid_or_local = ~valid_cosmo
    pseudo_abs_r = np.where(invalid_or_local, 0.0, pseudo_abs_r)
    physical_half_light_kpc = np.where(invalid_or_local, 0.0, physical_half_light_kpc)
    angular_half_light_arcsec = np.where(invalid_or_local, 0.0, angular_half_light_arcsec)
    size_to_seeing = np.where(invalid_or_local, 1.0, size_to_seeing)
    log_size_to_seeing = np.where(invalid_or_local, 0.0, log_size_to_seeing)
    soft_resolved_0p5 = np.where(invalid_or_local, 0.5, soft_resolved_0p5)
    soft_resolved_1p0 = np.where(invalid_or_local, 0.5, soft_resolved_1p0)
    soft_resolved_2p0 = np.where(invalid_or_local, 0.5, soft_resolved_2p0)
    surface_brightness_half_light = np.where(invalid_or_local, 0.0, surface_brightness_half_light)
    sb_margin_23 = np.where(invalid_or_local, 0.0, sb_margin_23)
    sb_margin_24p5 = np.where(invalid_or_local, 0.0, sb_margin_24p5)

    compact_bright = valid_cosmo & (r_values < 15.0) & (angular_half_light_arcsec < 2.0)

    return pd.DataFrame(
        {
            "z_negative": negative_redshift.astype(np.int8),
            "z_local_or_invalid": local_redshift.astype(np.int8),
            "cosmo_z_valid": valid_cosmo.astype(np.int8),
            "pseudo_abs_r": pseudo_abs_r,
            "pred_half_light_kpc": physical_half_light_kpc,
            "pred_half_light_arcsec": angular_half_light_arcsec,
            "size_to_seeing": size_to_seeing,
            "log_size_to_seeing": log_size_to_seeing,
            "soft_resolved_at_half_seeing": soft_resolved_0p5,
            "soft_resolved_at_seeing": soft_resolved_1p0,
            "soft_resolved_at_double_seeing": soft_resolved_2p0,
            "compact_bright_unresolved": compact_bright.astype(np.int8),
            "half_light_surface_brightness": surface_brightness_half_light,
            "surface_brightness_margin_23": sb_margin_23,
            "surface_brightness_margin_24p5": sb_margin_24p5,
        },
        index=raw.index,
    )


FEATURE_GROUPS = [
    {
        "name": "seeing_limited_resolvability_proxy",
        "fn": add_seeing_limited_resolvability_proxy,
        "depends_on": [],
        "description": "Predicts seeing-limited angular resolvability from redshift, r-band brightness, and catalog galaxy population.",
    }
]