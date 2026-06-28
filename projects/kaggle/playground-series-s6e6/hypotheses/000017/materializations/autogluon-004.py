import numpy as np
import pandas as pd

COSMOLOGY_H0 = 70.0
COSMOLOGY_OMEGA_M = 0.3
COSMOLOGY_OMEGA_LAMBDA = 0.7
COSMOLOGY_C_KM_S = 299792.458
REDSHIFT_EPS = 0.0001
DISTANCE_FLOOR_MPC = 0.000001
ABS_MAG_CLIP_LOW = -35.0
ABS_MAG_CLIP_HIGH = 15.0
COSMOLOGY_GRID_POINTS = 4097
COSMOLOGY_GRID_Z_MAX = 7.05
PHOTOMETRY_BANDS = ("u", "g", "r", "i", "z")
REGIME_THRESHOLDS = (-10.0, -18.0, -23.0, -27.0)


def add_redshift_luminosity_plausibility(raw, deps, aux):
    redshift = pd.to_numeric(raw["redshift"], errors="coerce").astype(float)
    z_raw = redshift.to_numpy(dtype=float, copy=True)
    z_eff = np.maximum(np.nan_to_num(z_raw, nan=REDSHIFT_EPS), REDSHIFT_EPS)

    z_grid = np.linspace(0.0, COSMOLOGY_GRID_Z_MAX, COSMOLOGY_GRID_POINTS, dtype=float)
    inv_e = 1.0 / np.sqrt(
        COSMOLOGY_OMEGA_M * np.power(1.0 + z_grid, 3.0) + COSMOLOGY_OMEGA_LAMBDA
    )
    dz = z_grid[1] - z_grid[0]
    cumulative = np.empty_like(z_grid)
    cumulative[0] = 0.0
    cumulative[1:] = np.cumsum(0.5 * (inv_e[:-1] + inv_e[1:]) * dz)

    comoving_integral = np.interp(
        np.minimum(z_eff, COSMOLOGY_GRID_Z_MAX),
        z_grid,
        cumulative,
    )
    comoving_distance_mpc = (COSMOLOGY_C_KM_S / COSMOLOGY_H0) * comoving_integral
    luminosity_distance_mpc = (1.0 + z_eff) * comoving_distance_mpc
    distance_modulus = 5.0 * np.log10(np.maximum(luminosity_distance_mpc, DISTANCE_FLOOR_MPC)) + 25.0

    mag_values = []
    for band in PHOTOMETRY_BANDS:
        mag_values.append(pd.to_numeric(raw[band], errors="coerce").to_numpy(dtype=float, copy=True))
    mags = np.column_stack(mag_values)

    abs_mags = np.clip(
        mags - distance_modulus[:, None],
        ABS_MAG_CLIP_LOW,
        ABS_MAG_CLIP_HIGH,
    )

    abs_u = abs_mags[:, 0]
    abs_g = abs_mags[:, 1]
    abs_r = abs_mags[:, 2]
    abs_i = abs_mags[:, 3]
    abs_z = abs_mags[:, 4]

    abs_mean = np.mean(abs_mags, axis=1)
    abs_median = np.median(abs_mags, axis=1)
    abs_min = np.min(abs_mags, axis=1)
    abs_max = np.max(abs_mags, axis=1)
    central_ri = 0.5 * (abs_r + abs_i)

    q75 = np.percentile(abs_mags, 75.0, axis=1)
    q25 = np.percentile(abs_mags, 25.0, axis=1)
    abs_iqr = q75 - q25
    abs_sd = np.std(abs_mags, axis=1)
    abs_range = abs_max - abs_min

    mr_regime = np.select(
        [
            abs_r > -10.0,
            (abs_r <= -10.0) & (abs_r > -18.0),
            (abs_r <= -18.0) & (abs_r > -23.0),
            (abs_r <= -23.0) & (abs_r > -27.0),
            abs_r <= -27.0,
        ],
        [0, 1, 2, 3, 4],
        default=2,
    ).astype(np.int8)

    central_ri_regime = np.select(
        [
            central_ri > -10.0,
            (central_ri <= -10.0) & (central_ri > -18.0),
            (central_ri <= -18.0) & (central_ri > -23.0),
            (central_ri <= -23.0) & (central_ri > -27.0),
            central_ri <= -27.0,
        ],
        [0, 1, 2, 3, 4],
        default=2,
    ).astype(np.int8)

    band_regimes = np.select(
        [
            abs_mags > -10.0,
            (abs_mags <= -10.0) & (abs_mags > -18.0),
            (abs_mags <= -18.0) & (abs_mags > -23.0),
            (abs_mags <= -23.0) & (abs_mags > -27.0),
            abs_mags <= -27.0,
        ],
        [0, 1, 2, 3, 4],
        default=2,
    ).astype(np.int8)

    out = pd.DataFrame(index=raw.index)

    out["z_eff"] = z_eff
    out["z_nonpos"] = z_raw <= 0.0
    out["z_very_low"] = z_raw < 0.01
    out["z_low"] = z_raw < 0.05
    out["z_high"] = z_raw >= 3.0
    out["luminosity_distance_mpc_log1p"] = np.log1p(luminosity_distance_mpc)
    out["distance_modulus"] = distance_modulus

    out["abs_mag_u"] = abs_u
    out["abs_mag_g"] = abs_g
    out["abs_mag_r"] = abs_r
    out["abs_mag_i"] = abs_i
    out["abs_mag_z"] = abs_z
    out["abs_mag_mean"] = abs_mean
    out["abs_mag_median"] = abs_median
    out["abs_mag_min"] = abs_min
    out["abs_mag_max"] = abs_max
    out["abs_mag_central_ri"] = central_ri

    out["abs_mag_sd"] = abs_sd
    out["abs_mag_range"] = abs_range
    out["abs_mag_iqr"] = abs_iqr
    out["abs_mag_u_minus_mean"] = abs_u - abs_mean
    out["abs_mag_g_minus_mean"] = abs_g - abs_mean
    out["abs_mag_r_minus_mean"] = abs_r - abs_mean
    out["abs_mag_i_minus_mean"] = abs_i - abs_mean
    out["abs_mag_z_minus_mean"] = abs_z - abs_mean

    for threshold in REGIME_THRESHOLDS:
        suffix = str(int(abs(threshold)))
        out["mr_margin_to_neg_" + suffix] = abs_r - threshold
        out["central_ri_margin_to_neg_" + suffix] = central_ri - threshold

    out["mr_luminosity_regime_ord"] = mr_regime
    out["central_ri_luminosity_regime_ord"] = central_ri_regime
    out["band_count_stellar_like_faint"] = np.sum(band_regimes == 0, axis=1).astype(np.int8)
    out["band_count_compact_intermediate"] = np.sum(band_regimes == 1, axis=1).astype(np.int8)
    out["band_count_normal_galaxy_like"] = np.sum(band_regimes == 2, axis=1).astype(np.int8)
    out["band_count_bright_galaxy_agn_like"] = np.sum(band_regimes == 3, axis=1).astype(np.int8)
    out["band_count_extreme_quasar_like"] = np.sum(band_regimes == 4, axis=1).astype(np.int8)

    out["distance_modulus_x_redshift"] = np.clip(distance_modulus * z_raw, -100.0, 1000.0)
    out["mr_x_z_eff"] = np.clip(abs_r * z_eff, -300.0, 150.0)
    out["central_ri_x_z_eff"] = np.clip(central_ri * z_eff, -300.0, 150.0)

    return out


FEATURE_GROUPS = [
    {
        "name": "redshift_luminosity_plausibility",
        "fn": add_redshift_luminosity_plausibility,
        "depends_on": [],
        "description": "Cosmology-scaled intrinsic-brightness plausibility features from redshift and ugriz photometry.",
    }
]