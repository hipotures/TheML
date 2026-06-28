import numpy as np
import pandas as pd


BAND_NAMES = ("u", "g", "r", "i", "z")
SDSS_CENTRAL_WAVELENGTHS = (3551.0, 4686.0, 6166.0, 7480.0, 8932.0)
BALMER_EDGE_REST_ANGSTROM = 3646.0
EPS = 1.0e-9


def _safe_sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50.0, 50.0)))


def add_balmer_jump_contaminant_geometry(raw, deps, aux):
    index = raw.index
    bands = list(BAND_NAMES)
    wavelengths = np.asarray(SDSS_CENTRAL_WAVELENGTHS, dtype=np.float64)
    log_wavelengths = np.log(wavelengths)

    mags = raw.loc[:, bands].astype(np.float64)
    mag_values = mags.to_numpy(dtype=np.float64, copy=True)
    median_ugriz = np.nanmedian(mag_values, axis=1)
    flux = np.power(10.0, -0.4 * (mag_values - median_ugriz[:, None]))
    flux = np.nan_to_num(flux, nan=0.0, posinf=1.0e9, neginf=0.0)
    flux = np.clip(flux, EPS, 1.0e9)

    redshift = raw["redshift"].astype(np.float64).to_numpy(copy=True)
    redshift_nonneg = np.maximum(redshift, 0.0)
    edge = BALMER_EDGE_REST_ANGSTROM * (1.0 + redshift_nonneg)
    edge = np.nan_to_num(edge, nan=BALMER_EDGE_REST_ANGSTROM, posinf=1.0e9, neginf=BALMER_EDGE_REST_ANGSTROM)
    log_edge = np.log(np.clip(edge, EPS, None))

    below_u = edge < wavelengths[0]
    between_u_g = (edge >= wavelengths[0]) & (edge < wavelengths[1])
    between_g_r = (edge >= wavelengths[1]) & (edge < wavelengths[2])
    between_r_i = (edge >= wavelengths[2]) & (edge < wavelengths[3])
    between_i_z = (edge >= wavelengths[3]) & (edge <= wavelengths[4])
    above_z = edge > wavelengths[4]
    in_coverage = ~below_u & ~above_z

    side_scale = 0.055
    blue_side = _safe_sigmoid((log_edge[:, None] - log_wavelengths[None, :]) / side_scale)
    red_side = _safe_sigmoid((log_wavelengths[None, :] - log_edge[:, None]) / side_scale)

    distance_scale = 0.18
    proximity = np.exp(-np.square((log_wavelengths[None, :] - log_edge[:, None]) / distance_scale))
    blue_weights = blue_side * proximity
    red_weights = red_side * proximity

    blue_weight_sum = np.maximum(blue_weights.sum(axis=1), EPS)
    red_weight_sum = np.maximum(red_weights.sum(axis=1), EPS)
    weighted_blue_flux = (flux * blue_weights).sum(axis=1) / blue_weight_sum
    weighted_red_flux = (flux * red_weights).sum(axis=1) / red_weight_sum

    weighted_jump = np.log((weighted_red_flux + EPS) / (weighted_blue_flux + EPS))
    weighted_jump = np.clip(weighted_jump, -8.0, 8.0)
    weighted_jump = np.where(in_coverage, weighted_jump, 0.0)

    log_flux = np.log(np.clip(flux, EPS, None))
    x = log_wavelengths
    x_centered = x - x.mean()
    denom = np.sum(np.square(x_centered))
    y_mean = log_flux.mean(axis=1)
    slope = ((log_flux - y_mean[:, None]) * x_centered[None, :]).sum(axis=1) / max(float(denom), EPS)
    intercept = y_mean - slope * x.mean()
    continuum = intercept[:, None] + slope[:, None] * x[None, :]
    residual_flux = np.exp(np.clip(log_flux - continuum, -20.0, 20.0))

    weighted_blue_resid = (residual_flux * blue_weights).sum(axis=1) / blue_weight_sum
    weighted_red_resid = (residual_flux * red_weights).sum(axis=1) / red_weight_sum
    continuum_removed_jump = np.log((weighted_red_resid + EPS) / (weighted_blue_resid + EPS))
    continuum_removed_jump = np.clip(continuum_removed_jump, -8.0, 8.0)
    continuum_removed_jump = np.where(in_coverage, continuum_removed_jump, 0.0)

    pair_left = np.select(
        [between_u_g, between_g_r, between_r_i, between_i_z],
        [0, 1, 2, 3],
        default=0,
    ).astype(np.int64)
    row_pos = np.arange(len(raw))
    left_flux = flux[row_pos, pair_left]
    right_flux = flux[row_pos, pair_left + 1]
    local_pair_contrast = np.log((right_flux + EPS) / (left_flux + EPS))
    local_pair_contrast = np.clip(local_pair_contrast, -8.0, 8.0)
    local_pair_contrast = np.where(in_coverage, local_pair_contrast, 0.0)

    pair_log_left = log_wavelengths[pair_left]
    pair_log_right = log_wavelengths[pair_left + 1]
    pair_position = (log_edge - pair_log_left) / np.maximum(pair_log_right - pair_log_left, EPS)
    pair_position = np.where(in_coverage, np.clip(pair_position, 0.0, 1.0), 0.0)
    edge_distance_to_nearest_band = np.min(np.abs(log_wavelengths[None, :] - log_edge[:, None]), axis=1)
    edge_distance_to_nearest_band = np.where(in_coverage, edge_distance_to_nearest_band, 0.0)

    u_minus_g = mag_values[:, 0] - mag_values[:, 1]
    g_minus_r = mag_values[:, 1] - mag_values[:, 2]
    observed_frame_balmer_color = u_minus_g - g_minus_r
    observed_frame_balmer_flux = np.log((flux[:, 1] + EPS) / (flux[:, 0] + EPS)) - np.log((flux[:, 2] + EPS) / (flux[:, 1] + EPS))
    near_zero_redshift_gate = _safe_sigmoid((0.01 - np.abs(redshift)) / 0.003)

    spectral = raw["spectral_type"].astype("string")
    population = raw["galaxy_population"].astype("string")
    af_gate = spectral.eq("A/F").astype(np.float64).to_numpy()
    ob_gate = spectral.eq("O/B").astype(np.float64).to_numpy()
    red_sequence_gate = population.eq("Red_Sequence").astype(np.float64).to_numpy()
    blue_cloud_gate = population.eq("Blue_Cloud").astype(np.float64).to_numpy()
    stellar_balmer_gate = np.maximum(af_gate, ob_gate)

    out = pd.DataFrame(index=index)
    out["edge_observed_angstrom"] = edge
    out["edge_log_position"] = log_edge
    out["edge_below_u"] = below_u
    out["edge_between_u_g"] = between_u_g
    out["edge_between_g_r"] = between_g_r
    out["edge_between_r_i"] = between_r_i
    out["edge_between_i_z"] = between_i_z
    out["edge_above_z"] = above_z
    out["edge_in_ugriz_coverage"] = in_coverage
    out["edge_pair_position"] = pair_position
    out["edge_distance_to_nearest_band_log"] = edge_distance_to_nearest_band
    out["weighted_blue_flux"] = weighted_blue_flux
    out["weighted_red_flux"] = weighted_red_flux
    out["weighted_red_blue_jump"] = weighted_jump
    out["continuum_removed_red_blue_jump"] = continuum_removed_jump
    out["local_edge_pair_contrast"] = local_pair_contrast
    out["near_zero_observed_balmer_color"] = observed_frame_balmer_color * near_zero_redshift_gate
    out["near_zero_observed_balmer_flux_contrast"] = observed_frame_balmer_flux * near_zero_redshift_gate
    out["af_gated_weighted_jump"] = weighted_jump * af_gate
    out["ob_gated_weighted_jump"] = weighted_jump * ob_gate
    out["stellar_gated_continuum_removed_jump"] = continuum_removed_jump * stellar_balmer_gate
    out["near_zero_af_balmer_color"] = observed_frame_balmer_color * near_zero_redshift_gate * af_gate
    out["red_sequence_gated_jump"] = weighted_jump * red_sequence_gate
    out["blue_cloud_gated_jump"] = weighted_jump * blue_cloud_gate
    out["red_sequence_coverage"] = in_coverage.astype(np.float64) * red_sequence_gate
    out["blue_cloud_coverage"] = in_coverage.astype(np.float64) * blue_cloud_gate
    return out


FEATURE_GROUPS = [
    {
        "name": "balmer_jump_contaminant_geometry",
        "fn": add_balmer_jump_contaminant_geometry,
        "depends_on": [],
        "description": "Encodes observed Balmer-edge broadband geometry and contaminant-context gates from ugriz fluxes, redshift, spectral type, and galaxy population.",
    }
]