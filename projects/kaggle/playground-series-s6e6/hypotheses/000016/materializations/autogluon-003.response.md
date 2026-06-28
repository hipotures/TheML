import numpy as np
import pandas as pd


GALACTIC_ROTATION_J2000 = (
    (-0.0548755604, -0.8734370902, -0.4838350155),
    (0.4941094279, -0.4448296300, 0.7469822445),
    (-0.8676661490, -0.1980763734, 0.4559837762),
)

SDSS_X_RA_DEG = 185.0
SDSS_X_DEC_DEG = 32.5
SDSS_Y_RA_DEG = 275.0
SDSS_Y_DEC_DEG = 0.0
SDSS_NODE_A_RA_DEG = 95.0
SDSS_NODE_B_RA_DEG = 275.0
SDSS_NODE_DEC_DEG = 0.0
SDSS_STRIPE_WIDTH_DEG = 2.5


def _as_finite_float_array(raw, column_name, default_value):
    values = pd.to_numeric(raw[column_name], errors="coerce").to_numpy(dtype=np.float64, copy=True)
    return np.nan_to_num(values, nan=default_value, posinf=default_value, neginf=default_value)


def _unit_vector_from_radec_rad(ra_rad, dec_rad):
    cos_dec = np.cos(dec_rad)
    x = cos_dec * np.cos(ra_rad)
    y = cos_dec * np.sin(ra_rad)
    z = np.sin(dec_rad)
    norm = np.sqrt(x * x + y * y + z * z)
    norm = np.maximum(norm, 1.0e-12)
    return x / norm, y / norm, z / norm


def _constant_unit_vector_from_radec_deg(ra_deg, dec_deg):
    ra_rad = np.deg2rad(ra_deg)
    dec_rad = np.deg2rad(dec_deg)
    cos_dec = np.cos(dec_rad)
    x = cos_dec * np.cos(ra_rad)
    y = cos_dec * np.sin(ra_rad)
    z = np.sin(dec_rad)
    norm = max(float(np.sqrt(x * x + y * y + z * z)), 1.0e-12)
    return x / norm, y / norm, z / norm


def add_sky_frame_position_geometry(raw, deps, aux):
    alpha = _as_finite_float_array(raw, "alpha", 0.0)
    delta = _as_finite_float_array(raw, "delta", 0.0)

    alpha_deg = np.mod(alpha, 360.0)
    delta_deg = np.clip(delta, -90.0, 90.0)

    alpha_rad = np.deg2rad(alpha_deg)
    delta_rad = np.deg2rad(delta_deg)

    p_x, p_y, p_z = _unit_vector_from_radec_rad(alpha_rad, delta_rad)

    sin_alpha = np.sin(alpha_rad)
    cos_alpha = np.cos(alpha_rad)
    sin_delta = np.sin(delta_rad)
    cos_delta = np.cos(delta_rad)

    gal_rotation = np.asarray(GALACTIC_ROTATION_J2000, dtype=np.float64)
    gal_x = gal_rotation[0, 0] * p_x + gal_rotation[0, 1] * p_y + gal_rotation[0, 2] * p_z
    gal_y = gal_rotation[1, 0] * p_x + gal_rotation[1, 1] * p_y + gal_rotation[1, 2] * p_z
    gal_z = gal_rotation[2, 0] * p_x + gal_rotation[2, 1] * p_y + gal_rotation[2, 2] * p_z

    gal_norm = np.sqrt(gal_x * gal_x + gal_y * gal_y + gal_z * gal_z)
    gal_norm = np.maximum(gal_norm, 1.0e-12)
    gal_x = gal_x / gal_norm
    gal_y = gal_y / gal_norm
    gal_z = gal_z / gal_norm

    gal_l = np.mod(np.arctan2(gal_y, gal_x), 2.0 * np.pi)
    gal_b = np.arcsin(np.clip(gal_z, -1.0, 1.0))
    sin_gal_l = np.sin(gal_l)
    cos_gal_l = np.cos(gal_l)
    sin_gal_b = np.sin(gal_b)
    cos_gal_b = np.cos(gal_b)
    abs_gal_b = np.abs(gal_b)
    gal_plane_distance = (0.5 * np.pi) - abs_gal_b
    gal_center_distance = np.arccos(np.clip(cos_gal_b * cos_gal_l, -1.0, 1.0))
    gal_anticenter_distance = np.arccos(np.clip(-cos_gal_b * cos_gal_l, -1.0, 1.0))

    sdss_x = _constant_unit_vector_from_radec_deg(SDSS_X_RA_DEG, SDSS_X_DEC_DEG)
    sdss_y = _constant_unit_vector_from_radec_deg(SDSS_Y_RA_DEG, SDSS_Y_DEC_DEG)

    z_x = sdss_x[1] * sdss_y[2] - sdss_x[2] * sdss_y[1]
    z_y = sdss_x[2] * sdss_y[0] - sdss_x[0] * sdss_y[2]
    z_z = sdss_x[0] * sdss_y[1] - sdss_x[1] * sdss_y[0]
    z_norm = max(float(np.sqrt(z_x * z_x + z_y * z_y + z_z * z_z)), 1.0e-12)
    sdss_z = (z_x / z_norm, z_y / z_norm, z_z / z_norm)

    dot_x = p_x * sdss_x[0] + p_y * sdss_x[1] + p_z * sdss_x[2]
    dot_y = p_x * sdss_y[0] + p_y * sdss_y[1] + p_z * sdss_y[2]
    dot_z = p_x * sdss_z[0] + p_y * sdss_z[1] + p_z * sdss_z[2]

    sdss_eta = np.arcsin(np.clip(dot_z, -1.0, 1.0))
    sdss_lambda = np.arctan2(dot_y, dot_x)

    sdss_node_a = _constant_unit_vector_from_radec_deg(SDSS_NODE_A_RA_DEG, SDSS_NODE_DEC_DEG)
    sdss_node_b = _constant_unit_vector_from_radec_deg(SDSS_NODE_B_RA_DEG, SDSS_NODE_DEC_DEG)
    node_a_dot = p_x * sdss_node_a[0] + p_y * sdss_node_a[1] + p_z * sdss_node_a[2]
    node_b_dot = p_x * sdss_node_b[0] + p_y * sdss_node_b[1] + p_z * sdss_node_b[2]

    eta_deg = np.rad2deg(sdss_eta)
    stripe_phase = np.mod(eta_deg + 90.0, SDSS_STRIPE_WIDTH_DEG) / SDSS_STRIPE_WIDTH_DEG
    stripe_index_centered = np.clip(np.floor((eta_deg + 90.0) / SDSS_STRIPE_WIDTH_DEG), 0.0, 71.0) - 35.5

    features = pd.DataFrame(
        {
            "eq_sin_alpha": sin_alpha,
            "eq_cos_alpha": cos_alpha,
            "eq_sin_delta": sin_delta,
            "eq_cos_delta": cos_delta,
            "eq_unit_x": p_x,
            "eq_unit_y": p_y,
            "eq_unit_z": p_z,
            "gal_sin_l": sin_gal_l,
            "gal_cos_l": cos_gal_l,
            "gal_sin_b": sin_gal_b,
            "gal_cos_b": cos_gal_b,
            "gal_abs_b": abs_gal_b,
            "gal_plane_distance": gal_plane_distance,
            "gal_center_distance": gal_center_distance,
            "gal_anticenter_distance": gal_anticenter_distance,
            "sdss_sin_lambda": np.sin(sdss_lambda),
            "sdss_cos_lambda": np.cos(sdss_lambda),
            "sdss_sin_eta": np.sin(sdss_eta),
            "sdss_cos_eta": np.cos(sdss_eta),
            "sdss_abs_eta": np.abs(sdss_eta),
            "sdss_node_95_distance": np.arccos(np.clip(node_a_dot, -1.0, 1.0)),
            "sdss_node_275_distance": np.arccos(np.clip(node_b_dot, -1.0, 1.0)),
            "sdss_stripe_phase_sin": np.sin(2.0 * np.pi * stripe_phase),
            "sdss_stripe_phase_cos": np.cos(2.0 * np.pi * stripe_phase),
            "sdss_stripe_index_centered": stripe_index_centered,
        },
        index=raw.index,
    )

    return features.replace([np.inf, -np.inf], 0.0).fillna(0.0)


FEATURE_GROUPS = [
    {
        "name": "sky_frame_position_geometry",
        "fn": add_sky_frame_position_geometry,
        "depends_on": [],
        "description": "Continuous equatorial, Galactic, and SDSS survey-frame sky geometry features derived from alpha and delta.",
    }
]