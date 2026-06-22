import numpy as np
import pandas as pd

_GALACTIC_ROTATION_MATRIX = (
    (-0.0548755604162154, -0.8734370902348850, -0.4838350155487132),
    (0.4941094278755837, -0.4448296299600112, 0.7469822444972189),
    (-0.8676661490190047, -0.1980763734312015, 0.4559837761750669),
)

_SURVEY_NODE_RA_DEG = 95.0
_SURVEY_POLE_RA_DEG = 185.0
_SURVEY_POLE_DEC_DEG = 32.5
_SURVEY_STRIPE_PERIOD_DEG = 2.5


def add_sky_frame_position_geometry(raw, deps, aux):
    alpha_deg = raw["alpha"].to_numpy(dtype=float)
    delta_deg = raw["delta"].to_numpy(dtype=float)

    alpha_wrapped_deg = np.mod(alpha_deg, 360.0)
    alpha_rad = np.deg2rad(alpha_wrapped_deg)
    delta_rad = np.deg2rad(delta_deg)

    alpha_sin = np.sin(alpha_rad)
    alpha_cos = np.cos(alpha_rad)
    delta_sin = np.sin(delta_rad)
    delta_cos = np.cos(delta_rad)

    eq_x = delta_cos * alpha_cos
    eq_y = delta_cos * alpha_sin
    eq_z = delta_sin

    mat = np.array(_GALACTIC_ROTATION_MATRIX, dtype=float)
    gal_x = mat[0, 0] * eq_x + mat[0, 1] * eq_y + mat[0, 2] * eq_z
    gal_y = mat[1, 0] * eq_x + mat[1, 1] * eq_y + mat[1, 2] * eq_z
    gal_z = mat[2, 0] * eq_x + mat[2, 1] * eq_y + mat[2, 2] * eq_z

    gal_b_rad = np.arcsin(np.clip(gal_z, -1.0, 1.0))
    gal_l_rad = np.arctan2(gal_y, gal_x)
    gal_l_rad = np.mod(gal_l_rad, 2.0 * np.pi)

    gal_b_deg = np.degrees(gal_b_rad)
    gal_l_deg = np.degrees(gal_l_rad)

    gal_b_sin = np.sin(gal_b_rad)
    gal_b_cos = np.cos(gal_b_rad)
    gal_l_sin = np.sin(gal_l_rad)
    gal_l_cos = np.cos(gal_l_rad)

    gal_abs_b_deg = np.abs(gal_b_deg)
    gal_to_plane_deg = gal_abs_b_deg
    gal_to_poles_deg = 90.0 - gal_abs_b_deg

    cos_b = np.clip(gal_b_cos, -1.0, 1.0)
    sep_galactic_center_rad = np.arccos(np.clip(cos_b * np.cos(gal_l_rad - 0.0), -1.0, 1.0))
    sep_anticenter_rad = np.arccos(np.clip(cos_b * np.cos(gal_l_rad - np.pi), -1.0, 1.0))
    sep_galactic_center_deg = np.degrees(sep_galactic_center_rad)
    sep_anticenter_deg = np.degrees(sep_anticenter_rad)

    node_ra_rad = np.deg2rad(_SURVEY_NODE_RA_DEG)
    pole_ra_rad = np.deg2rad(_SURVEY_POLE_RA_DEG)
    pole_dec_rad = np.deg2rad(_SURVEY_POLE_DEC_DEG)

    node_vec_x = np.cos(node_ra_rad)
    node_vec_y = np.sin(node_ra_rad)
    node_vec_z = 0.0
    node_vec = np.array([node_vec_x, node_vec_y, node_vec_z], dtype=float)

    pole_vec_x = np.cos(pole_dec_rad) * np.cos(pole_ra_rad)
    pole_vec_y = np.cos(pole_dec_rad) * np.sin(pole_ra_rad)
    pole_vec_z = np.sin(pole_dec_rad)
    pole_vec = np.array([pole_vec_x, pole_vec_y, pole_vec_z], dtype=float)

    survey_y_vec = np.cross(pole_vec, node_vec)
    survey_y_norm = np.linalg.norm(survey_y_vec)
    survey_y_vec = survey_y_vec / np.where(survey_y_norm == 0.0, 1.0, survey_y_norm)

    survey_lambda_raw = np.arctan2(
        eq_y * survey_y_vec[1] + eq_x * survey_y_vec[0] + eq_z * survey_y_vec[2],
        eq_x * node_vec[0] + eq_y * node_vec[1] + eq_z * node_vec[2],
    )
    survey_lambda_rad = np.mod(survey_lambda_raw, 2.0 * np.pi)
    survey_eta_rad = np.arcsin(
        np.clip(
            eq_x * pole_vec[0] + eq_y * pole_vec[1] + eq_z * pole_vec[2],
            -1.0,
            1.0,
        )
    )
    survey_eta_deg = np.degrees(survey_eta_rad)

    survey_lambda_sin = np.sin(survey_lambda_rad)
    survey_lambda_cos = np.cos(survey_lambda_rad)
    survey_eta_sin = np.sin(survey_eta_rad)
    survey_eta_cos = np.cos(survey_eta_rad)

    stripe_phase_arg = 2.0 * np.pi * (survey_eta_deg / _SURVEY_STRIPE_PERIOD_DEG)
    survey_stripe_phase_sin = np.sin(stripe_phase_arg)
    survey_stripe_phase_cos = np.cos(stripe_phase_arg)

    features = pd.DataFrame(
        {
            "sky_eq_x": eq_x,
            "sky_eq_y": eq_y,
            "sky_eq_z": eq_z,
            "sky_alpha_sin": alpha_sin,
            "sky_alpha_cos": alpha_cos,
            "sky_gal_l_sin": gal_l_sin,
            "sky_gal_l_cos": gal_l_cos,
            "sky_gal_b": gal_b_deg,
            "sky_gal_b_sin": gal_b_sin,
            "sky_gal_b_cos": gal_b_cos,
            "sky_gal_abs_b": gal_abs_b_deg,
            "sky_gal_plane_distance_deg": gal_to_plane_deg,
            "sky_gal_pole_proximity_deg": gal_to_poles_deg,
            "sky_sep_galactic_center_deg": sep_galactic_center_deg,
            "sky_sep_galactic_anticenter_deg": sep_anticenter_deg,
            "sky_survey_lambda_sin": survey_lambda_sin,
            "sky_survey_lambda_cos": survey_lambda_cos,
            "sky_survey_eta": survey_eta_deg,
            "sky_survey_eta_sin": survey_eta_sin,
            "sky_survey_eta_cos": survey_eta_cos,
            "sky_survey_stripe_phase_sin": survey_stripe_phase_sin,
            "sky_survey_stripe_phase_cos": survey_stripe_phase_cos,
        },
        index=raw.index,
    )

    return features.where(np.isfinite(features), 0.0)


FEATURE_GROUPS = [
    {
        "name": "sky_frame_position_geometry",
        "fn": add_sky_frame_position_geometry,
        "depends_on": [],
        "description": "Generate equatorial, Galactic, and survey-aligned geometric sky-frame features with wrapped longitudes and stripe-phase position encoding.",
    }
]