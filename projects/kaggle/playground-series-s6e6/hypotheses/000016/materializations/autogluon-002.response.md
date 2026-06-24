import numpy as np
import pandas as pd

_PI = 3.141592653589793
_FULL_CIRCLE_DEG = 360.0
_FULL_CIRCLE_RAD = 2.0 * _PI
_HALF_CIRCLE_RAD = _PI
_SDSS_STRIPE_WIDTH_DEG = 2.5
_SDSS_STRIPE_HALF_WIDTH_DEG = _SDSS_STRIPE_WIDTH_DEG / 2.0
_EPS = 1e-12

_EQ_TO_GAL_MATRIX = (
    (-0.0548755604, 0.4941094279, -0.8676661490),
    (-0.8734370902, -0.4448296300, -0.1980763734),
    (-0.4838350155, 0.7469822445, 0.4559837762),
)

_SDSS_X_AXIS = (185.0, 32.5)
_SDSS_V_AXIS = (275.0, 0.0)


def _to_finite(values, default=0.0):
    return np.nan_to_num(np.asarray(values, dtype=float), nan=default, posinf=default, neginf=default)


def _normalize_rows(vectors, eps=_EPS):
    vectors = _to_finite(vectors)
    norm = np.linalg.norm(vectors, axis=1, keepdims=True)
    denom = np.where(norm > eps, norm, 1.0)
    return vectors / denom


def _normalize_vector(vector, eps=_EPS):
    vector = _to_finite(vector)
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm <= eps:
        return np.zeros(3, dtype=float)
    return vector / norm


def _sph_to_cart(ra_rad, dec_rad):
    cos_dec = np.cos(dec_rad)
    x = cos_dec * np.cos(ra_rad)
    y = cos_dec * np.sin(ra_rad)
    z = np.sin(dec_rad)
    return np.column_stack((x, y, z))


def add_sky_frame_position_geometry(raw, deps, aux):
    idx = raw.index
    alpha = _to_finite(pd.to_numeric(raw["alpha"], errors="coerce").to_numpy(dtype=float))
    delta = _to_finite(pd.to_numeric(raw["delta"], errors="coerce").to_numpy(dtype=float))

    alpha_wrapped = np.mod(np.mod(alpha, _FULL_CIRCLE_DEG) + _FULL_CIRCLE_DEG, _FULL_CIRCLE_DEG)
    delta_clipped = np.clip(delta, -90.0, 90.0)

    alpha_rad = np.radians(alpha_wrapped)
    delta_rad = np.radians(delta_clipped)

    # Wrap-safe equatorial basis features
    eq_sin_alpha = _to_finite(np.sin(alpha_rad))
    eq_cos_alpha = _to_finite(np.cos(alpha_rad))
    eq_sin_delta = _to_finite(np.sin(delta_rad))
    eq_cos_delta = _to_finite(np.cos(delta_rad))

    eq_unit = _normalize_rows(_sph_to_cart(alpha_rad, delta_rad))
    eq_x = _to_finite(eq_unit[:, 0])
    eq_y = _to_finite(eq_unit[:, 1])
    eq_z = _to_finite(eq_unit[:, 2])

    # Galactic-frame geometry
    gal = eq_unit.dot(np.array(_EQ_TO_GAL_MATRIX, dtype=float).T)
    gx = _to_finite(np.clip(gal[:, 0], -1.0, 1.0))
    gy = _to_finite(np.clip(gal[:, 1], -1.0, 1.0))
    gz = _to_finite(np.clip(gal[:, 2], -1.0, 1.0))

    l_raw = np.arctan2(gy, gx)
    l = _to_finite(np.mod(l_raw + _FULL_CIRCLE_RAD, _FULL_CIRCLE_RAD))
    b = _to_finite(np.arctan2(gz, np.sqrt(np.clip(gx * gx + gy * gy, 0.0, np.inf))))

    gal_sin_l = _to_finite(np.sin(l))
    gal_cos_l = _to_finite(np.cos(l))
    gal_sin_b = _to_finite(np.sin(b))
    gal_cos_b = _to_finite(np.cos(b))
    gal_abs_b = _to_finite(np.abs(b))
    gal_plane_complement = _to_finite(_HALF_CIRCLE_RAD / 2.0 - gal_abs_b)

    gc_arg = np.clip(gal_cos_b * np.cos(l), -1.0, 1.0)
    anti_arg = np.clip(gal_cos_b * np.cos(l - _PI), -1.0, 1.0)
    gal_d_gc = _to_finite(np.arccos(gc_arg))
    gal_d_anticenter = _to_finite(np.arccos(anti_arg))

    # SDSS-frame geometry
    x_ra_rad = np.radians(np.mod(_SDSS_X_AXIS[0], _FULL_CIRCLE_DEG))
    x_dec_rad = np.radians(np.clip(_SDSS_X_AXIS[1], -90.0, 90.0))
    v_ra_rad = np.radians(np.mod(_SDSS_V_AXIS[0], _FULL_CIRCLE_DEG))
    v_dec_rad = np.radians(np.clip(_SDSS_V_AXIS[1], -90.0, 90.0))

    x_hat = _normalize_vector(
        np.array(
            (
                np.cos(x_dec_rad) * np.cos(x_ra_rad),
                np.cos(x_dec_rad) * np.sin(x_ra_rad),
                np.sin(x_dec_rad),
            ),
            dtype=float,
        )
    )
    v_hat = _normalize_vector(
        np.array(
            (
                np.cos(v_dec_rad) * np.cos(v_ra_rad),
                np.cos(v_dec_rad) * np.sin(v_ra_rad),
                np.sin(v_dec_rad),
            ),
            dtype=float,
        )
    )
    z_hat = _normalize_vector(np.cross(x_hat, v_hat))
    y_hat = _normalize_vector(np.cross(z_hat, x_hat))

    dot_x = _to_finite(eq_unit.dot(x_hat))
    dot_y = _to_finite(eq_unit.dot(y_hat))
    dot_z = _to_finite(eq_unit.dot(z_hat))

    eta = _to_finite(np.arcsin(np.clip(dot_z, -1.0, 1.0)))
    lam = _to_finite(np.arctan2(dot_y, dot_x))
    lam = _to_finite(np.mod(lam + _PI, _FULL_CIRCLE_RAD) - _PI)

    sdss_sin_eta = _to_finite(np.sin(eta))
    sdss_cos_eta = _to_finite(np.cos(eta))
    sdss_sin_lam = _to_finite(np.sin(lam))
    sdss_cos_lam = _to_finite(np.cos(lam))

    eta_deg = _to_finite(np.degrees(eta))
    eta_phase = _to_finite(np.mod(eta_deg, _SDSS_STRIPE_WIDTH_DEG))
    eta_phase_centered = _to_finite(eta_phase - _SDSS_STRIPE_HALF_WIDTH_DEG)
    stripe_phase_sin = _to_finite(np.sin(_FULL_CIRCLE_RAD * eta_phase_centered / _SDSS_STRIPE_WIDTH_DEG))
    stripe_phase_cos = _to_finite(np.cos(_FULL_CIRCLE_RAD * eta_phase_centered / _SDSS_STRIPE_WIDTH_DEG))
    stripe_band = np.floor(np.clip((eta_deg + 90.0) / _SDSS_STRIPE_WIDTH_DEG, -1e9, 1e9)).astype(np.int64)

    return pd.DataFrame(
        {
            "eq_alpha_sin": eq_sin_alpha,
            "eq_alpha_cos": eq_cos_alpha,
            "eq_delta_sin": eq_sin_delta,
            "eq_delta_cos": eq_cos_delta,
            "eq_x": eq_x,
            "eq_y": eq_y,
            "eq_z": eq_z,
            "gal_l": l,
            "gal_b": b,
            "gal_sin_l": gal_sin_l,
            "gal_cos_l": gal_cos_l,
            "gal_sin_b": gal_sin_b,
            "gal_cos_b": gal_cos_b,
            "gal_abs_b": gal_abs_b,
            "gal_plane_complement": gal_plane_complement,
            "gal_dist_to_gc": gal_d_gc,
            "gal_dist_to_anticenter": gal_d_anticenter,
            "sdss_eta": eta,
            "sdss_lambda": lam,
            "sdss_sin_eta": sdss_sin_eta,
            "sdss_cos_eta": sdss_cos_eta,
            "sdss_sin_lambda": sdss_sin_lam,
            "sdss_cos_lambda": sdss_cos_lam,
            "sdss_eta_phase": eta_phase,
            "sdss_eta_phase_centered": eta_phase_centered,
            "sdss_eta_phase_sin": stripe_phase_sin,
            "sdss_eta_phase_cos": stripe_phase_cos,
            "sdss_stripe_band": stripe_band,
        },
        index=idx,
    )


FEATURE_GROUPS = [
    {
        "name": "sky_frame_position_geometry",
        "fn": add_sky_frame_position_geometry,
        "depends_on": [],
        "description": "Builds continuous spatial geometry features in equatorial, Galactic, and SDSS survey frames with discontinuity-safe angular encoding and stripe-aware periodic terms.",
    }
]