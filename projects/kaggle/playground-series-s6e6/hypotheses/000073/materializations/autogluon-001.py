import numpy as np
import pandas as pd


SPEED_OF_LIGHT_KMS = 299792.458
REDSHIFT_CLIP_LOW = -0.02
REDSHIFT_CLIP_HIGH = 7.25
SOLAR_U_KMS = 11.1
SOLAR_V_KMS = 12.24
SOLAR_W_KMS = 7.25
GALACTIC_ROTATION_KMS = 220.0
VELOCITY_ENVELOPES_KMS = (50.0, 150.0, 500.0, 1200.0)
COSMOLOGICAL_REDSHIFT_CUTS = (0.005, 0.01, 0.03, 0.1, 0.5, 1.0)


def add_stellar_radial_velocity_frame_margins(raw, deps, aux):
    index = raw.index
    alpha = pd.to_numeric(raw["alpha"], errors="coerce").to_numpy(dtype=float)
    delta = pd.to_numeric(raw["delta"], errors="coerce").to_numpy(dtype=float)
    redshift = pd.to_numeric(raw["redshift"], errors="coerce").to_numpy(dtype=float)

    alpha = np.nan_to_num(alpha, nan=0.0, posinf=360.0, neginf=0.0)
    delta = np.nan_to_num(delta, nan=0.0, posinf=90.0, neginf=-90.0)
    redshift = np.nan_to_num(redshift, nan=0.0, posinf=REDSHIFT_CLIP_HIGH, neginf=REDSHIFT_CLIP_LOW)

    z_clip = np.clip(redshift, REDSHIFT_CLIP_LOW, REDSHIFT_CLIP_HIGH)
    one_plus_z_sq = np.square(1.0 + z_clip)
    radial_velocity = SPEED_OF_LIGHT_KMS * (one_plus_z_sq - 1.0) / (one_plus_z_sq + 1.0)

    ra_rad = np.deg2rad(np.mod(alpha, 360.0))
    dec_rad = np.deg2rad(np.clip(delta, -90.0, 90.0))
    cos_dec = np.cos(dec_rad)

    eq_x = cos_dec * np.cos(ra_rad)
    eq_y = cos_dec * np.sin(ra_rad)
    eq_z = np.sin(dec_rad)

    gal_x = (
        -0.0548755604162154 * eq_x
        - 0.8734370902348850 * eq_y
        - 0.4838350155487132 * eq_z
    )
    gal_y = (
        0.4941094278755837 * eq_x
        - 0.4448296299600112 * eq_y
        + 0.7469822444972189 * eq_z
    )
    gal_z = (
        -0.8676661490190047 * eq_x
        - 0.1980763734312015 * eq_y
        + 0.4559837761750669 * eq_z
    )

    norm = np.sqrt(gal_x * gal_x + gal_y * gal_y + gal_z * gal_z)
    norm = np.where(norm > 0.0, norm, 1.0)
    gal_x = gal_x / norm
    gal_y = gal_y / norm
    gal_z = np.clip(gal_z / norm, -1.0, 1.0)

    gal_l = np.arctan2(gal_y, gal_x)
    sin_l = np.sin(gal_l)
    cos_l = np.cos(gal_l)
    sin_b = gal_z
    cos_b = np.sqrt(np.maximum(0.0, 1.0 - sin_b * sin_b))

    solar_projection = (
        SOLAR_U_KMS * cos_l * cos_b
        + SOLAR_V_KMS * sin_l * cos_b
        + SOLAR_W_KMS * sin_b
    )
    rotation_projection = GALACTIC_ROTATION_KMS * sin_l * cos_b

    v_helio = radial_velocity
    v_lsr = v_helio + solar_projection
    v_gsr = v_lsr + rotation_projection

    abs_helio = np.abs(v_helio)
    abs_lsr = np.abs(v_lsr)
    abs_gsr = np.abs(v_gsr)

    features = {
        "z_clip": z_clip,
        "z_nonnegative": z_clip >= 0.0,
        "z_near_zero_abs": np.abs(z_clip),
        "rv_helio_kms": v_helio,
        "rv_lsr_kms": v_lsr,
        "rv_gsr_kms": v_gsr,
        "abs_rv_helio_kms": abs_helio,
        "abs_rv_lsr_kms": abs_lsr,
        "abs_rv_gsr_kms": abs_gsr,
        "rv_helio_positive": v_helio > 0.0,
        "rv_lsr_positive": v_lsr > 0.0,
        "rv_gsr_positive": v_gsr > 0.0,
        "galactic_sin_l": sin_l,
        "galactic_cos_l": cos_l,
        "galactic_sin_b": sin_b,
        "galactic_cos_b": cos_b,
        "solar_apex_projection_kms": solar_projection,
        "rotation_projection_kms": rotation_projection,
        "solar_projection_abs_kms": np.abs(solar_projection),
        "rotation_projection_abs_kms": np.abs(rotation_projection),
        "rv_times_solar_projection": v_helio * solar_projection,
        "rv_times_rotation_projection": v_helio * rotation_projection,
        "rv_solar_aligned": (v_helio * solar_projection) > 0.0,
        "rv_rotation_aligned": (v_helio * rotation_projection) > 0.0,
    }

    for envelope in VELOCITY_ENVELOPES_KMS:
        suffix = str(int(envelope))
        features["helio_margin_to_" + suffix + "kms"] = envelope - abs_helio
        features["lsr_margin_to_" + suffix + "kms"] = envelope - abs_lsr
        features["gsr_margin_to_" + suffix + "kms"] = envelope - abs_gsr
        features["helio_inside_" + suffix + "kms"] = abs_helio <= envelope
        features["lsr_inside_" + suffix + "kms"] = abs_lsr <= envelope
        features["gsr_inside_" + suffix + "kms"] = abs_gsr <= envelope
        features["helio_soft_margin_" + suffix + "kms"] = np.tanh((envelope - abs_helio) / envelope)
        features["lsr_soft_margin_" + suffix + "kms"] = np.tanh((envelope - abs_lsr) / envelope)
        features["gsr_soft_margin_" + suffix + "kms"] = np.tanh((envelope - abs_gsr) / envelope)

    for cut in COSMOLOGICAL_REDSHIFT_CUTS:
        label = str(cut).replace(".", "p")
        features["z_margin_to_" + label] = cut - z_clip
        features["z_above_" + label] = z_clip > cut
        features["z_soft_excess_" + label] = np.log1p(np.maximum(0.0, z_clip - cut))

    out = pd.DataFrame(features, index=index)
    return out.replace([np.inf, -np.inf], 0.0)


FEATURE_GROUPS = [
    {
        "name": "stellar_radial_velocity_frame_margins",
        "fn": add_stellar_radial_velocity_frame_margins,
        "depends_on": [],
        "description": "Relativistic redshift radial-velocity margins corrected by Galactic sky-frame projections.",
    }
]