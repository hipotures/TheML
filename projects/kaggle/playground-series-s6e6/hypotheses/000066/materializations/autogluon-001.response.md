import numpy as np
import pandas as pd


SPEED_OF_LIGHT_KMS = 299792.458
REDSHIFT_LOG_LOWER = -0.999
GRID_STEPS_KMS = (138.0, 276.0)
NORMALIZED_DISTANCE_CLIP = 5.0
STELLAR_VELOCITY_LIMIT_KMS = 1200.0
GALAXY_Z_MIN = -0.01
GALAXY_Z_MAX = 1.0
QUASAR_Z_MIN = 0.0333
QUASAR_Z_MAX = 7.0


def add_spectroscopic_redshift_grid_imprint(raw, deps, aux):
    index = raw.index
    z = pd.to_numeric(raw["redshift"], errors="coerce").to_numpy(dtype="float64", copy=True)

    z_safe = np.maximum(z, REDSHIFT_LOG_LOWER)
    one_plus_z = 1.0 + z_safe
    ln_one_plus_z = np.log(one_plus_z)

    one_plus_z_sq = one_plus_z * one_plus_z
    beta = (one_plus_z_sq - 1.0) / (one_plus_z_sq + 1.0)
    velocity_kms = SPEED_OF_LIGHT_KMS * beta
    log_velocity_kms = SPEED_OF_LIGHT_KMS * ln_one_plus_z

    features = pd.DataFrame(index=index)

    features["z_clipped_for_velocity"] = z_safe
    features["ln_one_plus_z"] = ln_one_plus_z
    features["relativistic_beta"] = beta
    features["velocity_kms"] = velocity_kms
    features["log_velocity_kms"] = log_velocity_kms

    for step in GRID_STEPS_KMS:
        step_name = str(int(step))
        phase_turns = log_velocity_kms / step
        nearest_grid = step * np.round(phase_turns)
        signed_residual = log_velocity_kms - nearest_grid
        normalized_abs_residual = np.abs(signed_residual) / (0.5 * step)

        features["log_velocity_grid_residual_kms_" + step_name] = signed_residual
        features["log_velocity_grid_abs_residual_halfstep_" + step_name] = normalized_abs_residual
        features["log_velocity_grid_sin_phase_" + step_name] = np.sin(2.0 * np.pi * phase_turns)
        features["log_velocity_grid_cos_phase_" + step_name] = np.cos(2.0 * np.pi * phase_turns)

    stellar_margin = STELLAR_VELOCITY_LIMIT_KMS - np.abs(velocity_kms)
    galaxy_lower_margin = z - GALAXY_Z_MIN
    galaxy_upper_margin = GALAXY_Z_MAX - z
    quasar_lower_margin = z - QUASAR_Z_MIN
    quasar_upper_margin = QUASAR_Z_MAX - z

    stellar_inside = stellar_margin >= 0.0
    galaxy_inside = (galaxy_lower_margin >= 0.0) & (galaxy_upper_margin >= 0.0)
    quasar_inside = (quasar_lower_margin >= 0.0) & (quasar_upper_margin >= 0.0)

    features["inside_stellar_velocity_window"] = stellar_inside
    features["inside_galaxy_redshift_window"] = galaxy_inside
    features["inside_quasar_redshift_window"] = quasar_inside

    features["stellar_velocity_margin_norm"] = np.clip(
        stellar_margin / STELLAR_VELOCITY_LIMIT_KMS,
        -NORMALIZED_DISTANCE_CLIP,
        NORMALIZED_DISTANCE_CLIP,
    )
    features["galaxy_redshift_lower_margin_norm"] = np.clip(
        galaxy_lower_margin / (GALAXY_Z_MAX - GALAXY_Z_MIN),
        -NORMALIZED_DISTANCE_CLIP,
        NORMALIZED_DISTANCE_CLIP,
    )
    features["galaxy_redshift_upper_margin_norm"] = np.clip(
        galaxy_upper_margin / (GALAXY_Z_MAX - GALAXY_Z_MIN),
        -NORMALIZED_DISTANCE_CLIP,
        NORMALIZED_DISTANCE_CLIP,
    )
    features["quasar_redshift_lower_margin_norm"] = np.clip(
        quasar_lower_margin / (QUASAR_Z_MAX - QUASAR_Z_MIN),
        -NORMALIZED_DISTANCE_CLIP,
        NORMALIZED_DISTANCE_CLIP,
    )
    features["quasar_redshift_upper_margin_norm"] = np.clip(
        quasar_upper_margin / (QUASAR_Z_MAX - QUASAR_Z_MIN),
        -NORMALIZED_DISTANCE_CLIP,
        NORMALIZED_DISTANCE_CLIP,
    )

    features["stellar_signed_boundary_distance_kms"] = np.clip(
        stellar_margin,
        -NORMALIZED_DISTANCE_CLIP * STELLAR_VELOCITY_LIMIT_KMS,
        NORMALIZED_DISTANCE_CLIP * STELLAR_VELOCITY_LIMIT_KMS,
    )
    features["galaxy_signed_boundary_distance_z"] = np.clip(
        np.minimum(galaxy_lower_margin, galaxy_upper_margin),
        -NORMALIZED_DISTANCE_CLIP,
        NORMALIZED_DISTANCE_CLIP,
    )
    features["quasar_signed_boundary_distance_z"] = np.clip(
        np.minimum(quasar_lower_margin, quasar_upper_margin),
        -NORMALIZED_DISTANCE_CLIP,
        NORMALIZED_DISTANCE_CLIP,
    )

    membership_count = (
        stellar_inside.astype("int8")
        + galaxy_inside.astype("int8")
        + quasar_inside.astype("int8")
    )
    membership_code = (
        stellar_inside.astype("int8")
        + 2 * galaxy_inside.astype("int8")
        + 4 * quasar_inside.astype("int8")
    )

    features["template_window_membership_count"] = membership_count
    features["template_window_membership_code"] = membership_code
    features["template_window_membership_pattern"] = pd.Categorical(
        np.select(
            [
                membership_code == 0,
                membership_code == 1,
                membership_code == 2,
                membership_code == 3,
                membership_code == 4,
                membership_code == 5,
                membership_code == 6,
                membership_code == 7,
            ],
            [
                "outside_all",
                "stellar_only",
                "galaxy_only",
                "stellar_galaxy",
                "quasar_only",
                "stellar_quasar",
                "galaxy_quasar",
                "stellar_galaxy_quasar",
            ],
            default="unknown",
        )
    )

    shared_boundaries = (
        -0.01,
        0.0,
        0.0333,
        1.0,
        7.0,
    )
    nearest_shared_boundary_distance = np.minimum.reduce(
        [np.abs(z - boundary) for boundary in shared_boundaries]
    )
    signed_nearest_shared_boundary_distance = nearest_shared_boundary_distance * np.where(
        z >= QUASAR_Z_MIN,
        1.0,
        -1.0,
    )

    features["nearest_shared_redshift_boundary_abs_distance"] = np.clip(
        nearest_shared_boundary_distance,
        0.0,
        NORMALIZED_DISTANCE_CLIP,
    )
    features["nearest_shared_redshift_boundary_signed_distance"] = np.clip(
        signed_nearest_shared_boundary_distance,
        -NORMALIZED_DISTANCE_CLIP,
        NORMALIZED_DISTANCE_CLIP,
    )

    return features


FEATURE_GROUPS = [
    {
        "name": "spectroscopic_redshift_grid_imprint",
        "fn": add_spectroscopic_redshift_grid_imprint,
        "depends_on": [],
        "description": "Encodes redshift velocity transforms, SDSS-style velocity-grid phase, and stellar/galaxy/quasar search-window geometry.",
    }
]