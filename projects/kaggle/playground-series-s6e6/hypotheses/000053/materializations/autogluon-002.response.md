import numpy as np
import pandas as pd

UGRIZ_WAVELENGTHS = (3551.0, 4670.0, 6170.0, 7480.0, 8930.0)
_SCALE_EPSILON = 1e-6


def _local_color_scale(mag_matrix):
    local_colors = np.column_stack(
        (
            mag_matrix[:, 0] - mag_matrix[:, 1],
            mag_matrix[:, 1] - mag_matrix[:, 2],
            mag_matrix[:, 2] - mag_matrix[:, 3],
            mag_matrix[:, 3] - mag_matrix[:, 4],
        )
    )
    return np.median(np.abs(local_colors), axis=1) + _SCALE_EPSILON


def _compute_break_features(redshift_tilde, mag_matrix, local_scale, break_wavelength):
    n = redshift_tilde.shape[0]
    wavelengths = np.array(UGRIZ_WAVELENGTHS, dtype=float)
    ln_w = np.log(wavelengths)

    observed_break = break_wavelength * (1.0 + redshift_tilde)
    intervals = np.searchsorted(wavelengths, observed_break, side="right") - 1
    valid = (intervals >= 0) & (intervals <= 3)

    miss = (~valid).astype(np.int8)
    jump_obs = np.zeros(n, dtype=float)
    residual = np.zeros(n, dtype=float)
    confidence = np.zeros(n, dtype=float)

    row_index = np.arange(n)
    for j in (0, 1, 2, 3):
        sel = valid & (intervals == j)
        if not np.any(sel):
            continue

        idx = row_index[sel]
        mj = mag_matrix[idx, j]
        mj1 = mag_matrix[idx, j + 1]
        obs_jump = mj - mj1
        jump_obs[idx] = obs_jump

        if j == 0:
            left_cont = 1
            right_cont = 2
        elif j == 1:
            left_cont = 0
            right_cont = 3
        elif j == 2:
            left_cont = 1
            right_cont = 4
        else:
            left_cont = 2
            right_cont = 4

        cont_left = mag_matrix[idx, left_cont]
        cont_right = mag_matrix[idx, right_cont]
        slope_cont = (cont_right - cont_left) / (ln_w[right_cont] - ln_w[left_cont])

        delta_ln = ln_w[j] - ln_w[j + 1]
        jump_exp = slope_cont * delta_ln
        residual[idx] = (obs_jump - jump_exp) / local_scale[idx]

        delta_t = (np.log(observed_break[idx]) - ln_w[j]) / (ln_w[j + 1] - ln_w[j])
        confidence[idx] = np.clip(1.0 - 2.0 * np.abs(delta_t - 0.5), 0.0, 1.0)

    residual = np.clip(residual, -8.0, 8.0)
    boundary = (confidence < 0.2).astype(np.int8)

    return {
        "jump_obs": jump_obs,
        "residual": residual,
        "confidence": confidence,
        "miss": miss,
        "boundary": boundary,
    }


def add_dual_restframe_break_alignment(raw, deps, aux):
    mags = raw[["u", "g", "r", "i", "z"]].to_numpy(dtype=float)
    redshift = pd.to_numeric(raw["redshift"], errors="coerce").to_numpy(dtype=float)
    redshift_tilde = np.clip(redshift, 0.0, 7.0)

    local_scale = _local_color_scale(mags)

    lya = _compute_break_features(redshift_tilde, mags, local_scale, 1216.0)
    balmer = _compute_break_features(redshift_tilde, mags, local_scale, 4000.0)

    lya_jump_conf = lya["jump_obs"] * lya["confidence"]
    balmer_jump_conf = balmer["jump_obs"] * balmer["confidence"]
    lya_resid_conf = lya["residual"] * lya["confidence"]
    balmer_resid_conf = balmer["residual"] * balmer["confidence"]

    lya_jump_conf_sc = lya_jump_conf * (1 - lya["miss"]).astype(float)
    balmer_jump_conf_sc = balmer_jump_conf * (1 - balmer["miss"]).astype(float)
    lya_resid_conf_sc = lya_resid_conf * (1 - lya["miss"]).astype(float)
    balmer_resid_conf_sc = balmer_resid_conf * (1 - balmer["miss"]).astype(float)

    break_balance = lya_resid_conf - balmer_resid_conf
    break_abs_diff = np.abs(lya_resid_conf) - np.abs(balmer_resid_conf)
    break_balance_sc = lya_resid_conf_sc - balmer_resid_conf_sc
    break_abs_diff_sc = np.abs(lya_resid_conf_sc) - np.abs(balmer_resid_conf_sc)

    lya_regime = (redshift_tilde >= 1.9) & (redshift_tilde <= 6.5)
    balmer_regime = (redshift_tilde >= 0.0) & (redshift_tilde <= 1.3)

    features = {
        "lya_jump_obs": lya["jump_obs"],
        "lya_residual_norm": lya["residual"],
        "lya_jump_conf": lya_jump_conf,
        "lya_residual_conf": lya_resid_conf,
        "lya_jump_conf_sc": lya_jump_conf_sc,
        "lya_residual_conf_sc": lya_resid_conf_sc,
        "lya_miss": lya["miss"],
        "lya_boundary": lya["boundary"],

        "balmer_jump_obs": balmer["jump_obs"],
        "balmer_residual_norm": balmer["residual"],
        "balmer_jump_conf": balmer_jump_conf,
        "balmer_residual_conf": balmer_resid_conf,
        "balmer_jump_conf_sc": balmer_jump_conf_sc,
        "balmer_residual_conf_sc": balmer_resid_conf_sc,
        "balmer_miss": balmer["miss"],
        "balmer_boundary": balmer["boundary"],

        "lya_regime": lya_regime.astype(np.int8),
        "balmer_regime": balmer_regime.astype(np.int8),

        "break_balance": break_balance,
        "break_abs_diff": break_abs_diff,
        "break_balance_sc": break_balance_sc,
        "break_abs_diff_sc": break_abs_diff_sc,
        "break_confidence_delta": lya["confidence"] - balmer["confidence"],
    }

    return pd.DataFrame(features, index=raw.index)


FEATURE_GROUPS = [
    {
        "name": "dual_restframe_break_alignment",
        "fn": add_dual_restframe_break_alignment,
        "depends_on": [],
        "description": "Constructs break-residual and confidence-weighted competition features around redshifted 1216Å and 4000Å discontinuities with regime and miss/boundary handling.",
    }
]