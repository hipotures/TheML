import numpy as np
import pandas as pd

MAG_COLUMN_ORDER = ("u", "g", "r", "i", "z")
OBSERVED_WAVELENGTHS_A = (3551.0, 4686.0, 6165.0, 7481.0, 8931.0)
LYMAN_BREAKS_A = (912.0, 1216.0)


def _compute_lyman_features(redshift, mags, break_wave):
    n_rows = mags.shape[0]
    obs = np.asarray(OBSERVED_WAVELENGTHS_A, dtype=np.float64)

    z_t = np.maximum(redshift, 0.0)
    lambda_rest = obs[None, :] / (1.0 + z_t)[:, None]

    # j = insertion index of break_wave in each row of sorted rest-frame centers
    j = np.sum(lambda_rest < break_wave, axis=1).astype(np.int8)
    edge_available = ((j > 0) & (j < lambda_rest.shape[1])).astype(np.int8)

    phase = np.zeros(n_rows, dtype=np.float64)
    edge_quality = np.zeros(n_rows, dtype=np.float64)
    jump = np.zeros(n_rows, dtype=np.float64)
    local = np.zeros(n_rows, dtype=np.float64)
    slope = np.zeros(n_rows, dtype=np.float64)

    active = edge_available.astype(bool)
    if active.any():
        active_idx = np.nonzero(active)[0]
        j_in = j[active_idx]

        left_idx = j_in - 1
        right_idx = j_in

        lambda_left = lambda_rest[active_idx, left_idx]
        lambda_right = lambda_rest[active_idx, right_idx]

        phase_vals = (break_wave - lambda_left) / (lambda_right - lambda_left)
        phase_vals = np.clip(phase_vals, 0.0, 1.0)
        phase[active_idx] = phase_vals
        edge_quality[active_idx] = np.clip(1.0 - 2.0 * np.abs(phase_vals - 0.5), 0.0, 1.0)

        local[active_idx] = mags[active_idx, left_idx] - mags[active_idx, right_idx]
        slope[active_idx] = (mags[active_idx, right_idx] - mags[active_idx, left_idx]) / (
            np.log(lambda_right) - np.log(lambda_left)
        )

        # Blue-side and red-side baselines by case of j
        if np.any(j_in == 1):
            mask = j_in == 1
            idx = active_idx[mask]
            m_blue = mags[idx, 0]
            m_red = (mags[idx, 1] + mags[idx, 2]) * 0.5
            jump[idx] = m_blue - m_red

        if np.any(j_in == 2):
            mask = j_in == 2
            idx = active_idx[mask]
            m_blue = (mags[idx, 0] + mags[idx, 1]) * 0.5
            m_red = (mags[idx, 2] + mags[idx, 3]) * 0.5
            jump[idx] = m_blue - m_red

        if np.any(j_in == 3):
            mask = j_in == 3
            idx = active_idx[mask]
            m_blue = (mags[idx, 1] + mags[idx, 2]) * 0.5
            m_red = (mags[idx, 3] + mags[idx, 4]) * 0.5
            jump[idx] = m_blue - m_red

        if np.any(j_in == 4):
            mask = j_in == 4
            idx = active_idx[mask]
            m_blue = (mags[idx, 2] + mags[idx, 3]) * 0.5
            m_red = mags[idx, 4]
            jump[idx] = m_blue - m_red

    break_bin = j.astype(np.int8)
    break_label = str(int(break_wave))
    prefix = break_label.replace(".", "")

    features = {
        f"edge_available_{prefix}": edge_available,
        f"phase_{prefix}": phase,
        f"edge_quality_{prefix}": edge_quality,
        f"break_bin_{prefix}": break_bin,
        f"jump_{prefix}": jump,
        f"local_{prefix}": local,
        f"slope_{prefix}": slope,
        f"jump_x_quality_{prefix}": jump * edge_quality,
        f"local_x_quality_{prefix}": local * edge_quality,
        f"slope_x_quality_{prefix}": slope * edge_quality,
    }

    for bin_id in range(6):
        features[f"break_bin_{prefix}_is_{bin_id}"] = (break_bin == bin_id).astype(np.uint8)

    return features


def add_redshifted_lyman_discontinuity(raw, deps, aux):
    redshift = raw["redshift"].to_numpy(dtype=np.float64, copy=False)
    mags = np.column_stack([raw[col].to_numpy(dtype=np.float64, copy=False) for col in MAG_COLUMN_ORDER])

    features = {}
    for break_wave in LYMAN_BREAKS_A:
        features.update(_compute_lyman_features(redshift, mags, break_wave))

    return pd.DataFrame(features, index=raw.index)


FEATURE_GROUPS = [
    {
        "name": "redshifted_lyman_discontinuity",
        "fn": add_redshifted_lyman_discontinuity,
        "depends_on": [],
        "description": "Encodes rest-frame Lyman-break phase, contrast, and local slope geometry across ugriz as a function of redshift with boundary-aware quality gating.",
    }
]