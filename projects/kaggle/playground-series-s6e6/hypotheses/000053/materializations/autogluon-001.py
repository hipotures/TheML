import numpy as np
import pandas as pd

_FEATURE_BANDS = ("u", "g", "r", "i", "z")
_SDSS_WAVELENGTHS_AA = (3551.0, 4670.0, 6170.0, 7480.0, 8930.0)
_LYA_BREAK_AA = 1216.0
_BALMER_BREAK_AA = 4000.0
_RESIDUAL_RANGE = (-8.0, 8.0)
_EPS = 1e-6


def _compute_break_features(raw_magnitudes, redshift, break_wavelength):
    wavelengths = np.array(_SDSS_WAVELENGTHS_AA, dtype=np.float64)
    log_wavelengths = np.log(wavelengths)
    n = len(redshift)

    clipped_z = np.clip(np.asarray(redshift, dtype=np.float64), 0.0, 7.0)
    rest = break_wavelength * (1.0 + clipped_z)

    pair_idx = np.full(n, -1, dtype=np.int8)
    for band in range(len(wavelengths) - 1):
        pair_mask = (rest > wavelengths[band]) & (rest <= wavelengths[band + 1])
        pair_idx[pair_mask] = band

    raw_jump = np.zeros(n, dtype=np.float64)
    norm_jump = np.zeros(n, dtype=np.float64)
    valid_pair = np.zeros(n, dtype=bool)

    for band in range(len(wavelengths) - 1):
        rows = np.flatnonzero(pair_idx == band)
        if rows.size == 0:
            continue

        mj = raw_magnitudes[rows, band]
        mj1 = raw_magnitudes[rows, band + 1]
        base_ok = np.isfinite(mj) & np.isfinite(mj1)
        if not np.any(base_ok):
            continue

        obs_jump = mj - mj1

        if band + 2 < len(wavelengths):
            mj2 = raw_magnitudes[rows, band + 2]
            ok = base_ok & np.isfinite(mj2)
            if np.any(ok):
                ridx = rows[ok]
                obs = obs_jump[ok]
                slope = (mj2[ok] - mj1[ok]) / (
                    log_wavelengths[band + 2] - log_wavelengths[band + 1]
                )
                predicted_left = mj1[ok] - slope * (
                    log_wavelengths[band] - log_wavelengths[band + 1]
                )
                expected_jump = predicted_left - mj1[ok]
                denom = np.abs(mj1[ok] - mj2[ok]) + _EPS
                raw_jump[ridx] = obs
                norm_jump[ridx] = (obs - expected_jump) / denom
                valid_pair[ridx] = True

        elif band == 0:
            mj2 = raw_magnitudes[rows, 2]
            ok = base_ok & np.isfinite(mj2)
            if np.any(ok):
                ridx = rows[ok]
                obs = obs_jump[ok]
                expected_jump = ((mj1[ok] + mj2[ok]) / 2.0) - mj1[ok]
                denom = np.abs(mj1[ok] - mj2[ok]) + _EPS
                raw_jump[ridx] = obs
                norm_jump[ridx] = (obs - expected_jump) / denom
                valid_pair[ridx] = True

        else:
            ridx = rows[base_ok]
            obs = obs_jump[base_ok]
            raw_jump[ridx] = obs
            norm_jump[ridx] = obs / 1.0
            valid_pair[ridx] = True

    return raw_jump, norm_jump, valid_pair


def add_dual_restframe_break_alignment(raw, deps, aux):
    magnitudes = raw.loc[:, _FEATURE_BANDS].to_numpy(dtype=np.float64, copy=False)
    redshift = np.asarray(raw["redshift"].to_numpy(dtype=np.float64), dtype=np.float64)
    clipped_z = np.clip(redshift, 0.0, 7.0)

    lya_raw, lya_norm, lya_valid = _compute_break_features(
        magnitudes, clipped_z, _LYA_BREAK_AA
    )
    balmer_raw, balmer_norm, balmer_valid = _compute_break_features(
        magnitudes, clipped_z, _BALMER_BREAK_AA
    )

    lya_norm = np.clip(lya_norm, _RESIDUAL_RANGE[0], _RESIDUAL_RANGE[1])
    balmer_norm = np.clip(balmer_norm, _RESIDUAL_RANGE[0], _RESIDUAL_RANGE[1])

    lya_visible = (clipped_z > 1.9) & (clipped_z < 6.5)
    balmer_visible = (clipped_z > -0.1) & (clipped_z < 1.3)

    break_balance = np.zeros(len(raw), dtype=np.float64)
    both_valid = lya_valid & balmer_valid
    np.divide(
        lya_norm,
        np.abs(balmer_norm) + 1.0,
        out=break_balance,
        where=both_valid,
    )
    break_balance = np.clip(break_balance, _RESIDUAL_RANGE[0], _RESIDUAL_RANGE[1])

    break_regime_flags = (
        lya_valid.astype(np.int8)
        | (balmer_valid.astype(np.int8) << 1)
        | (lya_visible.astype(np.int8) << 2)
        | (balmer_visible.astype(np.int8) << 3)
    )

    return pd.DataFrame(
        {
            "lya_raw": lya_raw,
            "lya_norm": lya_norm,
            "balmer_raw": balmer_raw,
            "balmer_norm": balmer_norm,
            "break_balance": break_balance,
            "break_regime_flags": break_regime_flags.astype(np.int16),
        },
        index=raw.index,
    )


FEATURE_GROUPS = [
    {
        "name": "dual_restframe_break_alignment",
        "fn": add_dual_restframe_break_alignment,
        "depends_on": [],
        "description": (
            "Builds dual rest-frame break features by locating Lyman and 4000 Å break "
            "positions in observed ugriz bands, computing jump residuals versus smooth local "
            "continuum trends, and emitting regime-aware flags."
        ),
    }
]