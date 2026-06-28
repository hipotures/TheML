import numpy as np
import pandas as pd


BANDS = ("u", "g", "r", "i", "z")
SDSS_EFFECTIVE_WAVELENGTHS = (3551.0, 4670.0, 6170.0, 7480.0, 8930.0)
BREAK_WAVELENGTHS = (1216.0, 4000.0)
BREAK_LABELS = ("lya", "balmer")


def _continuum_expected_jumps(mag, log_wavelengths):
    n_rows = mag.shape[0]
    expected = np.zeros((n_rows, 4), dtype=float)
    band_index = np.arange(5, dtype=int)

    for k in range(4):
        outside = (band_index != k) & (band_index != k + 1)
        x_fit = log_wavelengths[outside]
        y_fit = mag[:, outside]

        if x_fit.size < 2:
            if k <= 1:
                fit_idx = np.array((k + 2, k + 3), dtype=int)
            else:
                fit_idx = np.array((k - 2, k - 1), dtype=int)
            x_fit = log_wavelengths[fit_idx]
            y_fit = mag[:, fit_idx]

        x_centered = x_fit - np.mean(x_fit)
        denom = np.sum(x_centered * x_centered)

        if denom <= 0:
            slope = np.zeros(n_rows, dtype=float)
        else:
            y_centered = y_fit - np.mean(y_fit, axis=1, keepdims=True)
            slope = np.sum(y_centered * x_centered, axis=1) / denom

        expected[:, k] = slope * (log_wavelengths[k] - log_wavelengths[k + 1])

    return expected


def _break_features(label, break_wavelength, clipped_redshift, mag, log_wavelengths, jumps, expected_jumps, denom):
    n_rows = mag.shape[0]
    xb = np.log(break_wavelength * (1.0 + clipped_redshift))
    valid = (xb >= log_wavelengths[0]) & (xb <= log_wavelengths[-1])

    widths = log_wavelengths[1:] - log_wavelengths[:-1]
    centers = (log_wavelengths[:-1] + log_wavelengths[1:]) * 0.5
    nearest_widths = np.empty(4, dtype=float)
    nearest_widths[0] = widths[1]
    nearest_widths[1] = min(widths[0], widths[2])
    nearest_widths[2] = min(widths[1], widths[3])
    nearest_widths[3] = widths[2]
    half_support = 0.5 * widths + 0.5 * nearest_widths

    weights = np.maximum(0.0, 1.0 - np.abs(xb[:, None] - centers[None, :]) / half_support[None, :])
    weights[~valid, :] = 0.0
    weight_sum = weights.sum(axis=1)
    nonzero = weight_sum > 0.0
    weights[nonzero, :] = weights[nonzero, :] / weight_sum[nonzero, None]

    residuals = (jumps - expected_jumps) / denom[:, None]
    residuals = np.clip(residuals, -8.0, 8.0)

    raw_jump = np.sum(weights * jumps, axis=1)
    expected_jump = np.sum(weights * expected_jumps, axis=1)
    residual = np.sum(weights * residuals, axis=1)
    abs_residual = np.sum(weights * np.abs(residuals), axis=1)
    sharpness = np.max(weights, axis=1)
    edge = (sharpness < 0.45).astype(np.int8)
    dominant_interval = np.argmax(weights, axis=1).astype(np.int8)
    dominant_interval[~nonzero] = -1

    out = {
        f"{label}_miss": (~valid).astype(np.int8),
        f"{label}_raw_jump": raw_jump,
        f"{label}_expected_jump": expected_jump,
        f"{label}_residual": residual,
        f"{label}_abs_residual": abs_residual,
        f"{label}_alignment_sharpness": sharpness,
        f"{label}_edge": edge,
        f"{label}_dominant_interval": dominant_interval,
    }

    for k in range(4):
        out[f"{label}_interval_{k}_weight"] = weights[:, k]
        out[f"{label}_interval_{k}_dominant"] = ((dominant_interval == k) & nonzero).astype(np.int8)

    return out


def add_dual_restframe_break_alignment(raw, deps, aux):
    index = raw.index
    mag = raw.loc[:, BANDS].astype(float).to_numpy(copy=True)
    redshift = raw["redshift"].astype(float).to_numpy(copy=True)
    clipped_redshift = np.clip(np.nan_to_num(redshift, nan=0.0, posinf=7.0, neginf=0.0), 0.0, 7.0)

    wavelengths = np.asarray(SDSS_EFFECTIVE_WAVELENGTHS, dtype=float)
    log_wavelengths = np.log(wavelengths)

    jumps = mag[:, :-1] - mag[:, 1:]
    abs_jumps = np.abs(jumps)
    denom = np.median(abs_jumps, axis=1) + 0.05
    denom = np.where(np.isfinite(denom) & (denom > 0.0), denom, 0.05)

    expected_jumps = _continuum_expected_jumps(mag, log_wavelengths)

    feature_dict = {}
    break_outputs = {}

    for label, break_wavelength in zip(BREAK_LABELS, BREAK_WAVELENGTHS):
        block = _break_features(
            label,
            break_wavelength,
            clipped_redshift,
            mag,
            log_wavelengths,
            jumps,
            expected_jumps,
            denom,
        )
        feature_dict.update(block)
        break_outputs[label] = block

    lya_score = break_outputs["lya"]["lya_residual"] * break_outputs["lya"]["lya_alignment_sharpness"]
    balmer_score = break_outputs["balmer"]["balmer_residual"] * break_outputs["balmer"]["balmer_alignment_sharpness"]
    lya_abs_score = break_outputs["lya"]["lya_abs_residual"] * break_outputs["lya"]["lya_alignment_sharpness"]
    balmer_abs_score = break_outputs["balmer"]["balmer_abs_residual"] * break_outputs["balmer"]["balmer_alignment_sharpness"]

    feature_dict["lya_regime"] = ((clipped_redshift >= 1.8) & (clipped_redshift <= 6.6)).astype(np.int8)
    feature_dict["balmer_regime"] = ((clipped_redshift >= 0.0) & (clipped_redshift <= 1.35)).astype(np.int8)
    feature_dict["valid_pair"] = (
        (1 - break_outputs["lya"]["lya_miss"]) * (1 - break_outputs["balmer"]["balmer_miss"])
    ).astype(np.int8)
    feature_dict["lya_vs_balmer"] = lya_score - balmer_score
    feature_dict["abs_balance"] = lya_abs_score - balmer_abs_score
    feature_dict["signed_ratio"] = lya_score / (np.abs(balmer_score) + 1.0)

    return pd.DataFrame(feature_dict, index=index)


FEATURE_GROUPS = [
    {
        "name": "dual_restframe_break_alignment",
        "fn": add_dual_restframe_break_alignment,
        "depends_on": [],
        "description": "Soft redshift-aligned Lyman and 4000 Angstrom break contrast features from ugriz photometry.",
    }
]