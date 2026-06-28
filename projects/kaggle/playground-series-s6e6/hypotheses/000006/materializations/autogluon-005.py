import numpy as np
import pandas as pd


BAND_COLUMNS = ("u", "g", "r", "i", "z")
BREAK_POSITIONS = ("ug", "gr", "ri", "iz")
BREAK_EPS = 1e-8
BREAK_FLAT_THRESHOLD = 1e-6


def add_bandpass_break_localization(raw, deps, aux):
    u = pd.to_numeric(raw["u"], errors="coerce").to_numpy(dtype=float, copy=True)
    g = pd.to_numeric(raw["g"], errors="coerce").to_numpy(dtype=float, copy=True)
    r = pd.to_numeric(raw["r"], errors="coerce").to_numpy(dtype=float, copy=True)
    i = pd.to_numeric(raw["i"], errors="coerce").to_numpy(dtype=float, copy=True)
    z = pd.to_numeric(raw["z"], errors="coerce").to_numpy(dtype=float, copy=True)

    log_flux = np.column_stack((-0.4 * u, -0.4 * g, -0.4 * r, -0.4 * i, -0.4 * z))
    signed_slopes = log_flux[:, :-1] - log_flux[:, 1:]
    abs_slopes = np.abs(signed_slopes)

    total_abs = np.sum(abs_slopes, axis=1)
    break_present = total_abs >= BREAK_FLAT_THRESHOLD

    dominant_idx = np.argmax(abs_slopes, axis=1)
    row_idx = np.arange(len(raw))

    dominant_signed = signed_slopes[row_idx, dominant_idx]
    dominant_abs = abs_slopes[row_idx, dominant_idx]
    direction_sign = np.sign(dominant_signed)

    weights = abs_slopes / (total_abs[:, None] + BREAK_EPS)
    soft_position = np.sum(weights * np.array((1.0, 2.0, 3.0, 4.0)), axis=1)
    entropy = -np.sum(weights * np.log(np.maximum(weights, BREAK_EPS)), axis=1) / np.log(4.0)

    blue_signed_mean = np.zeros(len(raw), dtype=float)
    red_signed_mean = np.zeros(len(raw), dtype=float)
    blue_abs_mean = np.zeros(len(raw), dtype=float)
    red_abs_mean = np.zeros(len(raw), dtype=float)

    for pos_idx in range(4):
        mask = dominant_idx == pos_idx
        if not np.any(mask):
            continue
        if pos_idx > 0:
            blue_signed_mean[mask] = np.mean(signed_slopes[mask, :pos_idx], axis=1)
            blue_abs_mean[mask] = np.mean(abs_slopes[mask, :pos_idx], axis=1)
        if pos_idx < 3:
            red_signed_mean[mask] = np.mean(signed_slopes[mask, pos_idx + 1:], axis=1)
            red_abs_mean[mask] = np.mean(abs_slopes[mask, pos_idx + 1:], axis=1)

    avg_signed_side_mean = 0.5 * (blue_signed_mean + red_signed_mean)
    avg_abs_side_mean = 0.5 * (blue_abs_mean + red_abs_mean)

    signed_local_contrast = dominant_signed - avg_signed_side_mean
    abs_local_contrast = dominant_abs - avg_abs_side_mean
    asymmetry = blue_abs_mean - red_abs_mean

    neighbor_sign_change_count = np.zeros(len(raw), dtype=float)
    for pos_idx in range(4):
        mask = dominant_idx == pos_idx
        if not np.any(mask):
            continue
        count = np.zeros(np.sum(mask), dtype=float)
        if pos_idx > 0:
            count += (
                np.sign(signed_slopes[mask, pos_idx - 1])
                != np.sign(signed_slopes[mask, pos_idx])
            ).astype(float)
        if pos_idx < 3:
            count += (
                np.sign(signed_slopes[mask, pos_idx])
                != np.sign(signed_slopes[mask, pos_idx + 1])
            ).astype(float)
        neighbor_sign_change_count[mask] = count

    position_labels = np.array(BREAK_POSITIONS, dtype=object)[dominant_idx]
    position_labels = np.where(break_present, position_labels, "none")

    out = pd.DataFrame(index=raw.index)
    out["break_present"] = break_present.astype(np.int8)
    out["dominant_position"] = pd.Series(position_labels, index=raw.index, dtype="category")
    out["dominant_signed_amplitude"] = np.where(break_present, dominant_signed, 0.0)
    out["dominant_abs_amplitude"] = np.where(break_present, dominant_abs, 0.0)
    out["direction_sign"] = np.where(break_present, direction_sign, 0.0)
    out["sharpness"] = np.where(break_present, dominant_abs / (total_abs + BREAK_EPS), 0.0)
    out["soft_position"] = np.where(break_present, soft_position, 0.0)
    out["normalized_entropy"] = np.where(break_present, entropy, 0.0)
    out["blue_signed_mean"] = np.where(break_present, blue_signed_mean, 0.0)
    out["red_signed_mean"] = np.where(break_present, red_signed_mean, 0.0)
    out["blue_abs_mean"] = np.where(break_present, blue_abs_mean, 0.0)
    out["red_abs_mean"] = np.where(break_present, red_abs_mean, 0.0)
    out["signed_local_contrast"] = np.where(break_present, signed_local_contrast, 0.0)
    out["abs_local_contrast"] = np.where(break_present, abs_local_contrast, 0.0)
    out["asymmetry"] = np.where(break_present, asymmetry, 0.0)
    out["neighbor_sign_change_count"] = np.where(break_present, neighbor_sign_change_count, 0.0)

    for pos_idx, pos_name in enumerate(BREAK_POSITIONS):
        out["dominant_is_" + pos_name] = (
            break_present & (dominant_idx == pos_idx)
        ).astype(np.int8)

    return out


FEATURE_GROUPS = [
    {
        "name": "bandpass_break_localization",
        "fn": add_bandpass_break_localization,
        "depends_on": [],
        "description": "Encodes the strongest adjacent ugriz spectral break, its confidence, direction, position, and local context.",
    }
]