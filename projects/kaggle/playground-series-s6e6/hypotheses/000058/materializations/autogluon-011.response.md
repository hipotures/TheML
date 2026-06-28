import numpy as np
import pandas as pd


EPSILON = 1e-8
LOGIT_CLIP = 12.0
SIGMOID_WIDTH = 0.05
QSO_LOW_EDGE = 2.2
QSO_HIGH_EDGE = 3.5
MAG_DIFF_CLIP = 12.0
WINDOW_WIDTH = 0.8
WINDOW_STEP = 0.4


def _safe_numeric(raw, column, default=0.0):
    if column in raw.columns:
        return pd.to_numeric(raw[column], errors="coerce").astype(float)
    return pd.Series(default, index=raw.index, dtype=float)


def _safe_category(raw, column):
    if column in raw.columns:
        return raw[column].astype("string").fillna("")
    return pd.Series("", index=raw.index, dtype="string")


def _clip_series(values, lower, upper):
    return values.clip(lower=lower, upper=upper)


def _sigmoid(x):
    x = np.clip(x, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-x))


def _logit(p):
    p = np.clip(p, EPSILON, 1.0 - EPSILON)
    return np.clip(np.log(p / (1.0 - p)), -LOGIT_CLIP, LOGIT_CLIP)


def _softmax3(a, b, c):
    stacked = np.vstack([a, b, c])
    stacked = stacked - np.nanmax(stacked, axis=0)
    expv = np.exp(np.clip(stacked, -60.0, 60.0))
    denom = np.maximum(expv.sum(axis=0), EPSILON)
    return expv[0] / denom, expv[1] / denom, expv[2] / denom


def add_redshift_branch_deconvolved_flux_posteriors(raw, deps, aux):
    idx = raw.index

    u = _safe_numeric(raw, "u")
    g = _safe_numeric(raw, "g")
    r = _safe_numeric(raw, "r")
    i = _safe_numeric(raw, "i")
    zmag = _safe_numeric(raw, "z")
    redshift = _safe_numeric(raw, "redshift")
    alpha = _safe_numeric(raw, "alpha")
    delta = _safe_numeric(raw, "delta")
    spectral_type = _safe_category(raw, "spectral_type")
    galaxy_population = _safe_category(raw, "galaxy_population")

    ui = _clip_series(u - i, -MAG_DIFF_CLIP, MAG_DIFF_CLIP)
    gi = _clip_series(g - i, -MAG_DIFF_CLIP, MAG_DIFF_CLIP)
    ri = _clip_series(r - i, -MAG_DIFF_CLIP, MAG_DIFF_CLIP)
    zi = _clip_series(zmag - i, -MAG_DIFF_CLIP, MAG_DIFF_CLIP)
    ug = _clip_series(u - g, -MAG_DIFF_CLIP, MAG_DIFF_CLIP)
    gr = _clip_series(g - r, -MAG_DIFF_CLIP, MAG_DIFF_CLIP)
    rz = _clip_series(r - zmag, -MAG_DIFF_CLIP, MAG_DIFF_CLIP)

    flux_scale = -0.4 * np.log(10.0)
    log_flux_ui = flux_scale * ui
    log_flux_gi = flux_scale * gi
    log_flux_ri = flux_scale * ri
    log_flux_zi = flux_scale * zi

    color_slope_blue = ug + gr
    color_slope_red = ri + zi
    color_curvature = (u - g) - (g - r) + (r - i) - (i - zmag)
    flux_radius = np.sqrt(
        np.square(log_flux_ui)
        + np.square(log_flux_gi)
        + np.square(log_flux_ri)
        + np.square(log_flux_zi)
    )
    optical_compactness = np.sqrt(np.square(gr) + np.square(ri) + np.square(rz))

    finite_i = i.replace([np.inf, -np.inf], np.nan)
    i_min = finite_i.min(skipna=True)
    i_max = finite_i.max(skipna=True)
    if not np.isfinite(i_min) or not np.isfinite(i_max) or i_max <= i_min:
        brightness_position = pd.Series(0.5, index=idx, dtype=float)
        edge_distance = pd.Series(0.0, index=idx, dtype=float)
        window_density = pd.Series(1.0, index=idx, dtype=float)
    else:
        brightness_position = ((i - i_min) / max(i_max - i_min, EPSILON)).clip(0.0, 1.0)
        edge_distance = np.minimum(i - i_min, i_max - i).clip(lower=0.0)
        centers = np.arange(i_min, i_max + WINDOW_STEP, WINDOW_STEP)
        if centers.size == 0:
            window_density = pd.Series(1.0, index=idx, dtype=float)
        else:
            diffs = np.abs(i.to_numpy(dtype=float)[:, None] - centers[None, :])
            weights = np.maximum(0.0, 1.0 - diffs / max(WINDOW_WIDTH / 2.0, EPSILON))
            window_density = pd.Series(np.maximum(weights.sum(axis=1), EPSILON), index=idx)

    redshift_nonneg = redshift.clip(lower=0.0)
    low_gate_raw = 1.0 - _sigmoid((redshift_nonneg - QSO_LOW_EDGE) / SIGMOID_WIDTH)
    high_gate_raw = _sigmoid((redshift_nonneg - QSO_HIGH_EDGE) / SIGMOID_WIDTH)
    mid_gate_raw = np.maximum(
        EPSILON,
        _sigmoid((redshift_nonneg - QSO_LOW_EDGE) / SIGMOID_WIDTH)
        * (1.0 - _sigmoid((redshift_nonneg - QSO_HIGH_EDGE) / SIGMOID_WIDTH)),
    )
    branch_gate_sum = np.maximum(low_gate_raw + mid_gate_raw + high_gate_raw, EPSILON)
    qso_low_gate = low_gate_raw / branch_gate_sum
    qso_mid_gate = mid_gate_raw / branch_gate_sum
    qso_high_gate = high_gate_raw / branch_gate_sum

    is_ob = spectral_type.eq("O/B").astype(float)
    is_af = spectral_type.eq("A/F").astype(float)
    is_gk = spectral_type.eq("G/K").astype(float)
    is_m = spectral_type.eq("M").astype(float)
    is_red_sequence = galaxy_population.eq("Red_Sequence").astype(float)
    is_blue_cloud = galaxy_population.eq("Blue_Cloud").astype(float)

    star_score = (
        1.55 * is_m
        + 1.15 * is_gk
        + 0.55 * is_af
        - 1.70 * np.log1p(redshift_nonneg)
        - 0.18 * np.abs(gr - 0.75)
        - 0.16 * np.abs(ri - 0.35)
        - 0.10 * flux_radius
        + 0.12 * (1.0 - brightness_position)
    )

    galaxy_score = (
        1.25 * is_red_sequence
        + 0.55 * is_gk
        + 0.35 * is_m
        + 0.95 * np.exp(-np.square((redshift_nonneg - 0.45) / 0.55))
        + 0.35 * np.exp(-np.square((redshift_nonneg - 0.95) / 0.75))
        + 0.18 * np.maximum(color_slope_red, -2.0)
        - 0.09 * optical_compactness
        + 0.10 * brightness_position
    )

    qso_score = (
        1.20 * is_ob
        + 0.75 * is_af
        + 0.60 * is_blue_cloud
        + 0.72 * np.log1p(redshift_nonneg)
        + 0.28 * np.abs(color_curvature)
        + 0.16 * np.maximum(-color_slope_blue, -2.0)
        - 0.06 * np.abs(delta / 90.0)
        - 0.04 * np.abs(np.sin(np.deg2rad(alpha)))
    )

    star_prob, galaxy_prob, qso_prob = _softmax3(star_score, galaxy_score, qso_score)

    qso_low_prob = qso_prob * qso_low_gate
    qso_mid_prob = qso_prob * qso_mid_gate
    qso_high_prob = qso_prob * qso_high_gate

    branch_probs = np.vstack([qso_low_prob, qso_mid_prob, qso_high_prob])
    branch_max = np.maximum(np.nanmax(branch_probs, axis=0), EPSILON)
    branch_sum = np.maximum(branch_probs.sum(axis=0), EPSILON)
    branch_norm = branch_probs / branch_sum
    branch_entropy = -np.sum(branch_norm * np.log(np.clip(branch_norm, EPSILON, 1.0)), axis=0)
    branch_concentration = np.sum(np.square(branch_norm), axis=0)
    branch_halfmax_count = (branch_probs >= (0.5 * branch_max)).sum(axis=0).astype(float)

    out = pd.DataFrame(index=idx)
    out["log_flux_rel_u_i"] = log_flux_ui
    out["log_flux_rel_g_i"] = log_flux_gi
    out["log_flux_rel_r_i"] = log_flux_ri
    out["log_flux_rel_z_i"] = log_flux_zi
    out["flux_color_radius"] = flux_radius
    out["flux_color_curvature"] = color_curvature
    out["brightness_window_position"] = brightness_position
    out["brightness_edge_distance"] = edge_distance
    out["brightness_window_density"] = np.log1p(window_density)

    out["star_posterior_logit"] = _logit(star_prob)
    out["galaxy_posterior_logit"] = _logit(galaxy_prob)
    out["qso_posterior_logit"] = _logit(qso_prob)
    out["qso_low_branch_logit"] = _logit(qso_low_prob)
    out["qso_mid_branch_logit"] = _logit(qso_mid_prob)
    out["qso_high_branch_logit"] = _logit(qso_high_prob)

    out["star_minus_galaxy_margin"] = np.clip(np.log(np.clip(star_prob, EPSILON, 1.0)) - np.log(np.clip(galaxy_prob, EPSILON, 1.0)), -LOGIT_CLIP, LOGIT_CLIP)
    out["qso_minus_star_margin"] = np.clip(np.log(np.clip(qso_prob, EPSILON, 1.0)) - np.log(np.clip(star_prob, EPSILON, 1.0)), -LOGIT_CLIP, LOGIT_CLIP)
    out["qso_minus_galaxy_margin"] = np.clip(np.log(np.clip(qso_prob, EPSILON, 1.0)) - np.log(np.clip(galaxy_prob, EPSILON, 1.0)), -LOGIT_CLIP, LOGIT_CLIP)
    out["qso_low_minus_star_margin"] = np.clip(np.log(np.clip(qso_low_prob, EPSILON, 1.0)) - np.log(np.clip(star_prob, EPSILON, 1.0)), -LOGIT_CLIP, LOGIT_CLIP)
    out["qso_mid_minus_galaxy_margin"] = np.clip(np.log(np.clip(qso_mid_prob, EPSILON, 1.0)) - np.log(np.clip(galaxy_prob, EPSILON, 1.0)), -LOGIT_CLIP, LOGIT_CLIP)
    out["qso_high_minus_star_margin"] = np.clip(np.log(np.clip(qso_high_prob, EPSILON, 1.0)) - np.log(np.clip(star_prob, EPSILON, 1.0)), -LOGIT_CLIP, LOGIT_CLIP)

    out["qso_branch_entropy"] = branch_entropy
    out["qso_branch_concentration"] = branch_concentration
    out["qso_branch_halfmax_count"] = branch_halfmax_count

    return out.replace([np.inf, -np.inf], np.nan).fillna(0.0)


FEATURE_GROUPS = [
    {
        "name": "redshift_branch_deconvolved_flux_posteriors",
        "fn": add_redshift_branch_deconvolved_flux_posteriors,
        "depends_on": [],
        "description": "Leakage-safe covariate-only flux geometry and redshift-branch posterior proxy features.",
    }
]