import numpy as np
import pandas as pd


LOG_FLUX_SCALE = -0.4 * np.log(10.0)
EPS = 1e-8


def _sigmoid(x):
    x = np.clip(x, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-x))


def _safe_logit(p):
    p = np.clip(np.asarray(p, dtype=float), EPS, 1.0 - EPS)
    return np.clip(np.log(p / (1.0 - p)), -12.0, 12.0)


def _entropy3(p):
    p = np.clip(np.asarray(p, dtype=float), EPS, 1.0)
    return -np.sum(p * np.log(p), axis=1)


def add_redshift_branch_deconvolved_flux_posteriors(raw, deps, aux):
    idx = raw.index
    n = len(raw)

    cols = {}
    for c in ("u", "g", "r", "i", "z", "redshift"):
        if c in raw.columns:
            cols[c] = pd.to_numeric(raw[c], errors="coerce").to_numpy(dtype=float)
        else:
            cols[c] = np.full(n, np.nan, dtype=float)

    u = cols["u"]
    g = cols["g"]
    r = cols["r"]
    i_mag = cols["i"]
    z_mag = cols["z"]
    redshift = cols["redshift"]

    finite_i = np.isfinite(i_mag)
    i_fill = np.nanmedian(i_mag[finite_i]) if finite_i.any() else 20.0
    i_safe = np.where(finite_i, i_mag, i_fill)

    rel = np.column_stack(
        [
            LOG_FLUX_SCALE * (u - i_safe),
            LOG_FLUX_SCALE * (g - i_safe),
            LOG_FLUX_SCALE * (r - i_safe),
            LOG_FLUX_SCALE * (z_mag - i_safe),
        ]
    )

    for j in range(rel.shape[1]):
        x = rel[:, j]
        finite = np.isfinite(x)
        if finite.any():
            lo, hi = np.nanpercentile(x[finite], [0.1, 99.9])
            med = np.nanmedian(x[finite])
            rel[:, j] = np.clip(np.where(finite, x, med), lo, hi)
        else:
            rel[:, j] = 0.0

    finite_z = np.isfinite(redshift)
    z_fill = np.nanmedian(redshift[finite_z]) if finite_z.any() else 0.0
    z_safe = np.where(finite_z, redshift, z_fill)

    low_gate = 1.0 - _sigmoid((z_safe - 2.2) / 0.05)
    high_gate = _sigmoid((z_safe - 3.5) / 0.05)
    mid_gate = np.clip(1.0 - low_gate - high_gate, 0.0, 1.0)
    branch_raw = np.column_stack([low_gate, mid_gate, high_gate])
    branch_mass = branch_raw / np.clip(branch_raw.sum(axis=1, keepdims=True), EPS, None)

    color_spread = np.std(rel, axis=1)
    blue_slope = rel[:, 0] - rel[:, 3]
    gri_curve = rel[:, 1] - 2.0 * rel[:, 2] + rel[:, 3]
    uv_excess = rel[:, 0] - rel[:, 1]

    i_centered = i_safe - np.nanmedian(i_safe)
    i_scale = np.nanpercentile(np.abs(i_centered), 75) if finite_i.any() else 1.0
    i_scale = max(float(i_scale), 1.0)

    compact_star_score = (
        -0.75 * np.abs(blue_slope)
        -0.45 * color_spread
        -0.20 * np.abs(gri_curve)
        -0.12 * np.abs(z_safe)
        -0.05 * np.abs(i_centered / i_scale)
    )
    galaxy_score = (
        0.60 * np.maximum(z_safe, 0.0)
        + 0.35 * color_spread
        + 0.25 * np.maximum(gri_curve, 0.0)
        - 0.10 * np.abs(uv_excess)
    )
    qso_score = (
        0.42 * np.maximum(z_safe, 0.0)
        + 0.42 * np.abs(uv_excess)
        + 0.30 * np.abs(blue_slope)
        + 0.18 * color_spread
        - 0.08 * np.abs(gri_curve)
    )

    scores = np.column_stack([compact_star_score, galaxy_score, qso_score])
    scores = scores - np.nanmax(scores, axis=1, keepdims=True)
    exp_scores = np.exp(np.clip(scores, -50.0, 50.0))
    class_prob = exp_scores / np.clip(exp_scores.sum(axis=1, keepdims=True), EPS, None)

    star_p = class_prob[:, 0]
    galaxy_p = class_prob[:, 1]
    qso_p = class_prob[:, 2]

    qso_branch = branch_mass * qso_p[:, None]
    branch_norm = qso_branch / np.clip(qso_branch.sum(axis=1, keepdims=True), EPS, None)

    out = pd.DataFrame(index=idx)

    out["star_posterior_logit"] = _safe_logit(star_p)
    out["galaxy_posterior_logit"] = _safe_logit(galaxy_p)
    out["qso_posterior_logit"] = _safe_logit(qso_p)

    out["qso_low_branch_logit"] = _safe_logit(qso_branch[:, 0])
    out["qso_mid_branch_logit"] = _safe_logit(qso_branch[:, 1])
    out["qso_high_branch_logit"] = _safe_logit(qso_branch[:, 2])

    out["star_minus_galaxy_margin"] = np.clip(np.log(np.clip(star_p, EPS, 1.0)) - np.log(np.clip(galaxy_p, EPS, 1.0)), -12.0, 12.0)
    out["qso_minus_star_margin"] = np.clip(np.log(np.clip(qso_p, EPS, 1.0)) - np.log(np.clip(star_p, EPS, 1.0)), -12.0, 12.0)
    out["qso_minus_galaxy_margin"] = np.clip(np.log(np.clip(qso_p, EPS, 1.0)) - np.log(np.clip(galaxy_p, EPS, 1.0)), -12.0, 12.0)

    out["qso_low_minus_star_margin"] = np.clip(np.log(np.clip(qso_branch[:, 0], EPS, 1.0)) - np.log(np.clip(star_p, EPS, 1.0)), -12.0, 12.0)
    out["qso_mid_minus_star_margin"] = np.clip(np.log(np.clip(qso_branch[:, 1], EPS, 1.0)) - np.log(np.clip(star_p, EPS, 1.0)), -12.0, 12.0)
    out["qso_high_minus_star_margin"] = np.clip(np.log(np.clip(qso_branch[:, 2], EPS, 1.0)) - np.log(np.clip(star_p, EPS, 1.0)), -12.0, 12.0)

    out["qso_low_minus_galaxy_margin"] = np.clip(np.log(np.clip(qso_branch[:, 0], EPS, 1.0)) - np.log(np.clip(galaxy_p, EPS, 1.0)), -12.0, 12.0)
    out["qso_mid_minus_galaxy_margin"] = np.clip(np.log(np.clip(qso_branch[:, 1], EPS, 1.0)) - np.log(np.clip(galaxy_p, EPS, 1.0)), -12.0, 12.0)
    out["qso_high_minus_galaxy_margin"] = np.clip(np.log(np.clip(qso_branch[:, 2], EPS, 1.0)) - np.log(np.clip(galaxy_p, EPS, 1.0)), -12.0, 12.0)

    out["qso_branch_concentration"] = np.sum(branch_norm * branch_norm, axis=1)
    out["qso_branch_entropy"] = _entropy3(branch_norm)
    out["qso_branch_halfmax_count"] = (branch_norm >= (0.5 * np.max(branch_norm, axis=1, keepdims=True))).sum(axis=1).astype(np.int8)

    out["flux_color_spread"] = np.clip(color_spread, -12.0, 12.0)
    out["flux_blue_slope"] = np.clip(blue_slope, -12.0, 12.0)
    out["flux_gri_curvature"] = np.clip(gri_curve, -12.0, 12.0)
    out["flux_uv_excess"] = np.clip(uv_excess, -12.0, 12.0)

    return out


FEATURE_GROUPS = [
    {
        "name": "redshift_branch_deconvolved_flux_posteriors",
        "fn": add_redshift_branch_deconvolved_flux_posteriors,
        "depends_on": [],
        "description": "Leakage-safe flux-geometry posterior-style features for class overlap and quasar redshift branch structure.",
    }
]