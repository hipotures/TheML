import numpy as np
import pandas as pd

BANDS = ("u", "g", "r", "i", "z")
LUPTITUDE_B = (1.4e-10, 9.0e-11, 1.2e-10, 1.8e-10, 7.4e-10)
MAGNITUDE_CORRECTIONS = (-0.04, 0.0, 0.0, 0.0, 0.02)
WAVELENGTH_A = (3551.0, 4686.0, 6165.0, 7481.0, 8931.0)
SOFT_COEFF = -0.9210340371976183
EPS = 1e-12


def add_luptitude_regime_flux_features(raw, deps, aux):
    u = raw["u"].to_numpy(dtype=np.float64)
    g = raw["g"].to_numpy(dtype=np.float64)
    r = raw["r"].to_numpy(dtype=np.float64)
    i = raw["i"].to_numpy(dtype=np.float64)
    z = raw["z"].to_numpy(dtype=np.float64)

    m_u = u + MAGNITUDE_CORRECTIONS[0]
    m_g = g + MAGNITUDE_CORRECTIONS[1]
    m_r = r + MAGNITUDE_CORRECTIONS[2]
    m_i = i + MAGNITUDE_CORRECTIONS[3]
    m_z = z + MAGNITUDE_CORRECTIONS[4]

    m_corr = np.column_stack((m_u, m_g, m_r, m_i, m_z))
    b = np.asarray(LUPTITUDE_B, dtype=np.float64)
    log_b = np.log(b)

    f = 2.0 * b * np.sinh(SOFT_COEFF * m_corr - log_b)
    abs_f = np.abs(f)

    floor1 = (abs_f <= (2.0 * b)).astype(np.float64)
    floor2 = (abs_f <= (0.2 * b)).astype(np.float64)

    soft_count = np.sum(floor1, axis=1)
    soft_frac = soft_count / 5.0
    ultra_floor_count = np.sum(floor2, axis=1)
    neg_flux_count = np.sum(f < 0.0, axis=1).astype(np.float64)

    f_tilde = np.sign(f) * np.maximum(abs_f, b)
    v = np.log10(np.abs(f_tilde) + EPS)

    v_u = v[:, 0]
    v_g = v[:, 1]
    v_r = v[:, 2]
    v_i = v[:, 3]
    v_z = v[:, 4]

    d_ug = v_u - v_g
    d_gr = v_g - v_r
    d_ri = v_r - v_i
    d_iz = v_i - v_z

    t = np.asarray(WAVELENGTH_A, dtype=np.float64)
    t2 = t * t
    t3 = t2 * t
    t4 = t2 * t2

    w = 1.0 + 0.5 * floor1

    S0 = np.sum(w, axis=1)
    S1 = np.sum(w * t, axis=1)
    S2 = np.sum(w * t2, axis=1)
    S3 = np.sum(w * t3, axis=1)
    S4 = np.sum(w * t4, axis=1)

    rhs0 = np.sum(w * v, axis=1)
    rhs1 = np.sum(w * (t * v), axis=1)
    rhs2 = np.sum(w * (t2 * v), axis=1)

    A = np.empty((m_corr.shape[0], 3, 3), dtype=np.float64)
    A[:, 0, 0] = S0
    A[:, 0, 1] = S1
    A[:, 0, 2] = S2
    A[:, 1, 0] = S1
    A[:, 1, 1] = S2
    A[:, 1, 2] = S3
    A[:, 2, 0] = S2
    A[:, 2, 1] = S3
    A[:, 2, 2] = S4

    rhs = np.stack((rhs0, rhs1, rhs2), axis=1)
    beta = np.linalg.solve(A, rhs[:, :, None])
    beta = beta[:, :, 0]

    slope = beta[:, 1]
    curvature = beta[:, 2]

    kappa_ugri = d_ug - 2.0 * d_gr + d_ri
    kappa_griz = d_gr - 2.0 * d_ri + d_iz

    abs_log_f = np.log10(abs_f + EPS)
    abs_log_f_t = np.log10(np.abs(f_tilde) + EPS)

    shares = np.abs(f_tilde) / (np.sum(np.abs(f_tilde), axis=1)[:, None] + EPS)
    max_share = np.max(shares, axis=1)
    entropy_sh = -np.sum(shares * np.log(shares + EPS), axis=1)
    l2_share = np.sum(shares ** 2, axis=1)
    concentration = 1.0 - l2_share

    sign_pos = np.sum(f_tilde > 0.0, axis=1).astype(np.float64)
    sign_neg = np.sum(f_tilde < 0.0, axis=1).astype(np.float64)
    p_pos = (sign_pos + EPS) / (5.0 + 2.0 * EPS)
    p_neg = (sign_neg + EPS) / (5.0 + 2.0 * EPS)
    sign_entropy = -(p_pos * np.log(p_pos + EPS) + p_neg * np.log(p_neg + EPS))

    features = {}

    for idx, band in enumerate(BANDS):
        features[f"floor1_{band}"] = floor1[:, idx]
        features[f"floor2_{band}"] = floor2[:, idx]

    features["soft_count"] = soft_count
    features["soft_frac"] = soft_frac
    features["soft_regime_weight"] = soft_frac
    features["ultra_floor_count"] = ultra_floor_count
    features["neg_flux_count"] = neg_flux_count

    features["lupt_v_u"] = v_u
    features["lupt_v_g"] = v_g
    features["lupt_v_r"] = v_r
    features["lupt_v_i"] = v_i
    features["lupt_v_z"] = v_z

    features["dlog_flux_ug"] = d_ug
    features["dlog_flux_gr"] = d_gr
    features["dlog_flux_ri"] = d_ri
    features["dlog_flux_iz"] = d_iz

    features["slope_weighted_poly2"] = slope
    features["curvature_weighted_poly2"] = curvature
    features["slope_hi"] = slope * (1.0 - soft_frac)
    features["slope_lo"] = slope * soft_frac
    features["curvature_hi"] = curvature * (1.0 - soft_frac)
    features["curvature_lo"] = curvature * soft_frac

    features["curvature_kappa_ugri"] = kappa_ugri
    features["curvature_kappa_griz"] = kappa_griz

    features["shape_share_max"] = max_share
    features["shape_share_entropy"] = entropy_sh
    features["shape_share_l2"] = l2_share
    features["shape_share_concentration"] = concentration
    features["shape_sign_entropy"] = sign_entropy

    pair_names = ("ug", "gr", "ri", "iz")
    pair_idx = ((0, 1), (1, 2), (2, 3), (3, 4))

    for name, (a, b_idx) in zip(pair_names, pair_idx):
        mag_color = m_corr[:, a] - m_corr[:, b_idx]
        flux_color = 2.5 * (abs_log_f[:, a] - abs_log_f[:, b_idx])
        flux_color_t = 2.5 * (abs_log_f_t[:, a] - abs_log_f_t[:, b_idx])

        mismatch = mag_color - flux_color
        mismatch_t = mag_color - flux_color_t

        features[f"mag_color_{name}"] = mag_color
        features[f"flux_color_{name}"] = flux_color
        features[f"mismatch_{name}"] = mismatch
        features[f"mismatch_tilde_{name}"] = mismatch_t
        features[f"mismatch_hi_{name}"] = mismatch * (1.0 - soft_frac)
        features[f"mismatch_lo_{name}"] = mismatch * soft_frac
        features[f"mismatch_tilde_hi_{name}"] = mismatch_t * (1.0 - soft_frac)
        features[f"mismatch_tilde_lo_{name}"] = mismatch_t * soft_frac

    return pd.DataFrame(features, index=raw.index)


FEATURE_GROUPS = [
    {
        "name": "luptitude_regime_flux_features",
        "fn": add_luptitude_regime_flux_features,
        "depends_on": [],
        "description": "Build luptitude-derived flux-shape, floor-regime, and mismatch descriptors with regime gating for low-SNR behavior.",
    }
]