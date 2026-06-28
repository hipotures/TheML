import numpy as np
import pandas as pd

BANDS = ("u", "g", "r", "i", "z")
ADJACENT_PAIRS = (("u", "g"), ("g", "r"), ("r", "i"), ("i", "z"))
MAG_OFFSETS = {"u": -0.04, "g": 0.0, "r": 0.0, "i": 0.0, "z": 0.02}
SOFTENING_B = {"u": 1.4e-10, "g": 0.9e-10, "r": 1.2e-10, "i": 1.8e-10, "z": 7.4e-10}
EFFECTIVE_WAVELENGTH = {"u": 3551.0, "g": 4686.0, "r": 6165.0, "i": 7481.0, "z": 8931.0}
EPS = 1e-12
MIN_FIT_WEIGHT = 0.05
SINH_CLIP = 50.0


def add_luptitude_regime_flux_features(raw, deps, aux):
    index = raw.index
    n_rows = len(raw)
    ln10_over_25 = np.log(10.0) / 2.5

    m_corr = {}
    flux = {}
    abs_flux = {}
    q = {}
    floor = {}
    ultra_floor = {}
    neg = {}
    ftilde = {}
    log_abs_flux = {}

    for band in BANDS:
        mag = pd.to_numeric(raw[band], errors="coerce").to_numpy(dtype=np.float64, copy=True)
        m = mag + MAG_OFFSETS[band]
        b = SOFTENING_B[band]

        arg = -ln10_over_25 * m - np.log(b)
        arg = np.clip(arg, -SINH_CLIP, SINH_CLIP)
        f = 2.0 * b * np.sinh(arg)
        af = np.abs(f)

        sign = np.where(f < 0.0, -1.0, 1.0)
        ft = sign * np.maximum(af, b)

        m_corr[band] = m
        flux[band] = f
        abs_flux[band] = af
        q[band] = af / (af + 2.0 * b + EPS)
        floor[band] = (af <= 2.0 * b).astype(np.float64)
        ultra_floor[band] = (af <= 0.2 * b).astype(np.float64)
        neg[band] = (f < 0.0).astype(np.float64)
        ftilde[band] = ft
        log_abs_flux[band] = np.log10(np.abs(ft) + EPS)

    features = {}

    for band in BANDS:
        features[f"m_corr_{band}"] = m_corr[band]
        features[f"flux_{band}"] = flux[band]
        features[f"abs_flux_{band}"] = abs_flux[band]
        features[f"q_{band}"] = q[band]
        features[f"floor_{band}"] = floor[band]
        features[f"ultra_floor_{band}"] = ultra_floor[band]
        features[f"neg_flux_{band}"] = neg[band]
        features[f"log_abs_flux_{band}"] = log_abs_flux[band]

    q_matrix = np.column_stack([q[band] for band in BANDS])
    floor_matrix = np.column_stack([floor[band] for band in BANDS])
    ultra_floor_matrix = np.column_stack([ultra_floor[band] for band in BANDS])
    neg_matrix = np.column_stack([neg[band] for band in BANDS])
    log_flux_matrix = np.column_stack([log_abs_flux[band] for band in BANDS])
    abs_ftilde_matrix = np.column_stack([np.abs(ftilde[band]) for band in BANDS])

    soft_count = floor_matrix.sum(axis=1)
    ultra_floor_count = ultra_floor_matrix.sum(axis=1)
    floor_frac = soft_count / float(len(BANDS))
    neg_flux_count = neg_matrix.sum(axis=1)
    reliability_weight = 1.0 - floor_frac

    features["soft_count"] = soft_count
    features["ultra_floor_count"] = ultra_floor_count
    features["floor_frac"] = floor_frac
    features["neg_flux_count"] = neg_flux_count
    features["min_q"] = q_matrix.min(axis=1)
    features["mean_q"] = q_matrix.mean(axis=1)
    features["reliability_weight"] = reliability_weight

    flux_colors = {}
    mismatches = {}
    for left, right in ADJACENT_PAIRS:
        key = f"{left}{right}"
        flux_color = -2.5 * (log_abs_flux[left] - log_abs_flux[right])
        mag_color = m_corr[left] - m_corr[right]
        mismatch = mag_color - flux_color

        flux_colors[key] = flux_color
        mismatches[key] = mismatch
        features[f"flux_color_{key}"] = flux_color
        features[f"mag_color_{key}"] = mag_color
        features[f"mismatch_{key}"] = mismatch

    curvature_ugr = flux_colors["ug"] - flux_colors["gr"]
    curvature_gri = flux_colors["gr"] - flux_colors["ri"]
    curvature_riz = flux_colors["ri"] - flux_colors["iz"]
    curvature_blue_red = curvature_ugr - curvature_riz

    features["curvature_ugr"] = curvature_ugr
    features["curvature_gri"] = curvature_gri
    features["curvature_riz"] = curvature_riz
    features["curvature_blue_red"] = curvature_blue_red

    wavelengths = np.array([EFFECTIVE_WAVELENGTH[band] for band in BANDS], dtype=np.float64)
    x = np.log10(wavelengths)
    x = x - x.mean()
    design = np.column_stack([np.ones(len(BANDS), dtype=np.float64), x, x * x])

    ols_pinv = np.linalg.pinv(design)
    ols_coef = log_flux_matrix @ ols_pinv.T

    weighted_coef = np.empty((n_rows, 3), dtype=np.float64)
    weighted_resid = np.empty((n_rows, len(BANDS)), dtype=np.float64)

    base_weights = np.maximum(q_matrix, MIN_FIT_WEIGHT)
    for row_idx in range(n_rows):
        w_sqrt = np.sqrt(base_weights[row_idx])
        xw = design * w_sqrt[:, None]
        yw = log_flux_matrix[row_idx] * w_sqrt
        coef, _, _, _ = np.linalg.lstsq(xw, yw, rcond=None)
        weighted_coef[row_idx] = coef
        weighted_resid[row_idx] = log_flux_matrix[row_idx] - design @ coef

    features["logflux_ols_intercept"] = ols_coef[:, 0]
    features["logflux_ols_slope"] = ols_coef[:, 1]
    features["logflux_ols_curvature"] = ols_coef[:, 2]
    features["logflux_wls_intercept"] = weighted_coef[:, 0]
    features["logflux_wls_slope"] = weighted_coef[:, 1]
    features["logflux_wls_curvature"] = weighted_coef[:, 2]

    abs_resid = np.abs(weighted_resid)
    features["wls_max_abs_residual"] = abs_resid.max(axis=1)
    features["wls_mean_abs_residual"] = abs_resid.mean(axis=1)
    features["wls_blue_vs_red_residual"] = (
        weighted_resid[:, 0] + weighted_resid[:, 1] - weighted_resid[:, 3] - weighted_resid[:, 4]
    )

    sum_abs_ftilde = abs_ftilde_matrix.sum(axis=1)
    shares = abs_ftilde_matrix / (sum_abs_ftilde[:, None] + EPS)

    features["max_share"] = shares.max(axis=1)
    features["min_share"] = shares.min(axis=1)
    features["l2_share"] = np.sum(shares * shares, axis=1)
    features["entropy_share"] = -np.sum(shares * np.log(shares + EPS), axis=1)
    features["blue_share"] = shares[:, 0] + shares[:, 1]
    features["red_share"] = shares[:, 3] + shares[:, 4]
    features["center_share"] = shares[:, 2]

    sign_bits = np.where(neg_matrix > 0.5, -1.0, 1.0)
    features["sign_changes_across_bands"] = np.sum(sign_bits[:, 1:] != sign_bits[:, :-1], axis=1).astype(np.float64)

    positive_count = float(len(BANDS)) - neg_flux_count
    negative_share = neg_flux_count / float(len(BANDS))
    positive_share = positive_count / float(len(BANDS))
    features["sign_entropy"] = -(
        positive_share * np.log(positive_share + EPS) + negative_share * np.log(negative_share + EPS)
    )

    gated_descriptors = {
        "logflux_wls_slope": weighted_coef[:, 1],
        "logflux_wls_curvature": weighted_coef[:, 2],
        "entropy_share": features["entropy_share"],
        "max_share": features["max_share"],
    }
    for key, values in mismatches.items():
        gated_descriptors[f"mismatch_{key}"] = values

    for name, values in gated_descriptors.items():
        features[f"{name}_high_reliability"] = values * reliability_weight
        features[f"{name}_floor_dominated"] = values * floor_frac

    return pd.DataFrame(features, index=index)


FEATURE_GROUPS = [
    {
        "name": "luptitude_regime_flux_features",
        "fn": add_luptitude_regime_flux_features,
        "depends_on": [],
        "description": "Reliability-weighted luptitude flux geometry features from SDSS-like u, g, r, i, and z photometry.",
    }
]