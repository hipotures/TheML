import numpy as np
import pandas as pd


TRANSFORM_SOURCE_URL = "https://www.sdss3.org/dr8/algorithms/sdssUBVRITransform.php"


def _as_float(raw, column):
    return pd.to_numeric(raw[column], errors="coerce").astype("float64")


def _safe_divide(numerator, denominator):
    numerator = np.asarray(numerator, dtype="float64")
    denominator = np.asarray(denominator, dtype="float64")
    return np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator, dtype="float64"),
        where=np.abs(denominator) > 1.0e-12,
    )


def _signed_min_abs(values):
    stacked = np.vstack([np.asarray(v, dtype="float64") for v in values])
    idx = np.nanargmin(np.abs(stacked), axis=0)
    return np.take_along_axis(stacked, idx.reshape(1, -1), axis=0)[0]


def add_ubvri_transform_consistency(raw, deps, aux):
    u = _as_float(raw, "u")
    g = _as_float(raw, "g")
    r = _as_float(raw, "r")
    i = _as_float(raw, "i")
    z = _as_float(raw, "z")
    redshift = _as_float(raw, "redshift") if "redshift" in raw.columns else pd.Series(0.0, index=raw.index)

    ug = u - g
    gr = g - r
    ri = r - i
    iz = i - z
    gi = g - i
    rz = r - z

    # Jester et al. ugriz -> UBVRcIc equations.
    j_star_ub = 0.78 * ug - 0.88
    j_star_bv = 0.98 * gr + 0.22
    j_star_vr = 1.09 * ri + 0.22
    j_star_ri = 1.00 * ri + 0.21
    j_star_b = g + 0.39 * gr + 0.21
    j_star_v = g - 0.59 * gr - 0.01
    j_star_bg = j_star_b - g

    j_blue_ub = 0.77 * ug - 0.88
    j_blue_bv = 0.90 * gr + 0.21
    j_blue_vr = 0.96 * ri + 0.21
    j_blue_ri = 1.02 * ri + 0.21
    j_blue_b = g + 0.33 * gr + 0.20
    j_blue_v = g - 0.58 * gr - 0.01
    j_blue_bg = j_blue_b - g

    j_qso_ub = 0.75 * ug - 0.81
    j_qso_bv = 0.62 * gr + 0.15
    j_qso_vr = 0.38 * ri + 0.27
    j_qso_ri = 0.72 * ri + 0.27
    j_qso_b = g + 0.17 * ug + 0.11
    j_qso_v = g - 0.52 * gr - 0.03
    j_qso_bg = j_qso_b - g

    # Jordi et al. general, Population I, and Population II inverse equations.
    jordi_ub_ug = 0.79 * ug - 0.93
    jordi_ub_twocolor = 0.52 * ug + 0.53 * gr - 0.82
    jordi_bg_ug = 0.175 * ug + 0.150
    jordi_bg_gr = 0.313 * gr + 0.219
    jordi_vg = -0.565 * gr - 0.016
    jordi_vi = np.where(gi <= 2.1, 0.675 * gi + 0.364, 1.11 * gi - 0.52)
    jordi_rr = -0.153 * ri - 0.117
    jordi_ri = 0.930 * ri + 0.259
    jordi_ii = -0.386 * iz - 0.397
    jordi_v = g + jordi_vg
    jordi_r = r + jordi_rr
    jordi_i = i + jordi_ii
    jordi_b_from_ug = g + jordi_bg_ug
    jordi_b_from_gr = g + jordi_bg_gr
    jordi_bv_from_ug = jordi_b_from_ug - jordi_v
    jordi_bv_from_gr = jordi_b_from_gr - jordi_v

    pop1_bg_ug = 0.163 * ug + 0.170
    pop1_bg_gr = 0.312 * gr + 0.219
    pop1_vg = -0.573 * gr - 0.016
    pop1_vi = np.where(gi <= 2.1, 0.671 * gi + 0.359, 1.12 * gi - 0.53)
    pop1_rr = -0.257 * ri + 0.152
    pop1_ri = 0.977 * ri + 0.234
    pop1_ii = -0.409 * iz - 0.394
    pop1_v = g + pop1_vg
    pop1_b_from_ug = g + pop1_bg_ug
    pop1_b_from_gr = g + pop1_bg_gr
    pop1_bv_from_ug = pop1_b_from_ug - pop1_v
    pop1_bv_from_gr = pop1_b_from_gr - pop1_v

    pop2_bg_ug = 0.20 * ug + 0.15
    pop2_bg_gr = 0.349 * gr + 0.245
    pop2_vg = -0.569 * gr + 0.021
    pop2_vi = 0.674 * gi + 0.406
    pop2_rr = -0.25 * ri - 0.119
    pop2_ri = 0.80 * ri + 0.317
    pop2_v = g + pop2_vg
    pop2_b_from_ug = g + pop2_bg_ug
    pop2_b_from_gr = g + pop2_bg_gr
    pop2_bv_from_ug = pop2_b_from_ug - pop2_v
    pop2_bv_from_gr = pop2_b_from_gr - pop2_v

    # Karaali et al. two-color main-sequence B-V estimate.
    karaali_bv = 0.992 * gr - 0.0199 * ug + 0.202

    # Lupton matched-star equations provide an independent internal consistency check.
    lupton_b_ug = u - 0.8116 * ug + 0.1313
    lupton_b_gr = g + 0.3130 * gr + 0.2271
    lupton_v_ug = g - 0.2906 * ug + 0.0885
    lupton_v_gr = g - 0.5784 * gr - 0.0038
    lupton_r_gr = r - 0.1837 * gr - 0.0971
    lupton_r_ri = r - 0.2936 * ri - 0.1439
    lupton_i_ri = r - 1.2444 * ri - 0.3820
    lupton_i_iz = i - 0.3780 * iz - 0.3974
    lupton_bv_ug = lupton_b_ug - lupton_v_gr
    lupton_bv_gr = lupton_b_gr - lupton_v_gr
    lupton_ri_color = lupton_r_ri - lupton_i_ri

    ri_from_rz_j_star = (rz + 0.41) / 1.72
    ri_from_rz_j_blue = (rz + 0.42) / 1.69
    ri_from_rz_j_qso = (rz + 0.20) / 1.20
    ri_from_rz_jordi = (rz + 0.386) / 1.584
    ri_from_rz_pop1 = (rz + 0.370) / 1.568
    ri_from_rz_pop2 = (rz + 0.46) / 1.60

    ri_from_ri_j_star = (ri + 0.20) / 0.91
    ri_from_ri_j_blue = (ri + 0.22) / 0.98
    ri_from_ri_j_qso = (ri + 0.20) / 0.90
    ri_from_ri_jordi_forward = (ri + 0.236) / 1.007
    ri_from_ri_pop1_forward = (ri + 0.221) / 0.988
    ri_from_ri_pop2_forward = (ri + 0.30) / 1.06

    residuals = {
        "ub_star_minus_qso": j_star_ub - j_qso_ub,
        "ub_star_minus_jordi_ug": j_star_ub - jordi_ub_ug,
        "ub_jordi_twocolor_minus_ug": jordi_ub_twocolor - jordi_ub_ug,
        "ub_blue_minus_allstar": j_blue_ub - j_star_ub,
        "bv_star_minus_qso": j_star_bv - j_qso_bv,
        "bv_star_minus_blue": j_star_bv - j_blue_bv,
        "bv_star_minus_karaali": j_star_bv - karaali_bv,
        "bv_jordi_ug_minus_gr": jordi_bv_from_ug - jordi_bv_from_gr,
        "bv_pop1_ug_minus_gr": pop1_bv_from_ug - pop1_bv_from_gr,
        "bv_pop2_ug_minus_gr": pop2_bv_from_ug - pop2_bv_from_gr,
        "bv_lupton_ug_minus_gr": lupton_bv_ug - lupton_bv_gr,
        "bg_jester_star_minus_jordi_gr": j_star_bg - jordi_bg_gr,
        "bg_jester_star_minus_jordi_ug": j_star_bg - jordi_bg_ug,
        "bg_qso_minus_jordi_ug": j_qso_bg - jordi_bg_ug,
        "bg_jordi_ug_minus_gr": jordi_bg_ug - jordi_bg_gr,
        "bg_pop1_ug_minus_gr": pop1_bg_ug - pop1_bg_gr,
        "bg_pop2_ug_minus_gr": pop2_bg_ug - pop2_bg_gr,
        "bg_lupton_ug_minus_gr": (lupton_b_ug - g) - (lupton_b_gr - g),
        "vr_star_minus_qso": j_star_vr - j_qso_vr,
        "vi_jordi_minus_pop1": jordi_vi - pop1_vi,
        "vi_jordi_minus_pop2": jordi_vi - pop2_vi,
        "ri_star_minus_qso": j_star_ri - j_qso_ri,
        "ri_star_minus_jordi": j_star_ri - jordi_ri,
        "ri_jordi_minus_pop1": jordi_ri - pop1_ri,
        "ri_jordi_minus_pop2": jordi_ri - pop2_ri,
        "ri_lupton_minus_jordi": lupton_ri_color - jordi_ri,
        "rr_jordi_minus_pop1": jordi_rr - pop1_rr,
        "rr_jordi_minus_pop2": jordi_rr - pop2_rr,
        "ii_jordi_minus_pop1": jordi_ii - pop1_ii,
        "rz_vs_ri_jester_star": ri_from_rz_j_star - ri_from_ri_j_star,
        "rz_vs_ri_jester_blue": ri_from_rz_j_blue - ri_from_ri_j_blue,
        "rz_vs_ri_jester_qso": ri_from_rz_j_qso - ri_from_ri_j_qso,
        "rz_vs_ri_jordi": ri_from_rz_jordi - ri_from_ri_jordi_forward,
        "rz_vs_ri_pop1": ri_from_rz_pop1 - ri_from_ri_pop1_forward,
        "rz_vs_ri_pop2": ri_from_rz_pop2 - ri_from_ri_pop2_forward,
        "b_jordi_ug_minus_lupton_ug": jordi_b_from_ug - lupton_b_ug,
        "b_jordi_gr_minus_lupton_gr": jordi_b_from_gr - lupton_b_gr,
        "v_jordi_minus_lupton_gr": jordi_v - lupton_v_gr,
        "r_jordi_minus_lupton_ri": jordi_r - lupton_r_ri,
        "i_jordi_minus_lupton_iz": jordi_i - lupton_i_iz,
    }

    scaled = {
        "ub_star_qso_sigma": residuals["ub_star_minus_qso"] / np.sqrt(0.05 * 0.05 + 0.03 * 0.03),
        "ub_jordi_twocolor_sigma": residuals["ub_jordi_twocolor_minus_ug"] / np.sqrt(0.02 * 0.02 + 0.06 * 0.06),
        "bv_star_qso_sigma": residuals["bv_star_minus_qso"] / np.sqrt(0.04 * 0.04 + 0.07 * 0.07),
        "bv_star_karaali_sigma": residuals["bv_star_minus_karaali"] / 0.04,
        "bg_jordi_ug_gr_sigma": residuals["bg_jordi_ug_minus_gr"] / np.sqrt(0.002 * 0.002 + 0.003 * 0.003),
        "bg_pop1_ug_gr_sigma": residuals["bg_pop1_ug_minus_gr"] / np.sqrt(0.002 * 0.002 + 0.003 * 0.003),
        "bg_lupton_ug_gr_sigma": residuals["bg_lupton_ug_minus_gr"] / np.sqrt(0.0095 * 0.0095 + 0.0107 * 0.0107),
        "ri_star_qso_sigma": residuals["ri_star_minus_qso"] / np.sqrt(0.01 * 0.01 + 0.06 * 0.06),
        "ri_star_jordi_sigma": residuals["ri_star_minus_jordi"] / 0.01,
        "rz_ri_jester_star_sigma": residuals["rz_vs_ri_jester_star"] / np.sqrt(0.03 * 0.03 + 0.03 * 0.03),
        "rz_ri_jester_qso_sigma": residuals["rz_vs_ri_jester_qso"] / np.sqrt(0.18 * 0.18 + 0.07 * 0.07),
        "b_lupton_jordi_sigma": residuals["b_jordi_gr_minus_lupton_gr"] / np.sqrt(0.0107 * 0.0107 + 0.003 * 0.003),
        "v_lupton_jordi_sigma": residuals["v_jordi_minus_lupton_gr"] / np.sqrt(0.0054 * 0.0054 + 0.001 * 0.001),
        "r_lupton_jordi_sigma": residuals["r_jordi_minus_lupton_ri"] / np.sqrt(0.0072 * 0.0072 + 0.003 * 0.003),
        "i_lupton_jordi_sigma": residuals["i_jordi_minus_lupton_iz"] / np.sqrt(0.0063 * 0.0063 + 0.004 * 0.004),
    }

    stellar_abs = [
        np.abs(residuals["ub_blue_minus_allstar"]),
        np.abs(residuals["bv_star_minus_blue"]),
        np.abs(residuals["bv_star_minus_karaali"]),
        np.abs(residuals["bg_jordi_ug_minus_gr"]),
        np.abs(residuals["bg_pop1_ug_minus_gr"]),
        np.abs(residuals["ri_star_minus_jordi"]),
        np.abs(residuals["ri_jordi_minus_pop1"]),
        np.abs(residuals["rz_vs_ri_jester_star"]),
        np.abs(residuals["rz_vs_ri_jordi"]),
    ]
    qso_abs = [
        np.abs(residuals["ub_star_minus_qso"]),
        np.abs(residuals["bv_star_minus_qso"]),
        np.abs(residuals["bg_qso_minus_jordi_ug"]),
        np.abs(residuals["ri_star_minus_qso"]),
        np.abs(residuals["rz_vs_ri_jester_qso"]),
    ]
    lupton_abs = [
        np.abs(residuals["bg_lupton_ug_minus_gr"]),
        np.abs(residuals["bv_lupton_ug_minus_gr"]),
        np.abs(residuals["b_jordi_ug_minus_lupton_ug"]),
        np.abs(residuals["v_jordi_minus_lupton_gr"]),
        np.abs(residuals["r_jordi_minus_lupton_ri"]),
        np.abs(residuals["i_jordi_minus_lupton_iz"]),
    ]

    stellar_mean_abs = np.mean(np.vstack(stellar_abs), axis=0)
    stellar_max_abs = np.max(np.vstack(stellar_abs), axis=0)
    qso_mean_abs = np.mean(np.vstack(qso_abs), axis=0)
    qso_max_abs = np.max(np.vstack(qso_abs), axis=0)
    lupton_mean_abs = np.mean(np.vstack(lupton_abs), axis=0)

    signed_best_family = _signed_min_abs(
        [
            residuals["bv_star_minus_karaali"],
            residuals["bg_jordi_ug_minus_gr"],
            residuals["ri_star_minus_jordi"],
            residuals["rz_vs_ri_jester_star"],
            residuals["rz_vs_ri_jester_qso"],
        ]
    )

    rc_ic_margin = 1.15 - j_star_ri
    ub_negative_margin = -j_star_ub
    bv_low_margin = j_star_bv - 0.3
    bv_high_margin = 1.1 - j_star_bv
    bv_domain_margin = np.minimum(bv_low_margin, bv_high_margin)
    gi_piece_margin = 2.1 - gi
    qso_redshift_margin = 2.1 - redshift
    qso_valid_margin = np.minimum(qso_redshift_margin, 2.1 - np.maximum(np.abs(j_qso_ub), np.abs(j_qso_bv)))

    features = pd.DataFrame(index=raw.index)

    transformed = {
        "ub_jester_star": j_star_ub,
        "ub_jester_qso": j_qso_ub,
        "ub_jordi_ug": jordi_ub_ug,
        "ub_jordi_twocolor": jordi_ub_twocolor,
        "bv_jester_star": j_star_bv,
        "bv_jester_qso": j_qso_bv,
        "bv_karaali_twocolor": karaali_bv,
        "bv_jordi_from_ug": jordi_bv_from_ug,
        "bv_jordi_from_gr": jordi_bv_from_gr,
        "vr_jester_star": j_star_vr,
        "vr_jester_qso": j_qso_vr,
        "vi_jordi": jordi_vi,
        "vi_pop1": pop1_vi,
        "vi_pop2": pop2_vi,
        "ri_jester_star": j_star_ri,
        "ri_jester_qso": j_qso_ri,
        "ri_jordi": jordi_ri,
        "ri_pop1": pop1_ri,
        "ri_pop2": pop2_ri,
        "bg_jester_star": j_star_bg,
        "bg_jester_qso": j_qso_bg,
        "bg_jordi_ug": jordi_bg_ug,
        "bg_jordi_gr": jordi_bg_gr,
        "bg_pop1_ug": pop1_bg_ug,
        "bg_pop1_gr": pop1_bg_gr,
        "bg_pop2_ug": pop2_bg_ug,
        "bg_pop2_gr": pop2_bg_gr,
    }

    for name, value in transformed.items():
        features[name] = np.clip(np.asarray(value, dtype="float64"), -8.0, 8.0).astype("float32")

    for name, value in residuals.items():
        features["resid_" + name] = np.clip(np.asarray(value, dtype="float64"), -5.0, 5.0).astype("float32")
        features["abs_resid_" + name] = np.clip(np.abs(np.asarray(value, dtype="float64")), 0.0, 5.0).astype("float32")

    for name, value in scaled.items():
        features[name] = np.clip(np.asarray(value, dtype="float64"), -50.0, 50.0).astype("float32")
        features["abs_" + name] = np.clip(np.abs(np.asarray(value, dtype="float64")), 0.0, 50.0).astype("float32")

    features["stellar_consistency_mean_abs"] = np.clip(stellar_mean_abs, 0.0, 5.0).astype("float32")
    features["stellar_consistency_max_abs"] = np.clip(stellar_max_abs, 0.0, 5.0).astype("float32")
    features["qso_consistency_mean_abs"] = np.clip(qso_mean_abs, 0.0, 5.0).astype("float32")
    features["qso_consistency_max_abs"] = np.clip(qso_max_abs, 0.0, 5.0).astype("float32")
    features["lupton_consistency_mean_abs"] = np.clip(lupton_mean_abs, 0.0, 5.0).astype("float32")
    features["stellar_minus_qso_consistency"] = np.clip(stellar_mean_abs - qso_mean_abs, -5.0, 5.0).astype("float32")
    features["qso_minus_stellar_consistency"] = np.clip(qso_mean_abs - stellar_mean_abs, -5.0, 5.0).astype("float32")
    features["best_signed_consistency_resid"] = np.clip(signed_best_family, -5.0, 5.0).astype("float32")
    features["best_abs_consistency_resid"] = np.clip(np.abs(signed_best_family), 0.0, 5.0).astype("float32")
    features["stellar_to_qso_consistency_ratio"] = np.clip(_safe_divide(stellar_mean_abs, qso_mean_abs + 1.0e-6), 0.0, 100.0).astype("float32")
    features["qso_to_stellar_consistency_ratio"] = np.clip(_safe_divide(qso_mean_abs, stellar_mean_abs + 1.0e-6), 0.0, 100.0).astype("float32")

    margins = {
        "valid_margin_rc_ic_lt_1p15": rc_ic_margin,
        "valid_margin_ub_lt_0": ub_negative_margin,
        "valid_margin_bv_gt_0p3": bv_low_margin,
        "valid_margin_bv_lt_1p1": bv_high_margin,
        "valid_margin_bv_domain": bv_domain_margin,
        "valid_margin_gi_le_2p1": gi_piece_margin,
        "valid_margin_qso_redshift_le_2p1": qso_redshift_margin,
        "valid_margin_qso_combined": qso_valid_margin,
    }

    for name, value in margins.items():
        arr = np.asarray(value, dtype="float64")
        features[name] = np.clip(arr, -10.0, 10.0).astype("float32")
        features[name + "_inside"] = (arr >= 0.0)

    features["jordi_vi_high_color_branch"] = gi > 2.1
    features["qso_transform_redshift_valid"] = redshift <= 2.1
    features["stellar_transform_color_valid"] = (rc_ic_margin >= 0.0) & (ub_negative_margin >= 0.0)
    features["karaali_bv_domain_valid"] = (bv_low_margin > 0.0) & (bv_high_margin > 0.0)

    return features


FEATURE_GROUPS = [
    {
        "name": "ubvri_transform_consistency",
        "fn": add_ubvri_transform_consistency,
        "depends_on": [],
        "description": "Cross-system SDSS ugriz to Johnson-Cousins UBVRI transformation consistency residuals, validity margins, and family agreement scores.",
    }
]