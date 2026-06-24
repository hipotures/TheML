import numpy as np
import pandas as pd

SPECTRAL_TYPE_TO_EXPECTATION = {
    "O/B": 0.00,
    "A/F": 0.33,
    "G/K": 0.67,
    "M": 1.00,
}
GALAXY_POPULATION_TO_EXPECTATION = {
    "Blue_Cloud": 0.00,
    "Red_Sequence": 1.00,
}
COLOR_WEIGHT_TUPLE = (0.40, 0.30, 0.20, 0.10)


def _sanitize_score(series, fill_value=0.5):
    return series.where(np.isfinite(series), fill_value).astype("float64")


def _robust_scale_color(values, q02, q98):
    span = q98 - q02
    if not np.isfinite(span) or span == 0:
        scaled = pd.Series(0.5, index=values.index, dtype="float64")
        invalid = pd.Series(True, index=values.index, dtype="bool")
        return scaled, invalid

    raw_scaled = (values - q02) / span
    invalid = ~np.isfinite(raw_scaled)
    scaled = raw_scaled.clip(lower=0.0, upper=1.0)
    scaled = scaled.where(~invalid, 0.5)
    return scaled, invalid


def _residual_summary(d1, d2, d3, d4):
    residuals = pd.concat([d1, d2, d3, d4], axis=1)
    residuals = residuals.where(np.isfinite(residuals), 0.5)
    mean = residuals.mean(axis=1)
    median = residuals.median(axis=1)
    l1 = residuals.abs().sum(axis=1)
    l2 = (residuals ** 2).sum(axis=1)
    max_abs = residuals.abs().max(axis=1)
    return mean, median, l1, l2, max_abs


def add_catalog_tag_color_concordance(raw, deps, aux):
    u = pd.to_numeric(raw["u"], errors="coerce")
    g = pd.to_numeric(raw["g"], errors="coerce")
    r = pd.to_numeric(raw["r"], errors="coerce")
    i = pd.to_numeric(raw["i"], errors="coerce")
    z = pd.to_numeric(raw["z"], errors="coerce")
    redshift = pd.to_numeric(raw["redshift"], errors="coerce")

    c1 = u - g
    c2 = g - r
    c3 = r - i
    c4 = i - z

    q1_lo = c1.quantile(0.02)
    q1_hi = c1.quantile(0.98)
    q2_lo = c2.quantile(0.02)
    q2_hi = c2.quantile(0.98)
    q3_lo = c3.quantile(0.02)
    q3_hi = c3.quantile(0.98)
    q4_lo = c4.quantile(0.02)
    q4_hi = c4.quantile(0.98)

    r1, r1_invalid = _robust_scale_color(c1, q1_lo, q1_hi)
    r2, r2_invalid = _robust_scale_color(c2, q2_lo, q2_hi)
    r3, r3_invalid = _robust_scale_color(c3, q3_lo, q3_hi)
    r4, r4_invalid = _robust_scale_color(c4, q4_lo, q4_hi)

    spectral_type_exp = raw["spectral_type"].map(SPECTRAL_TYPE_TO_EXPECTATION)
    galaxy_pop_exp = raw["galaxy_population"].map(GALAXY_POPULATION_TO_EXPECTATION)

    te = pd.to_numeric(spectral_type_exp, errors="coerce").fillna(0.5)
    ge = pd.to_numeric(galaxy_pop_exp, errors="coerce").fillna(0.5)

    R_raw = (
        COLOR_WEIGHT_TUPLE[0] * r1
        + COLOR_WEIGHT_TUPLE[1] * r2
        + COLOR_WEIGHT_TUPLE[2] * r3
        + COLOR_WEIGHT_TUPLE[3] * r4
    )
    R = _sanitize_score(R_raw)

    mismatch_tag_raw = (R - te).abs()
    mismatch_pop_raw = (R - ge).abs()

    tag_mismatch = _sanitize_score(mismatch_tag_raw)
    pop_mismatch = _sanitize_score(mismatch_pop_raw)

    tag_agreement = _sanitize_score(1.0 - mismatch_tag_raw)
    pop_agreement = _sanitize_score(1.0 - mismatch_pop_raw)

    tag_gap = _sanitize_score(te - ge)
    tag_gap_abs = _sanitize_score((te - ge).abs())

    tag_resid_c1 = _sanitize_score((r1 - te).where(~r1_invalid, 0.5))
    tag_resid_c2 = _sanitize_score((r2 - te).where(~r2_invalid, 0.5))
    tag_resid_c3 = _sanitize_score((r3 - te).where(~r3_invalid, 0.5))
    tag_resid_c4 = _sanitize_score((r4 - te).where(~r4_invalid, 0.5))

    pop_resid_c1 = _sanitize_score((r1 - ge).where(~r1_invalid, 0.5))
    pop_resid_c2 = _sanitize_score((r2 - ge).where(~r2_invalid, 0.5))
    pop_resid_c3 = _sanitize_score((r3 - ge).where(~r3_invalid, 0.5))
    pop_resid_c4 = _sanitize_score((r4 - ge).where(~r4_invalid, 0.5))

    tag_resid_mean, tag_resid_median, tag_resid_l1, tag_resid_l2, tag_resid_maxabs = _residual_summary(
        tag_resid_c1, tag_resid_c2, tag_resid_c3, tag_resid_c4
    )
    pop_resid_mean, pop_resid_median, pop_resid_l1, pop_resid_l2, pop_resid_maxabs = _residual_summary(
        pop_resid_c1, pop_resid_c2, pop_resid_c3, pop_resid_c4
    )

    w_z = 1.0 - ((redshift - 0.45) / 1.25).clip(lower=0.0, upper=1.0)
    w_z = w_z.where(np.isfinite(w_z), 0.0).clip(lower=0.0, upper=1.0)

    invalid_indicator = (
        ~np.isfinite(R_raw)
        | ~np.isfinite(mismatch_tag_raw)
        | ~np.isfinite(mismatch_pop_raw)
        | ~np.isfinite(tag_resid_c1)
        | ~np.isfinite(tag_resid_c2)
        | ~np.isfinite(tag_resid_c3)
        | ~np.isfinite(tag_resid_c4)
        | ~np.isfinite(pop_resid_c1)
        | ~np.isfinite(pop_resid_c2)
        | ~np.isfinite(pop_resid_c3)
        | ~np.isfinite(pop_resid_c4)
        | r1_invalid
        | r2_invalid
        | r3_invalid
        | r4_invalid
    )
    valid_indicator = (~invalid_indicator).astype("bool")

    htc = ((te <= 0.34) & (R >= 0.67) & valid_indicator).astype("int8")
    ctc = ((te >= 0.67) & (R <= 0.33) & valid_indicator).astype("int8")
    rbc = ((ge >= 0.67) & (R <= 0.33) & valid_indicator).astype("int8")
    brc = ((ge <= 0.34) & (R >= 0.67) & valid_indicator).astype("int8")
    dual_contra = ((tag_mismatch >= 0.35) & (pop_mismatch >= 0.35) & valid_indicator).astype("int8")

    features = pd.DataFrame(index=raw.index)
    features["c1_u_minus_g"] = c1
    features["c2_g_minus_r"] = c2
    features["c3_r_minus_i"] = c3
    features["c4_i_minus_z"] = c4

    features["r1_u_minus_g"] = r1
    features["r2_g_minus_r"] = r2
    features["r3_r_minus_i"] = r3
    features["r4_i_minus_z"] = r4

    features["redness_R"] = R
    features["te"] = te
    features["ge"] = ge
    features["te_ge_signed_gap"] = tag_gap
    features["te_ge_abs_gap"] = tag_gap_abs
    features["te_minus_ge_signed"] = te - ge
    features["te_minus_ge_abs"] = (te - ge).abs()

    features["tag_agreement"] = tag_agreement
    features["pop_agreement"] = pop_agreement
    features["tag_mismatch"] = tag_mismatch
    features["pop_mismatch"] = pop_mismatch

    features["tag_resid_c1"] = tag_resid_c1
    features["tag_resid_c2"] = tag_resid_c2
    features["tag_resid_c3"] = tag_resid_c3
    features["tag_resid_c4"] = tag_resid_c4

    features["pop_resid_c1"] = pop_resid_c1
    features["pop_resid_c2"] = pop_resid_c2
    features["pop_resid_c3"] = pop_resid_c3
    features["pop_resid_c4"] = pop_resid_c4

    features["tag_resid_mean"] = tag_resid_mean
    features["tag_resid_median"] = tag_resid_median
    features["tag_resid_l1"] = tag_resid_l1
    features["tag_resid_l2"] = tag_resid_l2
    features["tag_resid_maxabs"] = tag_resid_maxabs

    features["pop_resid_mean"] = pop_resid_mean
    features["pop_resid_median"] = pop_resid_median
    features["pop_resid_l1"] = pop_resid_l1
    features["pop_resid_l2"] = pop_resid_l2
    features["pop_resid_maxabs"] = pop_resid_maxabs

    features["redshift_trust_gate"] = w_z

    features["htc"] = htc
    features["ctc"] = ctc
    features["rbc"] = rbc
    features["brc"] = brc
    features["dual_contra"] = dual_contra

    features["tag_agreement_wz"] = tag_agreement * w_z
    features["pop_agreement_wz"] = pop_agreement * w_z
    features["tag_mismatch_wz"] = tag_mismatch * w_z
    features["pop_mismatch_wz"] = pop_mismatch * w_z
    features["tag_gap_abs_wz"] = tag_gap_abs * w_z
    features["te_ge_abs_gap_wz"] = (tag_gap_abs) * w_z

    features["tag_resid_mean_wz"] = tag_resid_mean * w_z
    features["tag_resid_median_wz"] = tag_resid_median * w_z
    features["tag_resid_l1_wz"] = tag_resid_l1 * w_z
    features["tag_resid_l2_wz"] = tag_resid_l2 * w_z
    features["tag_resid_maxabs_wz"] = tag_resid_maxabs * w_z

    features["pop_resid_mean_wz"] = pop_resid_mean * w_z
    features["pop_resid_median_wz"] = pop_resid_median * w_z
    features["pop_resid_l1_wz"] = pop_resid_l1 * w_z
    features["pop_resid_l2_wz"] = pop_resid_l2 * w_z
    features["pop_resid_maxabs_wz"] = pop_resid_maxabs * w_z

    features["htc_wz"] = htc.astype("float64") * w_z
    features["ctc_wz"] = ctc.astype("float64") * w_z
    features["rbc_wz"] = rbc.astype("float64") * w_z
    features["brc_wz"] = brc.astype("float64") * w_z
    features["dual_contra_wz"] = dual_contra.astype("float64") * w_z

    return features


FEATURE_GROUPS = [
    {
        "name": "catalog_tag_color_concordance",
        "fn": add_catalog_tag_color_concordance,
        "depends_on": [],
        "description": "Build redshift-gated tag and population color-consistency features from robustly quantile-scaled ugriz color indices.",
    }
]