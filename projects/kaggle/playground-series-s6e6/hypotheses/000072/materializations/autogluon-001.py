import numpy as np
import pandas as pd


WDMS_COLOR_CLIP_LIMITS = {
    "ug": (-3.0, 5.0),
    "gr": (-2.0, 4.0),
    "ri": (-2.0, 3.0),
    "iz": (-2.0, 3.0),
}


def _as_numeric(raw, column, default=0.0):
    if column in raw:
        values = pd.to_numeric(raw[column], errors="coerce")
        return values.replace([np.inf, -np.inf], np.nan).fillna(default)
    return pd.Series(default, index=raw.index, dtype="float64")


def _signed_interval_margin(values, lower, upper):
    below = values - lower
    above = upper - values
    return np.minimum(below, above)


def add_wdms_color_bridge_signature(raw, deps, aux):
    u = _as_numeric(raw, "u")
    g = _as_numeric(raw, "g")
    r = _as_numeric(raw, "r")
    i = _as_numeric(raw, "i")
    z = _as_numeric(raw, "z")
    redshift = _as_numeric(raw, "redshift")

    ug_raw = u - g
    gr_raw = g - r
    ri_raw = r - i
    iz_raw = i - z

    ug_low, ug_high = WDMS_COLOR_CLIP_LIMITS["ug"]
    gr_low, gr_high = WDMS_COLOR_CLIP_LIMITS["gr"]
    ri_low, ri_high = WDMS_COLOR_CLIP_LIMITS["ri"]
    iz_low, iz_high = WDMS_COLOR_CLIP_LIMITS["iz"]

    ug = ug_raw.clip(ug_low, ug_high)
    gr = gr_raw.clip(gr_low, gr_high)
    ri = ri_raw.clip(ri_low, ri_high)
    iz = iz_raw.clip(iz_low, iz_high)

    ug_overflow = ((ug_raw < ug_low) | (ug_raw > ug_high)).astype("int8")
    gr_overflow = ((gr_raw < gr_low) | (gr_raw > gr_high)).astype("int8")
    ri_overflow = ((ri_raw < ri_low) | (ri_raw > ri_high)).astype("int8")
    iz_overflow = ((iz_raw < iz_low) | (iz_raw > iz_high)).astype("int8")
    any_color_overflow = (
        (ug_overflow + gr_overflow + ri_overflow + iz_overflow) > 0
    ).astype("int8")

    gr_wdms = gr.clip(-0.5, 1.3)
    gr_wdms_clipped_flag = ((gr < -0.5) | (gr > 1.3)).astype("int8")

    x = gr_wdms
    poly_boundary = (
        0.93
        - 0.27 * x
        - 4.7 * x**2
        + 12.38 * x**3
        + 3.08 * x**4
        - 22.19 * x**5
        + 16.67 * x**6
        - 3.89 * x**7
    )
    upper_boundary = pd.Series(
        np.where(x <= 0.52, poly_boundary, 0.4 + x),
        index=raw.index,
        dtype="float64",
    )

    g_interval_margin = _signed_interval_margin(g, 15.0, 19.0)
    ug_blue_margin = ug - (-0.6)
    gr_interval_margin = _signed_interval_margin(gr, -0.5, 1.3)
    upper_color_margin = upper_boundary - ug

    wdms_bridge_min_margin = pd.concat(
        [
            g_interval_margin,
            ug_blue_margin,
            gr_interval_margin,
            upper_color_margin,
        ],
        axis=1,
    ).min(axis=1)

    in_wdms_bridge = (
        (g_interval_margin >= 0.0)
        & (ug_blue_margin > 0.0)
        & (gr_interval_margin > 0.0)
        & (upper_color_margin > 0.0)
    ).astype("int8")

    blue_excess_strength = (-ug).clip(0.0, 3.0)
    red_tail_strength = (ri.clip(0.0, 3.0) + iz.clip(0.0, 3.0)) / 2.0
    hot_cool_product = blue_excess_strength * red_tail_strength

    middle_color = (gr + ri) / 2.0
    end_color_mean = (ug + iz) / 2.0
    valley_curvature_contrast = end_color_mean - middle_color
    color_bridge_span = iz - ug
    cool_tail_slope = iz - ri
    optical_color_curvature = ug - 2.0 * gr + ri

    abs_redshift = redshift.abs()
    stellar_redshift_margin = 0.03 - abs_redshift
    near_zero_redshift_weight = 1.0 / (1.0 + np.exp((abs_redshift - 0.03) / 0.01))
    extragalactic_redshift_margin = abs_redshift - 0.08
    bridge_stellar_gated_score = wdms_bridge_min_margin * near_zero_redshift_weight
    bridge_extragalactic_suppressed_score = wdms_bridge_min_margin * (
        1.0 - near_zero_redshift_weight
    )

    new_features = pd.DataFrame(
        {
            "ug_color": ug,
            "gr_color": gr,
            "ri_color": ri,
            "iz_color": iz,
            "ug_color_overflow": ug_overflow,
            "gr_color_overflow": gr_overflow,
            "ri_color_overflow": ri_overflow,
            "iz_color_overflow": iz_overflow,
            "any_color_overflow": any_color_overflow,
            "g_15_19_signed_margin": g_interval_margin,
            "ug_blue_cut_margin": ug_blue_margin,
            "gr_wdms_interval_margin": gr_interval_margin,
            "wdms_upper_boundary_ug_margin": upper_color_margin,
            "gr_wdms_boundary_clipped": gr_wdms_clipped_flag,
            "wdms_bridge_min_margin": wdms_bridge_min_margin,
            "in_wdms_bridge_box": in_wdms_bridge,
            "blue_excess_strength": blue_excess_strength,
            "red_tail_strength": red_tail_strength,
            "hot_cool_color_product": hot_cool_product,
            "valley_curvature_contrast": valley_curvature_contrast,
            "color_bridge_span": color_bridge_span,
            "cool_tail_slope": cool_tail_slope,
            "optical_color_curvature": optical_color_curvature,
            "abs_redshift": abs_redshift,
            "stellar_redshift_margin": stellar_redshift_margin,
            "near_zero_redshift_weight": near_zero_redshift_weight,
            "extragalactic_redshift_margin": extragalactic_redshift_margin,
            "bridge_stellar_gated_score": bridge_stellar_gated_score,
            "bridge_extragalactic_suppressed_score": bridge_extragalactic_suppressed_score,
        },
        index=raw.index,
    )

    return new_features.replace([np.inf, -np.inf], 0.0).fillna(0.0)


FEATURE_GROUPS = [
    {
        "name": "wdms_color_bridge_signature",
        "fn": add_wdms_color_bridge_signature,
        "depends_on": [],
        "description": "Encodes WDMS-style hot-plus-cool optical color bridge margins and redshift-gated stellar contaminant signatures.",
    }
]