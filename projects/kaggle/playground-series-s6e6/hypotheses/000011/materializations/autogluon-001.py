import pandas as pd
import numpy as np

_LRG_MARGIN_CLIP_BOUNDS = (-20.0, 20.0)


def add_lrg_target_cut_margins(raw, deps, aux):
    g_minus_r = raw["g"] - raw["r"]
    r_minus_i = raw["r"] - raw["i"]

    c_perp = r_minus_i - (g_minus_r / 4.0) - 0.177
    c_parallel = 0.7 * g_minus_r + 1.2 * (r_minus_i - 0.177)

    cut_i_cperp_margin = c_perp.abs() - 0.2
    cut_i_r_mag_margin = raw["r"] - 19.2
    cut_i_r_cpar_margin = raw["r"] - (13.116 + c_parallel / 0.3)

    cut_ii_cperp_margin = (0.449 - g_minus_r / 6.0) - c_perp
    cut_ii_gmri_margin = (1.296 + 0.25 * r_minus_i) - g_minus_r
    cut_ii_r_mag_margin = raw["r"] - 19.5

    cut_i_score = np.maximum.reduce(
        [
            cut_i_cperp_margin.to_numpy(),
            cut_i_r_mag_margin.to_numpy(),
            cut_i_r_cpar_margin.to_numpy(),
        ]
    )
    cut_ii_score = np.maximum.reduce(
        [
            cut_ii_cperp_margin.to_numpy(),
            cut_ii_gmri_margin.to_numpy(),
            cut_ii_r_mag_margin.to_numpy(),
        ]
    )

    new_features = pd.DataFrame(
        {
            "lrg_g_minus_r": g_minus_r,
            "lrg_r_minus_i": r_minus_i,
            "lrg_c_perp": c_perp,
            "lrg_c_parallel": c_parallel,
            "lrg_cut_i_margin_c_perp": cut_i_cperp_margin,
            "lrg_cut_i_margin_r_mag": cut_i_r_mag_margin,
            "lrg_cut_i_margin_r_parallel_surface": cut_i_r_cpar_margin,
            "lrg_cut_i_max_signed_violation": pd.Series(
                cut_i_score, index=raw.index, dtype="float64"
            ),
            "lrg_cut_ii_margin_c_perp": cut_ii_cperp_margin,
            "lrg_cut_ii_margin_color": cut_ii_gmri_margin,
            "lrg_cut_ii_margin_r_mag": cut_ii_r_mag_margin,
            "lrg_cut_ii_max_signed_violation": pd.Series(
                cut_ii_score, index=raw.index, dtype="float64"
            ),
        },
        index=raw.index,
    )

    lower, upper = _LRG_MARGIN_CLIP_BOUNDS
    for col in (
        "lrg_cut_i_margin_c_perp",
        "lrg_cut_i_margin_r_mag",
        "lrg_cut_i_margin_r_parallel_surface",
        "lrg_cut_i_max_signed_violation",
        "lrg_cut_ii_margin_c_perp",
        "lrg_cut_ii_margin_color",
        "lrg_cut_ii_margin_r_mag",
        "lrg_cut_ii_max_signed_violation",
    ):
        new_features[col] = new_features[col].clip(lower=lower, upper=upper)

    return new_features


FEATURE_GROUPS = [
    {
        "name": "lrg_target_cut_margins",
        "fn": add_lrg_target_cut_margins,
        "depends_on": [],
        "description": "Computes SDSS DR2/LRG Cut I and Cut II g-r/r-i rotated-color margins and compact max-violation scores for each photometric target selection family.",
    }
]