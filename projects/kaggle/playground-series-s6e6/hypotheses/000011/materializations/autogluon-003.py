import numpy as np
import pandas as pd


def add_lrg_target_cut_margins(raw, deps, aux):
    g = raw["g"].astype("float64")
    r = raw["r"].astype("float64")
    i = raw["i"].astype("float64")

    gmr = g - r
    rmi = r - i

    c_perp = rmi - (gmr / 4.0) - 0.177
    c_par = (0.7 * gmr) + (1.2 * (rmi - 0.177))

    margin_cutI_lum_raw = r - (13.116 + (c_par / 0.3))
    margin_cutI_r_raw = r - 19.2
    margin_cutI_c_raw = c_perp.abs() - 0.2
    score_cutI_raw = pd.concat(
        [margin_cutI_lum_raw, margin_cutI_r_raw, margin_cutI_c_raw],
        axis=1,
    ).max(axis=1)

    margin_cutII_c_raw = (0.449 - (gmr / 6.0)) - c_perp
    margin_cutII_red_raw = (1.296 + (0.25 * rmi)) - gmr
    margin_cutII_r_raw = r - 19.5
    score_cutII_raw = pd.concat(
        [margin_cutII_c_raw, margin_cutII_red_raw, margin_cutII_r_raw],
        axis=1,
    ).max(axis=1)

    score_lrg_any_raw = np.minimum(score_cutI_raw, score_cutII_raw)

    new_features = pd.DataFrame(index=raw.index)
    new_features["c_perp"] = c_perp
    new_features["c_par"] = c_par
    new_features["margin_cutI_lum"] = margin_cutI_lum_raw.clip(-20.0, 20.0)
    new_features["margin_cutI_r"] = margin_cutI_r_raw.clip(-20.0, 20.0)
    new_features["margin_cutI_c"] = margin_cutI_c_raw.clip(-20.0, 20.0)
    new_features["margin_cutII_c"] = margin_cutII_c_raw.clip(-20.0, 20.0)
    new_features["margin_cutII_red"] = margin_cutII_red_raw.clip(-20.0, 20.0)
    new_features["margin_cutII_r"] = margin_cutII_r_raw.clip(-20.0, 20.0)
    new_features["score_cutI"] = score_cutI_raw.clip(-20.0, 20.0)
    new_features["score_cutII"] = score_cutII_raw.clip(-20.0, 20.0)
    new_features["score_lrg_any"] = pd.Series(score_lrg_any_raw, index=raw.index).clip(-20.0, 20.0)
    new_features["pass_cutI"] = score_cutI_raw <= 0.0
    new_features["pass_cutII"] = score_cutII_raw <= 0.0
    new_features["pass_lrg_any"] = score_lrg_any_raw <= 0.0

    return new_features


FEATURE_GROUPS = [
    {
        "name": "lrg_target_cut_margins",
        "fn": add_lrg_target_cut_margins,
        "depends_on": [],
        "description": "Continuous SDSS luminous-red-galaxy cut-margin proxies from g, r, and i photometry.",
    }
]