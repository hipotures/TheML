import numpy as np
import pandas as pd


BANDS = ("u", "g", "r", "i", "z")
BAND_CENTERS = (3551.0, 4686.0, 6165.0, 7481.0, 8931.0)
BAND_EDGES = (3000.0, 4100.0, 5500.0, 7000.0, 8200.0, 9200.0)

ABSORPTION_LINES = (
    ("CaII_K", 3933.7, "blue", "cahk"),
    ("CaII_H", 3968.5, "blue", "cahk"),
    ("Hdelta", 4102.0, "blue", "balmer"),
    ("Gband", 4304.0, "blue", "other"),
    ("Hgamma", 4341.0, "blue", "balmer"),
    ("Hbeta", 4861.0, "red", "other"),
    ("Mgb", 5175.0, "red", "other"),
    ("NaD", 5893.0, "red", "other"),
)

REDSHIFT_REGIMES = (
    ("z000_045", 0.0, 0.45),
    ("z045_110", 0.45, 1.1),
    ("z110_200", 1.1, 2.0),
    ("z200_710", 2.0, 7.1),
)


def _weighted_mean(values, weights):
    numerator = np.sum(values * weights, axis=1)
    denominator = np.sum(weights, axis=1)
    return np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator, dtype=float),
        where=denominator > 0.0,
    )


def add_redshifted_absorption_trough_residuals(raw, deps, aux):
    n_rows = len(raw)
    index = raw.index

    mags = raw.loc[:, BANDS].to_numpy(dtype=float, copy=True)
    redshift = raw["redshift"].to_numpy(dtype=float, copy=True)
    z_eff = np.maximum(redshift, 0.0)

    centers = np.asarray(BAND_CENTERS, dtype=float)
    edges = np.asarray(BAND_EDGES, dtype=float)
    log_centers = np.log(centers)
    logf = -0.4 * np.log(10.0) * mags
    flux = np.exp(logf)

    out = pd.DataFrame(index=index)
    out["negative_redshift_flag"] = (redshift < 0.0).astype(np.int8)

    all_deficits = []
    all_excesses = []
    all_weights = []
    blue_deficits = []
    blue_weights = []
    red_deficits = []
    red_weights = []
    cahk_deficits = []
    cahk_weights = []
    balmer_deficits = []
    balmer_weights = []

    for line_name, rest_wavelength, color_group, line_group in ABSORPTION_LINES:
        lambda_obs = rest_wavelength * (1.0 + z_eff)
        visible = (lambda_obs >= edges[0]) & (lambda_obs <= edges[-1])
        band_idx = np.searchsorted(edges, lambda_obs, side="right") - 1
        band_idx = np.clip(band_idx, 0, len(BANDS) - 1)
        visible = visible & (band_idx >= 0) & (band_idx < len(BANDS))

        log_lambda_obs = np.log(np.maximum(lambda_obs, 1.0))
        logf_cont = np.zeros(n_rows, dtype=float)

        for j in range(len(BANDS)):
            mask = visible & (band_idx == j)
            if not np.any(mask):
                continue

            if 0 < j < len(BANDS) - 1:
                x0 = log_centers[j - 1]
                x1 = log_centers[j + 1]
                y0 = logf[mask, j - 1]
                y1 = logf[mask, j + 1]
                t = (log_lambda_obs[mask] - x0) / (x1 - x0)
                logf_cont[mask] = y0 + t * (y1 - y0)
            else:
                keep = np.asarray([k for k in range(len(BANDS)) if k != j], dtype=int)
                x = log_centers[keep]
                y = logf[mask][:, keep]
                x_mean = np.mean(x)
                y_mean = np.mean(y, axis=1)
                denom = np.sum((x - x_mean) ** 2)
                slope = np.sum((x - x_mean) * (y - y_mean[:, None]), axis=1) / denom
                intercept = y_mean - slope * x_mean
                logf_cont[mask] = intercept + slope * log_lambda_obs[mask]

        cont_flux = np.exp(logf_cont)
        observed_flux = flux[np.arange(n_rows), band_idx]
        residual = np.divide(
            cont_flux - observed_flux,
            cont_flux + 1e-12,
            out=np.zeros(n_rows, dtype=float),
            where=visible,
        )

        deficit = np.clip(np.maximum(residual, 0.0), 0.0, 3.0)
        excess = np.clip(np.maximum(-residual, 0.0), 0.0, 3.0)

        center_j = centers[band_idx]
        left_width = center_j - edges[band_idx]
        right_width = edges[band_idx + 1] - center_j
        reliability_width = np.maximum(left_width, right_width)
        reliability = np.clip(
            1.0 - 0.5 * np.abs(lambda_obs - center_j) / np.maximum(reliability_width, 1e-12),
            0.5,
            1.0,
        )
        weight = visible.astype(float) * reliability

        deficit = np.where(visible, deficit, 0.0)
        excess = np.where(visible, excess, 0.0)
        reliability = np.where(visible, reliability, 0.0)

        safe_name = line_name.lower()
        out[f"{safe_name}_visibility"] = visible.astype(np.int8)
        out[f"{safe_name}_deficit"] = deficit
        out[f"{safe_name}_excess"] = excess
        out[f"{safe_name}_reliability"] = reliability

        all_deficits.append(deficit)
        all_excesses.append(excess)
        all_weights.append(weight)

        if color_group == "blue":
            blue_deficits.append(deficit)
            blue_weights.append(weight)
        else:
            red_deficits.append(deficit)
            red_weights.append(weight)

        if line_group == "cahk":
            cahk_deficits.append(deficit)
            cahk_weights.append(weight)
        elif line_group == "balmer":
            balmer_deficits.append(deficit)
            balmer_weights.append(weight)

    all_deficits_arr = np.column_stack(all_deficits)
    all_excesses_arr = np.column_stack(all_excesses)
    all_weights_arr = np.column_stack(all_weights)
    blue_deficits_arr = np.column_stack(blue_deficits)
    blue_weights_arr = np.column_stack(blue_weights)
    red_deficits_arr = np.column_stack(red_deficits)
    red_weights_arr = np.column_stack(red_weights)
    cahk_deficits_arr = np.column_stack(cahk_deficits)
    cahk_weights_arr = np.column_stack(cahk_weights)
    balmer_deficits_arr = np.column_stack(balmer_deficits)
    balmer_weights_arr = np.column_stack(balmer_weights)

    blue_abs_deficit = _weighted_mean(blue_deficits_arr, blue_weights_arr)
    red_abs_deficit = _weighted_mean(red_deficits_arr, red_weights_arr)
    total_abs_deficit = _weighted_mean(all_deficits_arr, all_weights_arr)
    total_abs_excess = _weighted_mean(all_excesses_arr, all_weights_arr)
    cahk_deficit = _weighted_mean(cahk_deficits_arr, cahk_weights_arr)
    balmer_deficit = _weighted_mean(balmer_deficits_arr, balmer_weights_arr)

    out["blue_abs_deficit"] = blue_abs_deficit
    out["red_abs_deficit"] = red_abs_deficit
    out["total_abs_deficit"] = total_abs_deficit
    out["total_abs_excess"] = total_abs_excess
    out["metal_blanketing_skew"] = (
        (blue_abs_deficit - red_abs_deficit)
        / (blue_abs_deficit + red_abs_deficit + 1e-6)
    )
    out["absorption_excess_ratio"] = total_abs_excess / (total_abs_deficit + 1e-6)
    out["CaHK_to_Balmer"] = cahk_deficit / (balmer_deficit + 1e-6)
    out["visible_line_weight"] = np.sum(all_weights_arr, axis=1)

    gated_features = (
        "blue_abs_deficit",
        "red_abs_deficit",
        "total_abs_deficit",
        "total_abs_excess",
        "metal_blanketing_skew",
        "absorption_excess_ratio",
        "CaHK_to_Balmer",
        "visible_line_weight",
    )
    for regime_name, lower, upper in REDSHIFT_REGIMES:
        mask = ((z_eff >= lower) & (z_eff < upper)).astype(float)
        for feature_name in gated_features:
            out[f"{feature_name}_{regime_name}"] = out[feature_name].to_numpy(dtype=float) * mask

    return out


FEATURE_GROUPS = [
    {
        "name": "redshifted_absorption_trough_residuals",
        "fn": add_redshifted_absorption_trough_residuals,
        "depends_on": [],
        "description": "Broadband residual features measuring redshift-aligned absorption trough geometry across ugriz filters.",
    }
]