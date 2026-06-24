import numpy as np
import pandas as pd

_BAND_NAMES = ("u", "g", "r", "i", "z")
_BAND_EDGES = (3000.0, 4100.0, 5500.0, 7000.0, 8200.0, 9200.0)
_BAND_CENTERS = (3551.0, 4686.0, 6165.0, 7481.0, 8931.0)
_LINE_SPECS = (
    ("CaIIK", 3933.7, "blue"),
    ("CaIIH", 3968.5, "blue"),
    ("G_band", 4304.0, "blue"),
    ("Hdelta", 4102.0, "blue"),
    ("Hgamma", 4341.0, "blue"),
    ("Hbeta", 4861.0, "red"),
    ("MgB", 5175.0, "red"),
    ("NaID", 5893.0, "red"),
)
_REGIME_BINS = (
    (0.0, 0.45, "z_0_0_45"),
    (0.45, 1.1, "z_0_45_1_1"),
    (1.1, 2.0, "z_1_1_2_0"),
    (2.0, 7.0, "z_2_0_7_0"),
)
_EPS = 1e-12
_EPS_RATIO = 1e-6


def _fit_edge_continuum_log(log_flux_points, log_wave_points, wavelength):
    log_wave_points = np.asarray(log_wave_points, dtype=np.float64)
    wave_mean = log_wave_points.mean()
    wave_dev = log_wave_points - wave_mean
    denom = np.sum(wave_dev * wave_dev)
    y_mean = log_flux_points.mean(axis=1)
    slope = np.sum((log_flux_points - y_mean[:, None]) * wave_dev, axis=1) / denom
    intercept = y_mean - slope * wave_mean
    return intercept + slope * np.log(wavelength)


def add_redshifted_absorption_trough_residuals(raw, deps, aux):
    _ = (deps, aux)

    z = raw["redshift"].to_numpy(dtype=np.float64, copy=False)
    n = z.shape[0]

    band_edges = np.array(_BAND_EDGES, dtype=np.float64)
    band_centers = np.array(_BAND_CENTERS, dtype=np.float64)
    log_band_centers = np.log(band_centers)
    half_band_widths = (band_edges[1:] - band_edges[:-1]) / 2.0

    flux_matrix = np.column_stack(
        [np.power(10.0, -0.4 * raw[col].to_numpy(dtype=np.float64, copy=False)) for col in _BAND_NAMES]
    )
    log_flux_matrix = np.log(flux_matrix)

    d_blue = np.zeros(n, dtype=np.float64)
    d_red = np.zeros(n, dtype=np.float64)
    a_tot = np.zeros(n, dtype=np.float64)
    visible_line_count = np.zeros(n, dtype=np.float64)

    d_cakih = np.zeros(n, dtype=np.float64)
    d_caIIH = np.zeros(n, dtype=np.float64)
    d_hdelta = np.zeros(n, dtype=np.float64)
    d_hgamma = np.zeros(n, dtype=np.float64)

    feature_data = {}

    edge_fit_x_left = np.array((log_band_centers[1], log_band_centers[2], log_band_centers[3], log_band_centers[4]), dtype=np.float64)
    edge_fit_x_right = np.array((log_band_centers[0], log_band_centers[1], log_band_centers[2], log_band_centers[3]), dtype=np.float64)

    for line_name, rest_wave, region in _LINE_SPECS:
        obs_wave = rest_wave * (1.0 + z)
        visible = (z >= 0.0) & (obs_wave >= band_edges[0]) & (obs_wave <= band_edges[-1])
        line_rows = np.flatnonzero(visible)
        line_deficit = np.zeros(n, dtype=np.float64)

        if line_rows.size:
            obs_visible = obs_wave[line_rows]
            band_idx = np.searchsorted(band_edges[1:], obs_visible, side="left")
            band_idx = np.clip(band_idx, 0, len(_BAND_NAMES) - 1)

            center_visible = band_centers[band_idx]
            half_visible = half_band_widths[band_idx]
            q_vis = 1.0 - 0.5 * np.minimum(1.0, np.abs(obs_visible - center_visible) / half_visible)
            q_vis = np.clip(q_vis, 0.5, 1.0)

            log_obs = np.log(obs_visible)
            cont_log = np.zeros(line_rows.size, dtype=np.float64)

            for bj in range(len(_BAND_NAMES)):
                within_band = band_idx == bj
                if not within_band.any():
                    continue
                idx = np.flatnonzero(within_band)
                rows = line_rows[idx]
                log_lam = log_obs[idx]

                if bj == 0:
                    y = log_flux_matrix[rows][:, (1, 2, 3, 4)]
                    cont_log[idx] = _fit_edge_continuum_log(y, edge_fit_x_left, obs_visible[idx])
                elif bj == 4:
                    y = log_flux_matrix[rows][:, (0, 1, 2, 3)]
                    cont_log[idx] = _fit_edge_continuum_log(y, edge_fit_x_right, obs_visible[idx])
                else:
                    left = bj - 1
                    right = bj + 1
                    y_left = log_flux_matrix[rows, left]
                    y_right = log_flux_matrix[rows, right]
                    t = (log_lam - log_band_centers[left]) / (log_band_centers[right] - log_band_centers[left])
                    cont_log[idx] = y_left + (y_right - y_left) * t

            cont = np.exp(cont_log)
            fb = flux_matrix[line_rows, band_idx]
            residual = (cont - fb) / (cont + _EPS)

            d_vis = np.minimum(np.maximum(residual, 0.0), 3.0) * q_vis
            a_vis = np.minimum(np.maximum(-residual, 0.0), 3.0) * q_vis

            line_deficit[line_rows] = d_vis
            line_excess = a_vis

            if region == "blue":
                d_blue += d_vis
            else:
                d_red += d_vis

            a_tot += line_excess
            visible_line_count += q_vis

            if line_name == "CaIIK":
                d_cakih = line_deficit
            elif line_name == "CaIIH":
                d_caIIH = line_deficit
            elif line_name == "Hdelta":
                d_hdelta = line_deficit
            elif line_name == "Hgamma":
                d_hgamma = line_deficit

        feature_data[f"visible_{line_name}"] = visible.astype(np.int8)

    d_blue = np.clip(d_blue, 0.0, 3.0 * len(_LINE_SPECS))
    d_red = np.clip(d_red, 0.0, 3.0 * len(_LINE_SPECS))
    a_tot = np.clip(a_tot, 0.0, 3.0 * len(_LINE_SPECS))

    denom_abs = d_blue + d_red + _EPS_RATIO
    absorption_excess_ratio = a_tot / denom_abs
    metal_blanketing_skew = (d_blue - d_red) / denom_abs
    cak_vs_balmer_denom = d_hdelta + d_hgamma + _EPS_RATIO
    cak_cah_vs_balmer = (d_cakih + d_caIIH) / cak_vs_balmer_denom

    features = {
        "D_blue": d_blue,
        "D_red": d_red,
        "A_tot": a_tot,
        "absorption_excess_ratio": absorption_excess_ratio,
        "metal_blanketing_skew": metal_blanketing_skew,
        "CaK_CaH_vs_Balmer": cak_cah_vs_balmer,
        "visible_line_count": visible_line_count,
    }
    features.update(feature_data)

    for lo, hi, tag in _REGIME_BINS:
        regime_mask = (z >= lo) & (z < hi)
        m = regime_mask.astype(np.float64)
        features[f"D_blue_{tag}"] = d_blue * m
        features[f"D_red_{tag}"] = d_red * m
        features[f"A_tot_{tag}"] = a_tot * m
        features[f"absorption_excess_ratio_{tag}"] = absorption_excess_ratio * m
        features[f"metal_blanketing_skew_{tag}"] = metal_blanketing_skew * m
        features[f"CaK_CaH_vs_Balmer_{tag}"] = cak_cah_vs_balmer * m
        features[f"visible_line_count_{tag}"] = visible_line_count * m

    return pd.DataFrame(features, index=raw.index)


FEATURE_GROUPS = [
    {
        "name": "redshifted_absorption_trough_residuals",
        "fn": add_redshifted_absorption_trough_residuals,
        "depends_on": [],
        "description": "Builds redshift-aware absorption and excess residual geometry features from ugriz flux trough and edge-weighted line alignment.",
    },
]