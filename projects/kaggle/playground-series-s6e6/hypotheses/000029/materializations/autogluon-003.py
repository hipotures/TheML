import numpy as np
import pandas as pd


UGRIZ_EFFECTIVE_WAVELENGTHS = (3551.0, 4686.0, 6165.0, 7481.0, 8931.0)
UGRIZ_MAG_COLUMNS = ("u", "g", "r", "i", "z")
LYMAN_BREAKS = (("limit_912", 912.0), ("alpha_1216", 1216.0))
EPSILON_FLUX = 1e-12


def add_redshifted_lyman_discontinuity(raw, deps, aux):
    wavelengths = np.asarray(UGRIZ_EFFECTIVE_WAVELENGTHS, dtype=float)
    mags = raw.loc[:, UGRIZ_MAG_COLUMNS].to_numpy(dtype=float, copy=True)
    redshift = raw["redshift"].to_numpy(dtype=float, copy=True)
    z_eff = np.maximum(redshift, 0.0)

    features = pd.DataFrame(index=raw.index)
    n_rows = len(raw)

    for break_label, rest_wavelength in LYMAN_BREAKS:
        observed_break = rest_wavelength * (1.0 + z_eff)
        bin_index = np.searchsorted(wavelengths, observed_break, side="left").astype(np.int16)
        available = (bin_index > 0) & (bin_index < len(wavelengths))

        phase = np.zeros(n_rows, dtype=float)
        boundary_distance = np.zeros(n_rows, dtype=float)
        edge_quality = np.zeros(n_rows, dtype=float)

        local_color = np.zeros(n_rows, dtype=float)
        broad_jump = np.zeros(n_rows, dtype=float)
        flux_jump = np.zeros(n_rows, dtype=float)
        slope = np.zeros(n_rows, dtype=float)

        for j in range(1, len(wavelengths)):
            mask = bin_index == j
            if not np.any(mask):
                continue

            left_wavelength = wavelengths[j - 1]
            right_wavelength = wavelengths[j]
            interval_phase = np.clip(
                (observed_break[mask] - left_wavelength) / (right_wavelength - left_wavelength),
                0.0,
                1.0,
            )

            phase[mask] = interval_phase
            boundary_distance[mask] = np.minimum(interval_phase, 1.0 - interval_phase)
            edge_quality[mask] = np.clip(2.0 * boundary_distance[mask], 0.0, 1.0)

            blue_local = mags[mask, j - 1]
            red_local = mags[mask, j]
            local_color[mask] = blue_local - red_local

            blue_start = max(0, j - 2)
            red_end = min(len(wavelengths), j + 2)

            blue_mag_mean = np.mean(mags[mask, blue_start:j], axis=1)
            red_mag_mean = np.mean(mags[mask, j:red_end], axis=1)
            broad_jump[mask] = blue_mag_mean - red_mag_mean

            flux_values = np.power(10.0, -0.4 * mags[mask, :])
            blue_flux_mean = np.mean(flux_values[:, blue_start:j], axis=1)
            red_flux_mean = np.mean(flux_values[:, j:red_end], axis=1)
            flux_jump[mask] = np.log10((red_flux_mean + EPSILON_FLUX) / (blue_flux_mean + EPSILON_FLUX))

            slope[mask] = (red_local - blue_local) / (np.log(right_wavelength) - np.log(left_wavelength))

        features[f"{break_label}_observed_wavelength"] = observed_break
        features[f"{break_label}_bin_code"] = bin_index
        features[f"{break_label}_edge_available"] = available.astype(np.int8)
        features[f"{break_label}_outside_blueward"] = (bin_index == 0).astype(np.int8)
        features[f"{break_label}_outside_redward"] = (bin_index == len(wavelengths)).astype(np.int8)

        for j, interval_name in enumerate(("u_g", "g_r", "r_i", "i_z"), start=1):
            features[f"{break_label}_interval_{interval_name}"] = (bin_index == j).astype(np.int8)

        features[f"{break_label}_phase"] = phase
        features[f"{break_label}_boundary_distance"] = boundary_distance
        features[f"{break_label}_edge_quality"] = edge_quality

        features[f"{break_label}_local_color"] = local_color
        features[f"{break_label}_broad_jump"] = broad_jump
        features[f"{break_label}_flux_jump"] = flux_jump
        features[f"{break_label}_slope"] = slope

        features[f"{break_label}_local_color_weighted"] = local_color * edge_quality
        features[f"{break_label}_broad_jump_weighted"] = broad_jump * edge_quality
        features[f"{break_label}_flux_jump_weighted"] = flux_jump * edge_quality
        features[f"{break_label}_slope_weighted"] = slope * edge_quality

    return features


FEATURE_GROUPS = [
    {
        "name": "redshifted_lyman_discontinuity",
        "fn": add_redshifted_lyman_discontinuity,
        "depends_on": [],
        "description": "Rest-frame Lyman-limit and Lyman-alpha dropout geometry across fixed ugriz passbands.",
    }
]