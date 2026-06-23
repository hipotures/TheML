import numpy as np
import pandas as pd

_LYMAN_BANDS = ("u", "g", "r", "i", "z")
_LYMAN_WAVELENGTHS = (3551.0, 4686.0, 6165.0, 7481.0, 8931.0)
_LYMAN_BREAKS = (912.0, 1216.0)
_EPS = 1e-12
_MIN_DENOM = 1e-12


def _compute_single_break_features(rest_wavelength, flux_proxy, magnitudes, break_wave):
    n_rows, n_bands = magnitudes.shape
    rows = np.arange(n_rows, dtype=np.int64)

    has_edge = rest_wavelength >= break_wave
    available_mask = np.any(has_edge, axis=1)
    break_idx = np.where(available_mask, has_edge.argmax(axis=1), n_bands).astype(np.int16)
    edge_available = ((break_idx > 0) & (break_idx < n_bands)).astype(np.int8)

    jump = np.zeros(n_rows, dtype=np.float64)
    phase = np.zeros(n_rows, dtype=np.float64)
    local = np.zeros(n_rows, dtype=np.float64)

    active = edge_available.astype(bool)
    if np.any(active):
        active_rows = rows[active]
        idx = break_idx[active]
        left_idx = idx - 1

        left_wave = rest_wavelength[active_rows, left_idx]
        right_wave = rest_wavelength[active_rows, idx]
        phase_active = (break_wave - left_wave) / (right_wave - left_wave)
        phase[active_rows] = np.clip(phase_active, 0.0, 1.0)

        left_mag = magnitudes[active_rows, left_idx]
        right_mag = magnitudes[active_rows, idx]
        local[active_rows] = left_mag - right_mag

        active_flux = flux_proxy[active_rows]
        cumsum_flux = np.cumsum(active_flux, axis=1)
        low_sum = cumsum_flux[np.arange(active_flux.shape[0]), left_idx]
        high_sum = cumsum_flux[:, -1] - low_sum
        low_count = idx.astype(np.float64)
        high_count = float(n_bands) - idx.astype(np.float64)

        low_mean = low_sum / low_count
        high_mean = high_sum / high_count
        jump[active_rows] = np.log10((high_mean + _EPS) / (low_mean + _EPS))

    return break_idx, edge_available, jump, phase, local


def add_redshifted_lyman_discontinuity(raw, deps, aux):
    _ = (deps, aux)

    required_columns = set(_LYMAN_BANDS) | {"redshift"}
    missing_columns = required_columns - set(raw.columns)
    if missing_columns:
        raise KeyError(f"Missing required columns: {sorted(missing_columns)}")

    base = "redshifted_lyman_discontinuity"

    wavelengths = np.asarray(_LYMAN_WAVELENGTHS, dtype=np.float64)
    z = pd.to_numeric(raw["redshift"], errors="coerce").to_numpy(dtype=np.float64)
    denom = 1.0 + z
    denom = np.where(np.isfinite(denom) & (denom > _MIN_DENOM), denom, np.nan)

    mags = raw.loc[:, list(_LYMAN_BANDS)].to_numpy(dtype=np.float64)
    flux = np.power(10.0, -0.4 * mags)
    rest = wavelengths[None, :] / denom[:, None]

    features = {}

    for break_wave in _LYMAN_BREAKS:
        break_idx, edge_available, jump, phase, local = _compute_single_break_features(rest, flux, mags, break_wave)
        tag = str(int(break_wave))

        features[f"{base}_jump_{tag}"] = pd.Series(jump, index=raw.index, dtype="float64")
        features[f"{base}_phase_{tag}"] = pd.Series(phase, index=raw.index, dtype="float64")
        features[f"{base}_local_{tag}"] = pd.Series(local, index=raw.index, dtype="float64")
        features[f"{base}_break_band_{tag}"] = pd.Series(break_idx, index=raw.index, dtype="int16")
        features[f"{base}_edge_available_{tag}"] = pd.Series(edge_available, index=raw.index, dtype="int8")
        features[f"{base}_jump_x_break_band_{tag}"] = pd.Series(
            jump * break_idx.astype(np.float64), index=raw.index, dtype="float64"
        )
        features[f"{base}_local_x_edge_{tag}"] = pd.Series(
            local * edge_available.astype(np.float64), index=raw.index, dtype="float64"
        )

    return pd.DataFrame(features, index=raw.index)


FEATURE_GROUPS = [
    {
        "name": "redshifted_lyman_discontinuity",
        "fn": add_redshifted_lyman_discontinuity,
        "depends_on": [],
        "description": "Compute redshifted Lyman-limit and Lyman-alpha dropout geometry features from ugriz flux proxies and rest-frame band boundaries.",
    }
]