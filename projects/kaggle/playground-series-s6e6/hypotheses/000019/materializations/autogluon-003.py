import numpy as np
import pandas as pd


_BANDS = ("u", "g", "r", "i", "z")
_WAVELENGTHS_ANGSTROM = (3543.0, 4770.0, 6231.0, 7625.0, 9134.0)
_FALLBACK_LOWER = -1.0
_FALLBACK_UPPER = 35.0


def _fit_reference_rows(raw, aux):
    if aux is None or len(aux) != len(raw):
        return raw

    for col in ("fit_mask", "is_fit", "is_train_fold", "train_fold_mask"):
        if col in aux.columns:
            mask = aux[col].to_numpy(dtype=bool, copy=True)
            if mask.shape[0] == len(raw) and bool(np.any(mask)):
                return raw.loc[mask]

    if "is_train" in aux.columns:
        mask = aux["is_train"].to_numpy(dtype=bool, copy=True)
        if mask.shape[0] == len(raw) and bool(np.any(mask)):
            return raw.loc[mask]

    return raw


def _safe_stat(value, fallback):
    if np.isfinite(value):
        return float(value)
    return float(fallback)


def add_observed_sed_continuum_moments(raw, deps, aux):
    index = raw.index
    fit_raw = _fit_reference_rows(raw, aux)

    cleaned_columns = []
    for band in _BANDS:
        values = pd.to_numeric(raw[band], errors="coerce").to_numpy(dtype=float, copy=True)
        fit_values = pd.to_numeric(fit_raw[band], errors="coerce").to_numpy(dtype=float, copy=True)
        finite_fit = fit_values[np.isfinite(fit_values)]

        if finite_fit.size:
            median = _safe_stat(np.nanmedian(finite_fit), 17.5)
            lower = _safe_stat(np.nanquantile(finite_fit, 0.001), _FALLBACK_LOWER)
            upper = _safe_stat(np.nanquantile(finite_fit, 0.999), _FALLBACK_UPPER)
        else:
            median = 17.5
            lower = _FALLBACK_LOWER
            upper = _FALLBACK_UPPER

        if not lower < upper:
            lower = _FALLBACK_LOWER
            upper = _FALLBACK_UPPER

        values[~np.isfinite(values)] = median
        values = np.clip(values, lower, upper)
        cleaned_columns.append(values)

    mags = np.column_stack(cleaned_columns)
    flux_log = -0.4 * mags
    centered_flux = flux_log - np.mean(flux_log, axis=1, keepdims=True)

    wavelengths = np.asarray(_WAVELENGTHS_ANGSTROM, dtype=float)
    t = np.log10(wavelengths)
    t = t - np.mean(t)
    q = (t * t) - np.mean(t * t)
    design = np.column_stack((np.ones(len(t), dtype=float), t, q))
    pinv = np.linalg.pinv(design)

    beta = centered_flux @ pinv.T
    fitted = beta @ design.T
    residuals = centered_flux - fitted
    abs_residuals = np.abs(residuals)

    blue_slope = (centered_flux[:, 1] - centered_flux[:, 0]) / (t[1] - t[0])
    red_slope = (centered_flux[:, 4] - centered_flux[:, 2]) / (t[4] - t[2])

    data = {
        "continuum_tilt": beta[:, 1],
        "continuum_curvature": beta[:, 2],
        "residual_rms": np.sqrt(np.mean(residuals * residuals, axis=1)),
        "residual_abs_mean": np.mean(abs_residuals, axis=1),
        "residual_abs_max": np.max(abs_residuals, axis=1),
        "residual_u": residuals[:, 0],
        "residual_g": residuals[:, 1],
        "residual_r": residuals[:, 2],
        "residual_i": residuals[:, 3],
        "residual_z": residuals[:, 4],
        "blue_slope": blue_slope,
        "red_slope": red_slope,
        "red_minus_blue_slope": red_slope - blue_slope,
    }

    out = pd.DataFrame(data, index=index)
    return out.replace([np.inf, -np.inf], np.nan).fillna(0.0)


FEATURE_GROUPS = [
    {
        "name": "observed_sed_continuum_moments",
        "fn": add_observed_sed_continuum_moments,
        "depends_on": [],
        "description": "Brightness-normalized ugriz continuum moments and band residuals from a quadratic spectral trend.",
    }
]