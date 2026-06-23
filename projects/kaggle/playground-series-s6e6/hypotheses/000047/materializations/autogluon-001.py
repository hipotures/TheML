import numpy as np
import pandas as pd

_WAVELENGTHS_ANGSTROM = (3551.0, 4686.0, 6165.0, 7481.0, 8931.0)


def _to_float_array(values, fallback):
    arr = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(arr)
    if finite.all():
        return arr
    if finite.any():
        fallback = float(np.nanmedian(arr[finite]))
    return np.where(finite, arr, fallback)


def _sanitize(values, fallback, clip_low=None, clip_high=None):
    arr = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(arr)
    if finite.any():
        replacement = float(np.nanmedian(arr[finite]))
    else:
        replacement = float(fallback)
    arr = np.where(finite, arr, replacement)

    if clip_low is not None or clip_high is not None:
        low = -np.inf if clip_low is None else clip_low
        high = np.inf if clip_high is None else clip_high
        arr = np.clip(arr, low, high)
    return arr


def _top_two(p):
    sorted_p = np.sort(p, axis=1)
    return sorted_p[:, -1], sorted_p[:, -2]


def add_flux_allocation_entropy(raw, deps, aux):
    # External file can be provided through aux by the fixed wrapper; this group
    # uses only covariate-driven statistics from raw for its descriptors.
    _ = aux

    u = _to_float_array(raw["u"], 0.0)
    g = _to_float_array(raw["g"], 0.0)
    r = _to_float_array(raw["r"], 0.0)
    i = _to_float_array(raw["i"], 0.0)
    z = _to_float_array(raw["z"], 0.0)

    flux = np.column_stack(
        (
            np.power(10.0, -0.4 * u),
            np.power(10.0, -0.4 * g),
            np.power(10.0, -0.4 * r),
            np.power(10.0, -0.4 * i),
            np.power(10.0, -0.4 * z),
        )
    )

    finite_flux = np.isfinite(flux)
    if finite_flux.any():
        median_flux = float(np.nanmedian(flux[finite_flux]))
    else:
        median_flux = 1.0
    eps = 1e-12 * max(1.0, median_flux)

    flux = flux + eps
    row_sums = np.sum(flux, axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0.0, 1.0, row_sums)
    p = flux / row_sums

    p_u = p[:, 0]
    p_g = p[:, 1]
    p_r = p[:, 2]
    p_i = p[:, 3]
    p_z = p[:, 4]

    entropy = -np.sum(p * np.log(np.maximum(p, 1e-300)), axis=1)
    entropy_norm = entropy / np.log(5.0)
    gini = 1.0 - np.sum(p * p, axis=1)
    simpson = np.sum(p * p, axis=1)

    p_max, p_second = _top_two(p)
    top_ratio = p_max / (p_second + eps)

    wavelengths = np.array(_WAVELENGTHS_ANGSTROM, dtype=np.float64)
    ln_lambda = np.log(wavelengths)

    mu = np.dot(p, ln_lambda)
    centered = ln_lambda[None, :] - mu[:, None]
    var = np.sum(p * (centered ** 2), axis=1)
    m3 = np.sum(p * (centered ** 3), axis=1)
    m4 = np.sum(p * (centered ** 4), axis=1)

    redshift = _to_float_array(raw["redshift"], 0.0)
    z_scale = 1.0 + np.maximum(redshift, -0.999999) + 1e-6
    ln_lambda_rf = ln_lambda[None, :] - np.log(z_scale)[:, None]
    mu_rf = np.sum(p * ln_lambda_rf, axis=1)
    centered_rf = ln_lambda_rf - mu_rf[:, None]
    var_rf = np.sum(p * (centered_rf ** 2), axis=1)
    m3_rf = np.sum(p * (centered_rf ** 3), axis=1)
    m4_rf = np.sum(p * (centered_rf ** 4), axis=1)

    blue_flux = p_u + p_g
    red_flux = p_i + p_z
    center_flux = p_r

    new_features = pd.DataFrame(
        {
            "p_u": p_u,
            "p_g": p_g,
            "p_r": p_r,
            "p_i": p_i,
            "p_z": p_z,
            "entropy": entropy,
            "entropy_norm": entropy_norm,
            "gini": gini,
            "simpson": simpson,
            "top_ratio": top_ratio,
            "wavelength_mu": mu,
            "wavelength_variance": var,
            "wavelength_m3": m3,
            "wavelength_m4": m4,
            "wavelength_mu_rf": mu_rf,
            "wavelength_variance_rf": var_rf,
            "wavelength_m3_rf": m3_rf,
            "wavelength_m4_rf": m4_rf,
            "blue_flux": blue_flux,
            "red_flux": red_flux,
            "center_flux": center_flux,
            "blue_minus_red": blue_flux - red_flux,
            "p_u_minus_p_i": p_u - p_i,
            "p_g_minus_p_r": p_g - p_r,
            "p_z_minus_p_g": p_z - p_g,
        },
        index=raw.index,
    )

    new_features["entropy"] = _sanitize(new_features["entropy"].to_numpy(), 0.0, clip_low=0.0, clip_high=np.log(5.0))
    new_features["entropy_norm"] = _sanitize(
        new_features["entropy_norm"].to_numpy(),
        0.0,
        clip_low=0.0,
        clip_high=1.0,
    )
    new_features["gini"] = _sanitize(new_features["gini"].to_numpy(), 0.0, clip_low=0.0, clip_high=1.0)
    new_features["simpson"] = _sanitize(new_features["simpson"].to_numpy(), 0.0, clip_low=0.0, clip_high=1.0)
    new_features["top_ratio"] = _sanitize(
        new_features["top_ratio"].to_numpy(),
        1.0,
        clip_low=0.0,
    )
    new_features["wavelength_mu"] = _sanitize(new_features["wavelength_mu"].to_numpy(), 0.0)
    new_features["wavelength_variance"] = _sanitize(new_features["wavelength_variance"].to_numpy(), 0.0, clip_low=0.0)
    new_features["wavelength_m3"] = _sanitize(new_features["wavelength_m3"].to_numpy(), 0.0)
    new_features["wavelength_m4"] = _sanitize(new_features["wavelength_m4"].to_numpy(), 0.0)
    new_features["wavelength_mu_rf"] = _sanitize(new_features["wavelength_mu_rf"].to_numpy(), 0.0)
    new_features["wavelength_variance_rf"] = _sanitize(
        new_features["wavelength_variance_rf"].to_numpy(),
        0.0,
        clip_low=0.0,
    )
    new_features["wavelength_m3_rf"] = _sanitize(new_features["wavelength_m3_rf"].to_numpy(), 0.0)
    new_features["wavelength_m4_rf"] = _sanitize(new_features["wavelength_m4_rf"].to_numpy(), 0.0)

    return new_features


FEATURE_GROUPS = [
    {
        "name": "flux_allocation_entropy",
        "fn": add_flux_allocation_entropy,
        "depends_on": [],
        "description": "Builds normalized ugriz flux-allocation shares, entropy/concentration descriptors, wavelength-moment summaries, rest-frame variants, and contrast features from photometric bands.",
    }
]