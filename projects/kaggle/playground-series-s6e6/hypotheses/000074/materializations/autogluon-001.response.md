import numpy as np
import pandas as pd


SDSS_U_BAND_CONTINUUM_WAVELENGTHS = {
    "u": 3551.0,
    "g": 4686.0,
    "r": 6166.0,
    "i": 7480.0,
    "z": 8932.0,
}

U_BAND_CONTINUUM_THRESHOLDS = (-1.0, -0.5, -0.2, 0.2, 0.5, 1.0)


def add_u_band_continuum_anomaly(raw, deps, aux):
    index = raw.index

    u = raw["u"].to_numpy(dtype=float, copy=True)
    griz = raw[["g", "r", "i", "z"]].to_numpy(dtype=float, copy=True)

    wavelengths = SDSS_U_BAND_CONTINUUM_WAVELENGTHS
    lambda_r = wavelengths["r"]
    x_griz = np.log(
        np.asarray(
            [
                wavelengths["g"] / lambda_r,
                wavelengths["r"] / lambda_r,
                wavelengths["i"] / lambda_r,
                wavelengths["z"] / lambda_r,
            ],
            dtype=float,
        )
    )
    design = np.column_stack([np.ones(4, dtype=float), x_griz, x_griz * x_griz])
    pinv = np.linalg.pinv(design)

    x_u = np.log(wavelengths["u"] / lambda_r)
    u_vector = np.asarray([1.0, x_u, x_u * x_u], dtype=float)

    finite_griz = np.isfinite(griz).all(axis=1)
    finite_u = np.isfinite(u)
    good = finite_griz & finite_u

    safe_griz = np.where(np.isfinite(griz), griz, 0.0)
    coeffs = safe_griz @ pinv.T
    u_hat = coeffs @ u_vector
    griz_hat = coeffs @ design.T
    griz_rms = np.sqrt(np.mean((safe_griz - griz_hat) ** 2, axis=1))

    d_u = u - u_hat

    bad = ~good
    d_u = np.where(good, d_u, 0.0)
    griz_rms = np.where(good, griz_rms, 0.0)

    d_u_clipped = np.clip(d_u, -5.0, 5.0)
    abs_d_u = np.abs(d_u_clipped)
    uv_excess_mag = np.maximum(0.0, -d_u_clipped)
    uv_dropout_mag = np.maximum(0.0, d_u_clipped)
    log10_flux_ratio_proxy = -0.4 * d_u_clipped
    flux_ratio = np.clip(np.power(10.0, log10_flux_ratio_proxy), 0.001, 1000.0)

    faint_u_penalty = 0.03 * np.maximum(u - 22.0, 0.0)
    faint_u_penalty = np.where(np.isfinite(faint_u_penalty), faint_u_penalty, 0.0)
    normalized_d_u = d_u_clipped / (0.05 + griz_rms + faint_u_penalty)
    normalized_d_u = np.where(good & np.isfinite(normalized_d_u), normalized_d_u, 0.0)

    features = pd.DataFrame(
        {
            "u_continuum_resid": d_u_clipped,
            "u_continuum_abs_resid": abs_d_u,
            "u_continuum_uv_excess_mag": uv_excess_mag,
            "u_continuum_dropout_mag": uv_dropout_mag,
            "u_continuum_log10_flux_ratio_proxy": log10_flux_ratio_proxy,
            "u_continuum_flux_ratio": flux_ratio,
            "u_continuum_griz_rms": griz_rms,
            "u_continuum_norm_resid": normalized_d_u,
            "u_continuum_bad_flag": bad.astype(np.int8),
        },
        index=index,
    )

    for threshold in U_BAND_CONTINUUM_THRESHOLDS:
        if threshold < 0.0:
            name = "u_continuum_hinge_below_" + str(abs(threshold)).replace(".", "p")
            values = np.maximum(0.0, threshold - d_u_clipped)
        else:
            name = "u_continuum_hinge_above_" + str(threshold).replace(".", "p")
            values = np.maximum(0.0, d_u_clipped - threshold)
        features[name] = np.where(good, values, 0.0)

    return features


FEATURE_GROUPS = [
    {
        "name": "u_band_continuum_anomaly",
        "fn": add_u_band_continuum_anomaly,
        "depends_on": [],
        "description": "Measures u-band excess or dropout relative to a fixed quadratic optical continuum fit from g, r, i, and z magnitudes.",
    }
]