import numpy as np
import pandas as pd

BANDS = ("u", "g", "r", "i", "z")
REST_WAVELENGTHS_ANGSTROM = (3551.0, 4686.0, 6165.0, 7481.0, 8931.0)
LYMAN_FULL_WEIGHT = 1216.0
LYMAN_PARTIAL_WEIGHT = 912.0
WEIGHT_EPS = 1e-12
NUMERICAL_EPS = 1e-15


def _as_1d_float(values):
    return np.asarray(values, dtype=float).reshape(-1)


def _weighted_poly_block(x, y, weights):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    weights = np.asarray(weights, dtype=float)

    n = x.shape[0]
    rss_linear = np.full(n, np.nan, dtype=float)
    r2_linear = np.full(n, np.nan, dtype=float)
    slope_linear = np.full(n, np.nan, dtype=float)
    rss_quadratic = np.full(n, np.nan, dtype=float)
    r2_quadratic = np.full(n, np.nan, dtype=float)
    curvature = np.full(n, np.nan, dtype=float)
    abs_curvature = np.full(n, np.nan, dtype=float)
    curvature_gain = np.full(n, np.nan, dtype=float)

    linear_feasible = np.zeros(n, dtype=bool)
    quadratic_feasible = np.zeros(n, dtype=bool)

    for i in range(n):
        x_row = _as_1d_float(x[i])
        y_row = _as_1d_float(y[i])
        w_row = _as_1d_float(weights[i])

        active = w_row > WEIGHT_EPS
        k = int(np.count_nonzero(active))
        if k < 2:
            continue

        x_a = x_row[active]
        y_a = y_row[active]
        w_a = w_row[active]

        if x_a.size < 2:
            continue

        s0 = float(np.sum(w_a))
        if not np.isfinite(s0) or s0 <= WEIGHT_EPS:
            continue

        sx = np.dot(w_a, x_a)
        sx2 = np.dot(w_a, x_a * x_a)
        sx3 = np.dot(w_a, x_a * x_a * x_a)
        sx4 = np.dot(w_a, x_a * x_a * x_a * x_a)
        sy = np.dot(w_a, y_a)
        sxy = np.dot(w_a, x_a * y_a)
        sx2y = np.dot(w_a, x_a * x_a * y_a)

        det_lin = s0 * sx2 - sx * sx
        if abs(det_lin) > NUMERICAL_EPS:
            b1 = (s0 * sxy - sx * sy) / det_lin
            a1 = (sy - b1 * sx) / s0

            pred_lin = a1 + b1 * x_a
            resid_lin = y_a - pred_lin
            rss_lin = np.dot(w_a, resid_lin * resid_lin)
            y_mean = sy / s0
            sst = np.dot(w_a, (y_a - y_mean) ** 2)

            linear_feasible[i] = True
            slope_linear[i] = b1
            rss_linear[i] = rss_lin
            if sst > NUMERICAL_EPS:
                r2_linear[i] = 1.0 - (rss_lin / sst)
            elif rss_lin <= NUMERICAL_EPS:
                r2_linear[i] = 1.0

        if k >= 3:
            design = np.array(
                [
                    [s0, sx, sx2],
                    [sx, sx2, sx3],
                    [sx2, sx3, sx4],
                ],
                dtype=float,
            )
            rhs = np.array([sy, sxy, sx2y], dtype=float)
            try:
                a2, b2, c2 = np.linalg.solve(design, rhs)
            except np.linalg.LinAlgError:
                coeffs = np.linalg.lstsq(design, rhs, rcond=None)[0]
                a2, b2, c2 = coeffs

            if np.isfinite(a2) and np.isfinite(b2) and np.isfinite(c2):
                pred_quad = a2 + b2 * x_a + c2 * x_a * x_a
                resid_quad = y_a - pred_quad
                rss_quad = np.dot(w_a, resid_quad * resid_quad)
                y_mean = sy / s0
                sst = np.dot(w_a, (y_a - y_mean) ** 2)

                quadratic_feasible[i] = True
                rss_quadratic[i] = rss_quad
                curvature[i] = c2
                abs_curvature[i] = abs(c2)
                if sst > NUMERICAL_EPS:
                    r2_quadratic[i] = 1.0 - (rss_quad / sst)
                elif rss_quad <= NUMERICAL_EPS:
                    r2_quadratic[i] = 1.0

        if (
            linear_feasible[i]
            and quadratic_feasible[i]
            and np.isfinite(rss_linear[i])
            and rss_linear[i] > NUMERICAL_EPS
        ):
            curvature_gain[i] = (rss_linear[i] - rss_quadratic[i]) / rss_linear[i]

    return {
        "rss_linear": rss_linear,
        "r2_linear": r2_linear,
        "slope_linear": slope_linear,
        "rss_quadratic": rss_quadratic,
        "r2_quadratic": r2_quadratic,
        "curvature": curvature,
        "abs_curvature": abs_curvature,
        "curvature_gain": curvature_gain,
        "linear_feasible": linear_feasible,
        "quadratic_feasible": quadratic_feasible,
    }


def add_restframe_sed_family_fit(raw, deps, aux):
    mags = raw.loc[:, BANDS].to_numpy(dtype=float, copy=False)
    redshift = raw["redshift"].to_numpy(dtype=float, copy=False)

    zc = np.maximum(redshift, 0.0)
    z_low_flag = (redshift < 0.0).astype(np.uint8)

    wavelengths = np.array(REST_WAVELENGTHS_ANGSTROM, dtype=float)
    rest_wave = wavelengths[None, :] / (1.0 + zc)[:, None]
    x = np.log10(rest_wave)
    x_centered = x - x.mean(axis=1, keepdims=True)

    y = -0.4 * mags
    y_centered = y - y.mean(axis=1, keepdims=True)

    weights_unweighted = np.ones_like(x_centered, dtype=float)
    weights_lyman = np.where(
        rest_wave > LYMAN_FULL_WEIGHT,
        1.0,
        np.where(rest_wave > LYMAN_PARTIAL_WEIGHT, 0.5, 0.0),
    )

    fit_unweighted = _weighted_poly_block(x_centered, y_centered, weights_unweighted)
    fit_lyman = _weighted_poly_block(x_centered, y_centered, weights_lyman)

    lyman_instability = (
        ~fit_lyman["linear_feasible"] | ~fit_lyman["quadratic_feasible"]
    ).astype(np.uint8)

    lyman_rss_linear = fit_lyman["rss_linear"].copy()
    lyman_r2_linear = fit_lyman["r2_linear"].copy()
    lyman_slope_linear = fit_lyman["slope_linear"].copy()
    lyman_rss_quadratic = fit_lyman["rss_quadratic"].copy()
    lyman_r2_quadratic = fit_lyman["r2_quadratic"].copy()
    lyman_curvature = fit_lyman["curvature"].copy()
    lyman_abs_curvature = fit_lyman["abs_curvature"].copy()
    lyman_curvature_gain = fit_lyman["curvature_gain"].copy()

    unstable_idx = lyman_instability.astype(bool)
    if unstable_idx.any():
        lyman_rss_linear[unstable_idx] = fit_unweighted["rss_linear"][unstable_idx]
        lyman_r2_linear[unstable_idx] = fit_unweighted["r2_linear"][unstable_idx]
        lyman_slope_linear[unstable_idx] = fit_unweighted["slope_linear"][
            unstable_idx
        ]
        lyman_rss_quadratic[unstable_idx] = fit_lyman["rss_quadratic"][unstable_idx]
        lyman_r2_quadratic[unstable_idx] = fit_unweighted["r2_quadratic"][
            unstable_idx
        ]
        lyman_curvature[unstable_idx] = fit_unweighted["curvature"][unstable_idx]
        lyman_abs_curvature[unstable_idx] = fit_unweighted["abs_curvature"][
            unstable_idx
        ]
        lyman_curvature_gain[unstable_idx] = fit_unweighted["curvature_gain"][
            unstable_idx
        ]

    den_ug = x_centered[:, 0] - x_centered[:, 1]
    den_iz = x_centered[:, 3] - x_centered[:, 4]
    den_gr = x_centered[:, 1] - x_centered[:, 2]
    den_ri = x_centered[:, 2] - x_centered[:, 3]

    asym_unweighted = (y_centered[:, 0] - y_centered[:, 1]) / den_ug - (
        y_centered[:, 3] - y_centered[:, 4]
    ) / den_iz

    lyman_endpoints_ok = (
        (weights_lyman[:, 0] > WEIGHT_EPS)
        & (weights_lyman[:, 1] > WEIGHT_EPS)
        & (weights_lyman[:, 3] > WEIGHT_EPS)
        & (weights_lyman[:, 4] > WEIGHT_EPS)
    )
    asym_fallback = (y_centered[:, 1] - y_centered[:, 2]) / den_gr - (
        y_centered[:, 2] - y_centered[:, 3]
    ) / den_ri
    asym_lyman = np.where(lyman_endpoints_ok, asym_unweighted, asym_fallback)
    asym_lyman_fallback = (~lyman_endpoints_ok).astype(np.uint8)
    if unstable_idx.any():
        asym_lyman[unstable_idx] = asym_unweighted[unstable_idx]

    no_uv_downweight = (
        (weights_lyman > 1.0 - WEIGHT_EPS).all(axis=1)
    ).astype(np.uint8)

    delta_r2 = lyman_r2_quadratic - fit_unweighted["r2_quadratic"]
    delta_slope = lyman_slope_linear - fit_unweighted["slope_linear"]
    delta_curvature = lyman_curvature - fit_unweighted["curvature"]

    return pd.DataFrame(
        {
            "restframe_sed_unw_rss_linear": fit_unweighted["rss_linear"],
            "restframe_sed_unw_r2_linear": fit_unweighted["r2_linear"],
            "restframe_sed_unw_slope": fit_unweighted["slope_linear"],
            "restframe_sed_unw_rss_quadratic": fit_unweighted["rss_quadratic"],
            "restframe_sed_unw_r2_quadratic": fit_unweighted["r2_quadratic"],
            "restframe_sed_unw_curvature": fit_unweighted["curvature"],
            "restframe_sed_unw_abs_curvature": fit_unweighted["abs_curvature"],
            "restframe_sed_unw_curvature_gain": fit_unweighted["curvature_gain"],
            "restframe_sed_unw_asymmetry": asym_unweighted,
            "restframe_sed_unw_linear_feasible": fit_unweighted["linear_feasible"].astype(
                np.uint8
            ),
            "restframe_sed_unw_quadratic_feasible": fit_unweighted[
                "quadratic_feasible"
            ].astype(np.uint8),
            "restframe_sed_lyman_rss_linear": lyman_rss_linear,
            "restframe_sed_lyman_r2_linear": lyman_r2_linear,
            "restframe_sed_lyman_slope": lyman_slope_linear,
            "restframe_sed_lyman_rss_quadratic": lyman_rss_quadratic,
            "restframe_sed_lyman_r2_quadratic": lyman_r2_quadratic,
            "restframe_sed_lyman_curvature": lyman_curvature,
            "restframe_sed_lyman_abs_curvature": lyman_abs_curvature,
            "restframe_sed_lyman_curvature_gain": lyman_curvature_gain,
            "restframe_sed_lyman_asymmetry": asym_lyman,
            "restframe_sed_lyman_asymmetry_fallback": asym_lyman_fallback,
            "restframe_sed_lyman_linear_feasible_raw": fit_lyman[
                "linear_feasible"
            ].astype(np.uint8),
            "restframe_sed_lyman_quadratic_feasible_raw": fit_lyman[
                "quadratic_feasible"
            ].astype(np.uint8),
            "restframe_sed_lyman_instability": lyman_instability,
            "restframe_sed_delta_r2": delta_r2,
            "restframe_sed_delta_slope": delta_slope,
            "restframe_sed_delta_curvature": delta_curvature,
            "restframe_sed_no_uv_downweight": no_uv_downweight,
            "restframe_sed_z_low_flag": z_low_flag,
        },
        index=raw.index,
    )


FEATURE_GROUPS = [
    {
        "name": "restframe_sed_family_fit",
        "fn": add_restframe_sed_family_fit,
        "depends_on": [],
        "description": "Fits unweighted and Lyman-aware rest-frame ugriz log-flux continua with linear/quadratic models, emitting curvature, fit-quality, asymmetry, feasibility, stability, and weighted-vs-unweighted branch differences.",
    },
]