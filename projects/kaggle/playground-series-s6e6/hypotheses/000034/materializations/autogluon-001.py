import numpy as np
import pandas as pd

_BB_BANDS = ("u", "g", "r", "i", "z")
_BB_WAVELENGTHS = (3551.0, 4686.0, 6165.0, 7481.0, 8931.0)
_BB_TEMPERATURE_MIN = 1500.0
_BB_TEMPERATURE_MAX = 50000.0
_BB_TEMPERATURE_STEPS = 140
_BB_PLANCK_CONSTANT = 14387.7
_BB_EPS = 1e-12
_BB_BLOCK_SIZE = 8192


def add_blackbody_continuum_distance(raw, deps, aux):
    _ = deps
    _ = aux

    required_cols = ("redshift",) + _BB_BANDS
    missing_cols = [col for col in required_cols if col not in raw.columns]
    if missing_cols:
        raise KeyError(f"Missing required columns for blackbody_continuum_distance: {missing_cols}")

    n_rows = len(raw)
    index = raw.index

    wavelengths = np.asarray(_BB_WAVELENGTHS, dtype=np.float64)
    temps = np.logspace(
        np.log10(_BB_TEMPERATURE_MIN),
        np.log10(_BB_TEMPERATURE_MAX),
        _BB_TEMPERATURE_STEPS,
        dtype=np.float64,
    )
    n_t = temps.size
    if n_t < 2:
        raise ValueError("Temperature grid must contain at least two points.")

    band_flux = np.empty((n_rows, len(_BB_BANDS)), dtype=np.float64)
    for i, band in enumerate(_BB_BANDS):
        values = pd.to_numeric(raw[band], errors="coerce").to_numpy(dtype=np.float64)
        finite = np.isfinite(values)
        if not finite.all():
            if finite.any():
                fill = float(np.nanmedian(values[finite]))
            else:
                fill = 0.0
            values = values.copy()
            values[~finite] = fill
        band_flux[:, i] = np.clip(np.power(10.0, -0.4 * values), _BB_EPS, None)

    row_sum = band_flux.sum(axis=1, keepdims=True)
    row_sum = np.where(row_sum > _BB_EPS, row_sum, _BB_EPS)
    p = band_flux / row_sum

    row_norm2 = np.einsum("mb,mb->m", p, p)

    z = pd.to_numeric(raw["redshift"], errors="coerce").to_numpy(dtype=np.float64)
    z = np.where(np.isfinite(z), z, 0.0)
    z = np.maximum(z, 0.0)

    bb_logT = np.empty(n_rows, dtype=np.float64)
    bb_ssr = np.empty(n_rows, dtype=np.float64)
    bb_gap = np.empty(n_rows, dtype=np.float64)
    bb_second_margin = np.empty(n_rows, dtype=np.float64)
    bb_resid_u = np.empty(n_rows, dtype=np.float64)
    bb_resid_g = np.empty(n_rows, dtype=np.float64)
    bb_resid_r = np.empty(n_rows, dtype=np.float64)
    bb_resid_i = np.empty(n_rows, dtype=np.float64)
    bb_resid_z = np.empty(n_rows, dtype=np.float64)

    for start in range(0, n_rows, _BB_BLOCK_SIZE):
        stop = min(start + _BB_BLOCK_SIZE, n_rows)
        block_slice = slice(start, stop)

        p_block = p[block_slice]
        z_block = z[block_slice]
        m = stop - start

        lambda_rest = wavelengths[None, :] / (1.0 + z_block[:, None])
        lambda_rest = np.maximum(lambda_rest, _BB_EPS)
        inv_lam5 = np.power(lambda_rest, -5.0)

        denom_exp = np.expm1(_BB_PLANCK_CONSTANT / (lambda_rest[:, None, :] * temps[None, :, None]))
        denom_exp = np.where(denom_exp <= _BB_EPS, _BB_EPS, denom_exp)
        templates = inv_lam5[:, None, :] / denom_exp

        num = np.einsum("mb,mtb->mt", p_block, templates)
        den = np.einsum("mtb,mtb->mt", templates, templates)
        den = np.where(den <= _BB_EPS, _BB_EPS, den)

        scale = num / den
        ssr = row_norm2[block_slice][:, None] - 2.0 * scale * num + (scale * scale) * den

        top2 = np.partition(ssr, 1, axis=1)
        ssr_min = top2[:, 0]
        ssr_second = top2[:, 1]

        best_idx = np.argmin(ssr, axis=1)
        best_t = temps[best_idx]
        best_scale = scale[np.arange(m), best_idx]
        best_template = templates[np.arange(m), best_idx, :]
        residual = p_block - best_scale[:, None] * best_template

        bb_logT[block_slice] = np.log10(best_t)
        bb_ssr[block_slice] = ssr_min
        bb_gap[block_slice] = np.sqrt(ssr_min)
        bb_second_margin[block_slice] = np.log1p(np.maximum(ssr_second - ssr_min, 0.0))

        bb_resid_u[block_slice] = residual[:, 0]
        bb_resid_g[block_slice] = residual[:, 1]
        bb_resid_r[block_slice] = residual[:, 2]
        bb_resid_i[block_slice] = residual[:, 3]
        bb_resid_z[block_slice] = residual[:, 4]

    return pd.DataFrame(
        {
            "bb_logT": bb_logT,
            "bb_ssr": bb_ssr,
            "bb_gap": bb_gap,
            "bb_second_margin": bb_second_margin,
            "bb_resid_u": bb_resid_u,
            "bb_resid_g": bb_resid_g,
            "bb_resid_r": bb_resid_r,
            "bb_resid_i": bb_resid_i,
            "bb_resid_z": bb_resid_z,
        },
        index=index,
    )


FEATURE_GROUPS = [
    {
        "name": "blackbody_continuum_distance",
        "fn": add_blackbody_continuum_distance,
        "depends_on": [],
        "description": "Fits each source’s ugriz shape to a redshift-adjusted blackbody template grid and emits best-fit temperature and mismatch-based shape residual features.",
    }
]