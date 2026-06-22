import numpy as np
import pandas as pd

_GROUP_NAME = "aide_smooth_spline_sed_interactions"
_ALPHA_KNOTS = 7
_OTHER_KNOTS = 5
_BSPLINE_DEGREE = 3
_INTERACTION_BASIS = 2


def _coerce_series(values):
    return pd.to_numeric(values, errors="coerce")


def _bounds_from_series(main_series, auxiliary_series=None):
    combined = main_series if auxiliary_series is None else pd.concat([main_series, auxiliary_series], ignore_index=True)
    arr = combined.to_numpy(dtype=float)
    finite = np.isfinite(arr)
    if not np.any(finite):
        arr = main_series.to_numpy(dtype=float)
        finite = np.isfinite(arr)
    if not np.any(finite):
        return 0.0, 1.0
    lo = float(np.nanmin(arr[finite]))
    hi = float(np.nanmax(arr[finite]))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return 0.0, 1.0
    return lo, hi


def _difference_series(raw_df, aux_df, left_col, right_col):
    left = _coerce_series(raw_df[left_col])
    right = _coerce_series(raw_df[right_col])
    raw_series = left - right
    aux_series = None
    if isinstance(aux_df, pd.DataFrame) and not aux_df.empty and left_col in aux_df.columns and right_col in aux_df.columns:
        left_aux = _coerce_series(aux_df[left_col])
        right_aux = _coerce_series(aux_df[right_col])
        aux_series = left_aux - right_aux
    return raw_series, aux_series


def _manual_bspline_basis(values, lower, upper, n_internal_knots, degree=_BSPLINE_DEGREE):
    x = np.asarray(values, dtype=float)
    n = len(x)
    if not np.isfinite(lower) or not np.isfinite(upper) or upper <= lower or n == 0:
        out = np.full((n, 1), np.nan, dtype=np.float32)
        finite = np.isfinite(x)
        out[finite, 0] = 1.0
        return out

    n_internal = max(2, int(n_internal_knots))
    internal = np.linspace(lower, upper, n_internal + 2)[1:-1]
    knots = np.r_[np.repeat(lower, degree + 1), internal, np.repeat(upper, degree + 1)]
    n_basis = len(knots) - degree - 1
    out = np.full((n, n_basis), np.nan, dtype=np.float32)

    finite = np.isfinite(x)
    if not np.any(finite):
        return out

    xv = x.copy()
    xv[~finite] = lower
    xv = np.clip(xv, lower, upper)
    xi = xv[finite]
    m = len(knots)
    N = np.zeros((len(xi), m - 1), dtype=np.float64)

    left = xi[:, None] >= knots[:-1][None, :]
    right = xi[:, None] < knots[1:][None, :]
    N[:, : m - 1] = (left & right).astype(np.float64)
    N[xi == upper, n_basis - 1] = 1.0

    for d in range(1, degree + 1):
        n_cols = m - d - 1
        N_next = np.zeros((len(xi), n_cols), dtype=np.float64)
        for i in range(n_cols):
            d1 = knots[i + d] - knots[i]
            d2 = knots[i + d + 1] - knots[i + 1]
            left_term = 0.0
            right_term = 0.0
            if d1 != 0.0:
                left_term = ((xi - knots[i]) / d1) * N[:, i]
            if d2 != 0.0:
                right_term = ((knots[i + d + 1] - xi) / d2) * N[:, i + 1]
            N_next[:, i] = left_term + right_term
        N = N_next

    finite_idx = np.flatnonzero(finite)
    out[finite_idx] = N.astype(np.float32)
    return out


def _nonperiodic_cubic_basis(values, lower, upper, n_knots):
    arr = _coerce_series(values).to_numpy(dtype=float)
    n = len(arr)
    n_knots = int(max(2, n_knots))
    if n == 0:
        return np.empty((0, 0), dtype=np.float32)
    if not np.isfinite(lower) or not np.isfinite(upper) or upper <= lower:
        out = np.full((n, 1), np.nan, dtype=np.float32)
        finite = np.isfinite(arr)
        out[finite, 0] = 1.0
        return out

    internal = np.linspace(lower, upper, n_knots + 2)[1:-1]
    knots = np.r_[np.repeat(lower, _BSPLINE_DEGREE + 1), internal, np.repeat(upper, _BSPLINE_DEGREE + 1)]
    n_basis = len(knots) - _BSPLINE_DEGREE - 1
    out = np.full((n, n_basis), np.nan, dtype=np.float32)

    finite = np.isfinite(arr)
    if not np.any(finite):
        return out

    x = np.clip(arr[finite], lower, upper)
    try:
        from scipy.interpolate import BSpline

        coeff = np.eye(n_basis)
        spline = BSpline(knots, coeff, _BSPLINE_DEGREE, axis=0, extrapolate=False)
        basis = spline(x).astype(np.float64)
    except Exception:
        basis = _manual_bspline_basis(arr[finite], lower, upper, n_knots, degree=_BSPLINE_DEGREE)

    if basis.ndim == 1:
        basis = basis[:, None]
    finite_idx = np.flatnonzero(finite)
    out[finite_idx] = basis.astype(np.float32)
    return out


def _periodic_cubic_basis(values, lower, upper, n_knots):
    arr = _coerce_series(values).to_numpy(dtype=float)
    n = len(arr)
    n_knots = int(max(3, n_knots))
    if n == 0:
        return np.empty((0, 0), dtype=np.float32)
    if not np.isfinite(lower) or not np.isfinite(upper) or upper <= lower:
        out = np.full((n, 1), np.nan, dtype=np.float32)
        finite = np.isfinite(arr)
        out[finite, 0] = 1.0
        return out

    out = np.full((n, n_knots), np.nan, dtype=np.float32)
    finite = np.isfinite(arr)
    if not np.any(finite):
        return out

    period = upper - lower
    if not np.isfinite(period) or period <= 0.0:
        fallback = _manual_bspline_basis(arr[finite], lower, upper, n_knots, degree=_BSPLINE_DEGREE)
        fallback = np.asarray(fallback, dtype=np.float32)
        if fallback.shape[1] >= n_knots:
            fallback = fallback[:, :n_knots]
        else:
            pad = np.full((fallback.shape[0], n_knots - fallback.shape[1]), np.nan, dtype=np.float32)
            fallback = np.concatenate([fallback, pad], axis=1)
        out[np.flatnonzero(finite)] = fallback
        return out

    xw = ((arr[finite] - lower) % period) + lower
    try:
        from scipy.interpolate import CubicSpline

        nodes = np.linspace(lower, upper, n_knots + 1)
        y = np.zeros((len(nodes), n_knots), dtype=float)
        for i in range(n_knots):
            y[i, i] = 1.0
            if i == 0:
                y[-1, i] = 1.0
        spline = CubicSpline(nodes, y, bc_type="periodic", axis=0)
        basis = spline(xw).astype(np.float32)
    except Exception:
        basis = _manual_bspline_basis(xw, lower, upper, n_knots, degree=_BSPLINE_DEGREE)
        basis = np.asarray(basis, dtype=np.float32)
        if basis.shape[1] >= n_knots:
            basis = basis[:, :n_knots]
        else:
            pad = np.full((basis.shape[0], n_knots - basis.shape[1]), np.nan, dtype=np.float32)
            basis = np.concatenate([basis, pad], axis=1)

    out[np.flatnonzero(finite)] = basis
    return out


def _add_basis_columns(store, basis_matrix, prefix):
    for i in range(basis_matrix.shape[1]):
        store[f"{prefix}_spline_b{i + 1:02d}"] = basis_matrix[:, i]


def _add_interaction_columns(store, left_matrix, right_matrix, name, max_terms=_INTERACTION_BASIS):
    if left_matrix.size == 0 or right_matrix.size == 0:
        return
    left_terms = left_matrix[:, : min(max_terms, left_matrix.shape[1])]
    right_terms = right_matrix[:, : min(max_terms, right_matrix.shape[1])]
    for i in range(left_terms.shape[1]):
        for j in range(right_terms.shape[1]):
            store[f"{name}_b{i + 1:02d}_x{j + 1:02d}"] = left_terms[:, i] * right_terms[:, j]


def add_aide_smooth_spline_sed_interactions(raw, deps, aux):
    _ = deps
    aux_df = aux if isinstance(aux, pd.DataFrame) else pd.DataFrame()

    alpha = _coerce_series(raw["alpha"])
    delta = _coerce_series(raw["delta"])
    u = _coerce_series(raw["u"])
    g = _coerce_series(raw["g"])
    r = _coerce_series(raw["r"])
    i_band = _coerce_series(raw["i"])
    z_band = _coerce_series(raw["z"])
    redshift = _coerce_series(raw["redshift"])

    alpha_aux = _coerce_series(aux_df["alpha"]) if not aux_df.empty and "alpha" in aux_df.columns else None
    delta_aux = _coerce_series(aux_df["delta"]) if not aux_df.empty and "delta" in aux_df.columns else None
    u_aux = _coerce_series(aux_df["u"]) if not aux_df.empty and "u" in aux_df.columns else None
    g_aux = _coerce_series(aux_df["g"]) if not aux_df.empty and "g" in aux_df.columns else None
    r_aux = _coerce_series(aux_df["r"]) if not aux_df.empty and "r" in aux_df.columns else None
    i_aux = _coerce_series(aux_df["i"]) if not aux_df.empty and "i" in aux_df.columns else None
    z_aux = _coerce_series(aux_df["z"]) if not aux_df.empty and "z" in aux_df.columns else None
    redshift_aux = _coerce_series(aux_df["redshift"]) if not aux_df.empty and "redshift" in aux_df.columns else None

    u_g, u_g_aux = _difference_series(raw, aux_df, "u", "g")
    g_r, g_r_aux = _difference_series(raw, aux_df, "g", "r")
    r_i, r_i_aux = _difference_series(raw, aux_df, "r", "i")
    i_z, i_z_aux = _difference_series(raw, aux_df, "i", "z")
    u_r, u_r_aux = _difference_series(raw, aux_df, "u", "r")
    g_i, g_i_aux = _difference_series(raw, aux_df, "g", "i")
    r_z, r_z_aux = _difference_series(raw, aux_df, "r", "z")
    u_z, u_z_aux = _difference_series(raw, aux_df, "u", "z")

    redshift_hi = _bounds_from_series(redshift, redshift_aux)[1]
    if not np.isfinite(redshift_hi):
        redshift_hi = float(redshift.max(skipna=True))
    if not np.isfinite(redshift_hi):
        redshift_hi = 0.0
    clipped_redshift = redshift.clip(lower=0.0, upper=redshift_hi)
    clipped_redshift_aux = redshift_aux.clip(lower=0.0, upper=redshift_hi) if redshift_aux is not None else None

    feature_blocks = {}

    alpha_bounds = _bounds_from_series(alpha, alpha_aux)
    feature_blocks["alpha"] = _periodic_cubic_basis(alpha, alpha_bounds[0], alpha_bounds[1], _ALPHA_KNOTS)

    delta_bounds = _bounds_from_series(delta, delta_aux)
    feature_blocks["delta"] = _nonperiodic_cubic_basis(delta, delta_bounds[0], delta_bounds[1], _OTHER_KNOTS)

    u_bounds = _bounds_from_series(u, u_aux)
    feature_blocks["u"] = _nonperiodic_cubic_basis(u, u_bounds[0], u_bounds[1], _OTHER_KNOTS)

    g_bounds = _bounds_from_series(g, g_aux)
    feature_blocks["g"] = _nonperiodic_cubic_basis(g, g_bounds[0], g_bounds[1], _OTHER_KNOTS)

    r_bounds = _bounds_from_series(r, r_aux)
    feature_blocks["r"] = _nonperiodic_cubic_basis(r, r_bounds[0], r_bounds[1], _OTHER_KNOTS)

    i_bounds = _bounds_from_series(i_band, i_aux)
    feature_blocks["i"] = _nonperiodic_cubic_basis(i_band, i_bounds[0], i_bounds[1], _OTHER_KNOTS)

    z_bounds = _bounds_from_series(z_band, z_aux)
    feature_blocks["z"] = _nonperiodic_cubic_basis(z_band, z_bounds[0], z_bounds[1], _OTHER_KNOTS)

    clipped_z_bounds = _bounds_from_series(clipped_redshift, clipped_redshift_aux)
    feature_blocks["clipped_redshift"] = _nonperiodic_cubic_basis(
        clipped_redshift, clipped_z_bounds[0], clipped_z_bounds[1], _OTHER_KNOTS
    )

    u_g_bounds = _bounds_from_series(u_g, u_g_aux)
    feature_blocks["u_g"] = _nonperiodic_cubic_basis(u_g, u_g_bounds[0], u_g_bounds[1], _OTHER_KNOTS)

    g_r_bounds = _bounds_from_series(g_r, g_r_aux)
    feature_blocks["g_r"] = _nonperiodic_cubic_basis(g_r, g_r_bounds[0], g_r_bounds[1], _OTHER_KNOTS)

    r_i_bounds = _bounds_from_series(r_i, r_i_aux)
    feature_blocks["r_i"] = _nonperiodic_cubic_basis(r_i, r_i_bounds[0], r_i_bounds[1], _OTHER_KNOTS)

    i_z_bounds = _bounds_from_series(i_z, i_z_aux)
    feature_blocks["i_z"] = _nonperiodic_cubic_basis(i_z, i_z_bounds[0], i_z_bounds[1], _OTHER_KNOTS)

    u_r_bounds = _bounds_from_series(u_r, u_r_aux)
    feature_blocks["u_r"] = _nonperiodic_cubic_basis(u_r, u_r_bounds[0], u_r_bounds[1], _OTHER_KNOTS)

    g_i_bounds = _bounds_from_series(g_i, g_i_aux)
    feature_blocks["g_i"] = _nonperiodic_cubic_basis(g_i, g_i_bounds[0], g_i_bounds[1], _OTHER_KNOTS)

    r_z_bounds = _bounds_from_series(r_z, r_z_aux)
    feature_blocks["r_z"] = _nonperiodic_cubic_basis(r_z, r_z_bounds[0], r_z_bounds[1], _OTHER_KNOTS)

    u_z_bounds = _bounds_from_series(u_z, u_z_aux)
    feature_blocks["u_z"] = _nonperiodic_cubic_basis(u_z, u_z_bounds[0], u_z_bounds[1], _OTHER_KNOTS)

    out = {}
    for name, matrix in feature_blocks.items():
        _add_basis_columns(out, matrix, name)

    _add_interaction_columns(out, feature_blocks["clipped_redshift"], feature_blocks["u_g"], "redshift_x_u_g")
    _add_interaction_columns(out, feature_blocks["clipped_redshift"], feature_blocks["g_r"], "redshift_x_g_r")
    _add_interaction_columns(out, feature_blocks["clipped_redshift"], feature_blocks["r_i"], "redshift_x_r_i")
    _add_interaction_columns(out, feature_blocks["g_r"], feature_blocks["r_i"], "g_r_x_r_i")

    return pd.DataFrame(out, index=raw.index)


FEATURE_GROUPS = [
    {
        "name": _GROUP_NAME,
        "fn": add_aide_smooth_spline_sed_interactions,
        "depends_on": [],
        "description": "Add cubic spline expansions for sky coordinates, photometry, clipped redshift, and color indices with compact interactions over redshift-color and adjacent color terms.",
    }
]