import numpy as np
import pandas as pd

_BIN_WIDTH = 0.08
_MAD_SCALE = 1.4826
_MAD_EPS = 0.01
_RESID_OFFSET = 4.0
_RESID_SCALE = 2.0
_RESID_MAX = 20.0
_UVX_I_CENTER = 20.0
_UVX_I_WIDTH = 1.0
_REDSHIFT_MZ_LOW = 2.3
_REDSHIFT_MZ_HIGH = 3.2
_REDSHIFT_HIZ = 3.0
_REJECTION_SCALE_WD = 0.5
_REJECTION_SCALE_MWD = 1.0
_REJECTION_SCALE_A = 0.4
_NUMERIC_LIMITS = {
    "u": (-200.0, 120.0),
    "g": (-200.0, 120.0),
    "r": (-200.0, 120.0),
    "i": (-200.0, 120.0),
    "z": (-200.0, 120.0),
    "redshift": (-0.2, 10.0),
}
_AUX_REQUIRED = ("u", "g", "r", "i", "z", "redshift")
_DEFAULT_AXIS = (1.0, 0.0, 0.0)


def _extract_float(frame, column, limits):
    n = len(frame) if isinstance(frame, pd.DataFrame) else 0
    if n == 0 or not isinstance(frame, pd.DataFrame) or column not in frame.columns:
        return np.full(n, np.nan, dtype=float)
    arr = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float, copy=True)
    arr[~np.isfinite(arr)] = np.nan
    low, high = limits
    arr[(arr < low) | (arr > high)] = np.nan
    return arr


def _axis_bins(axis, axis_reference, width):
    axis = np.asarray(axis, dtype=float)
    n = axis.shape[0]
    bins = np.full(n, -1, dtype=np.int32)

    if n == 0 or width <= 0 or not np.isfinite(width):
        return bins, np.nan, np.nan

    ref = np.asarray(axis_reference, dtype=float)
    ref_valid = np.isfinite(ref)
    if not ref_valid.any():
        return bins, np.nan, np.nan

    lo, hi = np.nanpercentile(ref[ref_valid], [5.0, 95.0])
    if not np.isfinite(lo) or not np.isfinite(hi) or np.isclose(hi, lo):
        finite = np.isfinite(axis)
        bins[finite] = 0
        return bins, lo, hi

    finite = np.isfinite(axis)
    clamped = np.clip(axis, lo, hi)
    bins[finite] = np.floor((clamped[finite] - lo) / float(width)).astype(np.int32)
    return bins, lo, hi


def _axis_bins_with_bounds(axis, lo, hi, width):
    axis = np.asarray(axis, dtype=float)
    bins = np.full(axis.shape[0], -1, dtype=np.int32)
    if axis.shape[0] == 0 or width <= 0 or not np.isfinite(width):
        return bins
    if not np.isfinite(lo) or not np.isfinite(hi):
        return bins

    finite = np.isfinite(axis)
    if finite.any():
        clamped = np.clip(axis[finite], lo, hi)
        bins[finite] = np.floor((clamped - lo) / float(width)).astype(np.int32)
    return bins


def _ridge_stats(points):
    pts = np.asarray(points, dtype=float)
    valid = np.isfinite(pts).all(axis=1)
    pts = pts[valid]
    if pts.shape[0] == 0:
        return None

    center = np.median(pts, axis=0)
    scale = np.median(np.abs(pts - center), axis=0) * _MAD_SCALE
    scale = np.where((~np.isfinite(scale)) | (scale < _MAD_EPS), _MAD_EPS, scale)

    centered = (pts - center) / scale
    direction = np.array(_DEFAULT_AXIS, dtype=float)

    if centered.shape[0] >= 3:
        try:
            cov = np.dot(centered.T, centered) / max(centered.shape[0] - 1, 1)
            eigvals, eigvecs = np.linalg.eigh(cov)
            if np.isfinite(eigvals).all():
                direction = eigvecs[:, np.argmax(eigvals)]
        except np.linalg.LinAlgError:
            pass

    norm = np.linalg.norm(direction)
    if not np.isfinite(norm) or norm == 0:
        direction = np.array(_DEFAULT_AXIS, dtype=float)
        norm = 1.0
    return center, scale, direction / norm


def _ridge_distance(points, stats):
    pts = np.asarray(points, dtype=float)
    if pts.shape[0] == 0 or stats is None:
        return np.full(pts.shape[0], np.nan, dtype=float)

    center, scale, direction = stats
    valid = np.isfinite(pts).all(axis=1)
    dist = np.full(pts.shape[0], np.nan, dtype=float)
    if not valid.any():
        return dist

    s = (pts[valid] - center) / scale
    proj = np.dot(s, direction)
    perp = s - np.outer(proj, direction)
    dist[valid] = np.sqrt(np.sum(perp * perp, axis=1))
    return dist


def _local_ridge_residual(cube, axis, ref_cube=None, ref_axis=None):
    cube = np.asarray(cube, dtype=float)
    axis = np.asarray(axis, dtype=float)
    n = cube.shape[0]

    residual = np.full(n, np.nan, dtype=float)
    if n == 0 or cube.shape[1] != 3:
        return residual

    valid_raw = np.isfinite(cube).all(axis=1) & np.isfinite(axis)
    if not valid_raw.any():
        return residual

    ref_points = None
    ref_bins_axis = None
    if isinstance(ref_cube, pd.DataFrame):
        ref_cube = ref_cube.to_numpy(dtype=float)
    if isinstance(ref_axis, pd.Series):
        ref_axis = ref_axis.to_numpy(dtype=float)

    if ref_cube is not None and ref_axis is not None:
        ref_cube = np.asarray(ref_cube, dtype=float)
        ref_axis = np.asarray(ref_axis, dtype=float)
        if ref_cube.ndim == 2 and ref_cube.shape[1] == 3 and ref_cube.shape[0] == ref_axis.shape[0]:
            ref_valid = np.isfinite(ref_cube).all(axis=1) & np.isfinite(ref_axis)
            if ref_valid.any():
                ref_points = ref_cube[ref_valid]
                ref_bins_axis = ref_axis[ref_valid]

    axis_reference = axis[valid_raw]
    if ref_points is not None and ref_points.size:
        axis_reference = np.concatenate([axis_reference, ref_bins_axis], axis=0)

    bin_ids, lo, hi = _axis_bins(axis, axis_reference, _BIN_WIDTH)
    if not np.isfinite(lo) or not np.isfinite(hi):
        return residual

    if ref_points is not None and ref_points.size:
        ref_bins = _axis_bins_with_bounds(ref_bins_axis, lo, hi, _BIN_WIDTH)
    else:
        ref_bins = None

    global_points = cube[valid_raw]
    if ref_points is not None and ref_points.size:
        global_points = np.vstack((global_points, ref_points))
    global_stats = _ridge_stats(global_points)
    if global_stats is None:
        global_stats = _ridge_stats(cube[valid_raw])

    for b in np.unique(bin_ids[valid_raw]):
        idx = np.where(valid_raw & (bin_ids == b))[0]
        if idx.size == 0:
            continue

        local_points = cube[idx]
        if ref_points is not None and ref_points.size and ref_bins is not None:
            ref_idx = np.where(ref_bins == b)[0]
            if ref_idx.size > 0:
                local_points = np.vstack((local_points, ref_points[ref_idx]))

        stats = _ridge_stats(local_points)
        if stats is None:
            stats = global_stats
        residual[idx] = _ridge_distance(cube[idx], stats)

    return np.clip(residual, 0.0, _RESID_MAX)


def _signed_margin(values, low, high):
    v = np.asarray(values, dtype=float)
    return np.where(v < low, v - low, np.where(v > high, high - v, np.minimum(v - low, high - v)))


def _negative_inside_penalty(margin, scale):
    margin = np.asarray(margin, dtype=float)
    inside = np.where(np.isfinite(margin) & (margin > 0.0), margin / float(scale), 0.0)
    return -np.clip(inside, 0.0, 1.0)


def add_sdss_quasar_targeting_surface(raw, deps, aux):
    n = len(raw)
    if n == 0:
        return pd.DataFrame(index=raw.index)

    u = _extract_float(raw, "u", _NUMERIC_LIMITS["u"])
    g = _extract_float(raw, "g", _NUMERIC_LIMITS["g"])
    r = _extract_float(raw, "r", _NUMERIC_LIMITS["r"])
    i = _extract_float(raw, "i", _NUMERIC_LIMITS["i"])
    z = _extract_float(raw, "z", _NUMERIC_LIMITS["z"])
    redshift = _extract_float(raw, "redshift", _NUMERIC_LIMITS["redshift"])

    c1 = u - g
    c2 = g - r
    c3 = r - i
    c4 = i - z

    aux_cube_a = None
    aux_axis_a = None
    aux_cube_b = None
    aux_axis_b = None

    if isinstance(aux, pd.DataFrame) and not aux.empty:
        aux_lower = {str(c).lower(): c for c in aux.columns}
        mapped = {}
        required_ok = True
        for col in _AUX_REQUIRED:
            if col in aux_lower:
                mapped[col] = aux_lower[col]
            else:
                required_ok = False
                break

        if required_ok:
            au = _extract_float(aux, mapped["u"], _NUMERIC_LIMITS["u"])
            ag = _extract_float(aux, mapped["g"], _NUMERIC_LIMITS["g"])
            ar = _extract_float(aux, mapped["r"], _NUMERIC_LIMITS["r"])
            ai = _extract_float(aux, mapped["i"], _NUMERIC_LIMITS["i"])
            az = _extract_float(aux, mapped["z"], _NUMERIC_LIMITS["z"])

            if np.isfinite(au).any() and np.isfinite(ag).any() and np.isfinite(ar).any() and np.isfinite(ai).any() and np.isfinite(az).any():
                ac1 = au - ag
                ac2 = ag - ar
                ac3 = ar - ai
                ac4 = ai - az

                aux_cube_a = np.column_stack((ac1, ac2, ac3))
                aux_cube_b = np.column_stack((ac2, ac3, ac4))
                aux_axis_a = ac3
                aux_axis_b = ac2

    cube_a = np.column_stack((c1, c2, c3))
    cube_b = np.column_stack((c2, c3, c4))

    ridge_d_a = _local_ridge_residual(cube_a, c3, aux_cube_a, aux_axis_a)
    ridge_d_b = _local_ridge_residual(cube_b, c2, aux_cube_b, aux_axis_b)

    ridge_s_a = np.clip((ridge_d_a - _RESID_OFFSET) / _RESID_SCALE, 0.0, 1.0)
    ridge_s_b = np.clip((ridge_d_b - _RESID_OFFSET) / _RESID_SCALE, 0.0, 1.0)

    mz_term = ((c1 >= 0.65) & (c1 <= 1.5) & (c2 >= 0.0) & (c2 <= 0.2)).astype(float)
    mz_gate = (redshift >= _REDSHIFT_MZ_LOW) & (redshift <= _REDSHIFT_MZ_HIGH)
    mz_inclusion = mz_term * mz_gate.astype(float)

    uvx_fade = np.where(np.isfinite(i), np.clip((_UVX_I_CENTER - i) / _UVX_I_WIDTH, 0.0, 1.0), 0.0)
    uvx_inclusion = (c1 <= 0.6).astype(float) * uvx_fade

    hiz_term = (redshift > _REDSHIFT_HIZ) & (i < 20.0) & (c1 > 1.5)
    hiz_gate = redshift >= _REDSHIFT_HIZ
    hiz_inclusion = hiz_term.astype(float) * hiz_gate.astype(float)

    blue_reject = ((c1 < 0.9) & (c2 < 0.8) & (i > 19.0)).astype(float)

    wd_margin = np.minimum.reduce([
        _signed_margin(c2, -0.8, -0.2),
        _signed_margin(c3, -0.6, -0.2),
        _signed_margin(c4, -1.0, 0.0),
    ])
    mwd_margin = np.minimum.reduce([
        _signed_margin(c2, 0.0, 1.6),
        _signed_margin(c3, 0.6, 2.0),
    ])
    a_margin = np.minimum.reduce([
        _signed_margin(c1, 0.9, 1.5),
        _signed_margin(c2, -0.35, 0.0),
    ])

    reject_pen = _negative_inside_penalty(wd_margin, _REJECTION_SCALE_WD) \
        + _negative_inside_penalty(mwd_margin, _REJECTION_SCALE_MWD) \
        + _negative_inside_penalty(a_margin, _REJECTION_SCALE_A)

    ridge_core = 0.5 * (ridge_s_a + ridge_s_b)
    targeting_score = (
        ridge_core
        + 0.35 * mz_inclusion
        + 0.30 * hiz_inclusion
        + 0.12 * uvx_inclusion
        - 0.25 * blue_reject
        + 0.20 * reject_pen
    )
    targeting_score = np.clip(targeting_score, -2.0, 2.0)

    return pd.DataFrame(
        {
            "sdss_qso_ridge_dist_a": ridge_d_a,
            "sdss_qso_ridge_dist_b": ridge_d_b,
            "sdss_qso_ridge_score_a": ridge_s_a,
            "sdss_qso_ridge_score_b": ridge_s_b,
            "sdss_qso_ridge_score_gap": ridge_s_a - ridge_s_b,
            "sdss_qso_mz_inclusion": mz_inclusion,
            "sdss_qso_uvx_inclusion": uvx_inclusion,
            "sdss_qso_hiz_inclusion": hiz_inclusion,
            "sdss_qso_blue_reject": blue_reject,
            "sdss_qso_wd_reject_margin": wd_margin,
            "sdss_qso_mwd_reject_margin": mwd_margin,
            "sdss_qso_a_reject_margin": a_margin,
            "sdss_qso_rejection_penalty": reject_pen,
            "sdss_qso_targeting_score": targeting_score,
        },
        index=raw.index,
    )


FEATURE_GROUPS = [
    {
        "name": "sdss_quasar_targeting_surface",
        "fn": add_sdss_quasar_targeting_surface,
        "depends_on": [],
        "description": "Derive SDSS quasar target-surface geometry features from local color-cube ridge distances plus redshift-gated legacy inclusion/exclusion surfaces.",
    }
]