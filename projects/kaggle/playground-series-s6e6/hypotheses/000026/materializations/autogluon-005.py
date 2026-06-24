import numpy as np
import pandas as pd

from scipy.interpolate import CubicSpline

_GROUP_NAME = "aide_smooth_spline_sed_interactions"

_SPLINE_DEGREE = 3
_NONCYCLIC_BASIS_COUNT = 5
_NONCYCLIC_KEEP = 4
_ALPHA_BASIS_COUNT = 7
_ALPHA_KEEP = 6
_ALPHA_PERIOD = 360.0

_BASE_NUMERIC_FEATURES = ("alpha", "delta", "u", "g", "r", "i", "z", "redshift")
_COLOR_DEFINITIONS = (
    ("c1", "u", "g"),
    ("c2", "g", "r"),
    ("c3", "r", "i"),
    ("c4", "i", "z"),
    ("c5", "u", "r"),
    ("c6", "g", "i"),
    ("c7", "r", "z"),
    ("c8", "u", "z"),
)
_INTERACTION_PAIRS = (("redshift", "c1"), ("redshift", "c2"), ("redshift", "c3"), ("c2", "c3"))

_SPLINE_GROUP_STATE = {}


def _safe_numeric_array(values, median_fill):
    arr = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    if arr.size == 0:
        return arr
    arr = np.where(np.isfinite(arr), arr, median_fill)
    return arr


def _finite_median(arr):
    if arr.size == 0:
        return 0.0
    med = float(np.nanmedian(arr))
    if not np.isfinite(med):
        med = 0.0
    return med


def _safe_quantiles(arr):
    if arr.size == 0:
        return 0.0, 1.0
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return 0.0, 1.0
    q01 = float(np.nanquantile(finite, 0.01))
    q99 = float(np.nanquantile(finite, 0.99))
    if not (np.isfinite(q01) and np.isfinite(q99)) or q99 <= q01:
        q01 = float(np.min(finite))
        q99 = float(np.max(finite))
        if q99 <= q01:
            q99 = q01 + 1.0
    return q01, q99


def _uniform_open_knots(vmin, vmax, n_basis, degree):
    n_internal = int(n_basis - degree - 1)
    if n_internal < 1:
        n_internal = 1
    if not (np.isfinite(vmin) and np.isfinite(vmax)):
        vmin, vmax = 0.0, 1.0
    if vmax <= vmin:
        vmax = vmin + 1.0
    if n_internal > 0:
        internal = np.linspace(vmin, vmax, n_internal + 2)[1:-1]
        if internal.size == 0:
            internal = np.array([], dtype=float)
    else:
        internal = np.array([], dtype=float)
    return np.concatenate((np.full(degree + 1, float(vmin), dtype=float), internal, np.full(degree + 1, float(vmax), dtype=float)))


def _bspline_basis_matrix(x, knots, degree):
    t = np.asarray(knots, dtype=float)
    x = np.asarray(x, dtype=float)
    n_samples = x.shape[0]
    n_intervals = len(t) - 1
    if n_samples == 0:
        return np.empty((0, max(n_intervals - degree - 1, 0)), dtype=float)

    basis = np.zeros((n_samples, n_intervals), dtype=float)
    for i in range(n_intervals):
        left = t[i]
        right = t[i + 1]
        if i < n_intervals - 1:
            mask = (x >= left) & (x < right)
        else:
            mask = np.isclose(x, right)
        basis[:, i] = mask.astype(float)

    for d in range(1, degree + 1):
        n_next = basis.shape[1] - 1
        nxt = np.zeros((n_samples, n_next), dtype=float)
        for i in range(n_next):
            left_denom = t[i + d] - t[i]
            if left_denom != 0.0:
                nxt[:, i] += (x - t[i]) / left_denom * basis[:, i]
            right_denom = t[i + d + 1] - t[i + 1]
            if right_denom != 0.0:
                nxt[:, i] += (t[i + d + 1] - x) / right_denom * basis[:, i + 1]
        basis = nxt
    return basis


def _build_group_state(raw):
    state = {"features": {}}

    # Base numeric columns
    for feature in _BASE_NUMERIC_FEATURES:
        if feature == "alpha":
            raw_arr = _safe_numeric_array(raw[feature], 0.0) % _ALPHA_PERIOD
        elif feature == "redshift":
            raw_arr = _safe_numeric_array(raw[feature], 0.0)
            med = _finite_median(raw_arr)
            raw_arr = np.where(np.isfinite(raw_arr), raw_arr, med)
            lower = np.nanmax([0.0, float(np.nanquantile(raw_arr, 0.01))])
            raw_arr = np.maximum(raw_arr, lower)
            q01, q99 = _safe_quantiles(raw_arr)
            knots = _uniform_open_knots(q01, q99, _NONCYCLIC_BASIS_COUNT, _SPLINE_DEGREE)
            state["features"][feature] = {
                "median": med,
                "q01": q01,
                "q99": q99,
                "knots": tuple(float(v) for v in knots),
                "n_bases": _NONCYCLIC_BASIS_COUNT,
                "keep": _NONCYCLIC_KEEP,
                "redshift_floor": lower,
            }
            continue
        else:
            raw_arr = _safe_numeric_array(raw[feature], 0.0)

        med = _finite_median(raw_arr)
        raw_arr = np.where(np.isfinite(raw_arr), raw_arr, med)
        q01, q99 = _safe_quantiles(raw_arr)
        knots = _uniform_open_knots(q01, q99, _NONCYCLIC_BASIS_COUNT, _SPLINE_DEGREE)
        state["features"][feature] = {
            "median": med,
            "q01": q01,
            "q99": q99,
            "knots": tuple(float(v) for v in knots),
            "n_bases": _NONCYCLIC_BASIS_COUNT,
            "keep": _NONCYCLIC_KEEP,
            "redshift_floor": None,
        }

    # Colors from magnitudes
    u = _safe_numeric_array(raw["u"], state["features"]["u"]["median"])
    g = _safe_numeric_array(raw["g"], state["features"]["g"]["median"])
    r = _safe_numeric_array(raw["r"], state["features"]["r"]["median"])
    i = _safe_numeric_array(raw["i"], state["features"]["i"]["median"])
    z = _safe_numeric_array(raw["z"], state["features"]["z"]["median"])

    color_values = {
        "c1": u - g,
        "c2": g - r,
        "c3": r - i,
        "c4": i - z,
        "c5": u - r,
        "c6": g - i,
        "c7": r - z,
        "c8": u - z,
    }

    for cname, cvals in color_values.items():
        med = _finite_median(cvals)
        cvals = np.where(np.isfinite(cvals), cvals, med)
        q01, q99 = _safe_quantiles(cvals)
        knots = _uniform_open_knots(q01, q99, _NONCYCLIC_BASIS_COUNT, _SPLINE_DEGREE)
        state["features"][cname] = {
            "median": med,
            "q01": q01,
            "q99": q99,
            "knots": tuple(float(v) for v in knots),
            "n_bases": _NONCYCLIC_BASIS_COUNT,
            "keep": _NONCYCLIC_KEEP,
            "redshift_floor": None,
        }

    # Store cyclic alpha-specific dictionary
    alpha_q01 = state["features"]["alpha"]["q01"]
    alpha_q99 = state["features"]["alpha"]["q99"]
    state["features"]["alpha"]["cyclic_nodes"] = tuple(
        float(v) for v in np.linspace(0.0, _ALPHA_PERIOD, _ALPHA_BASIS_COUNT + 1)
    )
    template = np.zeros((_ALPHA_BASIS_COUNT + 1, _ALPHA_BASIS_COUNT), dtype=float)
    eye = np.eye(_ALPHA_BASIS_COUNT, dtype=float)
    template[:-1, :] = eye
    template[-1, :] = eye[0, :]
    state["features"]["alpha"]["cyclic_template"] = tuple(tuple(float(v) for v in row) for row in template)
    state["features"]["alpha"]["q01"] = alpha_q01
    state["features"]["alpha"]["q99"] = alpha_q99
    state["features"]["alpha"]["keep"] = _ALPHA_KEEP

    return state


def _noncyclic_spline_block(values, stats):
    arr = np.asarray(values, dtype=float)
    arr = np.where(np.isfinite(arr), arr, stats["median"])
    if stats.get("redshift_floor") is not None:
        arr = np.maximum(arr, float(stats["redshift_floor"]))
    arr = np.clip(arr, float(stats["q01"]), float(stats["q99"]))
    knots = np.asarray(stats["knots"], dtype=float)
    basis = _bspline_basis_matrix(arr, knots, _SPLINE_DEGREE)
    basis = np.clip(basis[:, : int(stats["keep"])], 0.0, None)
    return basis


def _cyclic_alpha_spline_block(values, stats):
    arr = np.asarray(values, dtype=float)
    arr = np.where(np.isfinite(arr), arr, stats["median"]) % _ALPHA_PERIOD
    arr = np.clip(arr, float(stats["q01"]), float(stats["q99"]))
    nodes = np.asarray(stats["cyclic_nodes"], dtype=float)
    template = np.asarray(stats["cyclic_template"], dtype=float)

    # Periodic cubic spline basis with cyclic continuity at 0/360.
    basis = np.empty((arr.shape[0], _ALPHA_BASIS_COUNT), dtype=float)
    for i in range(_ALPHA_BASIS_COUNT):
        spl = CubicSpline(nodes, template[:, i], bc_type="periodic")
        basis[:, i] = spl(arr)
    return np.clip(basis[:, : int(stats["keep"])], 0.0, None)


def add_aide_smooth_spline_sed_interactions(raw, deps, aux):
    _ = deps
    _ = aux
    state = _SPLINE_GROUP_STATE.get("state")
    if state is None:
        state = _build_group_state(raw)
        _SPLINE_GROUP_STATE["state"] = state

    stats = state["features"]

    # Deterministic numeric pre-quantities
    alpha = _safe_numeric_array(raw["alpha"], stats["alpha"]["median"])
    delta = _safe_numeric_array(raw["delta"], stats["delta"]["median"])
    u = _safe_numeric_array(raw["u"], stats["u"]["median"])
    g = _safe_numeric_array(raw["g"], stats["g"]["median"])
    r = _safe_numeric_array(raw["r"], stats["r"]["median"])
    i = _safe_numeric_array(raw["i"], stats["i"]["median"])
    z = _safe_numeric_array(raw["z"], stats["z"]["median"])
    redshift = _safe_numeric_array(raw["redshift"], stats["redshift"]["median"])

    c1 = u - g
    c2 = g - r
    c3 = r - i
    c4 = i - z
    c5 = u - r
    c6 = g - i
    c7 = r - z
    c8 = u - z

    feature_blocks = {}
    feature_blocks["alpha"] = _cyclic_alpha_spline_block(alpha, stats["alpha"])
    feature_blocks["delta"] = _noncyclic_spline_block(delta, stats["delta"])
    feature_blocks["u"] = _noncyclic_spline_block(u, stats["u"])
    feature_blocks["g"] = _noncyclic_spline_block(g, stats["g"])
    feature_blocks["r"] = _noncyclic_spline_block(r, stats["r"])
    feature_blocks["i"] = _noncyclic_spline_block(i, stats["i"])
    feature_blocks["z"] = _noncyclic_spline_block(z, stats["z"])
    feature_blocks["redshift"] = _noncyclic_spline_block(redshift, stats["redshift"])

    feature_blocks["c1"] = _noncyclic_spline_block(c1, stats["c1"])
    feature_blocks["c2"] = _noncyclic_spline_block(c2, stats["c2"])
    feature_blocks["c3"] = _noncyclic_spline_block(c3, stats["c3"])
    feature_blocks["c4"] = _noncyclic_spline_block(c4, stats["c4"])
    feature_blocks["c5"] = _noncyclic_spline_block(c5, stats["c5"])
    feature_blocks["c6"] = _noncyclic_spline_block(c6, stats["c6"])
    feature_blocks["c7"] = _noncyclic_spline_block(c7, stats["c7"])
    feature_blocks["c8"] = _noncyclic_spline_block(c8, stats["c8"])

    new_features = {}
    for name, matrix in feature_blocks.items():
        if name == "alpha":
            keep = _ALPHA_KEEP
            prefix = "aide_alpha_cyc"
        else:
            keep = _NONCYCLIC_KEEP
            prefix = f"aide_{name}"
        for i in range(keep):
            new_features[f"{prefix}_spline_{i}"] = matrix[:, i]

    for left, right in _INTERACTION_PAIRS:
        left_mat = feature_blocks[left][:, :2]
        right_mat = feature_blocks[right][:, :2]
        for i in range(2):
            for j in range(2):
                new_features[f"tensor_{left}_{right}_b{i}_b{j}"] = left_mat[:, i] * right_mat[:, j]

    return pd.DataFrame(new_features, index=raw.index)


FEATURE_GROUPS = [
    {
        "name": _GROUP_NAME,
        "fn": add_aide_smooth_spline_sed_interactions,
        "depends_on": [],
        "description": "Builds smooth cubic-spline basis expansions for numeric and color variables plus compact spline interaction terms.",
    }
]