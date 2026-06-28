import numpy as np
import pandas as pd


TRAIN_ID_MAX = 577346
GROUP_NAME = "aide_smooth_spline_sed_interactions"
MAG_COLOR_QUANTILES = (0.001, 0.999)
DELTA_QUANTILES = (0.001, 0.999)
REDSHIFT_QUANTILES = (0.001, 0.999)
NONCYCLIC_DF = 5
CYCLIC_ALPHA_DF = 7
SPLINE_DEGREE = 3
NUMERIC_COLUMNS = ("delta", "u", "g", "r", "i", "z", "redshift")
COLOR_PAIRS = (
    ("c_ug", "u", "g"),
    ("c_gr", "g", "r"),
    ("c_ri", "r", "i"),
    ("c_iz", "i", "z"),
    ("c_ur", "u", "r"),
    ("c_gi", "g", "i"),
    ("c_rz", "r", "z"),
    ("c_uz", "u", "z"),
)
INTERACTION_PAIRS = (
    ("redshift", "c_ug"),
    ("redshift", "c_gr"),
    ("redshift", "c_ri"),
    ("c_gr", "c_ri"),
)


def _training_mask(raw):
    if "id" not in raw.columns:
        return pd.Series(True, index=raw.index)
    ids = pd.to_numeric(raw["id"], errors="coerce")
    mask = ids.le(TRAIN_ID_MAX)
    if bool(mask.any()):
        return mask.fillna(False)
    return pd.Series(True, index=raw.index)


def _as_float_series(raw, name):
    return pd.to_numeric(raw[name], errors="coerce").astype("float64")


def _safe_quantile(values, q, fallback):
    clean = pd.Series(values).replace([np.inf, -np.inf], np.nan).dropna()
    if clean.empty:
        return float(fallback)
    return float(clean.quantile(q))


def _median(values, fallback):
    clean = pd.Series(values).replace([np.inf, -np.inf], np.nan).dropna()
    if clean.empty:
        return float(fallback)
    return float(clean.median())


def _bounds(values, lower_q, upper_q, fallback_lower, fallback_upper):
    lower = _safe_quantile(values, lower_q, fallback_lower)
    upper = _safe_quantile(values, upper_q, fallback_upper)
    if not np.isfinite(lower):
        lower = float(fallback_lower)
    if not np.isfinite(upper):
        upper = float(fallback_upper)
    if upper <= lower:
        upper = lower + 1.0
    return float(lower), float(upper)


def _prepare_clipped(values, train_values, lower_q, upper_q, fallback_lower, fallback_upper, redshift=False):
    median = _median(train_values, 0.0)
    lower, upper = _bounds(train_values, lower_q, upper_q, fallback_lower, fallback_upper)
    if redshift:
        lower = max(0.0, lower)
    arr = pd.Series(values, index=values.index).replace([np.inf, -np.inf], np.nan).fillna(median).to_numpy(dtype="float64")
    return np.clip(arr, lower, upper), lower, upper


def _make_open_knot_vector(train_values, lower, upper, df, degree):
    n_inner = max(0, df - degree - 1)
    knots = [float(lower)] * (degree + 1)
    if n_inner:
        qs = np.linspace(0.0, 1.0, n_inner + 2)[1:-1]
        inner = np.quantile(np.asarray(train_values, dtype="float64"), qs)
        for val in inner:
            val = float(np.clip(val, lower, upper))
            if lower < val < upper:
                knots.append(val)
    while len(knots) < df:
        frac = (len(knots) - degree) / max(1, df - degree)
        knots.append(float(lower + frac * (upper - lower)))
    knots.extend([float(upper)] * (degree + 1))
    return np.asarray(knots, dtype="float64")


def _bspline_basis(x, knots, degree):
    x = np.asarray(x, dtype="float64")
    n_basis = len(knots) - degree - 1
    basis = np.zeros((x.shape[0], n_basis), dtype="float64")

    for i in range(n_basis):
        basis[:, i] = ((x >= knots[i]) & (x < knots[i + 1])).astype("float64")
    basis[x == knots[-1], -1] = 1.0

    for d in range(1, degree + 1):
        next_basis = np.zeros((x.shape[0], n_basis), dtype="float64")
        for i in range(n_basis):
            left_den = knots[i + d] - knots[i]
            right_den = knots[i + d + 1] - knots[i + 1]

            if left_den > 0:
                next_basis[:, i] += ((x - knots[i]) / left_den) * basis[:, i]
            if right_den > 0 and i + 1 < n_basis:
                next_basis[:, i] += ((knots[i + d + 1] - x) / right_den) * basis[:, i + 1]
        basis = next_basis

    return basis


def _retain_centered_columns(full_basis, train_mask, prefix, start=0, max_cols=None):
    kept = {}
    train_basis = full_basis[np.asarray(train_mask, dtype=bool)]
    if train_basis.shape[0] == 0:
        train_basis = full_basis

    emitted = 0
    for j in range(start, full_basis.shape[1]):
        col_train = train_basis[:, j]
        if np.nanstd(col_train) <= 1e-12:
            continue
        mean = float(np.nanmean(col_train))
        kept[f"{prefix}_spline_{emitted}"] = full_basis[:, j] - mean
        emitted += 1
        if max_cols is not None and emitted >= max_cols:
            break
    return kept


def _cyclic_alpha_basis(alpha_values):
    alpha = np.mod(np.asarray(alpha_values, dtype="float64"), 360.0)
    theta = (2.0 * np.pi * alpha) / 360.0
    cols = []
    for k in range(1, 4):
        cols.append(np.sin(k * theta))
        cols.append(np.cos(k * theta))
    cols.append(np.sin(4 * theta))
    return np.column_stack(cols)


def add_aide_smooth_spline_sed_interactions(raw, deps, aux):
    train_mask = _training_mask(raw)
    train_mask_arr = train_mask.to_numpy(dtype=bool)
    features = {}

    alpha = _as_float_series(raw, "alpha").fillna(0.0).to_numpy(dtype="float64")
    alpha_basis = _cyclic_alpha_basis(alpha)
    features.update(_retain_centered_columns(alpha_basis, train_mask_arr, "alpha_cyclic", start=0, max_cols=CYCLIC_ALPHA_DF))

    base = {}
    for name in NUMERIC_COLUMNS:
        base[name] = _as_float_series(raw, name)

    derived = {}
    for color_name, left, right in COLOR_PAIRS:
        derived[color_name] = base[left] - base[right]

    all_variables = {}
    all_variables.update(base)
    all_variables.update(derived)

    retained_for_interactions = {}

    for name in ("delta", "u", "g", "r", "i", "z", "redshift", "c_ug", "c_gr", "c_ri", "c_iz", "c_ur", "c_gi", "c_rz", "c_uz"):
        values = all_variables[name]
        train_values = values.loc[train_mask]

        if name == "redshift":
            clipped, lower, upper = _prepare_clipped(
                values,
                train_values,
                REDSHIFT_QUANTILES[0],
                REDSHIFT_QUANTILES[1],
                0.0,
                7.01,
                redshift=True,
            )
        elif name == "delta":
            clipped, lower, upper = _prepare_clipped(
                values,
                train_values,
                DELTA_QUANTILES[0],
                DELTA_QUANTILES[1],
                -18.0,
                80.0,
                redshift=False,
            )
        else:
            clipped, lower, upper = _prepare_clipped(
                values,
                train_values,
                MAG_COLOR_QUANTILES[0],
                MAG_COLOR_QUANTILES[1],
                float(np.nanmin(values.to_numpy(dtype="float64"))) if values.notna().any() else -1.0,
                float(np.nanmax(values.to_numpy(dtype="float64"))) if values.notna().any() else 1.0,
                redshift=False,
            )

        train_clipped = clipped[train_mask_arr]
        if train_clipped.shape[0] == 0:
            train_clipped = clipped
        knots = _make_open_knot_vector(train_clipped, lower, upper, NONCYCLIC_DF, SPLINE_DEGREE)
        basis = _bspline_basis(clipped, knots, SPLINE_DEGREE)
        retained = _retain_centered_columns(basis, train_mask_arr, name, start=0, max_cols=None)
        features.update(retained)

        first_two = []
        for col_name in retained:
            first_two.append(retained[col_name])
            if len(first_two) == 2:
                break
        retained_for_interactions[name] = first_two

    for left, right in INTERACTION_PAIRS:
        left_cols = retained_for_interactions.get(left, [])[:2]
        right_cols = retained_for_interactions.get(right, [])[:2]
        for i, left_col in enumerate(left_cols):
            for j, right_col in enumerate(right_cols):
                values = left_col * right_col
                train_values = values[train_mask_arr]
                center = float(np.nanmean(train_values)) if train_values.size else float(np.nanmean(values))
                features[f"{left}_x_{right}_tensor_{i}_{j}"] = values - center

    return pd.DataFrame(features, index=raw.index)


FEATURE_GROUPS = [
    {
        "name": "aide_smooth_spline_sed_interactions",
        "fn": add_aide_smooth_spline_sed_interactions,
        "depends_on": [],
        "description": "Boundary-safe smooth spline bases for sky position, photometric SED shape, redshift, and compact color-redshift interactions.",
    }
]