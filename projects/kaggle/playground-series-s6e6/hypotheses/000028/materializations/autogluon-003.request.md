Fix one failed hypothesis materialization.

Return a corrected version of the same feature-group module. Do not invent a
new hypothesis or change the feature family. Keep the fix narrow: address the
observed failure while preserving the intended preprocessing behavior.

You are repairing Python preprocessing code for one stored hypothesis. The
existing file is a feature-group module imported by a fixed runtime wrapper.
Generate only the corrected module.

# Data Overview

-> test.csv has 247435 rows and 11 columns.
Here is some information about the columns:
id (int64) has range: 577347.00 - 824781.00, 0 nan values
alpha (float64) has range: 0.01 - 360.00, 0 nan values
delta (float64) has range: -17.96 - 79.17, 0 nan values
u (float64) has range: 13.90 - 27.84, 0 nan values
g (float64) has range: 13.37 - 27.17, 0 nan values
r (float64) has range: 10.39 - 25.29, 0 nan values
i (float64) has range: 10.03 - 24.57, 0 nan values
z (float64) has range: 10.63 - 25.70, 0 nan values
redshift (float64) has range: -0.01 - 7.01, 0 nan values
spectral_type (object) has 4 unique values: ['G/K', 'M', 'O/B', 'A/F'], 0 nan values
galaxy_population (object) has 2 unique values: ['Red_Sequence', 'Blue_Cloud'], 0 nan values

-> train.csv has 577347 rows and 12 columns.
Here is some information about the columns:
id (int64) has range: 0.00 - 577346.00, 0 nan values
alpha (float64) has range: 0.01 - 360.00, 0 nan values
delta (float64) has range: -17.97 - 79.16, 0 nan values
u (float64) has range: -0.14 - 28.25, 0 nan values
g (float64) has range: 13.54 - 27.62, 0 nan values
r (float64) has range: 12.58 - 25.25, 0 nan values
i (float64) has range: 11.96 - 27.91, 0 nan values
z (float64) has range: 11.68 - 26.83, 0 nan values
redshift (float64) has range: -0.01 - 7.01, 0 nan values
spectral_type (object) has 4 unique values: ['M', 'O/B', 'G/K', 'A/F'], 0 nan values
galaxy_population (object) has 2 unique values: ['Red_Sequence', 'Blue_Cloud'], 0 nan values

# Task

## Goal
Predict the stellar class for each test-set object.

## Evaluation
Submissions are evaluated using balanced accuracy between predicted class labels and the true class. Submission file must contain `id,class` with one label per test row, where `class` is one of GALAXY, STAR, or QSO.

## Data description
`train.csv` contains 577347 rows with 10 feature columns plus `id`, `galaxy_population`, `spectral_type`, and the target `class`. `test.csv` contains the same predictors without the target across 247435 rows. `sample_submission.csv` shows the required submission columns `id` and `class`. The target is a 3-class stellar classification problem with labels GALAXY, QSO, and STAR.

# Project Target
- ID column: id
- Target column: class
- Problem type: multiclass
- Evaluation metric: balanced_accuracy

# Failed Materialization

- Mode: autogluon
- Hypothesis ID: 000028
- Source file: autogluon-002.py
- Failed node: 
- Run: 

# Execution Error

```text
Invalid materialization for hypothesis 000028 (autogluon); response=hypotheses/000028/materializations/autogluon-002.response.md: Group materialization must not execute top-level code outside imports, definitions, and registry assignments
```

# Previous Code

```python
import numpy as np
import pandas as pd

try:
    from scipy.spatial import cKDTree
except Exception:
    cKDTree = None

_EXTINCTION_COEFFS = (5.155, 3.793, 2.751, 2.086, 1.479)
_SFD98_CORRECTION = 0.86
_EBV_CLIP_MAX = 1.2
_BIN_SIZE_DEG = 5.0
_BIN_KEY_SCALE = 1000
_NEAREST_BIN_RADIUS_DEG = 10.0
_PERCENTILE_CLIP = (0.5, 99.5)
_PERCENTILE_GRID = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
_EQ_TO_GAL_MATRIX = (
    (-0.0548755604, -0.8734370902, -0.4838350155),
    (0.4941094279, -0.4448296300, 0.7469822445),
    (-0.8676661490, -0.1980763734, 0.4559837762),
)

def _to_float_array(values):
    return pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)

def _choose_column(columns_map, aliases):
    for alias in aliases:
        if alias in columns_map:
            return columns_map[alias]
    return None

def _coerce_train_mask(series, expected_len):
    if series is None or len(series) != expected_len:
        return None

    arr = pd.Series(series).reset_index(drop=True)

    if pd.api.types.is_bool_dtype(arr):
        return arr.fillna(False).astype(bool).to_numpy()

    if pd.api.types.is_numeric_dtype(arr):
        vals = pd.to_numeric(arr, errors="coerce")
        finite = vals.dropna().to_numpy()
        if finite.size == 0:
            return None
        unique = np.unique(np.floor(finite + 1e-8).astype(int))
        if set(unique).issubset({0, 1}):
            return vals.fillna(0).round().clip(0, 1).astype(int).astype(bool).to_numpy()
        return None

    text = arr.astype(str).str.strip().str.lower()
    unique = set(text.dropna().unique())
    if unique.issubset({"train", "test"}):
        return (text == "train").to_numpy()
    if unique.issubset({"true", "false"}):
        return (text == "true").to_numpy()
    if unique.issubset({"yes", "no"}):
        return (text == "yes").to_numpy()
    if unique.issubset({"1", "0"}):
        return (text == "1").to_numpy()
    if unique.issubset({"tr", "va", "te"}):
        return ((text == "tr") | (text == "te")).to_numpy()
    return None

def _infer_train_mask(raw, aux):
    n = len(raw)
    if aux is None or len(aux) == 0:
        return np.ones(n, dtype=bool)

    aux_df = aux if isinstance(aux, pd.DataFrame) else pd.DataFrame(aux)
    if aux_df.empty or len(aux_df) != n:
        return np.ones(n, dtype=bool)

    col_map = {str(c).strip().lower(): c for c in aux_df.columns}
    preferred = ("is_train", "train", "is_train_row", "is_train_only", "train_mask", "in_train", "split", "subset")
    for name in preferred:
        candidate = _coerce_train_mask(aux_df[col_map[name]], n) if name in col_map else None
        if candidate is not None:
            return candidate

    for c in aux_df.columns:
        candidate = _coerce_train_mask(aux_df[c], n)
        if candidate is not None:
            return candidate

    return np.ones(n, dtype=bool)

def _spherical_vectors_from_lb(l_deg, b_deg):
    l = np.deg2rad(np.asarray(l_deg, dtype=float))
    b = np.deg2rad(np.asarray(b_deg, dtype=float))
    cb = np.cos(b)
    return np.column_stack((cb * np.cos(l), cb * np.sin(l), np.sin(b)))

def _equatorial_to_galactic(ra_deg, dec_deg):
    ra = np.deg2rad(np.asarray(ra_deg, dtype=float))
    dec = np.deg2rad(np.asarray(dec_deg, dtype=float))

    x = np.cos(dec) * np.cos(ra)
    y = np.cos(dec) * np.sin(ra)
    z = np.sin(dec)

    m00, m01, m02 = _EQ_TO_GAL_MATRIX[0]
    m10, m11, m12 = _EQ_TO_GAL_MATRIX[1]
    m20, m21, m22 = _EQ_TO_GAL_MATRIX[2]

    xg = m00 * x + m01 * y + m02 * z
    yg = m10 * x + m11 * y + m12 * z
    zg = m20 * x + m21 * y + m22 * z

    l = (np.degrees(np.arctan2(yg, xg)) + 360.0) % 360.0
    b = np.degrees(np.arcsin(np.clip(zg, -1.0, 1.0)))
    return l, b

def _nearest_indices(query_l, query_b, ref_l, ref_b):
    query_l = np.asarray(query_l, dtype=float)
    query_b = np.asarray(query_b, dtype=float)
    ref_l = np.asarray(ref_l, dtype=float)
    ref_b = np.asarray(ref_b, dtype=float)

    nq = len(query_l)
    nr = len(ref_l)
    idx = np.full(nq, -1, dtype=np.int64)
    dist_deg = np.full(nq, np.nan, dtype=float)

    if nq == 0 or nr == 0:
        return idx, dist_deg

    q = _spherical_vectors_from_lb(query_l, query_b)
    r = _spherical_vectors_from_lb(ref_l, ref_b)

    if cKDTree is not None:
        tree = cKDTree(r)
        chord, nearest = tree.query(q, k=1)
        dist_deg = np.degrees(2.0 * np.arcsin(np.clip(chord / 2.0, 0.0, 1.0)))
        idx = nearest.astype(np.int64)
        return idx, dist_deg

    chunk = 2048
    for start in range(0, nq, chunk):
        end = min(start + chunk, nq)
        block = q[start:end]
        dots = np.dot(block, r.T)
        dots = np.clip(dots, -1.0, 1.0)
        nearest = np.argmax(dots, axis=1)
        best_dot = dots[np.arange(block.shape[0]), nearest]
        idx[start:end] = nearest.astype(np.int64)
        dist_deg[start:end] = np.degrees(np.arccos(best_dot))

    return idx, dist_deg

def _nearest_values(query_l, query_b, ref_l, ref_b, ref_vals):
    query_l = np.asarray(query_l, dtype=float)
    query_b = np.asarray(query_b, dtype=float)
    ref_l = np.asarray(ref_l, dtype=float)
    ref_b = np.asarray(ref_b, dtype=float)
    ref_vals = np.asarray(ref_vals, dtype=float)

    out = np.full(len(query_l), np.nan, dtype=float)

    valid_ref = np.isfinite(ref_l) & np.isfinite(ref_b) & np.isfinite(ref_vals)
    valid_q = np.isfinite(query_l) & np.isfinite(query_b)

    if not np.any(valid_ref) or not np.any(valid_q):
        return out

    nearest_idx, _ = _nearest_indices(query_l[valid_q], query_b[valid_q], ref_l[valid_ref], ref_b[valid_ref])
    out_q = ref_vals[valid_ref][nearest_idx]
    out[valid_q] = out_q
    return out

def _bin_index_5deg(longitude_deg, latitude_deg, size_deg):
    l = np.asarray(longitude_deg, dtype=float)
    b = np.asarray(latitude_deg, dtype=float)
    out = np.full(len(l), -1, dtype=np.int64)

    valid = np.isfinite(l) & np.isfinite(b)
    if not np.any(valid):
        return out

    l_wrap = np.mod(l[valid], 360.0)
    l_wrap[l_wrap < 0.0] += 360.0
    b_clip = np.clip(b[valid], -89.999999, 89.999999)

    l_bin = np.floor_divide(l_wrap, size_deg).astype(np.int64)
    b_bin = np.floor_divide((b_clip + 90.0), size_deg).astype(np.int64)
    out[valid] = l_bin * _BIN_KEY_SCALE + b_bin
    return out

def _decode_bin_centers(keys):
    l_idx = (keys // _BIN_KEY_SCALE).astype(float)
    b_idx = (keys % _BIN_KEY_SCALE).astype(float)
    l_cent = (l_idx + 0.5) * _BIN_SIZE_DEG
    b_cent = (b_idx + 0.5) * _BIN_SIZE_DEG - 90.0
    return l_cent, b_cent

def _extract_map_ebv(raw_l, raw_b, aux):
    if aux is None or len(aux) == 0:
        return None

    aux_df = aux if isinstance(aux, pd.DataFrame) else pd.DataFrame(aux)
    if aux_df.empty:
        return None

    cols = {str(c).strip().lower(): c for c in aux_df.columns}

    ebv_col = _choose_column(
        cols,
        ("ebv", "ebv_sfd", "ebv_sfd98", "ebv_sfd98_scaled", "sfd_ebv", "sfd98_ebv", "e_bv", "e_bv_scaled"),
    )
    if ebv_col is None:
        return None

    ebv = _to_float_array(aux_df[ebv_col])

    if len(aux_df) == len(raw_l):
        return pd.Series(ebv, index=aux_df.index[: len(raw_l)]).reset_index(drop=True).to_numpy()

    l_col = _choose_column(cols, ("l", "l_deg", "galactic_longitude", "gal_lon", "glon", "lon"))
    b_col = _choose_column(cols, ("b", "b_deg", "galactic_latitude", "gal_lat", "glat", "lat"))
    if l_col is None or b_col is None:
        return None

    map_l = _to_float_array(aux_df[l_col])
    map_b = _to_float_array(aux_df[b_col])
    return _nearest_values(raw_l, raw_b, map_l, map_b, ebv)

def _impute_ebv(raw_l, raw_b, ebv_init, train_mask):
    ebv = np.asarray(ebv_init, dtype=float).copy()
    n = len(ebv)
    missing_init = ~np.isfinite(ebv)
    imputed = missing_init.copy()

    if n == 0:
        return ebv, imputed

    if not missing_init.any():
        return np.clip(np.maximum(ebv, 0.0), 0.0, _EBV_CLIP_MAX), imputed

    if len(train_mask) == n:
        train = np.asarray(train_mask, dtype=bool)
    else:
        train = np.ones(n, dtype=bool)

    train_ok = train & np.isfinite(ebv)
    if not np.any(train_ok):
        train_ok = np.isfinite(ebv)
    global_train_median = np.nan if not np.any(train_ok) else np.nanmedian(ebv[train_ok])

    bin_keys = _bin_index_5deg(raw_l, raw_b, _BIN_SIZE_DEG)
    train_key = pd.Series(bin_keys[train_ok], index=np.flatnonzero(train_ok), dtype="int64")
    train_ebv = pd.Series(ebv[train_ok], index=np.flatnonzero(train_ok), dtype=float)
    med_by_bin = pd.Series(train_ebv.to_numpy(), index=train_key.to_numpy(), dtype=float).groupby(level=0).median()
    med_by_bin_dict = med_by_bin.to_dict()

    miss_idx = np.flatnonzero(missing_init)
    if miss_idx.size > 0 and len(med_by_bin_dict) > 0:
        miss_keys = pd.Series(bin_keys[miss_idx], index=miss_idx, dtype="int64")
        fill_bin = miss_keys.map(med_by_bin_dict).to_numpy(dtype=float)
        fill_ok = np.isfinite(fill_bin)
        ebv[miss_idx[fill_ok]] = fill_bin[fill_ok]
        missing = ~np.isfinite(ebv)
        miss_idx = np.flatnonzero(missing)
    else:
        miss_idx = np.flatnonzero(~np.isfinite(ebv))

    if miss_idx.size > 0 and len(med_by_bin_dict) > 0:
        valid_keys = np.asarray(med_by_bin.index.to_numpy(), dtype=np.int64)
        valid_vals = med_by_bin.to_numpy(dtype=float)
        bin_l, bin_b = _decode_bin_centers(valid_keys)
        nn_idx, nn_deg = _nearest_indices(raw_l[miss_idx], raw_b[miss_idx], bin_l, bin_b)
        use = np.isfinite(nn_deg) & (nn_deg <= _NEAREST_BIN_RADIUS_DEG)
        if np.any(use):
            pick_rows = miss_idx[use]
            ebv[pick_rows] = valid_vals[nn_idx[use].astype(np.int64)]
        missing = ~np.isfinite(ebv)
        miss_idx = np.flatnonzero(missing)

    if miss_idx.size > 0:
        fill = 0.0 if not np.isfinite(global_train_median) else float(global_train_median)
        ebv[miss_idx] = fill

    ebv = np.clip(np.maximum(ebv, 0.0), 0.0, _EBV_CLIP_MAX)
    return ebv, imputed

def _latitude_bin(abs_b, train_mask, index):
    abs_b = np.asarray(abs_b, dtype=float)
    n = len(abs_b)
    bin_series = pd.Series(np.full(n, pd.NA, dtype="Int64"), index=index)

    if len(train_mask) == n:
        train = np.asarray(train_mask, dtype=bool)
    else:
        train = np.ones(n, dtype=bool)

    train_vals = abs_b[train & np.isfinite(abs_b)]
    if train_vals.size == 0:
        train_vals = abs_b[np.isfinite(abs_b)]
    if train_vals.size < 2:
        if n > 0:
            bin_series[:] = 0
        return bin_series

    try:
        _, edges = pd.qcut(train_vals, q=np.array(_PERCENTILE_GRID), labels=False, retbins=True, duplicates="drop")
        if edges.size < 3:
            bin_series[:] = 0
        else:
            binned = pd.cut(abs_b, bins=edges, labels=False, include_lowest=True)
            bin_series = pd.Series(binned, index=index).astype("Int64")
    except ValueError:
        bin_series[:] = 0

    return bin_series

def _clip_continuous_features(feature_df, train_mask):
    n = len(feature_df)
    if len(train_mask) == n:
        train = np.asarray(train_mask, dtype=bool)
    else:
        train = np.ones(n, dtype=bool)

    clipped = feature_df.copy()
    lo_q, hi_q = _PERCENTILE_CLIP

    for col in clipped.columns:
        if col in {"is_ebv_imputed", "latitude_bin"}:
            continue
        series = clipped[col]
        if not pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series):
            continue
        train_vals = pd.to_numeric(series[train], errors="coerce").to_numpy(dtype=float)
        finite = train_vals[np.isfinite(train_vals)]
        if finite.size < 3:
            continue
        lo, hi = np.nanpercentile(finite, [lo_q, hi_q])
        if np.isfinite(lo) and np.isfinite(hi) and lo < hi:
            clipped[col] = series.clip(lower=lo, upper=hi)

    return clipped

def add_foreground_reddening_geometry(raw, deps, aux):
    ra = _to_float_array(raw["alpha"])
    dec = _to_float_array(raw["delta"])
    l, b = _equatorial_to_galactic(ra, dec)

    l = np.asarray(l, dtype=float)
    b = np.asarray(b, dtype=float)

    train_mask = _infer_train_mask(raw, aux)

    map_ebv = _extract_map_ebv(l, b, aux)
    if map_ebv is None:
        map_ebv = np.full(len(raw), np.nan, dtype=float)
    map_ebv = np.asarray(map_ebv, dtype=float) * _SFD98_CORRECTION

    ebv, is_ebv_imputed = _impute_ebv(l, b, map_ebv, train_mask)

    a_u, a_g, a_r, a_i, a_z = _EXTINCTION_COEFFS
    A_u = a_u * ebv
    A_g = a_g * ebv
    A_r = a_r * ebv
    A_i = a_i * ebv
    A_z = a_z * ebv

    u = _to_float_array(raw["u"])
    g = _to_float_array(raw["g"])
    r = _to_float_array(raw["r"])
    i = _to_float_array(raw["i"])
    z = _to_float_array(raw["z"])
    redshift = _to_float_array(raw["redshift"])

    u0 = u - A_u
    g0 = g - A_g
    r0 = r - A_r
    i0 = i - A_i
    z0 = z - A_z

    c_ug = u - g
    c_gr = g - r
    c_ri = r - i
    c_iz = i - z

    c0_ug = u0 - g0
    c0_gr = g0 - r0
    c0_ri = r0 - i0
    c0_iz = i0 - z0

    colors = np.column_stack((c_ug, c_gr, c_ri, c_iz))
    colors0 = np.column_stack((c0_ug, c0_gr, c0_ri, c0_iz))

    reddening_delta = np.array((a_u - a_g, a_g - a_r, a_r - a_i, a_i - a_z), dtype=float)
    norm = np.linalg.norm(reddening_delta)
    if norm <= 0.0:
        norm = 1.0
    d_unit = reddening_delta / norm

    t = np.einsum("ij,j->i", colors, d_unit)
    t0 = np.einsum("ij,j->i", colors0, d_unit)
    dt = t - t0

    r0 = colors0 - np.outer(t0, d_unit)
    r0_u = r0[:, 0]
    r0_g = r0[:, 1]
    r0_i = r0[:, 2]
    r0_norm = np.linalg.norm(r0, axis=1)

    abs_b = np.abs(b)
    lat_bin = _latitude_bin(abs_b, train_mask, raw.index)

    features = pd.DataFrame(
        {
            "ebv_clipped": pd.Series(ebv, index=raw.index, dtype=float),
            "is_ebv_imputed": pd.Series(is_ebv_imputed, index=raw.index, dtype=bool),
            "A_u": pd.Series(A_u, index=raw.index, dtype=float),
            "A_g": pd.Series(A_g, index=raw.index, dtype=float),
            "A_r": pd.Series(A_r, index=raw.index, dtype=float),
            "A_i": pd.Series(A_i, index=raw.index, dtype=float),
            "A_z": pd.Series(A_z, index=raw.index, dtype=float),
            "u0": pd.Series(u0, index=raw.index, dtype=float),
            "g0": pd.Series(g0, index=raw.index, dtype=float),
            "r0_mag": pd.Series(r0, index=raw.index, dtype=float),
            "i0": pd.Series(i0, index=raw.index, dtype=float),
            "z0": pd.Series(z0, index=raw.index, dtype=float),
            "color_u_minus_g": pd.Series(c_ug, index=raw.index, dtype=float),
            "color_g_minus_r": pd.Series(c_gr, index=raw.index, dtype=float),
            "color_r_minus_i": pd.Series(c_ri, index=raw.index, dtype=float),
            "color_i_minus_z": pd.Series(c_iz, index=raw.index, dtype=float),
            "color0_u_minus_g": pd.Series(c0_ug, index=raw.index, dtype=float),
            "color0_g_minus_r": pd.Series(c0_gr, index=raw.index, dtype=float),
            "color0_r_minus_i": pd.Series(c0_ri, index=raw.index, dtype=float),
            "color0_i_minus_z": pd.Series(c0_iz, index=raw.index, dtype=float),
            "reddening_t": pd.Series(t, index=raw.index, dtype=float),
            "reddening_t0": pd.Series(t0, index=raw.index, dtype=float),
            "reddening_delta_t": pd.Series(dt, index=raw.index, dtype=float),
            "reddening_residual_u_minus_g": pd.Series(r0_u, index=raw.index, dtype=float),
            "reddening_residual_g_minus_r": pd.Series(r0_g, index=raw.index, dtype=float),
            "reddening_residual_r_minus_i": pd.Series(r0_i, index=raw.index, dtype=float),
            "reddening_residual_norm": pd.Series(r0_norm, index=raw.index, dtype=float),
            "abs_gal_latitude": pd.Series(abs_b, index=raw.index, dtype=float),
            "latitude_bin": lat_bin,
            "reddening_delta_t_x_abs_b": pd.Series(dt * abs_b, index=raw.index, dtype=float),
            "redshift_x_ebv": pd.Series(redshift * ebv, index=raw.index, dtype=float),
            "ebv_x_abs_b": pd.Series(ebv * abs_b, index=raw.index, dtype=float),
        },
        index=raw.index,
    )

    features = _clip_continuous_features(features, train_mask)
    return features

FEATURE_GROUPS = [
    {
        "name": "foreground_reddening_geometry",
        "fn": add_foreground_reddening_geometry,
        "depends_on": [],
        "description": "Build extinction-aware color decomposition features by estimating reddening from dust map values, imputing missing map coverage, and projecting observed/dereddened colors onto and orthogonal to the reddening direction with latitude-aware context features.",
    }
]
```

# Group Code Contract

Return only Python code. Do not use markdown fences.

Define semantic feature-group functions and `FEATURE_GROUPS`.

Generate only the feature-group module: Python definitions for feature-group
preprocessing. A separate fixed runtime wrapper imports this module and is
responsible for logging, timing, dependency ordering, output-column renaming,
final DataFrame assembly, and all non-preprocessing work.

Each feature function must use this signature:

```python
def add_group_name(raw, deps, aux):
    ...
    return new_features
```

Rules:
- `raw` is the raw/base train+test covariate frame without target labels. It includes ID columns.
- `deps` is a dict of dependency outputs by logical group name. Use it only when this group declares dependencies.
- `aux` is an auxiliary DataFrame when available, otherwise empty.
- Return a pandas DataFrame containing only new local feature columns with `index=raw.index`.
- Preserve row count, row order, and index exactly.
- Do not return raw/input columns.
- Do not mutate `raw`, `deps`, or `aux` in place.
- Use clear local feature names. The executor will rename returned columns after the function finishes.
- Outputs may be numeric, boolean, categorical, or string scalar columns. Do not return nested lists, dicts, tuples, or sets.
- You may compute covariate-only train+test statistics from `raw`; do not use target labels, validation labels, model outputs, or leaderboard feedback.
- Do not read project data files, write files, train models, create `main()`, concatenate final blocks, or implement orchestration.
- Do not implement timing decorators or logging wrappers. The group executor logs every group call and duration.
- Top-level code may contain only imports, function definitions, literal constants, and `FEATURE_GROUPS`.
- Do not call functions in top-level assignments. For example, do not write `EDGES = np.array(...)`, `CUTS = pd.IntervalIndex(...)`, or any other assignment whose right-hand side calls a function or constructor.
- If a constant needs conversion to a NumPy/Pandas object, store it as a literal tuple/list at module level and convert it inside the feature function.

Register groups like this:

```python
FEATURE_GROUPS = [
    {
        "name": "group_name",
        "fn": add_group_name,
        "depends_on": [],
        "description": "One sentence describing this feature group.",
    }
]
```

Mode-specific boundary:
- Do not train AutoGluon.
- Do not import or instantiate `TabularPredictor`.
- Do not call `.fit()`, `.predict()`, `.predict_proba()`, or `.leaderboard()`.
- Do not define `main()`.
- Do not read project data files.
- The fixed wrapper handles all non-preprocessing work.