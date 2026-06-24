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
- Hypothesis ID: 000051
- Source file: autogluon-004.py
- Failed node: 20260624T063323-7703ff40-360
- Run: 20260622T034843-eca160d1

# Execution Error

```text
failed.yaml error:
Process exited with 1

stderr.log:
Traceback (most recent call last):
  File "/home/xai/DEV/TheML/.worktrees/root-hypothesis-revisions/projects/kaggle/playground-series-s6e6/runs/20260622T034843-eca160d1/artifacts/20260624T063323-7703ff40-360/02-code.py", line 1169, in main
    transformed = run_feature_groups(
                  ^^^^^^^^^^^^^^^^^^^
  File "/home/xai/DEV/TheML/.worktrees/root-hypothesis-revisions/src/tml/features/groups.py", line 59, in run_feature_groups
    block = group["fn"](raw.copy(), {key: value.copy() for key, value in deps.items()}, aux.copy())
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/xai/DEV/TheML/.worktrees/root-hypothesis-revisions/projects/kaggle/playground-series-s6e6/runs/20260622T034843-eca160d1/artifacts/20260624T063323-7703ff40-360/02-code.py", line 523, in add_k_correction_residual_manifold
    blend_mask, blend_t, blend_left, blend_right = _build_blend_arrays(z_clip, regimes)
                                                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/xai/DEV/TheML/.worktrees/root-hypothesis-revisions/projects/kaggle/playground-series-s6e6/runs/20260622T034843-eca160d1/artifacts/20260624T063323-7703ff40-360/02-code.py", line 188, in _build_blend_arrays
    t = (z[local_pos] - (center[local_pos] - delta[local])) / (2.0 * delta[local[local_pos]] + EPS)
                                                                           ~~~~~^^^^^^^^^^^
IndexError: index 24245 is out of bounds for axis 0 with size 24243
Traceback (most recent call last):
  File "/home/xai/DEV/TheML/.worktrees/root-hypothesis-revisions/projects/kaggle/playground-series-s6e6/runs/20260622T034843-eca160d1/artifacts/20260624T063323-7703ff40-360/02-code.py", line 1293, in <module>
    main()
  File "/home/xai/DEV/TheML/.worktrees/root-hypothesis-revisions/projects/kaggle/playground-series-s6e6/runs/20260622T034843-eca160d1/artifacts/20260624T063323-7703ff40-360/02-code.py", line 1169, in main
    transformed = run_feature_groups(
                  ^^^^^^^^^^^^^^^^^^^
  File "/home/xai/DEV/TheML/.worktrees/root-hypothesis-revisions/src/tml/features/groups.py", line 59, in run_feature_groups
    block = group["fn"](raw.copy(), {key: value.copy() for key, value in deps.items()}, aux.copy())
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/xai/DEV/TheML/.worktrees/root-hypothesis-revisions/projects/kaggle/playground-series-s6e6/runs/20260622T034843-eca160d1/artifacts/20260624T063323-7703ff40-360/02-code.py", line 523, in add_k_correction_residual_manifold
    blend_mask, blend_t, blend_left, blend_right = _build_blend_arrays(z_clip, regimes)
                                                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/xai/DEV/TheML/.worktrees/root-hypothesis-revisions/projects/kaggle/playground-series-s6e6/runs/20260622T034843-eca160d1/artifacts/20260624T063323-7703ff40-360/02-code.py", line 188, in _build_blend_arrays
    t = (z[local_pos] - (center[local_pos] - delta[local])) / (2.0 * delta[local[local_pos]] + EPS)
                                                                           ~~~~~^^^^^^^^^^^
IndexError: index 24245 is out of bounds for axis 0 with size 24243

stdout.log:
AutoGluon materialization: loaded aux file star_classification.csv rows=100000 cols=18 passed_to_groups=True
AutoGluon materialization: starting feature groups
TheML feature group: start name=k_correction_residual_manifold
TheML feature group: failed name=k_correction_residual_manifold elapsed_s=0.145 rows=824782 cols=0
```

# Previous Code

```python
import numpy as np
import pandas as pd

BANDS = ("u", "g", "r", "i", "z")
REGIME_EDGES = (0.0, 0.2, 0.7, 1.6, 7.0)
NUM_Z_BINS = 30
NUM_COLOR_BINS = 12
MIN_SLICE_ROWS = 5000
MIN_BIN_ROWS = 200
HUBER_DELTA = 1.345
HUBER_ITERS = 8
BLEND_SCALE = 0.03
EPS = 1e-12

def _to_float(values):
    return pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)

def _robust_mad(values):
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return 0.0
    med = np.nanmedian(x)
    return float(np.nanmedian(np.abs(x - med)))

def _mad_series(values):
    return _robust_mad(values.to_numpy(dtype=float))

def _safe_quantile(values, q):
    try:
        return np.quantile(values, q, method="linear")
    except TypeError:
        return np.quantile(values, q, interpolation="linear")

def _quantile_edges(values, num_bins):
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.array([0.0, 1.0], dtype=float)

    v_min = float(np.nanmin(arr))
    v_max = float(np.nanmax(arr))
    if np.isclose(v_min, v_max):
        half_width = max(abs(v_min) * 1e-4, 1.0)
        return np.linspace(v_min - half_width, v_max + half_width, num_bins + 1, dtype=float)

    qs = np.linspace(0.0, 1.0, num_bins + 1)
    edges = _safe_quantile(arr, qs)
    edges = np.asarray(edges, dtype=float)
    edges = np.sort(edges)

    for i in range(1, edges.size):
        if edges[i] <= edges[i - 1]:
            edges[i] = edges[i - 1] + (1e-9 * (abs(edges[i - 1]) + 1.0))

    if not np.all(np.isfinite(edges)):
        return np.linspace(v_min, v_max, num_bins + 1, dtype=float)
    return edges

def _bin_index(values, edges):
    vals = np.asarray(values, dtype=float)
    valid = np.isfinite(vals) & (vals >= edges[0]) & (vals <= edges[-1])
    idx = np.full(vals.shape, -1, dtype=int)
    if np.any(valid):
        idx_valid = np.searchsorted(edges, vals[valid], side="right") - 1
        idx[valid] = np.clip(idx_valid, 0, len(edges) - 2)
    return idx, valid

def _fit_huber_polynomial(z, c, y):
    z = np.asarray(z, dtype=float)
    c = np.asarray(c, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(z) & np.isfinite(c) & np.isfinite(y)
    if np.sum(mask) < 6:
        return np.zeros(6, dtype=float), int(np.sum(mask))

    z = z[mask]
    c = c[mask]
    y = y[mask]
    X = np.column_stack((np.ones_like(z), z, z * z, c, c * c, z * c))

    try:
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
    except np.linalg.LinAlgError:
        return np.zeros(6, dtype=float), len(y)

    for _ in range(HUBER_ITERS):
        pred = X @ beta
        resid = y - pred
        scale = _robust_mad(resid)
        if scale <= 1e-10:
            break
        z_sc = resid / (scale / 0.6744897501960817 + EPS)
        w = np.ones_like(z_sc)
        outliers = np.abs(z_sc) > HUBER_DELTA
        w[outliers] = HUBER_DELTA / (np.abs(z_sc[outliers]) + EPS)
        sw = np.sqrt(w)
        try:
            beta_new = np.linalg.lstsq(X * sw[:, None], y * sw, rcond=None)[0]
        except np.linalg.LinAlgError:
            break
        if np.allclose(beta, beta_new, rtol=1e-7, atol=1e-7):
            beta = beta_new
            break
        beta = beta_new

    return beta.astype(float), int(mask.sum())

def _poly_predict(beta, z, color):
    z = np.asarray(z, dtype=float)
    c = np.asarray(color, dtype=float)
    return (
        beta[0]
        + beta[1] * z
        + beta[2] * z * z
        + beta[3] * c
        + beta[4] * c * c
        + beta[5] * z * c
    )

def _compute_regime(redshift_clip):
    reg_boundaries = np.array(REGIME_EDGES[1:-1], dtype=float)
    return np.searchsorted(reg_boundaries, redshift_clip, side="right").astype(int)

def _build_blend_arrays(redshift_clip, regimes):
    n = len(redshift_clip)
    blend_mask = np.zeros(n, dtype=bool)
    blend_t = np.zeros(n, dtype=float)
    blend_left = np.full(n, -1, dtype=int)
    blend_right = np.full(n, -1, dtype=int)

    delta = np.maximum(BLEND_SCALE * (1.0 + np.maximum(redshift_clip, 0.0)), 1e-6)

    for r in range(4):
        idx = np.where(regimes == r)[0]
        if idx.size == 0:
            continue
        z = redshift_clip[idx]
        d = np.zeros_like(z, dtype=float)
        left = np.full_like(idx, -1, dtype=int)
        right = np.full_like(idx, -1, dtype=int)
        center = np.zeros_like(z, dtype=float)

        if r == 0:
            boundary = REGIME_EDGES[1]
            left[:] = 0
            right[:] = 1
            d = np.abs(z - boundary)
            center[:] = boundary
        elif r == 3:
            boundary = REGIME_EDGES[2]
            left[:] = 2
            right[:] = 3
            d = np.abs(z - boundary)
            center[:] = boundary
        else:
            lower = REGIME_EDGES[r]
            upper = REGIME_EDGES[r + 1]
            d_lower = np.abs(z - lower)
            d_upper = np.abs(z - upper)
            use_lower = d_lower <= d_upper
            left_sub = np.where(use_lower, r - 1, r)
            right_sub = np.where(use_lower, r, r + 1)
            d = np.where(use_lower, d_lower, d_upper)
            center = np.where(use_lower, lower, upper)
            left = left_sub
            right = right_sub

        in_blend = d <= delta[idx]
        if not np.any(in_blend):
            continue

        local = idx[in_blend]
        local_pos = np.arange(idx.size)[in_blend]
        t = (z[local_pos] - (center[local_pos] - delta[local])) / (2.0 * delta[local[local_pos]] + EPS)
        t = np.clip(t, 0.0, 1.0)

        blend_mask[local] = True
        blend_t[local] = t
        blend_left[local] = left[local_pos]
        blend_right[local] = right[local_pos]

    return blend_mask, blend_t, blend_left, blend_right

def _select_fit_slice(h_codes, regimes, h_code, regime):
    mask_local = (h_codes == h_code) & (regimes == regime)
    if np.sum(mask_local) >= MIN_SLICE_ROWS:
        return mask_local, "local", (int(regime),), True

    adjacent = [int(regime)]
    if regime > 0:
        adjacent.append(int(regime - 1))
    if regime < 3:
        adjacent.append(int(regime + 1))
    adjacent = tuple(sorted(set(adjacent)))

    mask_adj_local = (h_codes == h_code) & np.isin(regimes, adjacent)
    if np.sum(mask_adj_local) >= MIN_SLICE_ROWS:
        return mask_adj_local, "local_adjacent", adjacent, True

    mask_adj_global = np.isin(regimes, adjacent)
    if np.sum(mask_adj_global) >= MIN_SLICE_ROWS:
        return mask_adj_global, "global_adjacent", adjacent, False

    return np.ones(len(regimes), dtype=bool), "global_all", (0, 1, 2, 3), False

def _build_models_for_band(z, color, mag, h_codes, regimes):
    unique_h = np.unique(h_codes)
    model_by_base = {}
    models = {}
    global_finite = np.isfinite(z) & np.isfinite(color) & np.isfinite(mag)

    for h_code in unique_h:
        for r in range(4):
            mask, scope, reg_set, by_strata = _select_fit_slice(h_codes, regimes, int(h_code), r)
            key = (scope, int(h_code), reg_set) if by_strata else (scope, reg_set)

            if key not in models:
                beta, nfit = _fit_huber_polynomial(z, color, mag)
                fit_mask = mask & np.isfinite(z) & np.isfinite(color) & np.isfinite(mag)
                if nfit > 0 and np.any(fit_mask):
                    z_min = float(np.nanmin(z[fit_mask]))
                    z_max = float(np.nanmax(z[fit_mask]))
                    c_min = float(np.nanmin(color[fit_mask]))
                    c_max = float(np.nanmax(color[fit_mask]))
                elif global_finite.any():
                    z_min = float(np.nanmin(z[global_finite]))
                    z_max = float(np.nanmax(z[global_finite]))
                    c_min = float(np.nanmin(color[global_finite]))
                    c_max = float(np.nanmax(color[global_finite]))
                else:
                    z_min = REGIME_EDGES[0]
                    z_max = REGIME_EDGES[-1]
                    c_min = -5.0
                    c_max = 5.0

                models[key] = {
                    "coef": beta,
                    "n": int(nfit),
                    "z_min": z_min,
                    "z_max": z_max,
                    "c_min": c_min,
                    "c_max": c_max,
                }

            model_by_base[(int(h_code), r)] = key

    # Optional MAD-based clipping of coefficients before use.
    coeff_rows = [m["coef"] for m in models.values()]
    if coeff_rows:
        coeff_matrix = np.vstack(coeff_rows)
        med = np.nanmedian(np.abs(coeff_matrix), axis=0)
        mad = np.nanmedian(np.abs(np.abs(coeff_matrix) - med), axis=0)
        clip = np.where(np.isfinite(med + 3.0 * mad), med + 3.0 * mad, 1e-4)
        clip = np.where(clip > 1e-6, clip, 1e-4)
        for m in models.values():
            m["coef"] = np.clip(m["coef"], -clip, clip)

    return model_by_base, models

def _in_model_support(info, z, color):
    return (
        np.isfinite(z)
        & np.isfinite(color)
        & (z >= info["z_min"])
        & (z <= info["z_max"])
        & (color >= info["c_min"])
        & (color <= info["c_max"])
    )

def _predict_band(redshift_clip, color, h_codes, regimes, blend_mask, blend_t, blend_left, blend_right, model_by_base, models):
    n = len(redshift_clip)
    pred = np.full(n, np.nan, dtype=float)
    out_hull = np.zeros(n, dtype=bool)

    unique_h = np.unique(h_codes)

    for h_code in unique_h:
        for r in range(4):
            idx = np.where((h_codes == h_code) & (regimes == r))[0]
            if idx.size == 0:
                continue

            base_key = model_by_base[(int(h_code), r)]
            base_info = models[base_key]
            pred[idx] = _poly_predict(base_info["coef"], redshift_clip[idx], color[idx])
            own_in = _in_model_support(base_info, redshift_clip[idx], color[idx])
            out_hull[idx] = ~own_in

            rel_blend = np.where(blend_mask[idx])[0]
            if rel_blend.size == 0:
                continue

            rows = idx[rel_blend]
            left_r = blend_left[rows]
            right_r = blend_right[rows]
            t = blend_t[rows].astype(float)

            own_out_local = ~own_in[rel_blend]
            left_pred = np.full(rows.size, np.nan, dtype=float)
            right_pred = np.full(rows.size, np.nan, dtype=float)
            left_out = np.array(own_out_local, dtype=bool)
            right_out = np.array(own_out_local, dtype=bool)

            for lr in np.unique(left_r):
                if lr < 0:
                    continue
                sel = left_r == lr
                if not np.any(sel):
                    continue
                left_key = model_by_base.get((int(h_code), int(lr)))
                if left_key is None:
                    continue
                info = models[left_key]
                sel_rows = rows[sel]
                lp = _poly_predict(info["coef"], redshift_clip[sel_rows], color[sel_rows])
                left_pred[sel] = lp
                left_out[sel] = ~_in_model_support(info, redshift_clip[sel_rows], color[sel_rows])

            for rr in np.unique(right_r):
                if rr < 0:
                    continue
                sel = right_r == rr
                if not np.any(sel):
                    continue
                right_key = model_by_base.get((int(h_code), int(rr)))
                if right_key is None:
                    continue
                info = models[right_key]
                sel_rows = rows[sel]
                rp = _poly_predict(info["coef"], redshift_clip[sel_rows], color[sel_rows])
                right_pred[sel] = rp
                right_out[sel] = ~_in_model_support(info, redshift_clip[sel_rows], color[sel_rows])

            left_pred[~np.isfinite(left_pred)] = pred[rows[~np.isfinite(left_pred)]]
            right_pred[~np.isfinite(right_pred)] = pred[rows[~np.isfinite(right_pred)]]
            pred[rows] = (1.0 - t) * left_pred + t * right_pred

            blend_out = own_out_local & left_out & right_out
            out_hull[rows] = blend_out

    return pred, out_hull

def _compute_anchor_offsets(pred, h_codes, redshift_clip):
    n_codes = int(np.nanmax(h_codes)) + 1 if h_codes.size else 0
    anchors = np.zeros(n_codes, dtype=float)
    finite = np.isfinite(pred)
    overall = float(np.nanmedian(pred[finite])) if np.any(finite) else 0.0
    for h_code in np.unique(h_codes):
        m = (h_codes == h_code) & finite & (redshift_clip < 0.02)
        if np.any(m):
            val = np.nanmedian(pred[m])
            anchors[int(h_code)] = float(val) if np.isfinite(val) else overall
        else:
            anchors[int(h_code)] = overall
    return anchors

def _compute_bin_stats(z_slice, color_slice, value_slice):
    z_slice = np.asarray(z_slice, dtype=float)
    color_slice = np.asarray(color_slice, dtype=float)
    value_slice = np.asarray(value_slice, dtype=float)

    finite_all = np.isfinite(z_slice) & np.isfinite(color_slice)
    if not finite_all.any():
        return {
            "edges_z": _quantile_edges(np.array([0.0, 1.0], dtype=float), NUM_Z_BINS),
            "edges_c": _quantile_edges(np.array([0.0, 1.0], dtype=float), NUM_COLOR_BINS),
            "bin_stats": {},
            "global_median": 0.0,
            "global_mad": 1e-6,
            "global_n": 0,
        }

    finite = finite_all & np.isfinite(value_slice)
    z_for_edges = z_slice[finite_all]
    c_for_edges = color_slice[finite_all]

    z_edges = _quantile_edges(z_for_edges, NUM_Z_BINS)
    c_edges = _quantile_edges(c_for_edges, NUM_COLOR_BINS)

    if not finite.any():
        global_vals = value_slice[finite_all]
        return {
            "edges_z": z_edges,
            "edges_c": c_edges,
            "bin_stats": {},
            "global_median": float(np.nanmedian(global_vals)) if np.any(finite_all) else 0.0,
            "global_mad": max(_robust_mad(global_vals), 1e-6),
            "global_n": int(np.sum(finite_all)),
        }

    zi, _ = _bin_index(z_slice, z_edges)
    ci, _ = _bin_index(color_slice, c_edges)

    tmp = pd.DataFrame({
        "zi": zi[finite].astype(int),
        "ci": ci[finite].astype(int),
        "v": value_slice[finite],
    }, copy=False)

    by_bin = tmp.groupby(["zi", "ci"])["v"]
    med = by_bin.median()
    mad = by_bin.apply(_mad_series)
    n = by_bin.size()

    bin_stats = {}
    for key in med.index:
        key_int = (int(key[0]), int(key[1]))
        cnt = int(n.loc[key])
        bin_stats[key_int] = {
            "median": float(med.loc[key]),
            "mad": max(float(mad.loc[key]), 1e-6),
            "n": cnt,
        }

    global_vals = value_slice[finite]
    g_med = float(np.nanmedian(global_vals))
    g_mad = max(_robust_mad(global_vals), 1e-6)

    return {
        "edges_z": z_edges,
        "edges_c": c_edges,
        "bin_stats": bin_stats,
        "global_median": g_med,
        "global_mad": g_mad,
        "global_n": int(np.sum(finite)),
    }

def _lookup_bin_baseline(stat, zi, ci, zi_ok, ci_ok):
    if not zi_ok or not ci_ok:
        return stat["global_median"], stat["global_mad"], False

    rec = stat["bin_stats"].get((int(zi), int(ci)))
    if rec is not None and rec["n"] >= MIN_BIN_ROWS:
        return rec["median"], rec["mad"], True

    nzi = len(stat["edges_z"]) - 1
    nci = len(stat["edges_c"]) - 1

    neigh = []
    for dz in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dz == 0 and dc == 0:
                continue
            zi2 = int(zi + dz)
            ci2 = int(ci + dc)
            if zi2 < 0 or zi2 >= nzi or ci2 < 0 or ci2 >= nci:
                continue
            nb = stat["bin_stats"].get((zi2, ci2))
            if nb is None:
                continue
            w = 1.0 / (abs(dz) + abs(dc) + 0.001)
            neigh.append((nb["median"], nb["mad"], max(nb["n"], 1), w))

    if neigh:
        num_med = 0.0
        num_mad = 0.0
        den = 0.0
        for med, mad, n, w in neigh:
            ww = w * n
            num_med += med * ww
            num_mad += mad * ww
            den += ww
        return num_med / den, max(num_mad / den, 1e-6), False

    return stat["global_median"], stat["global_mad"], False

def add_k_correction_residual_manifold(raw, deps, aux):
    required_cols = (
        "id",
        "alpha",
        "delta",
        "u",
        "g",
        "r",
        "i",
        "z",
        "redshift",
        "spectral_type",
        "galaxy_population",
    )
    if any(c not in raw.columns for c in required_cols):
        return pd.DataFrame(index=raw.index)

    n = len(raw)
    z = _to_float(raw["redshift"])
    z_clip = np.clip(z, REGIME_EDGES[0], REGIME_EDGES[-1] - 1e-12)
    regimes = _compute_regime(z_clip)

    strata = (raw["spectral_type"].astype(str) + "|" + raw["galaxy_population"].astype(str)).to_numpy()
    h_codes, _ = pd.factorize(strata, sort=False)

    mags = {band: _to_float(raw[band]) for band in BANDS}
    colors = {
        "u": mags["u"] - mags["g"],
        "g": mags["g"] - mags["r"],
        "r": mags["r"] - mags["i"],
        "i": mags["i"] - mags["z"],
        "z": mags["g"] - mags["r"],
    }

    blend_mask, blend_t, blend_left, blend_right = _build_blend_arrays(z_clip, regimes)

    kcorr = {}
    rest_mag = {}
    out_hull_model = np.zeros(n, dtype=bool)

    for band in BANDS:
        model_by_base, models = _build_models_for_band(
            z_clip,
            colors[band],
            mags[band],
            h_codes,
            regimes,
        )
        pred, out_hull = _predict_band(
            z_clip,
            colors[band],
            h_codes,
            regimes,
            blend_mask,
            blend_t,
            blend_left,
            blend_right,
            model_by_base,
            models,
        )
        anchors = _compute_anchor_offsets(pred, h_codes, z_clip)
        pred_neutral = pred - anchors[h_codes]
        kcorr[band] = pred_neutral
        rest_mag[band] = mags[band] - pred_neutral
        out_hull_model = np.logical_or(out_hull_model, out_hull)

    d1 = rest_mag["u"] - rest_mag["g"]
    d2 = rest_mag["g"] - rest_mag["r"]
    d3 = rest_mag["r"] - rest_mag["i"]
    d4 = rest_mag["i"] - rest_mag["z"]
    s1 = d1 - d2
    s2 = d2 - d3
    s3 = d3 - d4

    metric_to_color_band = {
        "M_u": "u",
        "M_g": "g",
        "M_r": "r",
        "M_i": "i",
        "M_z": "z",
        "d1": "u",
        "d2": "g",
        "d3": "r",
        "d4": "i",
        "s1": "g",
        "s2": "r",
        "s3": "i",
    }
    metric_values = {
        "M_u": rest_mag["u"],
        "M_g": rest_mag["g"],
        "M_r": rest_mag["r"],
        "M_i": rest_mag["i"],
        "M_z": rest_mag["z"],
        "d1": d1,
        "d2": d2,
        "d3": d3,
        "d4": d4,
        "s1": s1,
        "s2": s2,
        "s3": s3,
    }

    bin_stats = {}
    for h_code in np.unique(h_codes):
        for r in range(4):
            idx = np.where((h_codes == h_code) & (regimes == r))[0]
            if idx.size == 0:
                continue
            z_slice = z_clip[idx]
            for metric_name, color_band in metric_to_color_band.items():
                c_slice = colors[color_band][idx]
                v_slice = metric_values[metric_name][idx]
                stats = _compute_bin_stats(z_slice, c_slice, v_slice)
                bin_stats[(int(h_code), int(r), metric_name)] = stats

    out_of_hull_bin = np.zeros(n, dtype=bool)
    low_support = np.zeros(n, dtype=bool)

    features = {
        "kc_blend_zone": blend_mask.astype(bool),
        "kc_blend_weight": blend_t.astype(float),
        "kc_out_of_hull_model": out_hull_model.astype(bool),
        "kc_out_of_hull_bin": out_of_hull_bin,
        "kc_low_support": low_support,
        "kc_out_of_hull": np.logical_or(out_hull_model, out_of_hull_bin),
    }

    for band in BANDS:
        features[f"kc_{band}_kcorr"] = kcorr[band]
        features[f"kc_{band}_rest"] = rest_mag[band]

    for metric_name in metric_to_color_band.keys():
        features[f"kc_{metric_name}_resid"] = np.full(n, np.nan, dtype=float)
        features[f"kc_{metric_name}_zscore"] = np.full(n, np.nan, dtype=float)

    unique_h = np.unique(h_codes)
    for h_code in unique_h:
        for r in range(4):
            idx = np.where((h_codes == h_code) & (regimes == r))[0]
            if idx.size == 0:
                continue

            z_slice = z_clip[idx]

            for metric_name, color_band in metric_to_color_band.items():
                stats = bin_stats.get((int(h_code), int(r), metric_name))
                if stats is None:
                    features[f"kc_{metric_name}_resid"][idx] = np.nan
                    features[f"kc_{metric_name}_zscore"][idx] = np.nan
                    features["kc_low_support"][idx] = True
                    features["kc_out_of_hull_bin"][idx] = True
                    continue

                c_slice = colors[color_band][idx]
                v_slice = metric_values[metric_name][idx]

                zi, zi_ok = _bin_index(z_slice, stats["edges_z"])
                ci, ci_ok = _bin_index(c_slice, stats["edges_c"])

                for j, row_idx in enumerate(idx):
                    if not np.isfinite(v_slice[j]):
                        features[f"kc_{metric_name}_resid"][row_idx] = np.nan
                        features[f"kc_{metric_name}_zscore"][row_idx] = np.nan
                        features["kc_low_support"][row_idx] = True
                        continue

                    med, mad, good_support = _lookup_bin_baseline(
                        stats,
                        zi[j],
                        ci[j],
                        bool(zi_ok[j]),
                        bool(ci_ok[j]),
                    )
                    resid = v_slice[j] - med
                    zscore = resid / (mad if mad > 1e-6 else 1e-6)

                    features[f"kc_{metric_name}_resid"][row_idx] = resid
                    features[f"kc_{metric_name}_zscore"][row_idx] = zscore
                    if not good_support:
                        features["kc_low_support"][row_idx] = True
                    if not (bool(zi_ok[j]) and bool(ci_ok[j])):
                        features["kc_out_of_hull_bin"][row_idx] = True

    features["kc_out_of_hull"] = np.logical_or(
        features["kc_out_of_hull_model"],
        features["kc_out_of_hull_bin"],
    )

    return pd.DataFrame(features, index=raw.index)

FEATURE_GROUPS = [
    {
        "name": "k_correction_residual_manifold",
        "fn": add_k_correction_residual_manifold,
        "depends_on": [],
        "description": "Builds regime-and-stratum aware k-correction manifold predictions plus binwise residual and z-score geometry features.",
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