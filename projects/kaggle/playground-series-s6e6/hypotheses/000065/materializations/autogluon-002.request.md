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
- Hypothesis ID: 000065
- Source file: autogluon-001.py
- Failed node: 20260629T220046-67eefcc1-66
- Run: 20260629T213805-6aae5126

# Execution Error

```text
failed.yaml error:
Process exited with 1

stderr.log:
Traceback (most recent call last):
  File "/home/xai/DEV/TheML/projects/kaggle/playground-series-s6e6/runs/20260629T213805-6aae5126/artifacts/20260629T220046-67eefcc1-66/02-code.py", line 658, in main
    transformed = run_feature_groups(
                  ^^^^^^^^^^^^^^^^^^^
  File "/home/xai/DEV/TheML/src/tml/features/groups.py", line 59, in run_feature_groups
    block = group["fn"](raw.copy(), {key: value.copy() for key, value in deps.items()}, aux.copy())
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/xai/DEV/TheML/projects/kaggle/playground-series-s6e6/runs/20260629T213805-6aae5126/artifacts/20260629T220046-67eefcc1-66/02-code.py", line 95, in add_fiber_collision_crowding_context
    neighbor_indices, neighbor_dists = tree.query_radius(
                                       ^^^^^^^^^^^^^^^^^^
  File "/home/xai/DEV/TheML/projects/kaggle/playground-series-s6e6/runs/20260629T213805-6aae5126/artifacts/20260629T220046-67eefcc1-66/02-code.py", line 337, in _raise_timeout
    raise TimeoutError(f"AutoGluon preprocess exceeded {seconds} seconds")
TimeoutError: AutoGluon preprocess exceeded 90 seconds
Traceback (most recent call last):
  File "/home/xai/DEV/TheML/projects/kaggle/playground-series-s6e6/runs/20260629T213805-6aae5126/artifacts/20260629T220046-67eefcc1-66/02-code.py", line 782, in <module>
    main()
  File "/home/xai/DEV/TheML/projects/kaggle/playground-series-s6e6/runs/20260629T213805-6aae5126/artifacts/20260629T220046-67eefcc1-66/02-code.py", line 658, in main
    transformed = run_feature_groups(
                  ^^^^^^^^^^^^^^^^^^^
  File "/home/xai/DEV/TheML/src/tml/features/groups.py", line 59, in run_feature_groups
    block = group["fn"](raw.copy(), {key: value.copy() for key, value in deps.items()}, aux.copy())
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/xai/DEV/TheML/projects/kaggle/playground-series-s6e6/runs/20260629T213805-6aae5126/artifacts/20260629T220046-67eefcc1-66/02-code.py", line 95, in add_fiber_collision_crowding_context
    neighbor_indices, neighbor_dists = tree.query_radius(
                                       ^^^^^^^^^^^^^^^^^^
  File "/home/xai/DEV/TheML/projects/kaggle/playground-series-s6e6/runs/20260629T213805-6aae5126/artifacts/20260629T220046-67eefcc1-66/02-code.py", line 337, in _raise_timeout
    raise TimeoutError(f"AutoGluon preprocess exceeded {seconds} seconds")
TimeoutError: AutoGluon preprocess exceeded 90 seconds

stdout.log:
AutoGluon materialization: loaded aux file star_classification.csv rows=100000 cols=18 passed_to_groups=True
AutoGluon materialization: starting feature groups
TheML feature group: start name=fiber_collision_crowding_context
TheML feature group: failed name=fiber_collision_crowding_context elapsed_s=91.050 rows=824782 cols=0
```

# Previous Code

```python
import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

ARCMIN_PER_DEGREE = 60.0
ARCSEC_PER_DEGREE = 3600.0
ARCSEC_PER_RADIAN = 206264.80624709636
CROWDING_RADII_ARCSEC = (2.0, 3.0, 10.0, 55.0, 62.0, 120.0)
COLOR_PAIRS = (("u", "g"), ("g", "r"), ("r", "i"), ("i", "z"))

def _as_numeric(raw, column, default=0.0):
    if column in raw.columns:
        return pd.to_numeric(raw[column], errors="coerce").to_numpy(dtype=float)
    return np.full(len(raw), default, dtype=float)

def _finite_quantile(values, q, fallback):
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float(fallback)
    return float(np.nanquantile(arr, q))

def _context_frame(raw, aux):
    needed = ("alpha", "delta", "redshift", "r", "u", "g", "i", "z")
    frames = [raw.loc[:, [c for c in needed if c in raw.columns]]]
    if isinstance(aux, pd.DataFrame) and len(aux) > 0 and "alpha" in aux.columns and "delta" in aux.columns:
        frames.append(aux.loc[:, [c for c in needed if c in aux.columns]])
    return pd.concat(frames, axis=0, ignore_index=True, copy=False)

def add_fiber_collision_crowding_context(raw, deps, aux):
    n = len(raw)
    out = pd.DataFrame(index=raw.index)

    if n == 0 or "alpha" not in raw.columns or "delta" not in raw.columns:
        return out

    context = _context_frame(raw, aux)
    alpha = pd.to_numeric(raw["alpha"], errors="coerce").to_numpy(dtype=float)
    delta = pd.to_numeric(raw["delta"], errors="coerce").to_numpy(dtype=float)
    ctx_alpha = pd.to_numeric(context["alpha"], errors="coerce").to_numpy(dtype=float)
    ctx_delta = pd.to_numeric(context["delta"], errors="coerce").to_numpy(dtype=float)

    valid = np.isfinite(alpha) & np.isfinite(delta)
    ctx_valid = np.isfinite(ctx_alpha) & np.isfinite(ctx_delta)

    query_coords = np.column_stack((np.deg2rad(np.clip(delta[valid], -90.0, 90.0)), np.deg2rad(alpha[valid] % 360.0)))
    context_coords = np.column_stack((np.deg2rad(np.clip(ctx_delta[ctx_valid], -90.0, 90.0)), np.deg2rad(ctx_alpha[ctx_valid] % 360.0)))

    radii_rad = [r / ARCSEC_PER_RADIAN for r in CROWDING_RADII_ARCSEC]
    max_radius_rad = 120.0 / ARCSEC_PER_RADIAN

    counts = {r: np.zeros(n, dtype=float) for r in CROWDING_RADII_ARCSEC}
    nearest = np.full(n, 120.0, dtype=float)
    second_nearest = np.full(n, 120.0, dtype=float)
    has_neighbor_120 = np.zeros(n, dtype=bool)

    redshift = _as_numeric(raw, "redshift")
    rmag = _as_numeric(raw, "r")
    color_values = {}
    for a, b in COLOR_PAIRS:
        color_values[a + "_" + b] = _as_numeric(raw, a) - _as_numeric(raw, b)

    dz_med = np.full(n, _finite_quantile(np.abs(redshift - np.nanmedian(redshift[np.isfinite(redshift)])), 0.95, 1.0), dtype=float)
    dr_med = np.full(n, _finite_quantile(np.abs(rmag - np.nanmedian(rmag[np.isfinite(rmag)])), 0.95, 5.0), dtype=float)
    color_med = {
        name: np.full(
            n,
            _finite_quantile(np.abs(vals - np.nanmedian(vals[np.isfinite(vals)])), 0.95, 2.0),
            dtype=float,
        )
        for name, vals in color_values.items()
    }

    if query_coords.shape[0] > 0 and context_coords.shape[0] > 0:
        tree = BallTree(context_coords, metric="haversine")
        valid_positions = np.flatnonzero(valid)
        ctx_raw_valid_positions = np.flatnonzero(ctx_valid[:n])
        raw_ctx_lookup = np.full(n, -1, dtype=int)
        raw_ctx_lookup[ctx_raw_valid_positions] = np.arange(ctx_raw_valid_positions.size)

        for radius_arcsec, radius_rad in zip(CROWDING_RADII_ARCSEC, radii_rad):
            raw_counts = tree.query_radius(query_coords, r=radius_rad, count_only=True).astype(float)
            self_mask = raw_ctx_lookup[valid_positions] >= 0
            raw_counts[self_mask] -= 1.0
            counts[radius_arcsec][valid_positions] = np.maximum(raw_counts, 0.0)

        neighbor_indices, neighbor_dists = tree.query_radius(
            query_coords,
            r=max_radius_rad,
            return_distance=True,
            sort_results=True,
        )

        ctx_to_raw = np.full(len(context), -1, dtype=int)
        raw_context_indices = np.arange(n)
        ctx_to_raw[raw_context_indices[ctx_valid[:n]]] = raw_context_indices[ctx_valid[:n]]

        ctx_redshift = pd.to_numeric(context.get("redshift", pd.Series(np.nan, index=context.index)), errors="coerce").to_numpy(dtype=float)
        ctx_rmag = pd.to_numeric(context.get("r", pd.Series(np.nan, index=context.index)), errors="coerce").to_numpy(dtype=float)
        ctx_colors = {}
        for a, b in COLOR_PAIRS:
            ctx_a = pd.to_numeric(context.get(a, pd.Series(np.nan, index=context.index)), errors="coerce").to_numpy(dtype=float)
            ctx_b = pd.to_numeric(context.get(b, pd.Series(np.nan, index=context.index)), errors="coerce").to_numpy(dtype=float)
            ctx_colors[a + "_" + b] = ctx_a - ctx_b

        valid_context_original = np.flatnonzero(ctx_valid)

        for local_i, raw_i in enumerate(valid_positions):
            inds = neighbor_indices[local_i]
            dists = neighbor_dists[local_i]
            original_inds = valid_context_original[inds]

            self_ctx = raw_ctx_lookup[raw_i]
            if self_ctx >= 0:
                keep = inds != self_ctx
                inds = inds[keep]
                dists = dists[keep]
                original_inds = original_inds[keep]

            if len(dists) > 0:
                arcsec_dists = dists * ARCSEC_PER_RADIAN
                nearest[raw_i] = min(float(arcsec_dists[0]), 120.0)
                second_nearest[raw_i] = min(float(arcsec_dists[1]), 120.0) if len(arcsec_dists) > 1 else 120.0
                has_neighbor_120[raw_i] = True

                dz = np.abs(ctx_redshift[original_inds] - redshift[raw_i])
                dr = np.abs(ctx_rmag[original_inds] - rmag[raw_i])
                if np.isfinite(dz).any():
                    dz_med[raw_i] = float(np.nanmedian(dz))
                if np.isfinite(dr).any():
                    dr_med[raw_i] = float(np.nanmedian(dr))
                for name, vals in color_values.items():
                    dc = np.abs(ctx_colors[name][original_inds] - vals[raw_i])
                    if np.isfinite(dc).any():
                        color_med[name][raw_i] = float(np.nanmedian(dc))

    for radius in CROWDING_RADII_ARCSEC:
        label = str(int(radius))
        out["log1p_neighbors_within_" + label + "arcsec"] = np.log1p(counts[radius])
        out["has_neighbor_within_" + label + "arcsec"] = counts[radius] > 0

    out["nearest_neighbor_arcsec_capped_120"] = nearest
    out["second_nearest_neighbor_arcsec_capped_120"] = second_nearest
    out["log_nearest_over_55arcsec"] = np.log((nearest + 0.25) / 55.0)
    out["log_second_nearest_over_55arcsec"] = np.log((second_nearest + 0.25) / 55.0)
    out["margin_to_3arcsec"] = 3.0 - nearest
    out["margin_to_55arcsec"] = 55.0 - nearest
    out["margin_to_62arcsec"] = 62.0 - nearest
    out["neighbor_count_55_to_120arcsec_log1p"] = np.log1p(np.maximum(counts[120.0] - counts[55.0], 0.0))
    out["neighbor_count_62_to_120arcsec_log1p"] = np.log1p(np.maximum(counts[120.0] - counts[62.0], 0.0))
    out["has_any_neighbor_within_120arcsec"] = has_neighbor_120
    out["median_abs_redshift_diff_neighbors_120arcsec"] = dz_med
    out["median_abs_rmag_diff_neighbors_120arcsec"] = dr_med

    for name in sorted(color_med):
        out["median_abs_" + name + "_diff_neighbors_120arcsec"] = color_med[name]

    return out

FEATURE_GROUPS = [
    {
        "name": "fiber_collision_crowding_context",
        "fn": add_fiber_collision_crowding_context,
        "depends_on": [],
        "description": "Fiber-scale angular neighbor counts, collision-threshold margins, and companion-similarity summaries from unlabeled sky-position context.",
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