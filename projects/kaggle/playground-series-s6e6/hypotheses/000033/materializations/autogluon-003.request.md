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
- Hypothesis ID: 000033
- Source file: autogluon-002.py
- Failed node: 20260624T061610-f9fffdd2-341
- Run: 20260622T034843-eca160d1

# Execution Error

```text
failed.yaml error:
Process exited with 1

stderr.log:
Traceback (most recent call last):
  File "/home/xai/DEV/TheML/.worktrees/root-hypothesis-revisions/projects/kaggle/playground-series-s6e6/runs/20260622T034843-eca160d1/artifacts/20260624T061610-f9fffdd2-341/02-code.py", line 748, in main
    transformed = run_feature_groups(
                  ^^^^^^^^^^^^^^^^^^^
  File "/home/xai/DEV/TheML/.worktrees/root-hypothesis-revisions/src/tml/features/groups.py", line 59, in run_feature_groups
    block = group["fn"](raw.copy(), {key: value.copy() for key, value in deps.items()}, aux.copy())
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/xai/DEV/TheML/.worktrees/root-hypothesis-revisions/projects/kaggle/playground-series-s6e6/runs/20260622T034843-eca160d1/artifacts/20260624T061610-f9fffdd2-341/02-code.py", line 150, in add_restframe_sed_family_fit
    fit_unweighted = _weighted_poly_block(x_centered, y_centered, weights_unweighted)
                     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/xai/DEV/TheML/.worktrees/root-hypothesis-revisions/projects/kaggle/playground-series-s6e6/runs/20260622T034843-eca160d1/artifacts/20260624T061610-f9fffdd2-341/02-code.py", line 44, in _weighted_poly_block
    if s0 <= WEIGHT_EPS:
       ^^^^^^^^^^^^^^^^
ValueError: The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()
Traceback (most recent call last):
  File "/home/xai/DEV/TheML/.worktrees/root-hypothesis-revisions/projects/kaggle/playground-series-s6e6/runs/20260622T034843-eca160d1/artifacts/20260624T061610-f9fffdd2-341/02-code.py", line 872, in <module>
    main()
  File "/home/xai/DEV/TheML/.worktrees/root-hypothesis-revisions/projects/kaggle/playground-series-s6e6/runs/20260622T034843-eca160d1/artifacts/20260624T061610-f9fffdd2-341/02-code.py", line 748, in main
    transformed = run_feature_groups(
                  ^^^^^^^^^^^^^^^^^^^
  File "/home/xai/DEV/TheML/.worktrees/root-hypothesis-revisions/src/tml/features/groups.py", line 59, in run_feature_groups
    block = group["fn"](raw.copy(), {key: value.copy() for key, value in deps.items()}, aux.copy())
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/xai/DEV/TheML/.worktrees/root-hypothesis-revisions/projects/kaggle/playground-series-s6e6/runs/20260622T034843-eca160d1/artifacts/20260624T061610-f9fffdd2-341/02-code.py", line 150, in add_restframe_sed_family_fit
    fit_unweighted = _weighted_poly_block(x_centered, y_centered, weights_unweighted)
                     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/xai/DEV/TheML/.worktrees/root-hypothesis-revisions/projects/kaggle/playground-series-s6e6/runs/20260622T034843-eca160d1/artifacts/20260624T061610-f9fffdd2-341/02-code.py", line 44, in _weighted_poly_block
    if s0 <= WEIGHT_EPS:
       ^^^^^^^^^^^^^^^^
ValueError: The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()

stdout.log:
AutoGluon materialization: loaded aux file star_classification.csv rows=100000 cols=18 passed_to_groups=True
AutoGluon materialization: starting feature groups
TheML feature group: start name=restframe_sed_family_fit
TheML feature group: failed name=restframe_sed_family_fit elapsed_s=0.088 rows=824782 cols=0
```

# Previous Code

```python
import numpy as np
import pandas as pd

BANDS = ("u", "g", "r", "i", "z")
REST_WAVELENGTHS_ANGSTROM = (3551.0, 4686.0, 6165.0, 7481.0, 8931.0)
LYMAN_FULL_WEIGHT = 1216.0
LYMAN_PARTIAL_WEIGHT = 912.0
WEIGHT_EPS = 1e-12
NUMERICAL_EPS = 1e-15

def _weighted_poly_block(x, y, weights):
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
        x_row = x[i]
        y_row = y[i]
        w_row = weights[i]

        active = w_row > WEIGHT_EPS
        k = int(active.sum())
        if k < 2:
            continue

        x_a = x_row[active]
        y_a = y_row[active]
        w_a = w_row[active]

        s0 = np.dot(w_a, 1.0)
        if s0 <= WEIGHT_EPS:
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