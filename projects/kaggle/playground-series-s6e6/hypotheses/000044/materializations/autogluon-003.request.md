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
- Hypothesis ID: 000044
- Source file: autogluon-002.py
- Failed node: 20260624T062553-2596ce29-353
- Run: 20260622T034843-eca160d1

# Execution Error

```text
failed.yaml error:
Process exited with 1

stderr.log:
Traceback (most recent call last):
  File "/home/xai/DEV/TheML/.worktrees/root-hypothesis-revisions/projects/kaggle/playground-series-s6e6/runs/20260622T034843-eca160d1/artifacts/20260624T062553-2596ce29-353/02-code.py", line 665, in main
    transformed = run_feature_groups(
                  ^^^^^^^^^^^^^^^^^^^
  File "/home/xai/DEV/TheML/.worktrees/root-hypothesis-revisions/src/tml/features/groups.py", line 59, in run_feature_groups
    block = group["fn"](raw.copy(), {key: value.copy() for key, value in deps.items()}, aux.copy())
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/xai/DEV/TheML/.worktrees/root-hypothesis-revisions/projects/kaggle/playground-series-s6e6/runs/20260622T034843-eca160d1/artifacts/20260624T062553-2596ce29-353/02-code.py", line 124, in add_redshifted_absorption_trough_residuals
    d_blue += d_vis
ValueError: operands could not be broadcast together with shapes (824782,) (694249,) (824782,) 
Traceback (most recent call last):
  File "/home/xai/DEV/TheML/.worktrees/root-hypothesis-revisions/projects/kaggle/playground-series-s6e6/runs/20260622T034843-eca160d1/artifacts/20260624T062553-2596ce29-353/02-code.py", line 789, in <module>
    main()
  File "/home/xai/DEV/TheML/.worktrees/root-hypothesis-revisions/projects/kaggle/playground-series-s6e6/runs/20260622T034843-eca160d1/artifacts/20260624T062553-2596ce29-353/02-code.py", line 665, in main
    transformed = run_feature_groups(
                  ^^^^^^^^^^^^^^^^^^^
  File "/home/xai/DEV/TheML/.worktrees/root-hypothesis-revisions/src/tml/features/groups.py", line 59, in run_feature_groups
    block = group["fn"](raw.copy(), {key: value.copy() for key, value in deps.items()}, aux.copy())
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/xai/DEV/TheML/.worktrees/root-hypothesis-revisions/projects/kaggle/playground-series-s6e6/runs/20260622T034843-eca160d1/artifacts/20260624T062553-2596ce29-353/02-code.py", line 124, in add_redshifted_absorption_trough_residuals
    d_blue += d_vis
ValueError: operands could not be broadcast together with shapes (824782,) (694249,) (824782,)

stdout.log:
AutoGluon materialization: loaded aux file star_classification.csv rows=100000 cols=18 passed_to_groups=True
AutoGluon materialization: starting feature groups
TheML feature group: start name=redshifted_absorption_trough_residuals
TheML feature group: failed name=redshifted_absorption_trough_residuals elapsed_s=0.099 rows=824782 cols=0
```

# Previous Code

```python
import numpy as np
import pandas as pd

_BAND_NAMES = ("u", "g", "r", "i", "z")
_BAND_EDGES = (3000.0, 4100.0, 5500.0, 7000.0, 8200.0, 9200.0)
_BAND_CENTERS = (3551.0, 4686.0, 6165.0, 7481.0, 8931.0)
_LINE_SPECS = (
    ("CaIIK", 3933.7, "blue"),
    ("CaIIH", 3968.5, "blue"),
    ("G_band", 4304.0, "blue"),
    ("Hdelta", 4102.0, "blue"),
    ("Hgamma", 4341.0, "blue"),
    ("Hbeta", 4861.0, "red"),
    ("MgB", 5175.0, "red"),
    ("NaID", 5893.0, "red"),
)
_REGIME_BINS = (
    (0.0, 0.45, "z_0_0_45"),
    (0.45, 1.1, "z_0_45_1_1"),
    (1.1, 2.0, "z_1_1_2_0"),
    (2.0, 7.0, "z_2_0_7_0"),
)
_EPS = 1e-12
_EPS_RATIO = 1e-6

def _fit_edge_continuum_log(log_flux_points, log_wave_points, wavelength):
    log_wave_points = np.asarray(log_wave_points, dtype=np.float64)
    wave_mean = log_wave_points.mean()
    wave_dev = log_wave_points - wave_mean
    denom = np.sum(wave_dev * wave_dev)
    y_mean = log_flux_points.mean(axis=1)
    slope = np.sum((log_flux_points - y_mean[:, None]) * wave_dev, axis=1) / denom
    intercept = y_mean - slope * wave_mean
    return intercept + slope * np.log(wavelength)

def add_redshifted_absorption_trough_residuals(raw, deps, aux):
    _ = (deps, aux)

    z = raw["redshift"].to_numpy(dtype=np.float64, copy=False)
    n = z.shape[0]

    band_edges = np.array(_BAND_EDGES, dtype=np.float64)
    band_centers = np.array(_BAND_CENTERS, dtype=np.float64)
    log_band_centers = np.log(band_centers)
    half_band_widths = (band_edges[1:] - band_edges[:-1]) / 2.0

    flux_matrix = np.column_stack(
        [np.power(10.0, -0.4 * raw[col].to_numpy(dtype=np.float64, copy=False)) for col in _BAND_NAMES]
    )
    log_flux_matrix = np.log(flux_matrix)

    d_blue = np.zeros(n, dtype=np.float64)
    d_red = np.zeros(n, dtype=np.float64)
    a_tot = np.zeros(n, dtype=np.float64)
    visible_line_count = np.zeros(n, dtype=np.float64)

    d_cakih = np.zeros(n, dtype=np.float64)
    d_caIIH = np.zeros(n, dtype=np.float64)
    d_hdelta = np.zeros(n, dtype=np.float64)
    d_hgamma = np.zeros(n, dtype=np.float64)

    feature_data = {}

    edge_fit_x_left = np.array((log_band_centers[1], log_band_centers[2], log_band_centers[3], log_band_centers[4]), dtype=np.float64)
    edge_fit_x_right = np.array((log_band_centers[0], log_band_centers[1], log_band_centers[2], log_band_centers[3]), dtype=np.float64)

    for line_name, rest_wave, region in _LINE_SPECS:
        obs_wave = rest_wave * (1.0 + z)
        visible = (z >= 0.0) & (obs_wave >= band_edges[0]) & (obs_wave <= band_edges[-1])
        line_rows = np.flatnonzero(visible)
        line_deficit = np.zeros(n, dtype=np.float64)

        if line_rows.size:
            obs_visible = obs_wave[line_rows]
            band_idx = np.searchsorted(band_edges[1:], obs_visible, side="left")
            band_idx = np.clip(band_idx, 0, len(_BAND_NAMES) - 1)

            center_visible = band_centers[band_idx]
            half_visible = half_band_widths[band_idx]
            q_vis = 1.0 - 0.5 * np.minimum(1.0, np.abs(obs_visible - center_visible) / half_visible)
            q_vis = np.clip(q_vis, 0.5, 1.0)

            log_obs = np.log(obs_visible)
            cont_log = np.zeros(line_rows.size, dtype=np.float64)

            for bj in range(len(_BAND_NAMES)):
                within_band = band_idx == bj
                if not within_band.any():
                    continue
                idx = np.flatnonzero(within_band)
                rows = line_rows[idx]
                log_lam = log_obs[idx]

                if bj == 0:
                    y = log_flux_matrix[rows][:, (1, 2, 3, 4)]
                    cont_log[idx] = _fit_edge_continuum_log(y, edge_fit_x_left, obs_visible[idx])
                elif bj == 4:
                    y = log_flux_matrix[rows][:, (0, 1, 2, 3)]
                    cont_log[idx] = _fit_edge_continuum_log(y, edge_fit_x_right, obs_visible[idx])
                else:
                    left = bj - 1
                    right = bj + 1
                    y_left = log_flux_matrix[rows, left]
                    y_right = log_flux_matrix[rows, right]
                    t = (log_lam - log_band_centers[left]) / (log_band_centers[right] - log_band_centers[left])
                    cont_log[idx] = y_left + (y_right - y_left) * t

            cont = np.exp(cont_log)
            fb = flux_matrix[line_rows, band_idx]
            residual = (cont - fb) / (cont + _EPS)

            d_vis = np.minimum(np.maximum(residual, 0.0), 3.0) * q_vis
            a_vis = np.minimum(np.maximum(-residual, 0.0), 3.0) * q_vis

            line_deficit[line_rows] = d_vis
            line_excess = a_vis

            if region == "blue":
                d_blue += d_vis
            else:
                d_red += d_vis

            a_tot += line_excess
            visible_line_count += q_vis

            if line_name == "CaIIK":
                d_cakih = line_deficit
            elif line_name == "CaIIH":
                d_caIIH = line_deficit
            elif line_name == "Hdelta":
                d_hdelta = line_deficit
            elif line_name == "Hgamma":
                d_hgamma = line_deficit

        feature_data[f"visible_{line_name}"] = visible.astype(np.int8)

    d_blue = np.clip(d_blue, 0.0, 3.0 * len(_LINE_SPECS))
    d_red = np.clip(d_red, 0.0, 3.0 * len(_LINE_SPECS))
    a_tot = np.clip(a_tot, 0.0, 3.0 * len(_LINE_SPECS))

    denom_abs = d_blue + d_red + _EPS_RATIO
    absorption_excess_ratio = a_tot / denom_abs
    metal_blanketing_skew = (d_blue - d_red) / denom_abs
    cak_vs_balmer_denom = d_hdelta + d_hgamma + _EPS_RATIO
    cak_cah_vs_balmer = (d_cakih + d_caIIH) / cak_vs_balmer_denom

    features = {
        "D_blue": d_blue,
        "D_red": d_red,
        "A_tot": a_tot,
        "absorption_excess_ratio": absorption_excess_ratio,
        "metal_blanketing_skew": metal_blanketing_skew,
        "CaK_CaH_vs_Balmer": cak_cah_vs_balmer,
        "visible_line_count": visible_line_count,
    }
    features.update(feature_data)

    for lo, hi, tag in _REGIME_BINS:
        regime_mask = (z >= lo) & (z < hi)
        m = regime_mask.astype(np.float64)
        features[f"D_blue_{tag}"] = d_blue * m
        features[f"D_red_{tag}"] = d_red * m
        features[f"A_tot_{tag}"] = a_tot * m
        features[f"absorption_excess_ratio_{tag}"] = absorption_excess_ratio * m
        features[f"metal_blanketing_skew_{tag}"] = metal_blanketing_skew * m
        features[f"CaK_CaH_vs_Balmer_{tag}"] = cak_cah_vs_balmer * m
        features[f"visible_line_count_{tag}"] = visible_line_count * m

    return pd.DataFrame(features, index=raw.index)

FEATURE_GROUPS = [
    {
        "name": "redshifted_absorption_trough_residuals",
        "fn": add_redshifted_absorption_trough_residuals,
        "depends_on": [],
        "description": "Builds redshift-aware absorption and excess residual geometry features from ugriz flux trough and edge-weighted line alignment.",
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