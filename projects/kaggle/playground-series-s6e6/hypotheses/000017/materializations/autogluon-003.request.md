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
- Hypothesis ID: 000017
- Source file: autogluon-002.py
- Failed node: 20260624T055839-6aa59352-323
- Run: 20260622T034843-eca160d1

# Execution Error

```text
failed.yaml error:
Process exited with 1

stderr.log:
Traceback (most recent call last):
  File "/home/xai/DEV/TheML/.worktrees/root-hypothesis-revisions/projects/kaggle/playground-series-s6e6/runs/20260622T034843-eca160d1/artifacts/20260624T055839-6aa59352-323/02-code.py", line 613, in main
    transformed = run_feature_groups(
                  ^^^^^^^^^^^^^^^^^^^
  File "/home/xai/DEV/TheML/.worktrees/root-hypothesis-revisions/src/tml/features/groups.py", line 59, in run_feature_groups
    block = group["fn"](raw.copy(), {key: value.copy() for key, value in deps.items()}, aux.copy())
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/xai/DEV/TheML/.worktrees/root-hypothesis-revisions/projects/kaggle/playground-series-s6e6/runs/20260622T034843-eca160d1/artifacts/20260624T055839-6aa59352-323/02-code.py", line 62, in add_redshift_luminosity_plausibility
    abs_mag = mags_obs - (5.0 * np.log10(d_l_mpc) + 25.0)
              ~~~~~~~~~^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
ValueError: operands could not be broadcast together with shapes (824782,5) (824782,) 
Traceback (most recent call last):
  File "/home/xai/DEV/TheML/.worktrees/root-hypothesis-revisions/projects/kaggle/playground-series-s6e6/runs/20260622T034843-eca160d1/artifacts/20260624T055839-6aa59352-323/02-code.py", line 737, in <module>
    main()
  File "/home/xai/DEV/TheML/.worktrees/root-hypothesis-revisions/projects/kaggle/playground-series-s6e6/runs/20260622T034843-eca160d1/artifacts/20260624T055839-6aa59352-323/02-code.py", line 613, in main
    transformed = run_feature_groups(
                  ^^^^^^^^^^^^^^^^^^^
  File "/home/xai/DEV/TheML/.worktrees/root-hypothesis-revisions/src/tml/features/groups.py", line 59, in run_feature_groups
    block = group["fn"](raw.copy(), {key: value.copy() for key, value in deps.items()}, aux.copy())
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/xai/DEV/TheML/.worktrees/root-hypothesis-revisions/projects/kaggle/playground-series-s6e6/runs/20260622T034843-eca160d1/artifacts/20260624T055839-6aa59352-323/02-code.py", line 62, in add_redshift_luminosity_plausibility
    abs_mag = mags_obs - (5.0 * np.log10(d_l_mpc) + 25.0)
              ~~~~~~~~~^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
ValueError: operands could not be broadcast together with shapes (824782,5) (824782,)

stdout.log:
AutoGluon materialization: loaded aux file star_classification.csv rows=100000 cols=18 passed_to_groups=True
AutoGluon materialization: starting feature groups
TheML feature group: start name=redshift_luminosity_plausibility
TheML feature group: failed name=redshift_luminosity_plausibility elapsed_s=0.075 rows=824782 cols=0
```

# Previous Code

```python
import numpy as np
import pandas as pd

_BANDS = ("u", "g", "r", "i", "z")
_ZEFF_FLOOR = 1e-4
_M_MIN = -35.0
_M_MAX = 10.0
_LUM_DIST_GRID_STEPS = 4000
_LUM_DIST_MAX_Z = 10.0
_LIGHT_SPEED_KM_S = 299792.458
_H0_KM_S_MPC = 70.0
_OMEGA_M = 0.30
_OMEGA_L = 0.70

def _luminosity_distance_mpc(redshift):
    z = np.asarray(redshift, dtype=np.float64)
    z_eff = np.maximum(z, _ZEFF_FLOOR)

    z_grid = np.linspace(0.0, _LUM_DIST_MAX_Z, _LUM_DIST_GRID_STEPS + 1)
    e_inv = 1.0 / np.sqrt(_OMEGA_M * (1.0 + z_grid) ** 3 + _OMEGA_L)

    dz = z_grid[1] - z_grid[0]
    integral_grid = np.zeros_like(z_grid)
    integral_grid[1:] = np.cumsum((e_inv[:-1] + e_inv[1:]) * (0.5 * dz))

    z_lookup = np.minimum(z_eff, _LUM_DIST_MAX_Z)
    idx = np.searchsorted(z_grid, z_lookup, side="right") - 1
    idx = np.clip(idx, 0, _LUM_DIST_GRID_STEPS - 1)

    z0 = z_grid[idx]
    e0 = e_inv[idx]
    z1 = z_grid[idx + 1]
    e1 = e_inv[idx + 1]
    frac = (z_lookup - z0) / (z1 - z0)
    e_at_z = e0 + (e1 - e0) * frac

    integral = integral_grid[idx] + 0.5 * (e0 + e_at_z) * (z_lookup - z0)

    over_mask = z_eff > _LUM_DIST_MAX_Z
    if np.any(over_mask):
        z_tail = z_eff[over_mask]
        tail = (2.0 / np.sqrt(_OMEGA_M)) * (
            (1.0 + _LUM_DIST_MAX_Z) ** -0.5 - (1.0 + z_tail) ** -0.5
        )
        integral = integral.copy()
        integral[over_mask] = integral[over_mask] + tail

    d_c_mpc = (_LIGHT_SPEED_KM_S / _H0_KM_S_MPC) * integral
    d_l_mpc = (1.0 + z_eff) * d_c_mpc
    return d_l_mpc, z_eff

def add_redshift_luminosity_plausibility(raw, deps, aux):
    redshift = raw["redshift"].to_numpy(dtype=np.float64)
    d_l_mpc, z_eff = _luminosity_distance_mpc(redshift)

    mags_obs = raw.loc[:, list(_BANDS)].to_numpy(dtype=np.float64)
    abs_mag = mags_obs - (5.0 * np.log10(d_l_mpc) + 25.0)
    abs_mag = np.clip(abs_mag, _M_MIN, _M_MAX)

    m_u = abs_mag[:, 0]
    m_g = abs_mag[:, 1]
    m_r = abs_mag[:, 2]
    m_i = abs_mag[:, 3]
    m_z = abs_mag[:, 4]

    m_mean = np.nanmean(abs_mag, axis=1)
    m_median = np.nanmedian(abs_mag, axis=1)
    m_iqr = np.nanpercentile(abs_mag, 75, axis=1) - np.nanpercentile(abs_mag, 25, axis=1)
    m_sd = np.nanstd(abs_mag, axis=1)
    m_min = np.nanmin(abs_mag, axis=1)
    m_max = np.nanmax(abs_mag, axis=1)

    m_rg = m_r - m_g
    m_ri = m_r - m_i
    m_spread = m_max - m_min

    delta_star = m_r + 10.0
    delta_midgal = m_r + 18.0
    delta_brightgal = m_r + 23.0
    delta_qso = m_r + 27.0

    features = pd.DataFrame(
        {
            "z_nonpos": redshift <= 0.0,
            "z_low": redshift < 0.01,
            "z_hi": redshift >= 3.0,
            "z_eff": z_eff,
            "M_u": m_u,
            "M_g": m_g,
            "M_r": m_r,
            "M_i": m_i,
            "M_z": m_z,
            "M_mean": m_mean,
            "M_median": m_median,
            "M_iqr": m_iqr,
            "M_sd": m_sd,
            "M_min": m_min,
            "M_max": m_max,
            "M_spread": m_spread,
            "M_rg": m_rg,
            "M_ri": m_ri,
            "delta_star": delta_star,
            "delta_midgal": delta_midgal,
            "delta_brightgal": delta_brightgal,
            "delta_qso": delta_qso,
            "Mr_le_minus27": m_r <= -27.0,
            "Mr_between_minus27_and_minus23": (m_r > -27.0) & (m_r <= -23.0),
            "Mr_between_minus23_and_minus18": (m_r > -23.0) & (m_r <= -18.0),
            "Mr_between_minus18_and_minus10": (m_r > -18.0) & (m_r <= -10.0),
            "Mr_gt_minus10": m_r > -10.0,
            "Mr_x_z_eff": m_r * z_eff,
            "Mri_x_z_eff": m_ri * z_eff,
        },
        index=raw.index,
    )

    return features

FEATURE_GROUPS = [
    {
        "name": "redshift_luminosity_plausibility",
        "fn": add_redshift_luminosity_plausibility,
        "depends_on": [],
        "description": "Compute redshift-normalized absolute ugriz magnitudes with deterministic cosmology-based scaling and summarize cross-band absolute-photometry structure for luminosity-regime plausibility.",
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