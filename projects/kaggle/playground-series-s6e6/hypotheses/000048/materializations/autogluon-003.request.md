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
- Hypothesis ID: 000048
- Source file: autogluon-002.py
- Failed node: 20260624T063119-ac19a39c-357
- Run: 20260622T034843-eca160d1

# Execution Error

```text
failed.yaml error:
Process exited with 1

stderr.log:
Traceback (most recent call last):
  File "/home/xai/DEV/TheML/.worktrees/root-hypothesis-revisions/projects/kaggle/playground-series-s6e6/runs/20260622T034843-eca160d1/artifacts/20260624T063119-ac19a39c-357/02-code.py", line 658, in main
    transformed = run_feature_groups(
                  ^^^^^^^^^^^^^^^^^^^
  File "/home/xai/DEV/TheML/.worktrees/root-hypothesis-revisions/src/tml/features/groups.py", line 59, in run_feature_groups
    block = group["fn"](raw.copy(), {key: value.copy() for key, value in deps.items()}, aux.copy())
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/xai/DEV/TheML/.worktrees/root-hypothesis-revisions/projects/kaggle/playground-series-s6e6/runs/20260622T034843-eca160d1/artifacts/20260624T063119-ac19a39c-357/02-code.py", line 86, in add_luptitude_regime_flux_features
    beta = np.linalg.solve(A, rhs)
           ^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/xai/DEV/TheML/.worktrees/root-hypothesis-revisions/.venv/lib/python3.12/site-packages/numpy/linalg/_linalg.py", line 413, in solve
    r = gufunc(a, b, signature=signature)
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
ValueError: solve: Input operand 1 has a mismatch in its core dimension 0, with gufunc signature (m,m),(m,n)->(m,n) (size 824782 is different from 3)
Traceback (most recent call last):
  File "/home/xai/DEV/TheML/.worktrees/root-hypothesis-revisions/projects/kaggle/playground-series-s6e6/runs/20260622T034843-eca160d1/artifacts/20260624T063119-ac19a39c-357/02-code.py", line 782, in <module>
    main()
  File "/home/xai/DEV/TheML/.worktrees/root-hypothesis-revisions/projects/kaggle/playground-series-s6e6/runs/20260622T034843-eca160d1/artifacts/20260624T063119-ac19a39c-357/02-code.py", line 658, in main
    transformed = run_feature_groups(
                  ^^^^^^^^^^^^^^^^^^^
  File "/home/xai/DEV/TheML/.worktrees/root-hypothesis-revisions/src/tml/features/groups.py", line 59, in run_feature_groups
    block = group["fn"](raw.copy(), {key: value.copy() for key, value in deps.items()}, aux.copy())
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/xai/DEV/TheML/.worktrees/root-hypothesis-revisions/projects/kaggle/playground-series-s6e6/runs/20260622T034843-eca160d1/artifacts/20260624T063119-ac19a39c-357/02-code.py", line 86, in add_luptitude_regime_flux_features
    beta = np.linalg.solve(A, rhs)
           ^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/xai/DEV/TheML/.worktrees/root-hypothesis-revisions/.venv/lib/python3.12/site-packages/numpy/linalg/_linalg.py", line 413, in solve
    r = gufunc(a, b, signature=signature)
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
ValueError: solve: Input operand 1 has a mismatch in its core dimension 0, with gufunc signature (m,m),(m,n)->(m,n) (size 824782 is different from 3)

stdout.log:
AutoGluon materialization: loaded aux file star_classification.csv rows=100000 cols=18 passed_to_groups=True
AutoGluon materialization: starting feature groups
TheML feature group: start name=luptitude_regime_flux_features
TheML feature group: failed name=luptitude_regime_flux_features elapsed_s=0.316 rows=824782 cols=0
```

# Previous Code

```python
import numpy as np
import pandas as pd

BANDS = ("u", "g", "r", "i", "z")
LUPTITUDE_B = (1.4e-10, 9.0e-11, 1.2e-10, 1.8e-10, 7.4e-10)
MAGNITUDE_CORRECTIONS = (-0.04, 0.0, 0.0, 0.0, 0.02)
WAVELENGTH_A = (3551.0, 4686.0, 6165.0, 7481.0, 8931.0)
SOFT_COEFF = -0.9210340371976183
EPS = 1e-12

def add_luptitude_regime_flux_features(raw, deps, aux):
    u = raw["u"].to_numpy(dtype=np.float64)
    g = raw["g"].to_numpy(dtype=np.float64)
    r = raw["r"].to_numpy(dtype=np.float64)
    i = raw["i"].to_numpy(dtype=np.float64)
    z = raw["z"].to_numpy(dtype=np.float64)

    m_u = u + MAGNITUDE_CORRECTIONS[0]
    m_g = g + MAGNITUDE_CORRECTIONS[1]
    m_r = r + MAGNITUDE_CORRECTIONS[2]
    m_i = i + MAGNITUDE_CORRECTIONS[3]
    m_z = z + MAGNITUDE_CORRECTIONS[4]

    m_corr = np.column_stack((m_u, m_g, m_r, m_i, m_z))
    b = np.asarray(LUPTITUDE_B, dtype=np.float64)
    log_b = np.log(b)

    f = 2.0 * b * np.sinh(SOFT_COEFF * m_corr - log_b)
    abs_f = np.abs(f)

    floor1 = (abs_f <= (2.0 * b)).astype(np.float64)
    floor2 = (abs_f <= (0.2 * b)).astype(np.float64)

    soft_count = np.sum(floor1, axis=1)
    soft_frac = soft_count / 5.0
    ultra_floor_count = np.sum(floor2, axis=1)
    neg_flux_count = np.sum(f < 0.0, axis=1).astype(np.float64)

    f_tilde = np.sign(f) * np.maximum(abs_f, b)
    v = np.log10(np.abs(f_tilde) + EPS)

    v_u = v[:, 0]
    v_g = v[:, 1]
    v_r = v[:, 2]
    v_i = v[:, 3]
    v_z = v[:, 4]

    d_ug = v_u - v_g
    d_gr = v_g - v_r
    d_ri = v_r - v_i
    d_iz = v_i - v_z

    t = np.asarray(WAVELENGTH_A, dtype=np.float64)
    t2 = t * t
    t3 = t2 * t
    t4 = t2 * t2

    w = 1.0 + 0.5 * floor1

    S0 = np.sum(w, axis=1)
    S1 = np.sum(w * t, axis=1)
    S2 = np.sum(w * t2, axis=1)
    S3 = np.sum(w * t3, axis=1)
    S4 = np.sum(w * t4, axis=1)

    rhs0 = np.sum(w * v, axis=1)
    rhs1 = np.sum(w * (t * v), axis=1)
    rhs2 = np.sum(w * (t2 * v), axis=1)

    A = np.empty((m_corr.shape[0], 3, 3), dtype=np.float64)
    A[:, 0, 0] = S0
    A[:, 0, 1] = S1
    A[:, 0, 2] = S2
    A[:, 1, 0] = S1
    A[:, 1, 1] = S2
    A[:, 1, 2] = S3
    A[:, 2, 0] = S2
    A[:, 2, 1] = S3
    A[:, 2, 2] = S4

    rhs = np.stack((rhs0, rhs1, rhs2), axis=1)
    beta = np.linalg.solve(A, rhs)
    slope = beta[:, 1]
    curvature = beta[:, 2]

    kappa_ugri = d_ug - 2.0 * d_gr + d_ri
    kappa_griz = d_gr - 2.0 * d_ri + d_iz

    abs_log_f = np.log10(abs_f + EPS)
    abs_log_f_t = np.log10(np.abs(f_tilde) + EPS)

    shares = np.abs(f_tilde) / (np.sum(np.abs(f_tilde), axis=1)[:, None] + EPS)
    max_share = np.max(shares, axis=1)
    entropy_sh = -np.sum(shares * np.log(shares + EPS), axis=1)
    l2_share = np.sum(shares ** 2, axis=1)
    concentration = 1.0 - l2_share

    sign_pos = np.sum(f_tilde > 0.0, axis=1).astype(np.float64)
    sign_neg = np.sum(f_tilde < 0.0, axis=1).astype(np.float64)
    p_pos = (sign_pos + EPS) / (5.0 + 2.0 * EPS)
    p_neg = (sign_neg + EPS) / (5.0 + 2.0 * EPS)
    sign_entropy = -(p_pos * np.log(p_pos + EPS) + p_neg * np.log(p_neg + EPS))

    features = {}

    for idx, band in enumerate(BANDS):
        features[f"floor1_{band}"] = floor1[:, idx]
        features[f"floor2_{band}"] = floor2[:, idx]

    features["soft_count"] = soft_count
    features["soft_frac"] = soft_frac
    features["soft_regime_weight"] = soft_frac
    features["ultra_floor_count"] = ultra_floor_count
    features["neg_flux_count"] = neg_flux_count

    features["lupt_v_u"] = v_u
    features["lupt_v_g"] = v_g
    features["lupt_v_r"] = v_r
    features["lupt_v_i"] = v_i
    features["lupt_v_z"] = v_z

    features["dlog_flux_ug"] = d_ug
    features["dlog_flux_gr"] = d_gr
    features["dlog_flux_ri"] = d_ri
    features["dlog_flux_iz"] = d_iz

    features["slope_weighted_poly2"] = slope
    features["curvature_weighted_poly2"] = curvature
    features["slope_hi"] = slope * (1.0 - soft_frac)
    features["slope_lo"] = slope * soft_frac
    features["curvature_hi"] = curvature * (1.0 - soft_frac)
    features["curvature_lo"] = curvature * soft_frac

    features["curvature_kappa_ugri"] = kappa_ugri
    features["curvature_kappa_griz"] = kappa_griz

    features["shape_share_max"] = max_share
    features["shape_share_entropy"] = entropy_sh
    features["shape_share_l2"] = l2_share
    features["shape_share_concentration"] = concentration
    features["shape_sign_entropy"] = sign_entropy

    pair_names = ("ug", "gr", "ri", "iz")
    pair_idx = ((0, 1), (1, 2), (2, 3), (3, 4))

    for name, (a, b_idx) in zip(pair_names, pair_idx):
        mag_color = m_corr[:, a] - m_corr[:, b_idx]
        flux_color = 2.5 * (abs_log_f[:, a] - abs_log_f[:, b_idx])
        flux_color_t = 2.5 * (abs_log_f_t[:, a] - abs_log_f_t[:, b_idx])

        mismatch = mag_color - flux_color
        mismatch_t = mag_color - flux_color_t

        features[f"mag_color_{name}"] = mag_color
        features[f"flux_color_{name}"] = flux_color
        features[f"mismatch_{name}"] = mismatch
        features[f"mismatch_tilde_{name}"] = mismatch_t
        features[f"mismatch_hi_{name}"] = mismatch * (1.0 - soft_frac)
        features[f"mismatch_lo_{name}"] = mismatch * soft_frac
        features[f"mismatch_tilde_hi_{name}"] = mismatch_t * (1.0 - soft_frac)
        features[f"mismatch_tilde_lo_{name}"] = mismatch_t * soft_frac

    return pd.DataFrame(features, index=raw.index)

FEATURE_GROUPS = [
    {
        "name": "luptitude_regime_flux_features",
        "fn": add_luptitude_regime_flux_features,
        "depends_on": [],
        "description": "Build luptitude-derived flux-shape, floor-regime, and mismatch descriptors with regime gating for low-SNR behavior.",
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