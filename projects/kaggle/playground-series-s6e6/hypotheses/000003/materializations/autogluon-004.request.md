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
- Hypothesis ID: 000003
- Source file: autogluon-003.py
- Failed node: 20260624T054210-793bb2f3-308
- Run: 20260622T034843-eca160d1

# Execution Error

```text
failed.yaml error:
Process exited with 1

stderr.log:
Traceback (most recent call last):
  File "/home/xai/DEV/TheML/.worktrees/root-hypothesis-revisions/projects/kaggle/playground-series-s6e6/runs/20260622T034843-eca160d1/artifacts/20260624T054210-793bb2f3-308/02-code.py", line 622, in main
    transformed = run_feature_groups(
                  ^^^^^^^^^^^^^^^^^^^
  File "/home/xai/DEV/TheML/.worktrees/root-hypothesis-revisions/src/tml/features/groups.py", line 59, in run_feature_groups
    block = group["fn"](raw.copy(), {key: value.copy() for key, value in deps.items()}, aux.copy())
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/xai/DEV/TheML/.worktrees/root-hypothesis-revisions/projects/kaggle/playground-series-s6e6/runs/20260622T034843-eca160d1/artifacts/20260624T054210-793bb2f3-308/02-code.py", line 61, in add_galactic_sightline_context
    needs_fallback = np.isfinite(l_formula_rad) == False | np.isfinite(b_formula_rad) == False
                     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
ValueError: The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()
Traceback (most recent call last):
  File "/home/xai/DEV/TheML/.worktrees/root-hypothesis-revisions/projects/kaggle/playground-series-s6e6/runs/20260622T034843-eca160d1/artifacts/20260624T054210-793bb2f3-308/02-code.py", line 746, in <module>
    main()
  File "/home/xai/DEV/TheML/.worktrees/root-hypothesis-revisions/projects/kaggle/playground-series-s6e6/runs/20260622T034843-eca160d1/artifacts/20260624T054210-793bb2f3-308/02-code.py", line 622, in main
    transformed = run_feature_groups(
                  ^^^^^^^^^^^^^^^^^^^
  File "/home/xai/DEV/TheML/.worktrees/root-hypothesis-revisions/src/tml/features/groups.py", line 59, in run_feature_groups
    block = group["fn"](raw.copy(), {key: value.copy() for key, value in deps.items()}, aux.copy())
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/xai/DEV/TheML/.worktrees/root-hypothesis-revisions/projects/kaggle/playground-series-s6e6/runs/20260622T034843-eca160d1/artifacts/20260624T054210-793bb2f3-308/02-code.py", line 61, in add_galactic_sightline_context
    needs_fallback = np.isfinite(l_formula_rad) == False | np.isfinite(b_formula_rad) == False
                     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
ValueError: The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()

stdout.log:
AutoGluon materialization: loaded aux file star_classification.csv rows=100000 cols=18 passed_to_groups=True
AutoGluon materialization: starting feature groups
TheML feature group: start name=galactic_sightline_context
TheML feature group: failed name=galactic_sightline_context elapsed_s=0.096 rows=824782 cols=0
```

# Previous Code

```python
import numpy as np
import pandas as pd

ALPHA_NGP_DEG = 192.85948
DELTA_NGP_DEG = 27.12825
L_ASC_DEG = 122.93192
TWO_PI = 2.0 * np.pi

GAL_EQ2GAL_MATRIX = (
    (-0.0548755604, -0.8734370902, -0.4838350155),
    (0.4941094279, -0.4448296300, 0.7469822445),
    (-0.8676661490, -0.1980763734, 0.4559837762),
)

B_ABS_BAND_EDGES = (5.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 75.0, 90.0)
L_SECTOR_EDGES = (30.0, 60.0, 90.0, 120.0, 150.0, 180.0, 210.0, 240.0, 270.0, 300.0, 330.0, 360.0)

def add_galactic_sightline_context(raw, deps, aux):
    alpha_rad = np.deg2rad(np.mod(pd.to_numeric(raw["alpha"], errors="coerce").to_numpy(dtype=float), 360.0))
    delta_rad = np.deg2rad(pd.to_numeric(raw["delta"], errors="coerce").to_numpy(dtype=float))

    alpha_ngp_rad = np.deg2rad(ALPHA_NGP_DEG)
    delta_ngp_rad = np.deg2rad(DELTA_NGP_DEG)
    l_asc_rad = np.deg2rad(L_ASC_DEG)

    d_alpha = alpha_rad - alpha_ngp_rad
    sin_delta = np.sin(delta_rad)
    cos_delta = np.cos(delta_rad)
    sin_delta_ngp = np.sin(delta_ngp_rad)
    cos_delta_ngp = np.cos(delta_ngp_rad)

    sinb_formula = sin_delta * sin_delta_ngp + cos_delta * cos_delta_ngp * np.cos(d_alpha)
    sinb_formula = np.clip(sinb_formula, -1.0, 1.0)
    b_formula_rad = np.arcsin(sinb_formula)

    l_formula_rad = (
        l_asc_rad
        - np.arctan2(
            cos_delta * np.sin(d_alpha),
            cos_delta_ngp * sin_delta - sin_delta_ngp * cos_delta * np.cos(d_alpha),
        )
    ) % TWO_PI

    # Parallel unit-vector path (equatorial -> galactic Cartesian, then back to l,b).
    eq_x = cos_delta * np.cos(alpha_rad)
    eq_y = cos_delta * np.sin(alpha_rad)
    eq_z = sin_delta

    rot = np.array(GAL_EQ2GAL_MATRIX, dtype=float)
    gal_x = rot[0][0] * eq_x + rot[0][1] * eq_y + rot[0][2] * eq_z
    gal_y = rot[1][0] * eq_x + rot[1][1] * eq_y + rot[1][2] * eq_z
    gal_z = rot[2][0] * eq_x + rot[2][1] * eq_y + rot[2][2] * eq_z

    b_vec_rad = np.arcsin(np.clip(gal_z, -1.0, 1.0))
    l_vec_rad = np.mod(np.arctan2(gal_y, gal_x), TWO_PI)

    needs_fallback = np.isfinite(l_formula_rad) == False | np.isfinite(b_formula_rad) == False
    near_pole = np.abs(np.cos(b_formula_rad)) < 1e-10
    use_vec = needs_fallback | near_pole

    b_rad = np.where(use_vec, b_vec_rad, b_formula_rad)
    l_rad = np.where(use_vec, l_vec_rad, l_formula_rad)

    l_deg = np.mod(np.degrees(l_rad), 360.0)
    l_deg = np.where(np.isclose(l_deg, 360.0), 0.0, l_deg)
    b_deg = np.degrees(b_rad)
    b_abs_deg = np.abs(b_deg)

    b_sign = np.where(b_deg >= 0.0, 1, -1).astype(np.int8)
    b_sign_label = pd.Categorical(np.where(b_sign > 0, "north", "south"), categories=("south", "north"))

    sin_b = np.sin(b_rad)
    cos_b = np.cos(b_rad)
    sin_l = np.sin(l_rad)
    cos_l = np.cos(l_rad)

    cosb_cosl = np.clip(np.cos(b_rad) * np.cos(l_rad), -1.0, 1.0)
    d_gc_deg = np.degrees(np.arccos(cosb_cosl))
    d_ac_deg = np.degrees(np.arccos(np.clip(-cosb_cosl, -1.0, 1.0)))
    d_np_deg = np.degrees(np.arccos(np.clip(np.sin(b_rad), -1.0, 1.0)))
    d_sp_deg = np.degrees(np.arccos(np.clip(-np.sin(b_rad), -1.0, 1.0)))

    band_edges = np.array(B_ABS_BAND_EDGES, dtype=float)
    sector_edges = np.array(L_SECTOR_EDGES, dtype=float)

    b_band_id = np.searchsorted(band_edges, b_abs_deg, side="right")
    b_band_id = np.clip(b_band_id, 0, len(band_edges) - 1).astype(np.int16)

    l_sector_id = np.searchsorted(sector_edges, l_deg, side="right")
    l_sector_id = np.where(l_sector_id == sector_edges.size, 0, l_sector_id).astype(np.int16)

    band_sector_id = (b_band_id * sector_edges.size + l_sector_id).astype(np.int16)
    sign_x_band_id = (b_sign * (b_band_id + 1)).astype(np.int16)
    sign_x_sector_id = (b_sign * (l_sector_id + 1)).astype(np.int16)
    sign_x_band_label = pd.Categorical(
        np.where(b_sign > 0, "N_" + b_band_id.astype(str), "S_" + b_band_id.astype(str))
    )
    sign_x_sector_label = pd.Categorical(
        np.where(b_sign > 0, "N_" + l_sector_id.astype(str), "S_" + l_sector_id.astype(str))
    )

    return pd.DataFrame(
        {
            "gal_l_deg": l_deg,
            "gal_b_deg": b_deg,
            "gal_b_abs_deg": b_abs_deg,
            "gal_b_sign": b_sign,
            "gal_b_sign_label": b_sign_label,
            "gal_sin_b": sin_b,
            "gal_cos_b": cos_b,
            "gal_sin_l": sin_l,
            "gal_cos_l": cos_l,
            "gal_d_plane_deg": b_abs_deg,
            "gal_d_gc_deg": d_gc_deg,
            "gal_d_ac_deg": d_ac_deg,
            "gal_d_np_deg": d_np_deg,
            "gal_d_sp_deg": d_sp_deg,
            "gal_b_band_id": b_band_id,
            "gal_l_sector_id": l_sector_id,
            "gal_band_sector_id": band_sector_id,
            "gal_sign_x_band_id": sign_x_band_id,
            "gal_sign_x_sector_id": sign_x_sector_id,
            "gal_sign_x_band_label": sign_x_band_label,
            "gal_sign_x_sector_label": sign_x_sector_label,
        },
        index=raw.index,
    )

FEATURE_GROUPS = [
    {
        "name": "galactic_sightline_context",
        "fn": add_galactic_sightline_context,
        "depends_on": [],
        "description": "Compute robust Galactic-sky morphology descriptors from alpha/delta, including anisotropy-relevant angular distances, band/sector bins, and sign-aware interaction encodings.",
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