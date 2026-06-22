Fix one failed ROOT materialization.

Return a corrected version of the same feature-group module. Do not invent a
new hypothesis or change the feature family. Keep the fix narrow: address the
observed failure while preserving the intended preprocessing behavior.

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
- Source file: autogluon-001.py
- Failed node: 20260622T142156-b4b91f1e-7
- Run: 20260622T034843-eca160d1

# Execution Error

```text
failed.yaml error:
Process exited with 1

stderr.log:
Traceback (most recent call last):
  File "/home/xai/DEV/TheML/projects/kaggle/playground-series-s6e6/runs/20260622T034843-eca160d1/artifacts/20260622T142156-b4b91f1e-7/02-code.py", line 566, in main
    transformed = run_feature_groups(
                  ^^^^^^^^^^^^^^^^^^^
  File "/home/xai/DEV/TheML/src/tml/features/groups.py", line 59, in run_feature_groups
    block = group["fn"](raw.copy(), {key: value.copy() for key, value in deps.items()}, aux.copy())
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/xai/DEV/TheML/projects/kaggle/playground-series-s6e6/runs/20260622T034843-eca160d1/artifacts/20260622T142156-b4b91f1e-7/02-code.py", line 74, in add_galactic_sightline_context
    crossed_regime = b_band.astype("string").str.cat(l_sector.astype("string"), sep="|")
                     ^^^^^^^^^^^^^^^^^^^^^^^^^^^
AttributeError: 'StringArray' object has no attribute 'str'. Did you mean: 'std'?
Traceback (most recent call last):
  File "/home/xai/DEV/TheML/projects/kaggle/playground-series-s6e6/runs/20260622T034843-eca160d1/artifacts/20260622T142156-b4b91f1e-7/02-code.py", line 690, in <module>
    main()
  File "/home/xai/DEV/TheML/projects/kaggle/playground-series-s6e6/runs/20260622T034843-eca160d1/artifacts/20260622T142156-b4b91f1e-7/02-code.py", line 566, in main
    transformed = run_feature_groups(
                  ^^^^^^^^^^^^^^^^^^^
  File "/home/xai/DEV/TheML/src/tml/features/groups.py", line 59, in run_feature_groups
    block = group["fn"](raw.copy(), {key: value.copy() for key, value in deps.items()}, aux.copy())
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/xai/DEV/TheML/projects/kaggle/playground-series-s6e6/runs/20260622T034843-eca160d1/artifacts/20260622T142156-b4b91f1e-7/02-code.py", line 74, in add_galactic_sightline_context
    crossed_regime = b_band.astype("string").str.cat(l_sector.astype("string"), sep="|")
                     ^^^^^^^^^^^^^^^^^^^^^^^^^^^
AttributeError: 'StringArray' object has no attribute 'str'. Did you mean: 'std'?

stdout.log:
AutoGluon materialization: starting feature groups
TheML feature group: start name=galactic_sightline_context
TheML feature group: failed name=galactic_sightline_context elapsed_s=0.918 rows=824782 cols=0
```

# Previous Code

```python
import numpy as np
import pandas as pd

def add_galactic_sightline_context(raw, deps, aux):
    alpha_rad = np.deg2rad((pd.to_numeric(raw["alpha"], errors="coerce") % 360.0).to_numpy(dtype=float))
    delta_rad = np.deg2rad(pd.to_numeric(raw["delta"], errors="coerce").to_numpy(dtype=float))

    # Equatorial unit-sphere coordinates (wrap-safe via alpha modulo 360)
    cos_delta = np.cos(delta_rad)
    x_eq = cos_delta * np.cos(alpha_rad)
    y_eq = cos_delta * np.sin(alpha_rad)
    z_eq = np.sin(delta_rad)

    # J2000 equatorial -> Galactic rotation matrix
    x_gal = (
        -0.0548755604 * x_eq
        + -0.8734370902 * y_eq
        + -0.4838350155 * z_eq
    )
    y_gal = (
        0.4941094279 * x_eq
        + -0.4448296300 * y_eq
        + 0.7469822445 * z_eq
    )
    z_gal = (
        -0.8676661490 * x_eq
        + -0.1980763734 * y_eq
        + 0.4559837762 * z_eq
    )

    z_clip = np.clip(z_gal, -1.0, 1.0)
    b_rad = np.arcsin(z_clip)
    b_deg = np.degrees(b_rad)
    abs_b_deg = np.abs(b_deg)

    l_rad = np.arctan2(y_gal, x_gal)
    l_deg = np.degrees(l_rad) % 360.0
    l_rad = np.deg2rad(l_deg)

    sin_l = np.sin(l_rad)
    cos_l = np.cos(l_rad)
    sin_b = np.sin(b_rad)
    cos_b = np.cos(b_rad)

    dist_to_galactic_center = np.degrees(np.arccos(np.clip(x_gal, -1.0, 1.0)))
    dist_to_galactic_anticenter = np.degrees(np.arccos(np.clip(-x_gal, -1.0, 1.0)))
    dist_to_north_galactic_pole = np.degrees(np.arccos(np.clip(z_gal, -1.0, 1.0)))
    dist_to_south_galactic_pole = np.degrees(np.arccos(np.clip(-z_gal, -1.0, 1.0)))

    b_band_edges = [0.0, 10.0, 20.0, 35.0, 50.0, 70.0, 90.0000001]
    b_band_labels = ["0-10", "10-20", "20-35", "35-50", "50-70", "70-90"]
    b_band = pd.cut(
        abs_b_deg,
        bins=b_band_edges,
        labels=b_band_labels,
        right=False,
        include_lowest=True,
    )

    l_sector_edges = np.arange(0.0, 390.0, 30.0)
    l_sector_labels = [f"{int(start):03d}-{int(start+30):03d}" for start in l_sector_edges[:-1]]
    l_sector = pd.cut(
        l_deg,
        bins=l_sector_edges,
        labels=l_sector_labels,
        right=False,
        include_lowest=True,
    )

    crossed_regime = b_band.astype("string").str.cat(l_sector.astype("string"), sep="|")

    return pd.DataFrame(
        {
            "eq_x": x_eq,
            "eq_y": y_eq,
            "eq_z": z_eq,
            "gal_l_deg": l_deg,
            "gal_b_deg": b_deg,
            "gal_abs_b_deg": abs_b_deg,
            "gal_sin_l": sin_l,
            "gal_cos_l": cos_l,
            "gal_sin_b": sin_b,
            "gal_cos_b": cos_b,
            "gal_plane_angular_distance_deg": abs_b_deg,
            "gal_dist_to_galactic_center_deg": dist_to_galactic_center,
            "gal_dist_to_galactic_anticenter_deg": dist_to_galactic_anticenter,
            "gal_dist_to_north_pole_deg": dist_to_north_galactic_pole,
            "gal_dist_to_south_pole_deg": dist_to_south_galactic_pole,
            "gal_b_band": b_band.astype("string"),
            "gal_l_sector": l_sector.astype("string"),
            "gal_b_band_l_sector": crossed_regime,
        },
        index=raw.index,
    )

FEATURE_GROUPS = [
    {
        "name": "galactic_sightline_context",
        "fn": add_galactic_sightline_context,
        "depends_on": [],
        "description": "Creates Galactic coordinate-derived spatial features from sky position, including smooth angular encodings, center/pole distances, and coarse latitude/longitude regime bins.",
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