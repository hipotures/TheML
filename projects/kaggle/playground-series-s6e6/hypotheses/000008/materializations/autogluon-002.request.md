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
- Hypothesis ID: 000008
- Source file: autogluon-001.py
- Failed node: 20260622T142509-582c69c9-11
- Run: 20260622T034843-eca160d1

# Execution Error

```text
failed.yaml error:
Process exited with 1

stderr.log:
Traceback (most recent call last):
  File "/home/xai/DEV/TheML/projects/kaggle/playground-series-s6e6/runs/20260622T034843-eca160d1/artifacts/20260622T142509-582c69c9-11/02-code.py", line 697, in main
    transformed = run_feature_groups(
                  ^^^^^^^^^^^^^^^^^^^
  File "/home/xai/DEV/TheML/src/tml/features/groups.py", line 59, in run_feature_groups
    block = group["fn"](raw.copy(), {key: value.copy() for key, value in deps.items()}, aux.copy())
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/xai/DEV/TheML/projects/kaggle/playground-series-s6e6/runs/20260622T034843-eca160d1/artifacts/20260622T142509-582c69c9-11/02-code.py", line 203, in add_survey_manifold_rarity
    joint_train_count = _lookup_counts(all_keys, joint_cols, joint_counts)
                        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/xai/DEV/TheML/projects/kaggle/playground-series-s6e6/runs/20260622T034843-eca160d1/artifacts/20260622T142509-582c69c9-11/02-code.py", line 113, in _lookup_counts
    return pd.Series(idx).map(count_dict).fillna(0.0).to_numpy(dtype=float)
           ^^^^^^^^^^^^^^
  File "/home/xai/DEV/TheML/.venv/lib/python3.12/site-packages/pandas/core/series.py", line 506, in __init__
    raise NotImplementedError(
NotImplementedError: initializing a Series from a MultiIndex is not supported
Traceback (most recent call last):
  File "/home/xai/DEV/TheML/projects/kaggle/playground-series-s6e6/runs/20260622T034843-eca160d1/artifacts/20260622T142509-582c69c9-11/02-code.py", line 821, in <module>
    main()
  File "/home/xai/DEV/TheML/projects/kaggle/playground-series-s6e6/runs/20260622T034843-eca160d1/artifacts/20260622T142509-582c69c9-11/02-code.py", line 697, in main
    transformed = run_feature_groups(
                  ^^^^^^^^^^^^^^^^^^^
  File "/home/xai/DEV/TheML/src/tml/features/groups.py", line 59, in run_feature_groups
    block = group["fn"](raw.copy(), {key: value.copy() for key, value in deps.items()}, aux.copy())
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/xai/DEV/TheML/projects/kaggle/playground-series-s6e6/runs/20260622T034843-eca160d1/artifacts/20260622T142509-582c69c9-11/02-code.py", line 203, in add_survey_manifold_rarity
    joint_train_count = _lookup_counts(all_keys, joint_cols, joint_counts)
                        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/xai/DEV/TheML/projects/kaggle/playground-series-s6e6/runs/20260622T034843-eca160d1/artifacts/20260622T142509-582c69c9-11/02-code.py", line 113, in _lookup_counts
    return pd.Series(idx).map(count_dict).fillna(0.0).to_numpy(dtype=float)
           ^^^^^^^^^^^^^^
  File "/home/xai/DEV/TheML/.venv/lib/python3.12/site-packages/pandas/core/series.py", line 506, in __init__
    raise NotImplementedError(
NotImplementedError: initializing a Series from a MultiIndex is not supported

stdout.log:
AutoGluon materialization: starting feature groups
TheML feature group: start name=survey_manifold_rarity
TheML feature group: failed name=survey_manifold_rarity elapsed_s=1.036 rows=824782 cols=0
```

# Previous Code

```python
import numpy as np
import pandas as pd

_DECILE_QUANTILES = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
_REDSHIFT_BINS = (-np.inf, 0.002, 0.01, 0.05, 0.2, 0.6, 1.2, 2.5, np.inf)
_UNK = "__UNK__"
_TRAIN_ID_CUTOFF = 577347

def _infer_train_mask(raw, aux):
    n = len(raw)
    if n == 0:
        return np.array([], dtype=bool)

    if aux is not None and hasattr(aux, "empty") and not aux.empty and len(aux) == n:
        for col in ("is_train", "is_train_row", "is_train_mask", "train_mask"):
            if col in aux.columns:
                return aux[col].fillna(False).astype(bool).to_numpy()

    if "id" not in raw.columns:
        return np.ones(n, dtype=bool)

    ids = pd.to_numeric(raw["id"], errors="coerce")
    if ids.notna().sum() == 0:
        return np.ones(n, dtype=bool)

    ids_arr = ids.to_numpy()
    min_id = float(np.nanmin(ids_arr))
    max_id = float(np.nanmax(ids_arr))

    if min_id == 0.0:
        if max_id >= _TRAIN_ID_CUTOFF and n > _TRAIN_ID_CUTOFF:
            return ids_arr < _TRAIN_ID_CUTOFF

    if min_id >= _TRAIN_ID_CUTOFF:
        return np.zeros(n, dtype=bool)

    return np.ones(n, dtype=bool)

def _compute_photometric_coords(frame):
    u = pd.to_numeric(frame["u"], errors="coerce").to_numpy(dtype=float)
    g = pd.to_numeric(frame["g"], errors="coerce").to_numpy(dtype=float)
    r = pd.to_numeric(frame["r"], errors="coerce").to_numpy(dtype=float)
    i = pd.to_numeric(frame["i"], errors="coerce").to_numpy(dtype=float)
    z = pd.to_numeric(frame["z"], errors="coerce").to_numpy(dtype=float)

    flux = np.column_stack(
        (
            np.power(10.0, -0.4 * u),
            np.power(10.0, -0.4 * g),
            np.power(10.0, -0.4 * r),
            np.power(10.0, -0.4 * i),
            np.power(10.0, -0.4 * z),
        )
    )

    total_flux = flux.sum(axis=1)
    shares = np.zeros_like(flux, dtype=float)
    valid = total_flux > 0
    np.divide(flux, total_flux[:, None], out=shares, where=valid[:, None])

    blue_balance = (shares[:, 0] + shares[:, 1]) - (shares[:, 3] + shares[:, 4])
    flux_concentration = shares.max(axis=1)
    total_brightness = np.full_like(total_flux, np.nan, dtype=float)
    total_brightness[valid] = -2.5 * np.log10(total_flux[valid])

    return (
        pd.Series(blue_balance, index=frame.index),
        pd.Series(flux_concentration, index=frame.index),
        pd.Series(total_brightness, index=frame.index),
    )

def _compute_decile_edges(values):
    arr = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.array((-np.inf, np.inf), dtype=float)

    edges = np.quantile(arr, _DECILE_QUANTILES)
    edges = np.unique(np.asarray(edges, dtype=float))

    if edges.size <= 1:
        return np.array((-np.inf, np.inf), dtype=float)

    return np.r_[ -np.inf, edges[1:-1], np.inf ]

def _to_bin_code(values, edges):
    codes = pd.cut(
        pd.to_numeric(values, errors="coerce"),
        bins=edges,
        labels=False,
        include_lowest=True,
        right=True,
        duplicates="drop",
    )
    return codes.fillna(-1).astype(int).to_numpy()

def _normalize_category(series, train_levels):
    allowed = set(train_levels)
    s = series.astype("string").fillna(_UNK).astype(str)
    return s.where(s.isin(allowed), _UNK).astype(str)

def _lookup_counts(keys, columns, count_dict):
    idx = pd.MultiIndex.from_frame(keys[columns])
    return pd.Series(idx).map(count_dict).fillna(0.0).to_numpy(dtype=float)

def add_survey_manifold_rarity(raw, deps, aux):
    idx = raw.index
    n_total = len(raw)

    if n_total == 0:
        return pd.DataFrame(
            {
                "joint_log_density": np.array([], dtype=float),
                "rarity_score": np.array([], dtype=float),
                "interaction_surprise": np.array([], dtype=float),
            },
            index=idx,
        )

    train_mask = _infer_train_mask(raw, aux)
    if len(train_mask) != n_total:
        train_mask = np.ones(n_total, dtype=bool)

    train_df = raw.loc[train_mask]
    if train_df.empty:
        train_df = raw
        train_mask = np.ones(n_total, dtype=bool)

    train_blue, train_conc, train_bright = _compute_photometric_coords(train_df)
    blue_edges = _compute_decile_edges(train_blue)
    conc_edges = _compute_decile_edges(train_conc)
    bright_edges = _compute_decile_edges(train_bright)

    raw_blue, raw_conc, raw_bright = _compute_photometric_coords(raw)
    raw_blue_bin = _to_bin_code(raw_blue, blue_edges)
    raw_conc_bin = _to_bin_code(raw_conc, conc_edges)
    raw_bright_bin = _to_bin_code(raw_bright, bright_edges)

    train_blue_bin = _to_bin_code(train_blue, blue_edges)
    train_conc_bin = _to_bin_code(train_conc, conc_edges)
    train_bright_bin = _to_bin_code(train_bright, bright_edges)

    raw_redshift_bin = _to_bin_code(raw["redshift"], _REDSHIFT_BINS)
    train_redshift_bin = _to_bin_code(train_df["redshift"], _REDSHIFT_BINS)

    train_spec_levels = sorted(_normalize_category(train_df["spectral_type"], []).unique())
    train_gal_levels = sorted(_normalize_category(train_df["galaxy_population"], []).unique())

    raw_spectral = _normalize_category(raw["spectral_type"], train_spec_levels)
    raw_galaxy = _normalize_category(raw["galaxy_population"], train_gal_levels)
    train_spectral = _normalize_category(train_df["spectral_type"], train_spec_levels)
    train_galaxy = _normalize_category(train_df["galaxy_population"], train_gal_levels)

    train_keys = pd.DataFrame(
        {
            "redshift_bin": train_redshift_bin,
            "spectral_type": train_spectral,
            "galaxy_population": train_galaxy,
            "blue_bin": train_blue_bin,
            "flux_concentration_bin": train_conc_bin,
            "total_brightness_bin": train_bright_bin,
        },
        index=train_df.index,
    )

    all_keys = pd.DataFrame(
        {
            "redshift_bin": raw_redshift_bin,
            "spectral_type": raw_spectral,
            "galaxy_population": raw_galaxy,
            "blue_bin": raw_blue_bin,
            "flux_concentration_bin": raw_conc_bin,
            "total_brightness_bin": raw_bright_bin,
        },
        index=idx,
    )

    joint_cols = [
        "redshift_bin",
        "spectral_type",
        "galaxy_population",
        "blue_bin",
        "flux_concentration_bin",
        "total_brightness_bin",
    ]
    tag_cols = ["redshift_bin", "spectral_type", "galaxy_population"]
    photo_cols = ["blue_bin", "flux_concentration_bin", "total_brightness_bin"]

    joint_counts = train_keys[joint_cols].value_counts().to_dict()
    tag_counts = train_keys[tag_cols].value_counts().to_dict()
    photo_counts = train_keys[photo_cols].value_counts().to_dict()

    joint_train_count = _lookup_counts(all_keys, joint_cols, joint_counts)
    tag_count = _lookup_counts(all_keys, tag_cols, tag_counts)
    photo_count = _lookup_counts(all_keys, photo_cols, photo_counts)

    n_train = float(len(train_df))

    redshift_bins = len(_REDSHIFT_BINS) - 1
    blue_bins = max(len(blue_edges) - 1, 1)
    conc_bins = max(len(conc_edges) - 1, 1)
    bright_bins = max(len(bright_edges) - 1, 1)
    spectral_cells = len(train_spec_levels) + 1  # include UNK
    galaxy_cells = len(train_gal_levels) + 1      # include UNK

    k_joint = redshift_bins * spectral_cells * galaxy_cells * blue_bins * conc_bins * bright_bins

    joint_log_density = np.log((joint_train_count + 1.0) / (n_train + k_joint))
    rarity_score = -joint_log_density
    interaction_surprise = np.log((tag_count + 1.0) * (photo_count + 1.0)) - np.log((joint_train_count + 1.0) * n_train)

    return pd.DataFrame(
        {
            "joint_log_density": joint_log_density,
            "rarity_score": rarity_score,
            "interaction_surprise": interaction_surprise,
        },
        index=idx,
    )

FEATURE_GROUPS = [
    {
        "name": "survey_manifold_rarity",
        "fn": add_survey_manifold_rarity,
        "depends_on": [],
        "description": "Encodes manifold-based rarity and surprise from redshift-tag-photo joint frequency, with Laplace-smoothed density estimates.",
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