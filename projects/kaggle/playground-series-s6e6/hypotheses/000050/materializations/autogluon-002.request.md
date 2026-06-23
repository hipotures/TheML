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

# External Data Description for /home/xai/DEV/aideml/workspaces/2-liberal-celadon-panther/input/star_classification.csv

Original SDSS17 Stellar Classification Dataset.

This is the original real-world dataset that inspired the synthetic Playground
Series S6E6 competition data. It can be used as raw auxiliary data, but it is
not automatically merged with train.csv or test.csv.

Common columns with the competition data:
alpha, delta, u, g, r, i, z, redshift, class.

Columns present in this original dataset but not in the competition files:
obj_ID, run_ID, rerun_ID, cam_col, field_ID, spec_obj_ID, plate, MJD, fiber_ID.

Competition columns not present in this original dataset:
id, spectral_type, galaxy_population.

Generated code should decide whether and how to use this file. Any merge,
filtering, cleaning of sentinel magnitudes, or column mapping must be done
explicitly by the generated solution code.

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
- Hypothesis ID: 000050
- Source file: autogluon-001.py
- Failed node: 20260623T152642-1f352398-198
- Run: 20260622T034843-eca160d1

# Execution Error

```text
failed.yaml error:
Process exited with 1

stderr.log:
/home/xai/DEV/TheML/projects/kaggle/playground-series-s6e6/runs/20260622T034843-eca160d1/artifacts/20260623T152642-1f352398-198/02-code.py:38: RuntimeWarning: overflow encountered in power
  distance_pc = np.power(10.0, (mu + 5.0) / 5.0)
Traceback (most recent call last):
  File "/home/xai/DEV/TheML/projects/kaggle/playground-series-s6e6/runs/20260622T034843-eca160d1/artifacts/20260623T152642-1f352398-198/02-code.py", line 769, in main
    transformed = run_feature_groups(
                  ^^^^^^^^^^^^^^^^^^^
  File "/home/xai/DEV/TheML/src/tml/features/groups.py", line 59, in run_feature_groups
    block = group["fn"](raw.copy(), {key: value.copy() for key, value in deps.items()}, aux.copy())
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/xai/DEV/TheML/projects/kaggle/playground-series-s6e6/runs/20260622T034843-eca160d1/artifacts/20260623T152642-1f352398-198/02-code.py", line 192, in add_main_sequence_parallax_plausibility
    stats = _bin_reference_statistics(
            ^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/xai/DEV/TheML/projects/kaggle/playground-series-s6e6/runs/20260622T034843-eca160d1/artifacts/20260623T152642-1f352398-198/02-code.py", line 123, in _bin_reference_statistics
    bin_labels = bin_labels.astype("Int64")
                 ^^^^^^^^^^^^^^^^^^^^^^^^^^
TypeError: data type 'Int64' not understood
Traceback (most recent call last):
  File "/home/xai/DEV/TheML/projects/kaggle/playground-series-s6e6/runs/20260622T034843-eca160d1/artifacts/20260623T152642-1f352398-198/02-code.py", line 893, in <module>
    main()
  File "/home/xai/DEV/TheML/projects/kaggle/playground-series-s6e6/runs/20260622T034843-eca160d1/artifacts/20260623T152642-1f352398-198/02-code.py", line 769, in main
    transformed = run_feature_groups(
                  ^^^^^^^^^^^^^^^^^^^
  File "/home/xai/DEV/TheML/src/tml/features/groups.py", line 59, in run_feature_groups
    block = group["fn"](raw.copy(), {key: value.copy() for key, value in deps.items()}, aux.copy())
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/xai/DEV/TheML/projects/kaggle/playground-series-s6e6/runs/20260622T034843-eca160d1/artifacts/20260623T152642-1f352398-198/02-code.py", line 192, in add_main_sequence_parallax_plausibility
    stats = _bin_reference_statistics(
            ^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/xai/DEV/TheML/projects/kaggle/playground-series-s6e6/runs/20260622T034843-eca160d1/artifacts/20260623T152642-1f352398-198/02-code.py", line 123, in _bin_reference_statistics
    bin_labels = bin_labels.astype("Int64")
                 ^^^^^^^^^^^^^^^^^^^^^^^^^^
TypeError: data type 'Int64' not understood

stdout.log:
AutoGluon materialization: loaded aux file /home/xai/DEV/aideml/workspaces/2-liberal-celadon-panther/input/star_classification.csv rows=100000 cols=18 passed_to_groups=True
AutoGluon materialization: starting feature groups
TheML feature group: start name=main_sequence_parallax_plausibility
TheML feature group: failed name=main_sequence_parallax_plausibility elapsed_s=0.157 rows=824782 cols=0
```

# Previous Code

```python
from __future__ import annotations

import numpy as np
import pandas as pd

_POLY_COEFFS = (-5.06, 14.32, -12.97, 6.127, -1.267, 0.0967)
_X_LOW = 0.2
_X_HIGH = 4.0
_BIN_COUNT = 80
_MIN_BIN_SIZE = 12
_MAD_EPS = 1e-6

def _coerce_photometry(frame):
    if frame is None or not hasattr(frame, "columns"):
        return None
    lookup = {}
    for col in frame.columns:
        lookup[str(col).lower()] = col
    required = ("u", "g", "r", "i", "z")
    data = {}
    for col in required:
        source = lookup.get(col)
        if source is None:
            return None
        data[col] = pd.to_numeric(frame[source], errors="coerce").astype(float)
    return pd.DataFrame(data, index=frame.index)

def _main_sequence_values(photometry):
    x_raw = photometry["g"].to_numpy(dtype=float) - photometry["i"].to_numpy(dtype=float)
    x_clipped = np.clip(x_raw, _X_LOW, _X_HIGH)
    M_r = np.polyval(_POLY_COEFFS, x_clipped)
    mu = photometry["r"].to_numpy(dtype=float) - M_r
    distance_pc = np.power(10.0, (mu + 5.0) / 5.0)
    log10_distance_pc = np.log10(np.maximum(distance_pc, 1.0))
    return {
        "x_raw": x_raw,
        "x_clipped": x_clipped,
        "x_low_clip": x_raw < _X_LOW,
        "x_high_clip": x_raw > _X_HIGH,
        "M_r": M_r,
        "mu": mu,
        "distance_pc": distance_pc,
        "log10_distance_pc": log10_distance_pc,
        "M_u": photometry["u"].to_numpy(dtype=float) - mu,
        "M_g": photometry["g"].to_numpy(dtype=float) - mu,
        "M_i": photometry["i"].to_numpy(dtype=float) - mu,
        "M_z": photometry["z"].to_numpy(dtype=float) - mu,
    }

def _mad(values):
    s = pd.Series(values).dropna()
    if s.empty:
        return np.nan
    med = s.median()
    return np.median(np.abs(s - med))

def _neighbor_fallback_indices(bin_count, min_size):
    n_bins = len(bin_count)
    fallback = np.full(n_bins, -1, dtype=int)
    if n_bins == 0:
        return fallback

    for bin_idx in range(n_bins):
        if bin_count[bin_idx] >= min_size:
            fallback[bin_idx] = bin_idx
            continue

        for step in range(1, n_bins):
            left = bin_idx - step
            if left >= 0 and bin_count[left] >= min_size:
                fallback[bin_idx] = left
                break
            right = bin_idx + step
            if right < n_bins and bin_count[right] >= min_size:
                fallback[bin_idx] = right
                break

    return fallback

def _bin_reference_statistics(x_clipped, mu):
    x = np.asarray(x_clipped, dtype=float)
    mu = np.asarray(mu, dtype=float)

    valid = np.isfinite(x) & np.isfinite(mu)
    if not np.any(valid):
        return None

    x_valid = x[valid]
    mu_valid = mu[valid]

    x_min = float(np.nanmin(x_valid))
    x_max = float(np.nanmax(x_valid))

    if x_valid.size < 2 or x_min == x_max:
        edges = np.array([x_min - 0.5, x_max + 0.5], dtype=float)
    else:
        q = int(min(_BIN_COUNT, x_valid.size))
        q = max(2, q)
        try:
            _, edges = pd.qcut(
                pd.Series(x_valid), q=q, retbins=True, duplicates="drop"
            )
            edges = np.asarray(edges, dtype=float)
            edges = np.unique(edges)
        except (ValueError, TypeError):
            edges = np.array([x_min - 0.5, x_max + 0.5], dtype=float)

        if edges.size < 2 or not np.isfinite(edges).all():
            edges = np.array([x_min - 0.5, x_max + 0.5], dtype=float)

    if edges.size < 2:
        edges = np.array([x_min - 0.5, x_max + 0.5], dtype=float)

    bin_labels = pd.cut(x_valid, bins=edges, include_lowest=True, labels=False)
    bin_labels = bin_labels.astype("Int64")

    xbin = pd.DataFrame({"bin": bin_labels, "mu": mu_valid})
    xbin = xbin.dropna(subset=["bin", "mu"])

    n_bins = max(1, len(edges) - 1)
    bin_median = np.full(n_bins, np.nan, dtype=float)
    bin_mad = np.full(n_bins, np.nan, dtype=float)
    bin_count = np.zeros(n_bins, dtype=int)

    if xbin.empty:
        global_median = float(np.nanmedian(mu_valid))
        global_mad = float(_mad(mu_valid))
        bin_median[0] = global_median
        bin_mad[0] = global_mad
        bin_count[0] = int(mu_valid.size)
    else:
        grouped = xbin.groupby("bin")["mu"]
        for bin_id, group in grouped:
            b = int(bin_id)
            if 0 <= b < n_bins:
                values = group.to_numpy()
                bin_count[b] = int(values.size)
                bin_median[b] = float(np.median(values))
                bin_mad[b] = float(_mad(values))

        if np.all(np.isnan(bin_median)):
            global_median = float(np.nanmedian(mu_valid))
            global_mad = float(_mad(mu_valid))
            bin_median[:] = global_median
            bin_mad[:] = global_mad
            if bin_count.size:
                total = int(mu_valid.size)
                dominant = int(np.argmax(bin_count)) if n_bins > 0 else 0
                if n_bins > 0:
                    bin_count[dominant] = total

    global_median = float(np.nanmedian(mu_valid))
    global_mad = float(_mad(mu_valid))
    fallback = _neighbor_fallback_indices(bin_count, _MIN_BIN_SIZE)

    return {
        "bin_edges": edges,
        "bin_median": bin_median,
        "bin_mad": bin_mad,
        "bin_count": bin_count,
        "global_median": global_median,
        "global_mad": global_mad,
        "fallback": fallback,
    }

def add_main_sequence_parallax_plausibility(raw, deps, aux):
    _ = deps
    idx = raw.index
    n = len(raw)

    raw_phot = _coerce_photometry(raw)
    if raw_phot is None:
        return pd.DataFrame(index=idx)

    aux_phot = _coerce_photometry(aux)
    if aux_phot is None:
        context_phot = raw_phot
    else:
        context_phot = pd.concat((raw_phot, aux_phot), ignore_index=True)

    raw_feat = _main_sequence_values(raw_phot)
    context_feat = _main_sequence_values(context_phot)
    stats = _bin_reference_statistics(
        context_feat["x_clipped"], context_feat["mu"]
    )

    if stats is None:
        result = pd.DataFrame(index=idx)
        result["main_sequence_x_clipped"] = pd.Series(raw_feat["x_clipped"], index=idx)
        result["main_sequence_x_low_saturation"] = pd.Series(raw_feat["x_low_clip"], index=idx)
        result["main_sequence_x_high_saturation"] = pd.Series(raw_feat["x_high_clip"], index=idx)
        result["main_sequence_Mr"] = pd.Series(raw_feat["M_r"], index=idx)
        result["main_sequence_mu"] = pd.Series(raw_feat["mu"], index=idx)
        result["main_sequence_distance_pc"] = pd.Series(raw_feat["distance_pc"], index=idx)
        result["main_sequence_log10_distance_pc"] = pd.Series(raw_feat["log10_distance_pc"], index=idx)
        result["main_sequence_M_u"] = pd.Series(raw_feat["M_u"], index=idx)
        result["main_sequence_M_g"] = pd.Series(raw_feat["M_g"], index=idx)
        result["main_sequence_M_i"] = pd.Series(raw_feat["M_i"], index=idx)
        result["main_sequence_M_z"] = pd.Series(raw_feat["M_z"], index=idx)
        result["main_sequence_mu_zscore"] = pd.Series(np.nan, index=idx)
        result["main_sequence_mu_percentile_in_xbin"] = pd.Series(0.5, index=idx)
        return result

    edges = stats["bin_edges"]
    raw_bin = pd.cut(
        raw_feat["x_clipped"],
        bins=edges,
        include_lowest=True,
        labels=False,
    )
    raw_bin = pd.Series(raw_bin, index=idx)

    valid_rows = raw_bin.notna().to_numpy()
    raw_bin_code = raw_bin.fillna(-1).astype(int).to_numpy()
    fallback = np.full(n, -1, dtype=int)
    fallback[valid_rows] = stats["fallback"][raw_bin_code[valid_rows]]

    mu_median = np.full(n, stats["global_median"], dtype=float)
    mu_mad = np.full(n, stats["global_mad"], dtype=float)

    has_fallback = fallback >= 0
    if np.any(has_fallback):
        mapped = fallback[has_fallback]
        mu_median[has_fallback] = stats["bin_median"][mapped]
        mu_mad[has_fallback] = stats["bin_mad"][mapped]

    if not np.all(np.isfinite(mu_median)):
        mu_median[~np.isfinite(mu_median)] = stats["global_median"]
    if not np.all(np.isfinite(mu_mad)):
        mu_mad[~np.isfinite(mu_mad)] = stats["global_mad"]

    mu_zscore = (raw_feat["mu"] - mu_median) / (mu_mad + _MAD_EPS)

    rank_input = pd.DataFrame(
        {
            "bin": raw_bin.fillna(-1).astype(int),
            "mu": raw_feat["mu"],
        },
        index=idx,
    )
    rank_input = rank_input[(rank_input["bin"] >= 0) & np.isfinite(rank_input["mu"])]

    mu_percentile = pd.Series(0.5, index=idx, dtype=float)
    if not rank_input.empty:
        ranks = rank_input["mu"].groupby(rank_input["bin"]).rank(pct=True, method="average")
        mu_percentile.loc[ranks.index] = ranks.to_numpy(dtype=float)

        sizes = rank_input.groupby("bin").size()
        singletons = sizes[sizes == 1].index.to_numpy()
        if singletons.size:
            single_rows = rank_input.index[rank_input["bin"].isin(singletons)]
            mu_percentile.loc[single_rows] = 0.5

    result = pd.DataFrame(index=idx)
    result["main_sequence_x_clipped"] = pd.Series(raw_feat["x_clipped"], index=idx)
    result["main_sequence_x_low_saturation"] = pd.Series(raw_feat["x_low_clip"], index=idx)
    result["main_sequence_x_high_saturation"] = pd.Series(raw_feat["x_high_clip"], index=idx)
    result["main_sequence_Mr"] = pd.Series(raw_feat["M_r"], index=idx)
    result["main_sequence_mu"] = pd.Series(raw_feat["mu"], index=idx)
    result["main_sequence_distance_pc"] = pd.Series(raw_feat["distance_pc"], index=idx)
    result["main_sequence_log10_distance_pc"] = pd.Series(raw_feat["log10_distance_pc"], index=idx)
    result["main_sequence_M_u"] = pd.Series(raw_feat["M_u"], index=idx)
    result["main_sequence_M_g"] = pd.Series(raw_feat["M_g"], index=idx)
    result["main_sequence_M_i"] = pd.Series(raw_feat["M_i"], index=idx)
    result["main_sequence_M_z"] = pd.Series(raw_feat["M_z"], index=idx)
    result["main_sequence_mu_zscore"] = pd.Series(mu_zscore, index=idx)
    result["main_sequence_mu_percentile_in_xbin"] = mu_percentile

    return result

FEATURE_GROUPS = [
    {
        "name": "main_sequence_parallax_plausibility",
        "fn": add_main_sequence_parallax_plausibility,
        "depends_on": [],
        "description": "Generate main-sequence distance-plausibility features from ugriz photometry with clipped g-i binning, per-bin distance-modulus normalization, fallback neighbors, and in-bin mu ranking.",
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