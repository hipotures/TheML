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
- Hypothesis ID: 000061
- Source file: autogluon-001.py
- Failed node: 20260628T094425-6c1992a9-143
- Run: 20260624T200326-681cd81b

# Execution Error

```text
failed.yaml error:
Process exited with 1

stderr.log:
Traceback (most recent call last):
  File "/home/xai/DEV/TheML/projects/kaggle/playground-series-s6e6/runs/20260624T200326-681cd81b/artifacts/20260628T094425-6c1992a9-143/02-code.py", line 826, in main
    transformed = run_feature_groups(
                  ^^^^^^^^^^^^^^^^^^^
  File "/home/xai/DEV/TheML/src/tml/features/groups.py", line 59, in run_feature_groups
    block = group["fn"](raw.copy(), {key: value.copy() for key, value in deps.items()}, aux.copy())
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/xai/DEV/TheML/projects/kaggle/playground-series-s6e6/runs/20260624T200326-681cd81b/artifacts/20260628T094425-6c1992a9-143/02-code.py", line 215, in add_redshift_slice_angular_environment
    distances_rad, indices = tree.query(chunk_coords, k=k_query, return_distance=True)
                             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/xai/DEV/TheML/projects/kaggle/playground-series-s6e6/runs/20260624T200326-681cd81b/artifacts/20260628T094425-6c1992a9-143/02-code.py", line 505, in _raise_timeout
    raise TimeoutError(f"AutoGluon preprocess exceeded {seconds} seconds")
TimeoutError: AutoGluon preprocess exceeded 180 seconds
Traceback (most recent call last):
  File "/home/xai/DEV/TheML/projects/kaggle/playground-series-s6e6/runs/20260624T200326-681cd81b/artifacts/20260628T094425-6c1992a9-143/02-code.py", line 950, in <module>
    main()
  File "/home/xai/DEV/TheML/projects/kaggle/playground-series-s6e6/runs/20260624T200326-681cd81b/artifacts/20260628T094425-6c1992a9-143/02-code.py", line 826, in main
    transformed = run_feature_groups(
                  ^^^^^^^^^^^^^^^^^^^
  File "/home/xai/DEV/TheML/src/tml/features/groups.py", line 59, in run_feature_groups
    block = group["fn"](raw.copy(), {key: value.copy() for key, value in deps.items()}, aux.copy())
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/xai/DEV/TheML/projects/kaggle/playground-series-s6e6/runs/20260624T200326-681cd81b/artifacts/20260628T094425-6c1992a9-143/02-code.py", line 215, in add_redshift_slice_angular_environment
    distances_rad, indices = tree.query(chunk_coords, k=k_query, return_distance=True)
                             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/xai/DEV/TheML/projects/kaggle/playground-series-s6e6/runs/20260624T200326-681cd81b/artifacts/20260628T094425-6c1992a9-143/02-code.py", line 505, in _raise_timeout
    raise TimeoutError(f"AutoGluon preprocess exceeded {seconds} seconds")
TimeoutError: AutoGluon preprocess exceeded 180 seconds

stdout.log:
AutoGluon materialization: loaded aux file star_classification.csv rows=100000 cols=18 passed_to_groups=True
AutoGluon materialization: starting feature groups
TheML feature group: start name=redshift_slice_angular_environment
TheML feature group: failed name=redshift_slice_angular_environment elapsed_s=181.452 rows=824782 cols=0
```

# Previous Code

```python
import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

TRAIN_ID_MAX = 577346
C_LIGHT_KM_S = 299792.458
H0_KM_S_MPC = 70.0
OMEGA_M = 0.3
OMEGA_LAMBDA = 0.7
PHYSICAL_RADII_MPC = (0.5, 1.0, 3.0, 10.0)
LOW_REDSHIFT_RADII_DEG = (0.05, 0.2, 1.0, 2.0)
RADIUS_LABELS = ("r0p5", "r1", "r3", "r10")
COLOR_SCALE_LABELS = ("r1", "r3")
COLOR_SCALE_INDEXES = (1, 2)
NN_RANKS = (1, 5, 20)
K_NEIGHBORS = 256
CHUNK_SIZE = 10000
COSMOLOGY_GRID_SIZE = 768
EPSILON = 0.000000001

def _numeric_array(frame, column, default):
    if column not in frame.columns:
        return np.full(len(frame), default, dtype=np.float64)
    values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=np.float64, copy=True)
    return np.nan_to_num(values, nan=default, posinf=default, neginf=default)

def _infer_train_mask(raw, aux):
    n_rows = len(raw)

    if isinstance(aux, pd.DataFrame) and len(aux) == n_rows and not aux.empty:
        for column in ("is_train", "_is_train", "__is_train", "train"):
            if column in aux.columns:
                values = aux[column]
                if values.dtype == bool:
                    return values.to_numpy(dtype=bool, copy=True)
                numeric = pd.to_numeric(values, errors="coerce")
                if numeric.notna().any():
                    return numeric.fillna(0).to_numpy(dtype=np.float64) > 0

        for column in ("is_test", "_is_test", "__is_test", "test"):
            if column in aux.columns:
                values = aux[column]
                if values.dtype == bool:
                    return ~values.to_numpy(dtype=bool, copy=True)
                numeric = pd.to_numeric(values, errors="coerce")
                if numeric.notna().any():
                    return numeric.fillna(0).to_numpy(dtype=np.float64) <= 0

        for column in ("split", "dataset", "source"):
            if column in aux.columns:
                values = aux[column].astype(str).str.lower()
                if values.str.contains("train", regex=False).any():
                    return values.str.contains("train", regex=False).to_numpy(dtype=bool, copy=True)

    if "id" in raw.columns:
        ids = pd.to_numeric(raw["id"], errors="coerce").to_numpy(dtype=np.float64, copy=True)
        finite = np.isfinite(ids)
        if finite.any() and np.nanmin(ids[finite]) <= TRAIN_ID_MAX and np.nanmax(ids[finite]) > TRAIN_ID_MAX:
            return ids <= TRAIN_ID_MAX

    return np.ones(n_rows, dtype=bool)

def _angular_diameter_distance_mpc(redshift):
    z_pos = np.maximum(redshift, 0.0)
    finite = np.isfinite(z_pos)
    if not finite.any():
        return np.zeros_like(z_pos, dtype=np.float64)

    z_max = float(np.max(z_pos[finite]))
    if z_max <= 0.0:
        return np.zeros_like(z_pos, dtype=np.float64)

    grid = np.linspace(0.0, z_max, COSMOLOGY_GRID_SIZE, dtype=np.float64)
    inv_e = 1.0 / np.sqrt(OMEGA_M * np.power(1.0 + grid, 3.0) + OMEGA_LAMBDA)
    dz_grid = np.diff(grid)
    integral = np.empty_like(grid)
    integral[0] = 0.0
    integral[1:] = np.cumsum(0.5 * (inv_e[1:] + inv_e[:-1]) * dz_grid)
    comoving = (C_LIGHT_KM_S / H0_KM_S_MPC) * np.interp(z_pos, grid, integral)
    return comoving / (1.0 + z_pos)

def _spherical_cap_area(radius_deg):
    radius_rad = np.deg2rad(np.maximum(radius_deg, 0.0))
    return 2.0 * np.pi * (1.0 - np.cos(radius_rad))

def _median_mad_by_row(diff_values, mask):
    valid_mask = mask & np.isfinite(diff_values)
    has_values = valid_mask.any(axis=1)

    medians = np.zeros(diff_values.shape[0], dtype=np.float32)
    mads = np.zeros(diff_values.shape[0], dtype=np.float32)
    empty = (~has_values).astype(np.int8)

    if has_values.any():
        selected = np.where(valid_mask[has_values], diff_values[has_values], np.nan)
        median_values = np.nanmedian(selected, axis=1)
        median_values = np.nan_to_num(median_values, nan=0.0, posinf=0.0, neginf=0.0)
        mad_values = np.nanmedian(np.abs(selected - median_values[:, None]), axis=1)
        mad_values = np.nan_to_num(mad_values, nan=0.0, posinf=0.0, neginf=0.0)
        medians[has_values] = median_values.astype(np.float32)
        mads[has_values] = mad_values.astype(np.float32)

    return medians, mads, empty

def add_redshift_slice_angular_environment(raw, deps, aux):
    n_rows = len(raw)
    index = raw.index

    if n_rows == 0:
        return pd.DataFrame(index=index)

    alpha = _numeric_array(raw, "alpha", 0.0)
    delta = _numeric_array(raw, "delta", 0.0)
    redshift = _numeric_array(raw, "redshift", 0.0)
    g_mag = _numeric_array(raw, "g", 0.0)
    r_mag = _numeric_array(raw, "r", 0.0)
    i_mag = _numeric_array(raw, "i", 0.0)

    redshift_nonnegative = np.maximum(redshift, 0.0)
    dz_window = np.maximum(0.01, 0.03 * (1.0 + redshift_nonnegative)).astype(np.float32)
    invalid_physical_scale = (redshift <= 0.003).astype(np.int8)

    angular_diameter_distance = _angular_diameter_distance_mpc(redshift)
    physical_radii = np.asarray(PHYSICAL_RADII_MPC, dtype=np.float64)
    low_redshift_radii = np.asarray(LOW_REDSHIFT_RADII_DEG, dtype=np.float64)

    theta_deg = np.empty((n_rows, len(PHYSICAL_RADII_MPC)), dtype=np.float32)
    valid_distance = (redshift > 0.003) & (angular_diameter_distance > EPSILON)
    physical_theta = np.rad2deg(physical_radii[None, :] / np.maximum(angular_diameter_distance[:, None], EPSILON))
    physical_theta = np.clip(physical_theta, 0.02, 2.0)
    theta_deg[:, :] = np.where(valid_distance[:, None], physical_theta, low_redshift_radii[None, :]).astype(np.float32)

    ra_rad = np.deg2rad(np.mod(alpha, 360.0))
    dec_rad = np.deg2rad(np.clip(delta, -90.0, 90.0))
    coords = np.column_stack((dec_rad, ra_rad)).astype(np.float64, copy=False)

    train_mask = _infer_train_mask(raw, aux)
    if train_mask.shape[0] != n_rows or not train_mask.any():
        train_mask = np.ones(n_rows, dtype=bool)

    ref_positions = np.flatnonzero(train_mask)
    ref_coords = coords[train_mask]
    ref_redshift = redshift[train_mask]
    ref_gr = (g_mag - r_mag)[train_mask]
    ref_ri = (r_mag - i_mag)[train_mask]
    n_ref = len(ref_positions)

    self_ref_position = np.full(n_rows, -1, dtype=np.int64)
    self_ref_position[ref_positions] = np.arange(n_ref, dtype=np.int64)

    if n_ref == 0:
        return pd.DataFrame(index=index)

    tree = BallTree(ref_coords, metric="haversine")
    k_query = min(n_ref, K_NEIGHBORS + 1)

    sorted_ref_redshift = np.sort(ref_redshift)
    left = np.searchsorted(sorted_ref_redshift, redshift - dz_window, side="left")
    right = np.searchsorted(sorted_ref_redshift, redshift + dz_window, side="right")
    redshift_ref_count = right - left
    redshift_ref_count = redshift_ref_count - (self_ref_position >= 0)
    redshift_ref_count = np.maximum(redshift_ref_count, 0)
    redshift_ref_denominator = np.maximum(n_ref - (self_ref_position >= 0).astype(np.int64), 1)
    redshift_ref_fraction = (redshift_ref_count / redshift_ref_denominator).astype(np.float32)

    slice_counts = np.zeros((n_rows, len(RADIUS_LABELS)), dtype=np.int32)
    annulus_counts = np.zeros((n_rows, len(RADIUS_LABELS)), dtype=np.int32)
    allz_counts = np.zeros((n_rows, len(RADIUS_LABELS)), dtype=np.int32)

    nn_distances = np.zeros((n_rows, len(NN_RANKS)), dtype=np.float32)
    nn_missing = np.ones((n_rows, len(NN_RANKS)), dtype=np.int8)
    nn_dist_ratios = np.zeros((n_rows, len(NN_RANKS)), dtype=np.float32)
    nn_rank_ratios = np.zeros((n_rows, len(NN_RANKS)), dtype=np.float32)

    gr_medians = np.zeros((n_rows, len(COLOR_SCALE_INDEXES)), dtype=np.float32)
    gr_mads = np.zeros((n_rows, len(COLOR_SCALE_INDEXES)), dtype=np.float32)
    gr_empty = np.ones((n_rows, len(COLOR_SCALE_INDEXES)), dtype=np.int8)
    ri_medians = np.zeros((n_rows, len(COLOR_SCALE_INDEXES)), dtype=np.float32)
    ri_mads = np.zeros((n_rows, len(COLOR_SCALE_INDEXES)), dtype=np.float32)
    ri_empty = np.ones((n_rows, len(COLOR_SCALE_INDEXES)), dtype=np.int8)

    candidate_truncated = np.zeros(n_rows, dtype=np.int8)

    for start in range(0, n_rows, CHUNK_SIZE):
        end = min(start + CHUNK_SIZE, n_rows)
        chunk = slice(start, end)
        m_rows = end - start

        chunk_coords = coords[chunk]
        chunk_theta = theta_deg[chunk].astype(np.float64, copy=False)
        chunk_self_ref = self_ref_position[chunk]
        chunk_redshift = redshift[chunk]
        chunk_dz = dz_window[chunk].astype(np.float64, copy=False)

        for radius_idx in range(len(RADIUS_LABELS)):
            counts = tree.query_radius(
                chunk_coords,
                r=np.deg2rad(chunk_theta[:, radius_idx]),
                count_only=True,
            )
            counts = counts.astype(np.int32, copy=False)
            counts = counts - (chunk_self_ref >= 0).astype(np.int32)
            allz_counts[chunk, radius_idx] = np.maximum(counts, 0)

        distances_rad, indices = tree.query(chunk_coords, k=k_query, return_distance=True)
        distances_deg = np.rad2deg(distances_rad).astype(np.float32, copy=False)

        valid_all = np.isfinite(distances_deg)
        valid_all &= indices != chunk_self_ref[:, None]

        ref_z_neighbors = ref_redshift[indices]
        same_redshift = valid_all & (np.abs(ref_z_neighbors - chunk_redshift[:, None]) <= chunk_dz[:, None])

        for radius_idx in range(len(RADIUS_LABELS)):
            radius = chunk_theta[:, radius_idx][:, None]
            slice_counts[chunk, radius_idx] = np.sum(same_redshift & (distances_deg <= radius), axis=1).astype(np.int32)
            annulus_counts[chunk, radius_idx] = np.sum(
                same_redshift & (distances_deg > 3.0 * radius) & (distances_deg <= 6.0 * radius),
                axis=1,
            ).astype(np.int32)

        distance_all = np.where(valid_all, distances_deg, np.inf)
        distance_same = np.where(same_redshift, distances_deg, np.inf)
        all_cumulative_rank = np.cumsum(valid_all, axis=1)
        same_cumulative_rank = np.cumsum(same_redshift, axis=1)

        max_rank = max(NN_RANKS)
        all_partition = np.partition(distance_all, min(max_rank - 1, distance_all.shape[1] - 1), axis=1)
        same_partition = np.partition(distance_same, min(max_rank - 1, distance_same.shape[1] - 1), axis=1)

        for rank_idx, rank in enumerate(NN_RANKS):
            if rank <= distance_same.shape[1]:
                same_distance = same_partition[:, rank - 1]
                all_distance = all_partition[:, rank - 1]
                available = np.isfinite(same_distance)

                nn_distances[chunk, rank_idx] = np.where(available, same_distance, 0.0).astype(np.float32)
                nn_missing[chunk, rank_idx] = (~available).astype(np.int8)
                nn_dist_ratios[chunk, rank_idx] = np.where(
                    available & np.isfinite(all_distance),
                    same_distance / np.maximum(all_distance, EPSILON),
                    0.0,
                ).astype(np.float32)

                reached = same_cumulative_rank >= rank
                first_pos = np.argmax(reached, axis=1)
                any_reached = reached.any(axis=1)
                row_numbers = np.arange(m_rows)
                angular_rank = all_cumulative_rank[row_numbers, first_pos]
                nn_rank_ratios[chunk, rank_idx] = np.where(
                    any_reached,
                    angular_rank / float(rank),
                    0.0,
                ).astype(np.float32)

        if k_query > K_NEIGHBORS:
            farthest_candidate = np.max(np.where(valid_all, distances_deg, 0.0), axis=1)
            needed_outer_radius = 6.0 * np.max(chunk_theta, axis=1)
            candidate_truncated[chunk] = (farthest_candidate < needed_outer_radius).astype(np.int8)

        ref_gr_neighbors = ref_gr[indices]
        ref_ri_neighbors = ref_ri[indices]
        chunk_gr = (g_mag[chunk] - r_mag[chunk])[:, None]
        chunk_ri = (r_mag[chunk] - i_mag[chunk])[:, None]

        for scale_out_idx, radius_idx in enumerate(COLOR_SCALE_INDEXES):
            color_radius = chunk_theta[:, radius_idx][:, None]
            color_mask = same_redshift & (distances_deg <= color_radius)

            gr_diff = ref_gr_neighbors - chunk_gr
            gr_med, gr_mad, gr_is_empty = _median_mad_by_row(gr_diff, color_mask)
            gr_medians[chunk, scale_out_idx] = gr_med
            gr_mads[chunk, scale_out_idx] = gr_mad
            gr_empty[chunk, scale_out_idx] = gr_is_empty

            ri_diff = ref_ri_neighbors - chunk_ri
            ri_med, ri_mad, ri_is_empty = _median_mad_by_row(ri_diff, color_mask)
            ri_medians[chunk, scale_out_idx] = ri_med
            ri_mads[chunk, scale_out_idx] = ri_mad
            ri_empty[chunk, scale_out_idx] = ri_is_empty

    features = {
        "redshift_half_window": dz_window,
        "redshift_reference_fraction": redshift_ref_fraction,
        "invalid_physical_scale": invalid_physical_scale,
        "knn_background_truncated": candidate_truncated,
    }

    for radius_idx, label in enumerate(RADIUS_LABELS):
        radius = theta_deg[:, radius_idx].astype(np.float64, copy=False)
        inner_area = _spherical_cap_area(radius)
        annulus_area = np.maximum(_spherical_cap_area(6.0 * radius) - _spherical_cap_area(3.0 * radius), EPSILON)

        annulus_expected = annulus_counts[:, radius_idx].astype(np.float64) * inner_area / annulus_area
        allz_expected = allz_counts[:, radius_idx].astype(np.float64) * redshift_ref_fraction.astype(np.float64)

        features[f"theta_deg_{label}"] = theta_deg[:, radius_idx]
        features[f"slice_count_{label}"] = slice_counts[:, radius_idx]
        features[f"annulus_count_{label}"] = annulus_counts[:, radius_idx]
        features[f"allz_count_{label}"] = allz_counts[:, radius_idx]
        features[f"log_overdensity_annulus_{label}"] = (
            np.log1p(slice_counts[:, radius_idx].astype(np.float64)) - np.log1p(annulus_expected)
        ).astype(np.float32)
        features[f"log_overdensity_allz_{label}"] = (
            np.log1p(slice_counts[:, radius_idx].astype(np.float64)) - np.log1p(allz_expected)
        ).astype(np.float32)
        features[f"slice_fraction_allz_{label}"] = (
            slice_counts[:, radius_idx].astype(np.float64) / np.maximum(allz_counts[:, radius_idx].astype(np.float64), 1.0)
        ).astype(np.float32)

    for rank_idx, rank in enumerate(NN_RANKS):
        features[f"slice_nn{rank}_deg"] = nn_distances[:, rank_idx]
        features[f"slice_nn{rank}_missing"] = nn_missing[:, rank_idx]
        features[f"slice_nn{rank}_dist_ratio_allz"] = nn_dist_ratios[:, rank_idx]
        features[f"slice_nn{rank}_angular_rank_ratio"] = nn_rank_ratios[:, rank_idx]

    for scale_idx, scale_label in enumerate(COLOR_SCALE_LABELS):
        features[f"gr_diff_median_{scale_label}"] = gr_medians[:, scale_idx]
        features[f"gr_diff_mad_{scale_label}"] = gr_mads[:, scale_idx]
        features[f"gr_diff_empty_{scale_label}"] = gr_empty[:, scale_idx]
        features[f"ri_diff_median_{scale_label}"] = ri_medians[:, scale_idx]
        features[f"ri_diff_mad_{scale_label}"] = ri_mads[:, scale_idx]
        features[f"ri_diff_empty_{scale_label}"] = ri_empty[:, scale_idx]

    return pd.DataFrame(features, index=index)

FEATURE_GROUPS = [
    {
        "name": "redshift_slice_angular_environment",
        "fn": add_redshift_slice_angular_environment,
        "depends_on": [],
        "description": "Redshift-sliced spherical-neighborhood density, nearest-neighbor, and color-coherence features from the training sky reference.",
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