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
- Hypothesis ID: 000056
- Source file: autogluon-001.py
- Failed node: 20260623T153205-d52347d0-203
- Run: 20260622T034843-eca160d1

# Execution Error

```text
failed.yaml error:
Process exited with 1

stderr.log:
Traceback (most recent call last):
  File "/home/xai/DEV/TheML/projects/kaggle/playground-series-s6e6/runs/20260622T034843-eca160d1/artifacts/20260623T153205-d52347d0-203/02-code.py", line 865, in main
    transformed = run_feature_groups(
                  ^^^^^^^^^^^^^^^^^^^
  File "/home/xai/DEV/TheML/src/tml/features/groups.py", line 59, in run_feature_groups
    block = group["fn"](raw.copy(), {key: value.copy() for key, value in deps.items()}, aux.copy())
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/xai/DEV/TheML/projects/kaggle/playground-series-s6e6/runs/20260622T034843-eca160d1/artifacts/20260623T153205-d52347d0-203/02-code.py", line 215, in add_local_reddening_free_locus_offsets
    count_ctx_row = count_ctx_anchor.reindex(raw_context_key).to_numpy(dtype=np.float64)
                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/xai/DEV/TheML/.venv/lib/python3.12/site-packages/pandas/core/series.py", line 5172, in reindex
    return super().reindex(
           ^^^^^^^^^^^^^^^^
  File "/home/xai/DEV/TheML/.venv/lib/python3.12/site-packages/pandas/core/generic.py", line 5632, in reindex
    return self._reindex_axes(
           ^^^^^^^^^^^^^^^^^^^
  File "/home/xai/DEV/TheML/.venv/lib/python3.12/site-packages/pandas/core/generic.py", line 5655, in _reindex_axes
    new_index, indexer = ax.reindex(
                         ^^^^^^^^^^^
  File "/home/xai/DEV/TheML/.venv/lib/python3.12/site-packages/pandas/core/indexes/base.py", line 4440, in reindex
    target = self._wrap_reindex_result(target, indexer, preserve_names)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/xai/DEV/TheML/.venv/lib/python3.12/site-packages/pandas/core/indexes/multi.py", line 2736, in _wrap_reindex_result
    target = MultiIndex.from_tuples(target)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/xai/DEV/TheML/.venv/lib/python3.12/site-packages/pandas/core/indexes/multi.py", line 223, in new_meth
    return meth(self_or_cls, *args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/xai/DEV/TheML/.venv/lib/python3.12/site-packages/pandas/core/indexes/multi.py", line 618, in from_tuples
    arrays = list(lib.tuples_to_object_array(tuples).T)
                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "pandas/_libs/lib.pyx", line 3056, in pandas._libs.lib.tuples_to_object_array
ValueError: Buffer dtype mismatch, expected 'Python object' but got 'int'
Traceback (most recent call last):
  File "/home/xai/DEV/TheML/projects/kaggle/playground-series-s6e6/runs/20260622T034843-eca160d1/artifacts/20260623T153205-d52347d0-203/02-code.py", line 989, in <module>
    main()
  File "/home/xai/DEV/TheML/projects/kaggle/playground-series-s6e6/runs/20260622T034843-eca160d1/artifacts/20260623T153205-d52347d0-203/02-code.py", line 865, in main
    transformed = run_feature_groups(
                  ^^^^^^^^^^^^^^^^^^^
  File "/home/xai/DEV/TheML/src/tml/features/groups.py", line 59, in run_feature_groups
    block = group["fn"](raw.copy(), {key: value.copy() for key, value in deps.items()}, aux.copy())
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/xai/DEV/TheML/projects/kaggle/playground-series-s6e6/runs/20260622T034843-eca160d1/artifacts/20260623T153205-d52347d0-203/02-code.py", line 215, in add_local_reddening_free_locus_offsets
    count_ctx_row = count_ctx_anchor.reindex(raw_context_key).to_numpy(dtype=np.float64)
                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/xai/DEV/TheML/.venv/lib/python3.12/site-packages/pandas/core/series.py", line 5172, in reindex
    return super().reindex(
           ^^^^^^^^^^^^^^^^
  File "/home/xai/DEV/TheML/.venv/lib/python3.12/site-packages/pandas/core/generic.py", line 5632, in reindex
    return self._reindex_axes(
           ^^^^^^^^^^^^^^^^^^^
  File "/home/xai/DEV/TheML/.venv/lib/python3.12/site-packages/pandas/core/generic.py", line 5655, in _reindex_axes
    new_index, indexer = ax.reindex(
                         ^^^^^^^^^^^
  File "/home/xai/DEV/TheML/.venv/lib/python3.12/site-packages/pandas/core/indexes/base.py", line 4440, in reindex
    target = self._wrap_reindex_result(target, indexer, preserve_names)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/xai/DEV/TheML/.venv/lib/python3.12/site-packages/pandas/core/indexes/multi.py", line 2736, in _wrap_reindex_result
    target = MultiIndex.from_tuples(target)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/xai/DEV/TheML/.venv/lib/python3.12/site-packages/pandas/core/indexes/multi.py", line 223, in new_meth
    return meth(self_or_cls, *args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/xai/DEV/TheML/.venv/lib/python3.12/site-packages/pandas/core/indexes/multi.py", line 618, in from_tuples
    arrays = list(lib.tuples_to_object_array(tuples).T)
                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "pandas/_libs/lib.pyx", line 3056, in pandas._libs.lib.tuples_to_object_array
ValueError: Buffer dtype mismatch, expected 'Python object' but got 'int'

stdout.log:
AutoGluon materialization: loaded aux file /home/xai/DEV/aideml/workspaces/2-liberal-celadon-panther/input/star_classification.csv rows=100000 cols=18 passed_to_groups=True
AutoGluon materialization: starting feature groups
TheML feature group: start name=local_reddening_free_locus_offsets
TheML feature group: failed name=local_reddening_free_locus_offsets elapsed_s=2.117 rows=824782 cols=0
```

# Previous Code

```python
import math
import numpy as np
import pandas as pd

N_SIDE = 32
N_LATITUDE_BANDS = 2 * N_SIDE
N_LONGITUDE_BANDS = 6 * N_SIDE
N_CONTEXT_NEIGHBORS = 9
Q_GRI_COEFF = 1.582
Q_RIZ_COEFF = 0.987
REDSHIFT_BINS = (0.01, 0.10, 0.50, 1.0, 2.0)
MIN_CONTEXT_COUNT = 40
LOCAL_STD_CLIP = 8.0
MAD_FLOOR = 1e-8
INVALID_CONTEXT = -1
AREA_PER_CELL = (4.0 * math.pi) / (12.0 * (N_SIDE * N_SIDE))

def _to_float_vector(df, name):
    return pd.to_numeric(df[name], errors="coerce").to_numpy(dtype=np.float64)

def _locus_features(u, g, r, i, z):
    qgri = (g - r) - Q_GRI_COEFF * (r - i)
    qriz = (r - i) - Q_RIZ_COEFF * (i - z)
    ug = u - g
    gr = g - r
    ri = r - i
    iz = i - z
    ur = u - r
    gi = g - i
    return {
        "qgri": qgri,
        "qriz": qriz,
        "u_minus_g": ug,
        "g_minus_r": gr,
        "r_minus_i": ri,
        "i_minus_z": iz,
        "u_minus_r": ur,
        "g_minus_i": gi,
    }

def _redshift_strata(redshift):
    return np.digitize(redshift, REDSHIFT_BINS, right=False).astype(np.int8)

def _sky_context(alpha, delta):
    lon = np.mod(np.asarray(alpha, dtype=np.float64), 360.0)
    lon_idx = np.floor((lon / 360.0) * N_LONGITUDE_BANDS).astype(np.int32)
    lon_idx = np.clip(lon_idx, 0, N_LONGITUDE_BANDS - 1)

    sin_dec = np.sin(np.deg2rad(np.clip(np.asarray(delta, dtype=np.float64), -90.0, 90.0)))
    lat_idx = np.floor((sin_dec + 1.0) * 0.5 * N_LATITUDE_BANDS).astype(np.int32)
    lat_idx = np.clip(lat_idx, 0, N_LATITUDE_BANDS - 1)

    context_id = lat_idx * N_LONGITUDE_BANDS + lon_idx
    return context_id.astype(np.int32), lat_idx.astype(np.int32), lon_idx.astype(np.int32)

def _context_neighbor_matrix(lat_idx, lon_idx):
    mats = []
    for dlat in (-1, 0, 1):
        lat_shift = lat_idx + dlat
        valid = (lat_shift >= 0) & (lat_shift < N_LATITUDE_BANDS)
        lat_shift = np.where(valid, lat_shift, 0)
        for dlon in (-1, 0, 1):
            lon_shift = np.mod(lon_idx + dlon, N_LONGITUDE_BANDS)
            neighbor = np.where(valid, lat_shift * N_LONGITUDE_BANDS + lon_shift, INVALID_CONTEXT)
            mats.append(neighbor.astype(np.int32))
    return np.stack(mats, axis=1)

def _expand_context_memberships(context_neighbors, strat):
    flat_context = context_neighbors.ravel()
    valid = flat_context != INVALID_CONTEXT
    flat_context = flat_context[valid]
    flat_strat = np.repeat(strat, N_CONTEXT_NEIGHBORS)[valid]
    return flat_context.astype(np.int32), flat_strat.astype(np.int16)

def _median_mad_by_group(context_ids, strat_ids, values):
    idx = pd.MultiIndex.from_arrays([context_ids, strat_ids], names=("context_id", "z_bin"))
    s = pd.Series(values, index=idx, dtype=np.float64)

    med = s.groupby(level=["context_id", "z_bin"], observed=True).median()
    mad = (s - med.reindex(s.index)).abs().groupby(level=["context_id", "z_bin"], observed=True).median()
    cnt = s.groupby(level=["context_id", "z_bin"], observed=True).size()
    return med, mad, cnt

def _safe_global_stats(values):
    med = float(np.nanmedian(values))
    mad = float(np.nanmedian(np.abs(values - med)))
    if not np.isfinite(med):
        med = 0.0
    if (not np.isfinite(mad)) or (mad < MAD_FLOOR):
        mad = 1.0
    return med, mad

def add_local_reddening_free_locus_offsets(raw, deps, aux):
    del deps
    n = len(raw)
    index = raw.index

    alpha = _to_float_vector(raw, "alpha")
    delta = _to_float_vector(raw, "delta")
    u = _to_float_vector(raw, "u")
    g = _to_float_vector(raw, "g")
    r = _to_float_vector(raw, "r")
    i = _to_float_vector(raw, "i")
    z = _to_float_vector(raw, "z")
    redshift = _to_float_vector(raw, "redshift")

    valid_raw = (
        np.isfinite(alpha)
        & np.isfinite(delta)
        & np.isfinite(u)
        & np.isfinite(g)
        & np.isfinite(r)
        & np.isfinite(i)
        & np.isfinite(z)
        & np.isfinite(redshift)
    )

    raw_ctx = np.full(n, INVALID_CONTEXT, dtype=np.int32)
    raw_lat = np.full(n, -1, dtype=np.int16)
    raw_strat = np.full(n, -1, dtype=np.int8)
    if np.any(valid_raw):
        ctx, lat, lon = _sky_context(alpha[valid_raw], delta[valid_raw])
        raw_ctx[valid_raw] = ctx
        raw_lat[valid_raw] = lat
        raw_strat[valid_raw] = _redshift_strata(redshift[valid_raw])

    raw_features = _locus_features(u, g, r, i, z)

    s_alpha = alpha[valid_raw].copy()
    s_delta = delta[valid_raw].copy()
    s_u = u[valid_raw].copy()
    s_g = g[valid_raw].copy()
    s_r = r[valid_raw].copy()
    s_i = i[valid_raw].copy()
    s_z = z[valid_raw].copy()
    s_redshift = redshift[valid_raw].copy()

    if isinstance(aux, pd.DataFrame) and not aux.empty:
        aux_cols = ("alpha", "delta", "u", "g", "r", "i", "z", "redshift")
        if set(aux_cols).issubset(set(aux.columns)):
            a_alpha = _to_float_vector(aux, "alpha")
            a_delta = _to_float_vector(aux, "delta")
            a_u = _to_float_vector(aux, "u")
            a_g = _to_float_vector(aux, "g")
            a_r = _to_float_vector(aux, "r")
            a_i = _to_float_vector(aux, "i")
            a_z = _to_float_vector(aux, "z")
            a_redshift = _to_float_vector(aux, "redshift")

            aux_valid = (
                np.isfinite(a_alpha)
                & np.isfinite(a_delta)
                & np.isfinite(a_u)
                & np.isfinite(a_g)
                & np.isfinite(a_r)
                & np.isfinite(a_i)
                & np.isfinite(a_z)
                & np.isfinite(a_redshift)
            )

            if np.any(aux_valid):
                s_alpha = np.concatenate([s_alpha, a_alpha[aux_valid]], axis=0)
                s_delta = np.concatenate([s_delta, a_delta[aux_valid]], axis=0)
                s_u = np.concatenate([s_u, a_u[aux_valid]], axis=0)
                s_g = np.concatenate([s_g, a_g[aux_valid]], axis=0)
                s_r = np.concatenate([s_r, a_r[aux_valid]], axis=0)
                s_i = np.concatenate([s_i, a_i[aux_valid]], axis=0)
                s_z = np.concatenate([s_z, a_z[aux_valid]], axis=0)
                s_redshift = np.concatenate([s_redshift, a_redshift[aux_valid]], axis=0)

    if s_alpha.size == 0:
        return pd.DataFrame(
            {
                "local_reddening_free_qgri_std": np.full(n, np.nan, dtype=np.float64),
                "local_reddening_free_qriz_std": np.full(n, np.nan, dtype=np.float64),
                "local_reddening_free_locus_distance": np.full(n, np.nan, dtype=np.float64),
                "local_reddening_free_ur_std": np.full(n, np.nan, dtype=np.float64),
                "local_reddening_free_gi_std": np.full(n, np.nan, dtype=np.float64),
                "local_reddening_free_context_density": np.full(n, np.nan, dtype=np.float64),
                "local_reddening_free_context_density_z": np.full(n, np.nan, dtype=np.float64),
                "local_reddening_free_fallback_level": np.full(n, np.nan, dtype=np.float64),
            },
            index=index,
        )

    support_features = _locus_features(s_u, s_g, s_r, s_i, s_z)
    s_ctx, s_lat, s_lon = _sky_context(s_alpha, s_delta)
    s_strat = _redshift_strata(s_redshift)
    s_neighbors = _context_neighbor_matrix(s_lat, s_lon)
    rep_ctx, rep_strat = _expand_context_memberships(s_neighbors, s_strat)

    repeat_rep = np.tile(support_features["qgri"], N_CONTEXT_NEIGHBORS)
    rep_values = repeat_rep[np.flatnonzero((s_neighbors.ravel() != INVALID_CONTEXT))]
    med_cs_anchor, mad_cs_anchor, count_cs_anchor = _median_mad_by_group(rep_ctx, rep_strat, rep_values)

    rep_strat_all = np.full(rep_ctx.shape[0], -1, dtype=np.int16)
    _, _, count_ctx_anchor = _median_mad_by_group(rep_ctx, rep_strat_all, rep_values)

    raw_key = pd.MultiIndex.from_arrays((raw_ctx, raw_strat), names=("context_id", "z_bin"))
    raw_context_key = pd.Index(raw_ctx, name="context_id")

    count_cs_row = count_cs_anchor.reindex(raw_key).to_numpy(dtype=np.float64)
    count_ctx_row = count_ctx_anchor.reindex(raw_context_key).to_numpy(dtype=np.float64)

    have_cs = np.isfinite(count_cs_row) & (count_cs_row >= MIN_CONTEXT_COUNT)
    have_ctx = np.isfinite(count_ctx_row) & (count_ctx_row >= MIN_CONTEXT_COUNT)
    fallback_density_count = np.where(
        have_cs,
        np.maximum(count_cs_row, 0.0),
        np.where(have_ctx, np.maximum(count_ctx_row, 0.0), float(len(s_alpha))),
    )

    context_cells = np.where(
        raw_ctx == INVALID_CONTEXT,
        np.nan,
        np.where((raw_lat == 0) | (raw_lat == (N_LATITUDE_BANDS - 1)), 6.0, 9.0),
    )
    context_density = np.log1p(fallback_density_count / (context_cells * AREA_PER_CELL))
    context_density = np.where(np.isfinite(context_density), context_density, np.nan)

    density_valid = np.isfinite(context_density)
    density_mean = float(np.nanmean(context_density))
    density_std = float(np.nanstd(context_density))
    if not np.isfinite(density_mean):
        density_mean = 0.0
    if (not np.isfinite(density_std)) or (density_std < MAD_FLOOR):
        density_std = 1.0
    context_density_z = (context_density - density_mean) / density_std

    fallback_level = np.where(
        have_cs,
        0.0,
        np.where(have_ctx, 1.0, 2.0),
    )
    fallback_level = np.where(valid_raw, fallback_level, np.nan)

    # standardized color-offset features
    raw_qgri = raw_features["qgri"]
    raw_qriz = raw_features["qriz"]
    raw_ur = raw_features["u_minus_r"]
    raw_gi = raw_features["g_minus_i"]

    std_qgri = np.full(n, np.nan, dtype=np.float64)
    std_qriz = np.full(n, np.nan, dtype=np.float64)
    std_ur = np.full(n, np.nan, dtype=np.float64)
    std_gi = np.full(n, np.nan, dtype=np.float64)

    # 1) qgri
    rep_values = np.tile(support_features["qgri"], N_CONTEXT_NEIGHBORS)
    rep_values = rep_values[np.flatnonzero((s_neighbors.ravel() != INVALID_CONTEXT))]
    med_cs, mad_cs, _ = _median_mad_by_group(rep_ctx, rep_strat, rep_values)
    rep_values_ctx_only = np.tile(support_features["qgri"], N_CONTEXT_NEIGHBORS)
    rep_values_ctx_only = rep_values_ctx_only[np.flatnonzero((s_neighbors.ravel() != INVALID_CONTEXT))]
    med_ctx, mad_ctx, _ = _median_mad_by_group(rep_ctx, rep_strat_all, rep_values_ctx_only)
    g_med, g_mad = _safe_global_stats(support_features["qgri"])

    med_cs_row = med_cs.reindex(raw_key).to_numpy(dtype=np.float64)
    mad_cs_row = mad_cs.reindex(raw_key).to_numpy(dtype=np.float64)
    med_ctx_row = med_ctx.reindex(raw_context_key).to_numpy(dtype=np.float64)
    mad_ctx_row = mad_ctx.reindex(raw_context_key).to_numpy(dtype=np.float64)

    use_cs = np.isfinite(med_cs_row) & np.isfinite(mad_cs_row) & (count_cs_row >= MIN_CONTEXT_COUNT)
    use_ctx = np.isfinite(med_ctx_row) & np.isfinite(mad_ctx_row) & (count_ctx_row >= MIN_CONTEXT_COUNT)

    med_row = np.where(use_cs, med_cs_row, med_ctx_row)
    mad_row = np.where(use_cs, mad_cs_row, mad_ctx_row)
    fallback_mask = ~use_cs & ~use_ctx
    med_row = np.where(fallback_mask, g_med, med_row)
    mad_row = np.where(fallback_mask, g_mad, mad_row)
    mad_row = np.where(np.isfinite(mad_row) & (mad_row > MAD_FLOOR), mad_row, g_mad)
    std_qgri = np.clip((raw_qgri - med_row) / mad_row, -LOCAL_STD_CLIP, LOCAL_STD_CLIP)

    # 2) qriz
    rep_values = np.tile(support_features["qriz"], N_CONTEXT_NEIGHBORS)
    rep_values = rep_values[np.flatnonzero((s_neighbors.ravel() != INVALID_CONTEXT))]
    med_cs, mad_cs, _ = _median_mad_by_group(rep_ctx, rep_strat, rep_values)
    rep_values_ctx_only = np.tile(support_features["qriz"], N_CONTEXT_NEIGHBORS)
    rep_values_ctx_only = rep_values_ctx_only[np.flatnonzero((s_neighbors.ravel() != INVALID_CONTEXT))]
    med_ctx, mad_ctx, _ = _median_mad_by_group(rep_ctx, rep_strat_all, rep_values_ctx_only)
    g_med, g_mad = _safe_global_stats(support_features["qriz"])

    med_cs_row = med_cs.reindex(raw_key).to_numpy(dtype=np.float64)
    mad_cs_row = mad_cs.reindex(raw_key).to_numpy(dtype=np.float64)
    med_ctx_row = med_ctx.reindex(raw_context_key).to_numpy(dtype=np.float64)
    mad_ctx_row = mad_ctx.reindex(raw_context_key).to_numpy(dtype=np.float64)

    use_cs = np.isfinite(med_cs_row) & np.isfinite(mad_cs_row) & (count_cs_row >= MIN_CONTEXT_COUNT)
    use_ctx = np.isfinite(med_ctx_row) & np.isfinite(mad_ctx_row) & (count_ctx_row >= MIN_CONTEXT_COUNT)

    med_row = np.where(use_cs, med_cs_row, med_ctx_row)
    mad_row = np.where(use_cs, mad_cs_row, mad_ctx_row)
    fallback_mask = ~use_cs & ~use_ctx
    med_row = np.where(fallback_mask, g_med, med_row)
    mad_row = np.where(fallback_mask, g_mad, mad_row)
    mad_row = np.where(np.isfinite(mad_row) & (mad_row > MAD_FLOOR), mad_row, g_mad)
    std_qriz = np.clip((raw_qriz - med_row) / mad_row, -LOCAL_STD_CLIP, LOCAL_STD_CLIP)

    # 3) u-r residual
    rep_values = np.tile(support_features["u_minus_r"], N_CONTEXT_NEIGHBORS)
    rep_values = rep_values[np.flatnonzero((s_neighbors.ravel() != INVALID_CONTEXT))]
    med_cs, mad_cs, _ = _median_mad_by_group(rep_ctx, rep_strat, rep_values)
    rep_values_ctx_only = np.tile(support_features["u_minus_r"], N_CONTEXT_NEIGHBORS)
    rep_values_ctx_only = rep_values_ctx_only[np.flatnonzero((s_neighbors.ravel() != INVALID_CONTEXT))]
    med_ctx, mad_ctx, _ = _median_mad_by_group(rep_ctx, rep_strat_all, rep_values_ctx_only)
    g_med, g_mad = _safe_global_stats(support_features["u_minus_r"])

    med_cs_row = med_cs.reindex(raw_key).to_numpy(dtype=np.float64)
    mad_cs_row = mad_cs.reindex(raw_key).to_numpy(dtype=np.float64)
    med_ctx_row = med_ctx.reindex(raw_context_key).to_numpy(dtype=np.float64)
    mad_ctx_row = mad_ctx.reindex(raw_context_key).to_numpy(dtype=np.float64)

    use_cs = np.isfinite(med_cs_row) & np.isfinite(mad_cs_row) & (count_cs_row >= MIN_CONTEXT_COUNT)
    use_ctx = np.isfinite(med_ctx_row) & np.isfinite(mad_ctx_row) & (count_ctx_row >= MIN_CONTEXT_COUNT)

    med_row = np.where(use_cs, med_cs_row, med_ctx_row)
    mad_row = np.where(use_cs, mad_cs_row, mad_ctx_row)
    fallback_mask = ~use_cs & ~use_ctx
    med_row = np.where(fallback_mask, g_med, med_row)
    mad_row = np.where(fallback_mask, g_mad, mad_row)
    mad_row = np.where(np.isfinite(mad_row) & (mad_row > MAD_FLOOR), mad_row, g_mad)
    std_ur = np.clip((raw_ur - med_row) / mad_row, -LOCAL_STD_CLIP, LOCAL_STD_CLIP)

    # 4) g-i residual
    rep_values = np.tile(support_features["g_minus_i"], N_CONTEXT_NEIGHBORS)
    rep_values = rep_values[np.flatnonzero((s_neighbors.ravel() != INVALID_CONTEXT))]
    med_cs, mad_cs, _ = _median_mad_by_group(rep_ctx, rep_strat, rep_values)
    rep_values_ctx_only = np.tile(support_features["g_minus_i"], N_CONTEXT_NEIGHBORS)
    rep_values_ctx_only = rep_values_ctx_only[np.flatnonzero((s_neighbors.ravel() != INVALID_CONTEXT))]
    med_ctx, mad_ctx, _ = _median_mad_by_group(rep_ctx, rep_strat_all, rep_values_ctx_only)
    g_med, g_mad = _safe_global_stats(support_features["g_minus_i"])

    med_cs_row = med_cs.reindex(raw_key).to_numpy(dtype=np.float64)
    mad_cs_row = mad_cs.reindex(raw_key).to_numpy(dtype=np.float64)
    med_ctx_row = med_ctx.reindex(raw_context_key).to_numpy(dtype=np.float64)
    mad_ctx_row = mad_ctx.reindex(raw_context_key).to_numpy(dtype=np.float64)

    use_cs = np.isfinite(med_cs_row) & np.isfinite(mad_cs_row) & (count_cs_row >= MIN_CONTEXT_COUNT)
    use_ctx = np.isfinite(med_ctx_row) & np.isfinite(mad_ctx_row) & (count_ctx_row >= MIN_CONTEXT_COUNT)

    med_row = np.where(use_cs, med_cs_row, med_ctx_row)
    mad_row = np.where(use_cs, mad_cs_row, mad_ctx_row)
    fallback_mask = ~use_cs & ~use_ctx
    med_row = np.where(fallback_mask, g_med, med_row)
    mad_row = np.where(fallback_mask, g_mad, mad_row)
    mad_row = np.where(np.isfinite(mad_row) & (mad_row > MAD_FLOOR), mad_row, g_mad)
    std_gi = np.clip((raw_gi - med_row) / mad_row, -LOCAL_STD_CLIP, LOCAL_STD_CLIP)

    locus_distance = np.sqrt(np.square(std_qgri) + np.square(std_qriz))

    return pd.DataFrame(
        {
            "local_reddening_free_qgri_std": std_qgri,
            "local_reddening_free_qriz_std": std_qriz,
            "local_reddening_free_locus_distance": locus_distance,
            "local_reddening_free_ur_std": std_ur,
            "local_reddening_free_gi_std": std_gi,
            "local_reddening_free_context_density": context_density,
            "local_reddening_free_context_density_z": context_density_z,
            "local_reddening_free_fallback_level": fallback_level,
        },
        index=index,
    )

FEATURE_GROUPS = [
    {
        "name": "local_reddening_free_locus_offsets",
        "fn": add_local_reddening_free_locus_offsets,
        "depends_on": [],
        "description": "Compute local sky-cell reddening-free locus offsets from neighborhood medians and MADs with sparse-context fallback.",
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