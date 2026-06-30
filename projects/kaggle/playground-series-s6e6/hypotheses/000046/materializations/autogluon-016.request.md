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
- Hypothesis ID: 000046
- Source file: autogluon-011.py
- Failed node: 20260629T215318-6e9c3ef0-47
- Run: 20260629T213805-6aae5126

# Execution Error

```text
failed.yaml error:
Process exited with 1

stderr.log:
Traceback (most recent call last):
  File "/home/xai/DEV/TheML/projects/kaggle/playground-series-s6e6/runs/20260629T213805-6aae5126/artifacts/20260629T215318-6e9c3ef0-47/02-code.py", line 1021, in main
    transformed = run_feature_groups(
                  ^^^^^^^^^^^^^^^^^^^
  File "/home/xai/DEV/TheML/src/tml/features/groups.py", line 59, in run_feature_groups
    block = group["fn"](raw.copy(), {key: value.copy() for key, value in deps.items()}, aux.copy())
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/xai/DEV/TheML/projects/kaggle/playground-series-s6e6/runs/20260629T213805-6aae5126/artifacts/20260629T215318-6e9c3ef0-47/02-code.py", line 504, in add_local_manifold_codimension
    s2 = _compute_space_features(raw, X_s2, train_mask, category_codes)
         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/xai/DEV/TheML/projects/kaggle/playground-series-s6e6/runs/20260629T213805-6aae5126/artifacts/20260629T215318-6e9c3ef0-47/02-code.py", line 389, in _compute_space_features
    all_dist, all_idx = nn.kneighbors(X_scaled[all_indices], n_neighbors=nn_neighbors, return_distance=True)
                        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/xai/DEV/TheML/.venv/lib/python3.12/site-packages/sklearn/neighbors/_base.py", line 923, in kneighbors
    chunked_results = Parallel(n_jobs, prefer="threads")(
                      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/xai/DEV/TheML/.venv/lib/python3.12/site-packages/sklearn/utils/parallel.py", line 82, in __call__
    return super().__call__(iterable_with_config_and_warning_filters)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/xai/DEV/TheML/.venv/lib/python3.12/site-packages/joblib/parallel.py", line 2072, in __call__
    return output if self.return_generator else list(output)
                                                ^^^^^^^^^^^^
  File "/home/xai/DEV/TheML/.venv/lib/python3.12/site-packages/joblib/parallel.py", line 1682, in _get_outputs
    yield from self._retrieve()
  File "/home/xai/DEV/TheML/.venv/lib/python3.12/site-packages/joblib/parallel.py", line 1800, in _retrieve
    time.sleep(0.01)
  File "/home/xai/DEV/TheML/projects/kaggle/playground-series-s6e6/runs/20260629T213805-6aae5126/artifacts/20260629T215318-6e9c3ef0-47/02-code.py", line 700, in _raise_timeout
    raise TimeoutError(f"AutoGluon preprocess exceeded {seconds} seconds")
TimeoutError: AutoGluon preprocess exceeded 90 seconds
Traceback (most recent call last):
  File "/home/xai/DEV/TheML/projects/kaggle/playground-series-s6e6/runs/20260629T213805-6aae5126/artifacts/20260629T215318-6e9c3ef0-47/02-code.py", line 1145, in <module>
    main()
  File "/home/xai/DEV/TheML/projects/kaggle/playground-series-s6e6/runs/20260629T213805-6aae5126/artifacts/20260629T215318-6e9c3ef0-47/02-code.py", line 1021, in main
    transformed = run_feature_groups(
                  ^^^^^^^^^^^^^^^^^^^
  File "/home/xai/DEV/TheML/src/tml/features/groups.py", line 59, in run_feature_groups
    block = group["fn"](raw.copy(), {key: value.copy() for key, value in deps.items()}, aux.copy())
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/xai/DEV/TheML/projects/kaggle/playground-series-s6e6/runs/20260629T213805-6aae5126/artifacts/20260629T215318-6e9c3ef0-47/02-code.py", line 504, in add_local_manifold_codimension
    s2 = _compute_space_features(raw, X_s2, train_mask, category_codes)
         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/xai/DEV/TheML/projects/kaggle/playground-series-s6e6/runs/20260629T213805-6aae5126/artifacts/20260629T215318-6e9c3ef0-47/02-code.py", line 389, in _compute_space_features
    all_dist, all_idx = nn.kneighbors(X_scaled[all_indices], n_neighbors=nn_neighbors, return_distance=True)
                        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/xai/DEV/TheML/.venv/lib/python3.12/site-packages/sklearn/neighbors/_base.py", line 923, in kneighbors
    chunked_results = Parallel(n_jobs, prefer="threads")(
                      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/xai/DEV/TheML/.venv/lib/python3.12/site-packages/sklearn/utils/parallel.py", line 82, in __call__
    return super().__call__(iterable_with_config_and_warning_filters)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/xai/DEV/TheML/.venv/lib/python3.12/site-packages/joblib/parallel.py", line 2072, in __call__
    return output if self.return_generator else list(output)
                                                ^^^^^^^^^^^^
  File "/home/xai/DEV/TheML/.venv/lib/python3.12/site-packages/joblib/parallel.py", line 1682, in _get_outputs
    yield from self._retrieve()
  File "/home/xai/DEV/TheML/.venv/lib/python3.12/site-packages/joblib/parallel.py", line 1800, in _retrieve
    time.sleep(0.01)
  File "/home/xai/DEV/TheML/projects/kaggle/playground-series-s6e6/runs/20260629T213805-6aae5126/artifacts/20260629T215318-6e9c3ef0-47/02-code.py", line 700, in _raise_timeout
    raise TimeoutError(f"AutoGluon preprocess exceeded {seconds} seconds")
TimeoutError: AutoGluon preprocess exceeded 90 seconds

stdout.log:
AutoGluon materialization: loaded aux file star_classification.csv rows=100000 cols=18 passed_to_groups=True
AutoGluon materialization: starting feature groups
TheML feature group: start name=local_manifold_codimension
TheML feature group: failed name=local_manifold_codimension elapsed_s=89.952 rows=824782 cols=0
```

# Previous Code

```python
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

_K_NEIGHBORS = 64
_MIN_STABLE_K = 6
_LARGE_TRAIN_THRESHOLD = 200_000
_SMALL_TRAIN_K_CAP = 32
_LARGE_TRAIN_K_CAP = 24
_EPSILON = 1e-12
_MAD_CLIP = 1e-6
_SPACE_PREFIX = "manifold_codim"

_FEATURE_KEYS = (
    "r2",
    "residual2",
    "a12",
    "a23",
    "tail",
    "dim95",
    "rk",
    "r_med",
    "r_mean",
    "density_proxy",
    "r_ratio",
    "frac_in_box",
    "asym_mad",
    "asym_mad_ratio",
    "degenerate",
    "n_neighbors",
)

_SPARSE_REPLACE_KEYS = (
    "r2",
    "residual2",
    "a12",
    "a23",
    "tail",
    "dim95",
    "rk",
    "r_mean",
    "density_proxy",
    "r_ratio",
)

def _neighbor_cap(train_size):
    if train_size <= 1:
        return 0
    if train_size >= _LARGE_TRAIN_THRESHOLD:
        return min(_K_NEIGHBORS, _LARGE_TRAIN_K_CAP)
    return min(_K_NEIGHBORS, _SMALL_TRAIN_K_CAP)

def _safe_median_1d(values):
    arr = np.asarray(values, dtype=np.float64)
    n = arr.size
    if n == 0:
        return np.nan
    if n == 1:
        return float(arr[0])

    half = n // 2
    if n % 2 == 1:
        return float(np.partition(arr, half)[half])

    partitioned = np.partition(arr, (half - 1, half))
    return float(0.5 * (partitioned[half - 1] + partitioned[half]))

def _safe_median_rows(values):
    arr = np.asarray(values, dtype=np.float64)
    n, d = arr.shape
    if n == 0:
        return np.full(d, np.nan, dtype=np.float64)
    if n == 1:
        return arr[0].astype(np.float64, copy=True)

    half = n // 2
    if n % 2 == 1:
        return np.partition(arr, half, axis=0)[half]

    partitioned = np.partition(arr, (half - 1, half), axis=0)
    return 0.5 * (partitioned[half - 1] + partitioned[half])

def _coerce_train_mask_from_series(series, positive_is_train=True):
    s = pd.Series(series)

    if s.empty:
        return None

    if pd.api.types.is_bool_dtype(s.dtype):
        values = s.fillna(False).astype(bool).to_numpy(dtype=bool)
        return values if positive_is_train else ~values

    numeric = pd.to_numeric(s, errors="coerce")
    numeric_non_na = numeric.dropna()
    if len(numeric_non_na) > 0:
        uniq = pd.unique(numeric_non_na)
        if len(uniq) <= 3 and np.all(np.isin(uniq, [0, 1])):
            values = numeric.to_numpy(dtype=float)
            values = np.where(np.isnan(values), 0.0, values)
            train_mask = values > 0.5
            return train_mask.astype(bool) if positive_is_train else ~train_mask.astype(bool)

    strings = s.astype("string").str.lower().str.strip().fillna("")
    uniq = set(strings.unique())
    if not uniq:
        return None

    train_tokens = {"train", "tr", "training", "valid", "validation", "val", "1", "true", "t", "yes", "y"}
    test_tokens = {"test", "te", "0", "false", "f", "no", "n"}
    all_tokens = train_tokens | test_tokens

    if uniq.issubset(all_tokens):
        base_mask = strings.isin(train_tokens).to_numpy(dtype=bool)
        return base_mask if positive_is_train else ~base_mask

    return None

def _extract_train_mask(raw, aux):
    n = len(raw)

    candidate_frames = []
    if isinstance(aux, pd.DataFrame) and len(aux) == n and not aux.empty:
        candidate_frames.append(aux)
    if isinstance(raw, pd.DataFrame) and len(raw) == n:
        candidate_frames.append(raw)

    explicit_train_cols = {
        "is_train",
        "train",
        "is_train_row",
        "train_row",
        "is_train_mask",
        "train_mask",
        "training",
    }
    explicit_test_cols = {
        "is_test",
        "test",
        "is_test_row",
        "test_row",
        "is_test_mask",
        "test_mask",
        "testing",
    }
    split_cols = {"split", "data_split", "partition", "subset", "phase", "group"}

    for frame in candidate_frames:
        for col in frame.columns:
            key = str(col).lower()
            if key in explicit_train_cols:
                parsed = _coerce_train_mask_from_series(frame[col], positive_is_train=True)
                if parsed is not None:
                    return parsed
            if key in explicit_test_cols:
                parsed = _coerce_train_mask_from_series(frame[col], positive_is_train=False)
                if parsed is not None:
                    return parsed
            if key in split_cols:
                parsed = _coerce_train_mask_from_series(frame[col], positive_is_train=True)
                if parsed is not None:
                    return parsed

    if "id" in raw.columns:
        ids = pd.to_numeric(raw["id"], errors="coerce")
        if ids.notna().all():
            ids_sorted = np.sort(ids.to_numpy(dtype=np.float64))
            if ids_sorted.size >= 2:
                gaps = np.diff(ids_sorted)
                jumps = np.flatnonzero(gaps > 1)
                if jumps.size > 0:
                    cutoff = ids_sorted[jumps[0]] + 1.0
                    return (ids.to_numpy(dtype=np.float64) < cutoff).astype(bool)

    return np.ones(n, dtype=bool)

def _category_codes(raw):
    n = len(raw)
    if "spectral_type" not in raw.columns or "galaxy_population" not in raw.columns:
        return np.zeros(n, dtype=np.int64)

    spectral = raw["spectral_type"].astype("string").fillna("__NA__").astype(str).str.strip()
    population = raw["galaxy_population"].astype("string").fillna("__NA__").astype(str).str.strip()
    labels = (spectral + "|" + population).to_numpy()
    codes, _ = pd.factorize(labels)
    return codes.astype(np.int64)

def _global_eigen_stats(X_train):
    d = X_train.shape[1]
    if X_train.shape[0] == 0:
        mean = np.zeros(d, dtype=np.float64)
        cov = np.eye(d, dtype=np.float64)
    else:
        mean = np.nanmean(X_train, axis=0)
        centered = X_train - mean
        if X_train.shape[0] > 1:
            cov = (centered.T @ centered) / float(X_train.shape[0] - 1)
        else:
            cov = np.zeros((d, d), dtype=np.float64)

    cov = 0.5 * (cov + cov.T) + _EPSILON * np.eye(d, dtype=np.float64)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    return mean, eigvals[order], eigvecs[:, order]

def _linear_quantile_bounds(neighbors):
    n, d = neighbors.shape

    if n == 0:
        nan = np.full(d, np.nan, dtype=np.float64)
        return nan, nan

    if n == 1:
        vec = neighbors[0].astype(np.float64)
        return vec, vec.copy()

    pos_low = int(np.floor(0.05 * (n - 1)))
    pos_high = int(np.floor(0.95 * (n - 1)))
    lo_low = int(np.floor(pos_low))
    hi_low = int(np.ceil(pos_low))
    lo_high = int(np.floor(pos_high))
    hi_high = int(np.ceil(pos_high))

    idx = sorted(set((lo_low, hi_low, lo_high, hi_high)))
    partitioned = np.partition(neighbors, idx, axis=0)

    q05 = partitioned[lo_low]
    if hi_low != lo_low:
        q05 = q05 + (partitioned[hi_low] - q05) * (pos_low - lo_low)

    q95 = partitioned[lo_high]
    if hi_high != lo_high:
        q95 = q95 + (partitioned[hi_high] - q95) * (pos_high - lo_high)

    return q05, q95

def _compute_row_metrics(row_vec, neighbors, neighbor_dists, global_stats):
    global_mean, global_eigvals, global_eigvecs = global_stats
    d = row_vec.shape[0]

    metrics = {
        "r2": np.nan,
        "residual2": np.nan,
        "a12": np.nan,
        "a23": np.nan,
        "tail": np.nan,
        "dim95": np.nan,
        "rk": np.nan,
        "r_med": np.nan,
        "r_mean": np.nan,
        "density_proxy": np.nan,
        "r_ratio": np.nan,
        "frac_in_box": np.nan,
        "asym_mad": np.nan,
        "asym_mad_ratio": np.nan,
        "degenerate": 1.0,
        "n_neighbors": 0.0,
    }

    if neighbors is None or neighbors.shape[0] == 0:
        eigvals = global_eigvals
        eigvecs = global_eigvecs
        centroid = global_mean
    else:
        metrics["n_neighbors"] = float(neighbors.shape[0])

        k = neighbors.shape[0]
        centroid = np.mean(neighbors, axis=0)

        if k > 1:
            centered = neighbors - centroid
            cov = (centered.T @ centered) / float(k - 1)
        else:
            cov = np.zeros((d, d), dtype=np.float64)

        cov = 0.5 * (cov + cov.T) + _EPSILON * np.eye(d, dtype=np.float64)
        eigvals, eigvecs = np.linalg.eigh(cov)
        order = np.argsort(eigvals)[::-1]
        eigvals = eigvals[order]
        eigvecs = eigvecs[:, order]

        eig_sum = float(np.sum(eigvals))
        min_eig = float(np.min(eigvals))
        if eig_sum <= _EPSILON or min_eig <= _EPSILON or not np.isfinite(eig_sum):
            eigvals = global_eigvals
            eigvecs = global_eigvecs
            centroid = global_mean
            metrics["degenerate"] = 1.0
        else:
            metrics["degenerate"] = 0.0

    eig_sum = float(np.sum(eigvals))
    if eig_sum <= _EPSILON:
        metrics["r2"] = np.nan
        metrics["tail"] = np.nan
        metrics["a12"] = np.nan
        metrics["a23"] = np.nan
        metrics["dim95"] = np.nan
        metrics["residual2"] = np.nan
    else:
        metrics["r2"] = float((eigvals[0] + eigvals[1]) / eig_sum)
        metrics["tail"] = float(eigvals[-1] / eig_sum)
        metrics["a12"] = float(np.log((eigvals[0] + _EPSILON) / (eigvals[1] + _EPSILON)))
        metrics["a23"] = float(np.log((eigvals[1] + _EPSILON) / (max(eigvals[2], _EPSILON) + _EPSILON)))
        cum = np.cumsum(eigvals / eig_sum)
        metrics["dim95"] = float(np.searchsorted(cum, 0.95, side="left") + 1)

        row_delta = row_vec - centroid
        v2 = eigvecs[:, :2]
        orth = row_delta - (v2 @ (v2.T @ row_delta))
        residual_denom = float(np.sum(eigvals[2:]) + _EPSILON)
        metrics["residual2"] = float((orth @ orth) / residual_denom)

    if neighbor_dists is not None and neighbor_dists.size > 0:
        metrics["rk"] = float(np.max(neighbor_dists))
        metrics["r_med"] = _safe_median_1d(neighbor_dists)
        metrics["r_mean"] = float(np.mean(neighbor_dists))
        metrics["density_proxy"] = float(1.0 / (metrics["r_med"] + _EPSILON))
        metrics["r_ratio"] = float(metrics["rk"] / (metrics["r_med"] + _EPSILON))

        q05, q95 = _linear_quantile_bounds(neighbors)
        within_box = (row_vec >= q05) & (row_vec <= q95)
        metrics["frac_in_box"] = float(np.mean(within_box.astype(np.float64)))
    else:
        metrics["r_ratio"] = np.nan

    if neighbors is not None and neighbors.shape[0] > 0:
        nbr_center = np.mean(neighbors, axis=0)
        nbr_mad = np.mean(np.abs(neighbors - nbr_center), axis=0)
        nbr_mad = np.where(nbr_mad < _MAD_CLIP, _MAD_CLIP, nbr_mad)
        metrics["asym_mad"] = float(np.mean(np.abs(row_vec - nbr_center)))
        metrics["asym_mad_ratio"] = float(metrics["asym_mad"] / (float(np.mean(nbr_mad)) + _EPSILON))

    return metrics

def _write_metrics(output, pos, values):
    for key in _FEATURE_KEYS:
        output[key][pos] = values[key]

def _compute_space_features(raw, feature_matrix, train_mask, category_codes):
    n = feature_matrix.shape[0]
    train_mask = np.asarray(train_mask, dtype=bool)
    if train_mask.size != n:
        train_mask = np.ones(n, dtype=bool)

    train_indices = np.flatnonzero(train_mask)
    if train_indices.size == 0:
        train_indices = np.arange(n, dtype=np.int64)
        train_mask = np.ones(n, dtype=bool)

    X = np.asarray(feature_matrix, dtype=np.float64)
    train_X = X[train_indices]

    med = np.nanmedian(train_X, axis=0)
    mad = np.nanmedian(np.abs(train_X - med), axis=0)
    mad = np.where(mad < _MAD_CLIP, _MAD_CLIP, mad)
    X_scaled = (X - med) / mad

    global_stats = _global_eigen_stats(X_scaled[train_indices])
    all_indices = np.arange(n, dtype=np.int64)

    n_train = int(train_indices.size)
    base_k = _neighbor_cap(n_train)
    nn_neighbors = int(min(n_train, base_k + 1))
    if nn_neighbors <= 0:
        all_idx = np.empty((n, 0), dtype=np.int32)
        all_dist = np.empty((n, 0), dtype=np.float32)
    else:
        nn = NearestNeighbors(
            n_neighbors=nn_neighbors,
            algorithm="kd_tree",
            metric="euclidean",
            n_jobs=-1,
        )
        nn.fit(X_scaled[train_indices])
        all_dist, all_idx = nn.kneighbors(X_scaled[all_indices], n_neighbors=nn_neighbors, return_distance=True)
        all_dist = np.asarray(all_dist, dtype=np.float32)
        all_idx = np.asarray(all_idx, dtype=np.int32)

    global_k = np.where(
        train_mask,
        np.minimum(base_k, np.maximum(n_train - 1, 0)),
        np.minimum(base_k, n_train),
    ).astype(np.int64)

    cat_codes = np.asarray(category_codes, dtype=np.int64)
    if cat_codes.size != n:
        cat_codes = np.zeros(n, dtype=np.int64)

    max_code = int(cat_codes.max(initial=0))
    train_cat_codes = cat_codes[train_mask]
    if train_cat_codes.size > 0:
        cat_counts = np.bincount(train_cat_codes, minlength=max_code + 1)
    else:
        cat_counts = np.array([0], dtype=np.int64)

    cat_counts_per_row = cat_counts[cat_codes] if cat_counts.size > 0 else np.zeros(n, dtype=np.int64)
    local_k = np.where(train_mask, cat_counts_per_row - 1, cat_counts_per_row).astype(np.int64)
    local_k = np.clip(local_k, 0, base_k)

    sparse_mask = local_k < _MIN_STABLE_K
    fallback_mask = (local_k == 0) | (cat_counts_per_row < (2 * local_k))
    fallback_mask = fallback_mask.astype(bool)

    space_features = {key: np.full(n, np.nan, dtype=np.float64) for key in _FEATURE_KEYS}

    for pos in range(n):
        row_vec = X_scaled[pos]
        row_code = cat_codes[pos]
        row_local_k = int(local_k[pos])

        row_idx = all_idx[pos]
        row_dist = all_dist[pos]

        if train_mask[pos] and row_idx.size > 0 and row_idx[0] == pos:
            row_idx = row_idx[1:]
            row_dist = row_dist[1:]
        elif train_mask[pos] and row_idx.size > 0:
            keep = row_idx != pos
            row_idx = row_idx[keep]
            row_dist = row_dist[keep]

        use_global = True
        sparse = bool(sparse_mask[pos])

        if not fallback_mask[pos] and row_local_k > 0 and row_idx.size >= row_local_k:
            same_cat = cat_codes[row_idx] == row_code
            local_candidates = row_idx[same_cat]
            local_dist = row_dist[same_cat]
            if local_candidates.size >= row_local_k:
                local_metrics = _compute_row_metrics(
                    row_vec,
                    X_scaled[local_candidates[:row_local_k]],
                    local_dist[:row_local_k],
                    global_stats,
                )
                _write_metrics(space_features, pos, local_metrics)
                use_global = sparse
                if not sparse:
                    continue
            else:
                use_global = True

        if use_global:
            g_k = int(global_k[pos])
            g_count = int(min(g_k, row_idx.size))
            if g_k <= 0 or g_count <= 0:
                global_metrics = _compute_row_metrics(row_vec, None, None, global_stats)
            else:
                global_metrics = _compute_row_metrics(
                    row_vec,
                    X_scaled[row_idx[:g_count]],
                    row_dist[:g_count],
                    global_stats,
                )

            if fallback_mask[pos] or sparse is True:
                _write_metrics(space_features, pos, global_metrics)
            else:
                _write_metrics(space_features, pos, global_metrics)
                for key in _SPARSE_REPLACE_KEYS:
                    space_features[key][pos] = global_metrics[key]

    space_features["sparse_flag"] = sparse_mask.astype(np.float64)
    return space_features

def add_local_manifold_codimension(raw, deps, aux):
    if raw is None or len(raw) == 0:
        return pd.DataFrame(index=pd.Index([]) if raw is None else raw.index)

    u = raw["u"].to_numpy(dtype=np.float64)
    g = raw["g"].to_numpy(dtype=np.float64)
    r = raw["r"].to_numpy(dtype=np.float64)
    i = raw["i"].to_numpy(dtype=np.float64)
    z = raw["z"].to_numpy(dtype=np.float64)
    redshift = raw["redshift"].to_numpy(dtype=np.float64)

    c1 = u - g
    c2 = g - r
    c3 = r - i
    c4 = i - z

    X_s1 = np.column_stack((c1, c2, c3, c4, redshift))
    X_s2 = np.column_stack((u, g, r, i, z))

    train_mask = _extract_train_mask(raw, aux)
    category_codes = _category_codes(raw)

    s1 = _compute_space_features(raw, X_s1, train_mask, category_codes)
    s2 = _compute_space_features(raw, X_s2, train_mask, category_codes)

    out = pd.DataFrame(index=raw.index)

    for key in _FEATURE_KEYS:
        out[f"{_SPACE_PREFIX}_s1_{key}"] = s1[key]
        out[f"{_SPACE_PREFIX}_s2_{key}"] = s2[key]

    out[f"{_SPACE_PREFIX}_s1_sparse_flag"] = s1["sparse_flag"]
    out[f"{_SPACE_PREFIX}_s2_sparse_flag"] = s2["sparse_flag"]

    out[f"{_SPACE_PREFIX}_abs_delta_r2"] = (out[f"{_SPACE_PREFIX}_s1_r2"] - out[f"{_SPACE_PREFIX}_s2_r2"]).abs()
    out[f"{_SPACE_PREFIX}_abs_delta_residual2"] = (
        out[f"{_SPACE_PREFIX}_s1_residual2"] - out[f"{_SPACE_PREFIX}_s2_residual2"]
    ).abs()
    out[f"{_SPACE_PREFIX}_abs_delta_density_proxy"] = (
        out[f"{_SPACE_PREFIX}_s1_density_proxy"] - out[f"{_SPACE_PREFIX}_s2_density_proxy"]
    ).abs()
    out[f"{_SPACE_PREFIX}_abs_delta_r_ratio"] = (
        out[f"{_SPACE_PREFIX}_s1_r_ratio"] - out[f"{_SPACE_PREFIX}_s2_r_ratio"]
    ).abs()
    out[f"{_SPACE_PREFIX}_abs_delta_frac_in_box"] = (
        out[f"{_SPACE_PREFIX}_s1_frac_in_box"] - out[f"{_SPACE_PREFIX}_s2_frac_in_box"]
    ).abs()

    return out

FEATURE_GROUPS = [
    {
        "name": "local_manifold_codimension",
        "fn": add_local_manifold_codimension,
        "depends_on": [],
        "description": "Builds manifold curvature and neighborhood-support descriptors from scaled photometric spaces with category-aware fallback and timeout-aware neighbor budgeting.",
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