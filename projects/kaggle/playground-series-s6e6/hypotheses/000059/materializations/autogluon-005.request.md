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
- Hypothesis ID: 000059
- Source file: autogluon-004.py
- Failed node: 20260628T071319-d5f39997-138
- Run: 20260624T200326-681cd81b

# Execution Error

```text
failed.yaml error:
Process exited with 1

stderr.log:
Traceback (most recent call last):
  File "/home/xai/DEV/TheML/projects/kaggle/playground-series-s6e6/runs/20260624T200326-681cd81b/artifacts/20260628T071319-d5f39997-138/02-code.py", line 799, in main
    transformed = run_feature_groups(
                  ^^^^^^^^^^^^^^^^^^^
  File "/home/xai/DEV/TheML/src/tml/features/groups.py", line 59, in run_feature_groups
    block = group["fn"](raw.copy(), {key: value.copy() for key, value in deps.items()}, aux.copy())
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/xai/DEV/TheML/projects/kaggle/playground-series-s6e6/runs/20260624T200326-681cd81b/artifacts/20260628T071319-d5f39997-138/02-code.py", line 182, in add_tag_redshift_compatibility_residuals
    global_profile = _profile_from_colors(finite_train_colors, global_diag, sigma_floor)
                     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/xai/DEV/TheML/projects/kaggle/playground-series-s6e6/runs/20260624T200326-681cd81b/artifacts/20260628T071319-d5f39997-138/02-code.py", line 116, in _profile_from_colors
    jitter = max(float(np.nanmedian(global_diag)) * JITTER_FRACTION, sigma_floor ** 2 * JITTER_FRACTION, 1e-8)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
ValueError: The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()
Traceback (most recent call last):
  File "/home/xai/DEV/TheML/projects/kaggle/playground-series-s6e6/runs/20260624T200326-681cd81b/artifacts/20260628T071319-d5f39997-138/02-code.py", line 923, in <module>
    main()
  File "/home/xai/DEV/TheML/projects/kaggle/playground-series-s6e6/runs/20260624T200326-681cd81b/artifacts/20260628T071319-d5f39997-138/02-code.py", line 799, in main
    transformed = run_feature_groups(
                  ^^^^^^^^^^^^^^^^^^^
  File "/home/xai/DEV/TheML/src/tml/features/groups.py", line 59, in run_feature_groups
    block = group["fn"](raw.copy(), {key: value.copy() for key, value in deps.items()}, aux.copy())
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/xai/DEV/TheML/projects/kaggle/playground-series-s6e6/runs/20260624T200326-681cd81b/artifacts/20260628T071319-d5f39997-138/02-code.py", line 182, in add_tag_redshift_compatibility_residuals
    global_profile = _profile_from_colors(finite_train_colors, global_diag, sigma_floor)
                     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/xai/DEV/TheML/projects/kaggle/playground-series-s6e6/runs/20260624T200326-681cd81b/artifacts/20260628T071319-d5f39997-138/02-code.py", line 116, in _profile_from_colors
    jitter = max(float(np.nanmedian(global_diag)) * JITTER_FRACTION, sigma_floor ** 2 * JITTER_FRACTION, 1e-8)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
ValueError: The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()

stdout.log:
AutoGluon materialization: loaded aux file star_classification.csv rows=100000 cols=18 passed_to_groups=True
AutoGluon materialization: starting feature groups
TheML feature group: start name=tag_redshift_compatibility_residuals
TheML feature group: failed name=tag_redshift_compatibility_residuals elapsed_s=0.448 rows=824782 cols=0
```

# Previous Code

```python
import numpy as np
import pandas as pd

MIN_BIN_ROWS = 3000
MIN_PROFILE_ROWS = 150
N_CANDIDATE_BINS = 160
COV_SHRINKAGE = 0.35
JITTER_FRACTION = 0.0001
SIGMA_FLOOR_FRACTION = 0.03
DISTANCE_CLIP_LO = 0.5
DISTANCE_CLIP_HI = 99.5
TRAIN_ID_MAX = 577346
COLOR_COLUMNS = ("u_g", "g_r", "r_i", "i_z")
STATE_SEP = "__"

def _as_numeric_array(frame, columns):
    values = []
    for col in columns:
        values.append(pd.to_numeric(frame[col], errors="coerce").to_numpy(dtype=float))
    return values

def _safe_median(values, fallback=0.0):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float(fallback)
    return float(np.median(values))

def _safe_quantile(values, q, fallback):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float(fallback)
    return float(np.percentile(values, q))

def _make_redshift_bins(t_train):
    t_train = np.asarray(t_train, dtype=float)
    t_train = t_train[np.isfinite(t_train)]
    if t_train.size == 0:
        return np.array([-np.inf, np.inf], dtype=float)

    quantiles = np.linspace(0.0, 1.0, N_CANDIDATE_BINS + 1)
    edges = np.quantile(t_train, quantiles)
    edges = np.unique(edges)

    if edges.size < 2:
        center = float(edges[0]) if edges.size else 0.0
        return np.array([-np.inf, center, np.inf], dtype=float)

    candidate = np.searchsorted(edges[1:-1], t_train, side="right")
    counts = np.bincount(candidate, minlength=edges.size - 1)

    merged_edges = [float(edges[0])]
    running = 0
    for idx, count in enumerate(counts):
        running += int(count)
        is_last = idx == len(counts) - 1
        if running >= MIN_BIN_ROWS or is_last:
            merged_edges.append(float(edges[idx + 1]))
            running = 0

    if len(merged_edges) < 2:
        merged_edges = [float(edges[0]), float(edges[-1])]

    merged_edges[0] = -np.inf
    merged_edges[-1] = np.inf
    return np.asarray(merged_edges, dtype=float)

def _bin_values(t_values, edges):
    bins = np.searchsorted(edges[1:-1], t_values, side="right")
    return np.clip(bins, 0, max(0, len(edges) - 2)).astype(int)

def _profile_from_colors(colors, global_diag, sigma_floor):
    colors = np.asarray(colors, dtype=float)
    finite = np.all(np.isfinite(colors), axis=1)
    colors = colors[finite]
    n_rows = int(colors.shape[0])

    if n_rows == 0:
        mu = np.zeros(4, dtype=float)
        sigma = np.maximum(np.sqrt(np.maximum(global_diag, 0.0)), sigma_floor)
        inv_cov = np.diag(1.0 / np.maximum(global_diag, sigma_floor ** 2))
        return {"n": 0, "mu": mu, "sigma": sigma, "inv_cov": inv_cov}

    mu = np.median(colors, axis=0)
    mad = np.median(np.abs(colors - mu), axis=0)
    sigma = np.maximum(1.4826 * mad, sigma_floor)

    lo = np.percentile(colors, 1.0, axis=0)
    hi = np.percentile(colors, 99.0, axis=0)
    clipped = np.clip(colors, lo, hi)

    if n_rows >= 2:
        cov = np.cov(clipped, rowvar=False)
        if cov.shape == ():
            cov = np.diag(np.repeat(float(cov), 4))
    else:
        cov = np.diag(np.square(sigma))

    cov = np.asarray(cov, dtype=float)
    if cov.shape != (4, 4) or not np.all(np.isfinite(cov)):
        cov = np.diag(np.square(sigma))

    target = np.diag(np.maximum(global_diag, sigma_floor ** 2))
    shrunk = (1.0 - COV_SHRINKAGE) * cov + COV_SHRINKAGE * target
    jitter = max(float(np.nanmedian(global_diag)) * JITTER_FRACTION, sigma_floor ** 2 * JITTER_FRACTION, 1e-8)
    shrunk = shrunk + np.eye(4) * jitter

    try:
        inv_cov = np.linalg.pinv(shrunk)
    except np.linalg.LinAlgError:
        inv_cov = np.diag(1.0 / np.maximum(np.diag(shrunk), sigma_floor ** 2))

    return {"n": n_rows, "mu": mu, "sigma": sigma, "inv_cov": inv_cov}

def _state_key(spectral, population):
    return spectral.astype(str) + STATE_SEP + population.astype(str)

def _state_parts(state):
    parts = str(state).split(STATE_SEP, 1)
    if len(parts) == 2:
        return parts[0], parts[1]
    return str(state), ""

def add_tag_redshift_compatibility_residuals(raw, deps, aux):
    out_index = raw.index
    required = ["id", "u", "g", "r", "i", "z", "redshift", "spectral_type", "galaxy_population"]
    missing = [col for col in required if col not in raw.columns]
    if missing:
        return pd.DataFrame(index=out_index)

    frame = raw.loc[:, required].copy()

    ids = pd.to_numeric(frame["id"], errors="coerce")
    train_mask = ids.le(TRAIN_ID_MAX).to_numpy()
    if int(np.sum(train_mask)) < MIN_PROFILE_ROWS:
        train_mask = np.zeros(len(frame), dtype=bool)
        train_mask[: max(1, len(frame) // 2)] = True

    u, g, r, i_mag, z_mag = _as_numeric_array(frame, ["u", "g", "r", "i", "z"])
    colors = np.column_stack([u - g, g - r, r - i_mag, i_mag - z_mag])
    colors = np.where(np.isfinite(colors), colors, np.nan)

    redshift = pd.to_numeric(frame["redshift"], errors="coerce").to_numpy(dtype=float)
    redshift_t = np.log1p(np.maximum(np.nan_to_num(redshift, nan=0.0), 0.0))

    train_colors = colors[train_mask]
    finite_train_colors = train_colors[np.all(np.isfinite(train_colors), axis=1)]
    if finite_train_colors.size == 0:
        finite_train_colors = np.zeros((1, 4), dtype=float)

    global_diag = np.nanvar(finite_train_colors, axis=0)
    global_diag = np.where(np.isfinite(global_diag) & (global_diag > 1e-8), global_diag, 1.0)
    global_scale = np.sqrt(np.maximum(global_diag, 1e-8))
    sigma_floor = np.maximum(global_scale * SIGMA_FLOOR_FRACTION, 1e-4)

    edges = _make_redshift_bins(redshift_t[train_mask])
    bins = _bin_values(redshift_t, edges)
    n_bins = max(1, len(edges) - 1)

    states = _state_key(frame["spectral_type"], frame["galaxy_population"]).to_numpy()
    train_states = states[train_mask]
    unique_states = sorted(pd.unique(train_states).astype(str).tolist())
    if len(unique_states) == 0:
        unique_states = sorted(pd.unique(states).astype(str).tolist())

    profiles = {}
    state_profiles = {}
    global_profile = _profile_from_colors(finite_train_colors, global_diag, sigma_floor)

    for state in unique_states:
        state_train_mask = train_mask & (states == state)
        state_profiles[state] = _profile_from_colors(colors[state_train_mask], global_diag, sigma_floor)
        for bin_id in range(n_bins):
            cell_mask = state_train_mask & (bins == bin_id)
            if int(np.sum(cell_mask)) > 0:
                profiles[(state, bin_id)] = _profile_from_colors(colors[cell_mask], global_diag, sigma_floor)

    def select_profile(state, bin_id):
        exact = profiles.get((state, int(bin_id)))
        if exact is not None and exact["n"] >= MIN_PROFILE_ROWS:
            return exact, 0

        for radius in range(1, n_bins + 1):
            best = None
            best_count = -1
            left = int(bin_id) - radius
            right = int(bin_id) + radius
            for candidate_bin in (left, right):
                if 0 <= candidate_bin < n_bins:
                    candidate = profiles.get((state, candidate_bin))
                    if candidate is not None and candidate["n"] >= MIN_PROFILE_ROWS and candidate["n"] > best_count:
                        best = candidate
                        best_count = candidate["n"]
            if best is not None:
                return best, min(radius, 99)

        state_profile = state_profiles.get(state)
        if state_profile is not None and state_profile["n"] >= MIN_PROFILE_ROWS:
            return state_profile, 100

        return global_profile, 101

    n_rows = len(frame)
    n_states = len(unique_states)
    all_d2 = np.empty((n_rows, n_states), dtype=float)
    all_depth = np.empty((n_rows, n_states), dtype=float)
    all_z = np.empty((n_rows, n_states, 4), dtype=float)

    for state_idx, state in enumerate(unique_states):
        for bin_id in range(n_bins):
            row_mask = bins == bin_id
            if not np.any(row_mask):
                continue
            profile, depth = select_profile(state, bin_id)
            residual = colors[row_mask] - profile["mu"]
            residual = np.where(np.isfinite(residual), residual, 0.0)
            d2 = np.einsum("ij,jk,ik->i", residual, profile["inv_cov"], residual)
            all_d2[row_mask, state_idx] = d2
            all_depth[row_mask, state_idx] = float(depth)
            all_z[row_mask, state_idx, :] = residual / profile["sigma"]

    if n_states == 0:
        return pd.DataFrame(index=out_index)

    clip_lo = _safe_quantile(all_d2[train_mask].ravel(), DISTANCE_CLIP_LO, 0.0)
    clip_hi = _safe_quantile(all_d2[train_mask].ravel(), DISTANCE_CLIP_HI, max(clip_lo + 1.0, 1.0))
    if clip_hi <= clip_lo:
        clip_hi = clip_lo + 1.0
    all_d2 = np.clip(all_d2, clip_lo, clip_hi)

    state_to_idx = {state: idx for idx, state in enumerate(unique_states)}
    self_idx = np.array([state_to_idx.get(str(state), -1) for state in states], dtype=int)
    unknown_self = self_idx < 0
    if np.any(unknown_self):
        self_idx[unknown_self] = np.argmin(all_d2[unknown_self], axis=1)

    row_numbers = np.arange(n_rows)
    d_self = all_d2[row_numbers, self_idx]
    depth_self = all_depth[row_numbers, self_idx]
    z_self = all_z[row_numbers, self_idx, :]

    alt_d2 = all_d2.copy()
    alt_d2[row_numbers, self_idx] = np.inf
    best_alt_idx = np.argmin(alt_d2, axis=1)
    d_best_alt = alt_d2[row_numbers, best_alt_idx]

    alt_d2_second = alt_d2.copy()
    alt_d2_second[row_numbers, best_alt_idx] = np.inf
    d_second_alt = np.min(alt_d2_second, axis=1)
    if n_states <= 2:
        d_second_alt = d_best_alt.copy()

    depth_best_alt = all_depth[row_numbers, best_alt_idx]
    z_best_alt = all_z[row_numbers, best_alt_idx, :]

    rank_self = 1 + np.sum(all_d2 < d_self[:, None], axis=1)
    centered = all_d2 - np.min(all_d2, axis=1, keepdims=True)
    weights = np.exp(-0.5 * np.clip(centered, 0.0, 100.0))
    weights_sum = np.sum(weights, axis=1, keepdims=True)
    probs = weights / np.maximum(weights_sum, 1e-12)
    entropy = -np.sum(probs * np.log(np.maximum(probs, 1e-12)), axis=1)
    concentration = np.max(probs, axis=1)

    best_alt_states = np.array(unique_states, dtype=object)[best_alt_idx]
    best_alt_spectral = []
    best_alt_population = []
    for state in best_alt_states:
        spectral, population = _state_parts(state)
        best_alt_spectral.append(spectral)
        best_alt_population.append(population)

    result = pd.DataFrame(index=out_index)
    result["d_self"] = d_self
    result["d_best_alt"] = d_best_alt
    result["d_second_alt"] = d_second_alt
    result["margin_best_alt_minus_self"] = d_best_alt - d_self
    result["margin_second_alt_minus_self"] = d_second_alt - d_self
    result["self_distance_rank"] = rank_self.astype(np.int16)
    result["state_distance_entropy"] = entropy
    result["state_softmin_concentration"] = concentration

    for idx, col in enumerate(COLOR_COLUMNS):
        result["self_z_" + col] = np.clip(z_self[:, idx], -25.0, 25.0)
    for idx, col in enumerate(COLOR_COLUMNS):
        result["best_alt_z_" + col] = np.clip(z_best_alt[:, idx], -25.0, 25.0)

    result["fallback_depth_self"] = depth_self
    result["fallback_depth_best_alt"] = depth_best_alt
    result["best_alt_spectral_type"] = pd.Series(best_alt_spectral, index=out_index, dtype="object")
    result["best_alt_galaxy_population"] = pd.Series(best_alt_population, index=out_index, dtype="object")
    result["best_alt_matches_spectral_type"] = result["best_alt_spectral_type"].to_numpy() == frame["spectral_type"].astype(str).to_numpy()
    result["best_alt_matches_galaxy_population"] = result["best_alt_galaxy_population"].to_numpy() == frame["galaxy_population"].astype(str).to_numpy()

    return result

FEATURE_GROUPS = [
    {
        "name": "tag_redshift_compatibility_residuals",
        "fn": add_tag_redshift_compatibility_residuals,
        "depends_on": [],
        "description": "Robust redshift-binned tag compatibility distances and residual margins from broadband color profiles.",
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