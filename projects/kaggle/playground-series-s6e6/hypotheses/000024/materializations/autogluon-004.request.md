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
- Hypothesis ID: 000024
- Source file: autogluon-003.py
- Failed node: 
- Run: 

# Execution Error

```text
Invalid materialization for hypothesis 000024 (autogluon); response=hypotheses/000024/materializations/autogluon-003.response.md: Group materialization must not call functions in top-level assignments
```

# Previous Code

```python
import numpy as np
import pandas as pd

_TRAIN_ID_SPLIT = 577346
_TRAIN_ROWS_ESTIMATE = 577347
_SMOOTHING_ALPHA = 1.0
_Q12_LEVELS = tuple(i / 12 for i in range(13))
_PAIR_SEPARATOR = "||"
_CONTEXT_FEATURES = ("u", "g", "r", "i", "z", "redshift_clip", "redshift_log")

def _infer_train_mask(raw):
    ids = raw["id"] if "id" in raw.columns else pd.Series(np.arange(len(raw)), index=raw.index)
    ids_numeric = pd.to_numeric(ids, errors="coerce")

    if ids_numeric.notna().all() and float(ids_numeric.min()) == 0.0:
        if float(ids_numeric.max()) >= float(_TRAIN_ID_SPLIT + 1):
            return ids_numeric.le(_TRAIN_ID_SPLIT)

    n_train = min(_TRAIN_ROWS_ESTIMATE, len(raw))
    if n_train <= 0:
        n_train = len(raw)

    train_mask = pd.Series(False, index=raw.index)
    train_mask.iloc[:n_train] = True
    return train_mask

def _laplace_smoothed_freq(train_series, query_series, alpha):
    train_series = train_series.astype("object")
    query_series = query_series.astype("object")
    n_train = float(len(train_series))
    counts = train_series.value_counts(dropna=False)
    vocab = max(len(counts), 1)
    denom = n_train + alpha * vocab
    mapped = counts.reindex(query_series, fill_value=0.0)
    return (mapped + alpha) / denom

def _build_rank_lookup(train_values):
    train_float = pd.to_numeric(train_values, errors="coerce").dropna()
    train_clean = train_float.to_numpy(dtype=float)
    n = int(train_clean.size)

    if n == 0:
        return np.array([], dtype=float), np.array([0.5], dtype=float)

    train_sorted = np.sort(train_clean)
    uniq, counts = np.unique(train_sorted, return_counts=True)
    cumulative = np.cumsum(counts)
    start = cumulative - counts + 1.0
    avg_pos = (start + cumulative) / 2.0
    if n == 1:
        ranks = np.array([1.0], dtype=float)
    else:
        ranks = (avg_pos - 0.5) / (n - 0.5)
    return uniq.astype(float), ranks.astype(float)

def _map_empirical_rank(all_values, train_values):
    train_x, train_r = _build_rank_lookup(train_values)
    out = pd.Series(np.nan, index=all_values.index, dtype=float)
    if train_x.size == 0:
        return out

    values = pd.to_numeric(all_values, errors="coerce")
    valid = values[np.isfinite(values)]
    if valid.empty:
        return out

    mapped = np.interp(valid.to_numpy(dtype=float), train_x, train_r, left=train_r[0], right=train_r[-1])
    out.loc[valid.index] = mapped
    return out

def _withingroup_ranks(values, group_values, train_mask, global_ranks):
    out = global_ranks.copy()
    if group_values.empty:
        return out

    train_groups = group_values.loc[train_mask].dropna().unique()
    for group_value in train_groups:
        group_rows = group_values.eq(group_value)
        group_train_mask = train_mask & group_rows
        n_group = int(group_train_mask.sum())
        if n_group <= 1:
            continue
        mapped = _map_empirical_rank(values[group_rows], values.loc[group_train_mask])
        out.loc[group_rows] = mapped
    return out

def _quantile_bins_and_freq(train_values, all_values, q_levels):
    train_values = pd.to_numeric(train_values, errors="coerce")
    all_values = pd.to_numeric(all_values, errors="coerce")
    index = all_values.index

    finite_train = train_values[np.isfinite(train_values)]
    if finite_train.empty:
        return (
            pd.Series(0, index=index, dtype="int64"),
            pd.Series(0.0, index=index, dtype=float),
        )

    edges = np.quantile(finite_train.to_numpy(), np.array(q_levels, dtype=float), interpolation="linear")
    edges = np.unique(np.asarray(edges, dtype=float))

    if edges.size < 2:
        center = float(finite_train.min())
        if not np.isfinite(center):
            edges = np.array([0.0, 1.0], dtype=float)
        else:
            width = abs(center) * 1e-6
            if not np.isfinite(width) or width == 0.0:
                width = 1.0
            edges = np.array([center - width, center + width], dtype=float)

    train_clipped = train_values.clip(lower=edges[0], upper=edges[-1])
    all_clipped = all_values.clip(lower=edges[0], upper=edges[-1])

    train_bins = pd.cut(train_clipped, bins=edges, labels=False, include_lowest=True)
    all_bins = pd.cut(all_clipped, bins=edges, labels=False, include_lowest=True)
    all_bins = all_bins.fillna(-1).astype(int)

    denom = float(len(train_clipped))
    if denom <= 0.0:
        denom = 1.0
    freq_by_bin = train_bins.value_counts(dropna=False) / denom
    bin_freq = all_bins.map(freq_by_bin).fillna(0.0)

    return all_bins, bin_freq.astype(float)

def add_aide_catalog_rank_frequency_context(raw, deps, aux):
    train_mask = _infer_train_mask(raw)
    n_train = int(train_mask.sum())
    n_train_safe = max(n_train, 1)

    spectral = raw["spectral_type"].astype("object")
    galaxy = raw["galaxy_population"].astype("object")
    pair = spectral.astype("string").str.cat(galaxy.astype("string"), sep=_PAIR_SEPARATOR)

    spectral_freq = _laplace_smoothed_freq(spectral[train_mask], spectral, _SMOOTHING_ALPHA)
    galaxy_freq = _laplace_smoothed_freq(galaxy[train_mask], galaxy, _SMOOTHING_ALPHA)
    pair_freq = _laplace_smoothed_freq(pair[train_mask], pair, _SMOOTHING_ALPHA)

    features = pd.DataFrame(index=raw.index)
    features["spectral_type_freq"] = spectral_freq
    features["spectral_type_logfreq"] = np.log(spectral_freq)
    features["galaxy_population_freq"] = galaxy_freq
    features["galaxy_population_logfreq"] = np.log(galaxy_freq)
    features["spectral_galaxy_pair_freq"] = pair_freq
    features["spectral_galaxy_pair_logfreq"] = np.log(pair_freq)

    redshift = pd.to_numeric(raw["redshift"], errors="coerce")
    train_redshift = redshift.loc[train_mask].replace([np.inf, -np.inf], np.nan).dropna()
    if train_redshift.empty:
        redshift_cap = 0.0
    else:
        redshift_cap = float(np.nanquantile(train_redshift.to_numpy(dtype=float), 0.995))
        if not np.isfinite(redshift_cap) or redshift_cap < 0.0:
            redshift_cap = 0.0

    redshift_clip = redshift.clip(lower=0.0, upper=redshift_cap)
    redshift_log = np.log1p(redshift_clip)

    base_numeric = {
        "u": pd.to_numeric(raw["u"], errors="coerce"),
        "g": pd.to_numeric(raw["g"], errors="coerce"),
        "r": pd.to_numeric(raw["r"], errors="coerce"),
        "i": pd.to_numeric(raw["i"], errors="coerce"),
        "z": pd.to_numeric(raw["z"], errors="coerce"),
        "redshift_clip": redshift_clip,
        "redshift_log": redshift_log,
    }

    rank_cols = []
    within_st_cols = []
    within_gp_cols = []
    within_pair_cols = []
    bin_freq_cols = []

    for feat_name in _CONTEXT_FEATURES:
        train_values = base_numeric[feat_name].loc[train_mask]
        all_values = base_numeric[feat_name]

        global_rank = _map_empirical_rank(all_values, train_values)
        global_name = f"{feat_name}_global_rank"
        features[global_name] = global_rank
        rank_cols.append(global_name)

        within_st_name = f"{feat_name}_spectral_type_rank"
        within_gp_name = f"{feat_name}_galaxy_population_rank"
        within_pair_name = f"{feat_name}_spectral_galaxy_pair_rank"

        features[within_st_name] = _withingroup_ranks(all_values, spectral, train_mask, global_rank)
        features[within_gp_name] = _withingroup_ranks(all_values, galaxy, train_mask, global_rank)
        features[within_pair_name] = _withingroup_ranks(all_values, pair, train_mask, global_rank)

        within_st_cols.append(within_st_name)
        within_gp_cols.append(within_gp_name)
        within_pair_cols.append(within_pair_name)

        bin_col = f"{feat_name}_q12_bin"
        bin_freq_col = f"{feat_name}_q12_bin_freq"
        bins, freqs = _quantile_bins_and_freq(train_values, all_values, _Q12_LEVELS)

        features[bin_col] = bins
        features[bin_freq_col] = freqs
        bin_freq_cols.append(bin_freq_col)

    features["global_rank_mean_7"] = features[rank_cols].mean(axis=1)
    features["global_rank_std_7"] = features[rank_cols].std(axis=1, ddof=0)

    features["spectral_type_rank_mean_7"] = features[within_st_cols].mean(axis=1)
    features["galaxy_population_rank_mean_7"] = features[within_gp_cols].mean(axis=1)
    features["spectral_galaxy_pair_rank_mean_7"] = features[within_pair_cols].mean(axis=1)

    train_bin_values = features.loc[train_mask, bin_freq_cols].to_numpy(dtype=float).ravel()
    if train_bin_values.size == 0 or not np.isfinite(train_bin_values).any():
        low_threshold = 0.0
    else:
        low_threshold = float(np.nanquantile(train_bin_values, 0.10))

    features["low_freq_bin_count_10p"] = (features[bin_freq_cols].lt(low_threshold).sum(axis=1)).astype(int)

    return features

FEATURE_GROUPS = [
    {
        "name": "aide_catalog_rank_frequency_context",
        "fn": add_aide_catalog_rank_frequency_context,
        "depends_on": [],
        "description": "Builds catalog-frequency and rank/quantile context fields for catalog tags and numeric observables with fallback-safe within-group ranking.",
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