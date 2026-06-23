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
- Hypothesis ID: 000040
- Source file: autogluon-001.py
- Failed node: 20260623T151012-7dc6fdd8-189
- Run: 20260622T034843-eca160d1

# Execution Error

```text
failed.yaml error:
Process exited with 1

stderr.log:
Traceback (most recent call last):
  File "/home/xai/DEV/TheML/projects/kaggle/playground-series-s6e6/runs/20260622T034843-eca160d1/artifacts/20260623T151012-7dc6fdd8-189/02-code.py", line 850, in main
    transformed = run_feature_groups(
                  ^^^^^^^^^^^^^^^^^^^
  File "/home/xai/DEV/TheML/src/tml/features/groups.py", line 59, in run_feature_groups
    block = group["fn"](raw.copy(), {key: value.copy() for key, value in deps.items()}, aux.copy())
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/xai/DEV/TheML/projects/kaggle/playground-series-s6e6/runs/20260622T034843-eca160d1/artifacts/20260623T151012-7dc6fdd8-189/02-code.py", line 277, in add_class_conditional_color_density_posteriors
    ref = _build_reference_from(aux)
          ^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/xai/DEV/TheML/projects/kaggle/playground-series-s6e6/runs/20260622T034843-eca160d1/artifacts/20260623T151012-7dc6fdd8-189/02-code.py", line 74, in _build_reference_from
    cls = _normalize_class_series(source[class_col])
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/xai/DEV/TheML/projects/kaggle/playground-series-s6e6/runs/20260622T034843-eca160d1/artifacts/20260623T151012-7dc6fdd8-189/02-code.py", line 53, in _normalize_class_series
    normalized = mapped.fillna(mapped.where(False))
                               ^^^^^^^^^^^^^^^^^^^
  File "/home/xai/DEV/TheML/.venv/lib/python3.12/site-packages/pandas/core/generic.py", line 11048, in where
    return self._where(cond, other, inplace, axis, level)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/xai/DEV/TheML/.venv/lib/python3.12/site-packages/pandas/core/generic.py", line 10712, in _where
    raise ValueError("Array conditional must be same shape as self")
ValueError: Array conditional must be same shape as self
Traceback (most recent call last):
  File "/home/xai/DEV/TheML/projects/kaggle/playground-series-s6e6/runs/20260622T034843-eca160d1/artifacts/20260623T151012-7dc6fdd8-189/02-code.py", line 974, in <module>
    main()
  File "/home/xai/DEV/TheML/projects/kaggle/playground-series-s6e6/runs/20260622T034843-eca160d1/artifacts/20260623T151012-7dc6fdd8-189/02-code.py", line 850, in main
    transformed = run_feature_groups(
                  ^^^^^^^^^^^^^^^^^^^
  File "/home/xai/DEV/TheML/src/tml/features/groups.py", line 59, in run_feature_groups
    block = group["fn"](raw.copy(), {key: value.copy() for key, value in deps.items()}, aux.copy())
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/xai/DEV/TheML/projects/kaggle/playground-series-s6e6/runs/20260622T034843-eca160d1/artifacts/20260623T151012-7dc6fdd8-189/02-code.py", line 277, in add_class_conditional_color_density_posteriors
    ref = _build_reference_from(aux)
          ^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/xai/DEV/TheML/projects/kaggle/playground-series-s6e6/runs/20260622T034843-eca160d1/artifacts/20260623T151012-7dc6fdd8-189/02-code.py", line 74, in _build_reference_from
    cls = _normalize_class_series(source[class_col])
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/xai/DEV/TheML/projects/kaggle/playground-series-s6e6/runs/20260622T034843-eca160d1/artifacts/20260623T151012-7dc6fdd8-189/02-code.py", line 53, in _normalize_class_series
    normalized = mapped.fillna(mapped.where(False))
                               ^^^^^^^^^^^^^^^^^^^
  File "/home/xai/DEV/TheML/.venv/lib/python3.12/site-packages/pandas/core/generic.py", line 11048, in where
    return self._where(cond, other, inplace, axis, level)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/xai/DEV/TheML/.venv/lib/python3.12/site-packages/pandas/core/generic.py", line 10712, in _where
    raise ValueError("Array conditional must be same shape as self")
ValueError: Array conditional must be same shape as self

stdout.log:
AutoGluon materialization: loaded aux file /home/xai/DEV/aideml/workspaces/2-liberal-celadon-panther/input/star_classification.csv rows=100000 cols=18 passed_to_groups=True
AutoGluon materialization: starting feature groups
TheML feature group: start name=class_conditional_color_density_posteriors
TheML feature group: failed name=class_conditional_color_density_posteriors elapsed_s=0.107 rows=824782 cols=0
```

# Previous Code

```python
import numpy as np
import pandas as pd

_REDSHIFT_BINS = (0.0, 0.3, 1.2, 2.6, 4.0, 7.0)
_I_BIN_WIDTH = 0.5
_COLOR_BINS = 14
_COLOR_CELLS = _COLOR_BINS ** 3
_CLASS_ORDER = ("GALAXY", "QSO", "STAR")
_ALPHA = 1.0
_BETA = 1.0
_EPS = 1e-12

def _find_class_column(frame):
    if not isinstance(frame, pd.DataFrame):
        return None
    lower_map = {str(col).lower(): col for col in frame.columns}
    candidates = ("class", "target", "label", "y", "y_true", "target_class")
    for candidate in candidates:
        if candidate in lower_map:
            return lower_map[candidate]
    return None

def _normalize_class_series(series):
    if series is None:
        return pd.Series(dtype="string")

    numeric = pd.to_numeric(series, errors="coerce")
    normalized = pd.Series(pd.NA, index=series.index, dtype="string")
    unique_numeric = pd.Index(pd.Series(numeric).dropna().unique())

    if len(unique_numeric) == 3 and set(unique_numeric) == {0.0, 1.0, 2.0}:
        mapping = {0.0: "GALAXY", 1.0: "QSO", 2.0: "STAR"}
        normalized = numeric.map(mapping).astype("string")
    elif len(unique_numeric) == 3 and set(unique_numeric) == {1.0, 2.0, 3.0}:
        mapping = {1.0: "GALAXY", 2.0: "QSO", 3.0: "STAR"}
        normalized = numeric.map(mapping).astype("string")
    else:
        text = series.astype("string").str.strip().str.upper()
        mapped = pd.Series(pd.NA, index=series.index, dtype="string")
        mapped.loc[text.isin({"GALAXY", "GAL"})] = "GALAXY"
        mapped.loc[text.isin({"QSO", "QUASAR"})] = "QSO"
        mapped.loc[text.isin({"STAR"})] = "STAR"
        mapped.loc[text.str.contains("GALAXY", na=False)] = "GALAXY"
        mapped.loc[text.str.contains("QSO", na=False)] = "QSO"
        mapped.loc[text.str.contains("QUASAR", na=False)] = "QSO"
        mapped.loc[text.str.contains("STAR", na=False)] = "STAR"
        mapped = mapped.mask(mapped.eq("STAR"), "STAR")
        normalized = mapped.fillna(mapped.where(False))
        normalized = mapped

    if normalized.notna().any():
        normalized = normalized.where(~normalized.str.fullmatch(r"^\s*$", case=False, na=False), pd.NA)

    return normalized

def _build_reference_from(source):
    if not isinstance(source, pd.DataFrame) or source.empty:
        return pd.DataFrame(columns=["u", "g", "r", "i", "redshift", "class"])

    class_col = _find_class_column(source)
    required = ("u", "g", "r", "i", "redshift")
    if class_col is None:
        return pd.DataFrame(columns=["u", "g", "r", "i", "redshift", "class"])

    if not set(required).issubset(set(source.columns)):
        return pd.DataFrame(columns=["u", "g", "r", "i", "redshift", "class"])

    cls = _normalize_class_series(source[class_col])
    if cls.notna().sum() == 0:
        return pd.DataFrame(columns=["u", "g", "r", "i", "redshift", "class"])

    ref = source.loc[cls.notna(), list(required)].copy()
    ref["class"] = cls.loc[cls.notna()]
    for col in required:
        ref[col] = pd.to_numeric(ref[col], errors="coerce")
    ref = ref.replace([np.inf, -np.inf], np.nan)
    ref["class"] = ref["class"].astype("string")
    ref = ref.dropna(subset=list(required) + ["class"]).copy()
    ref["class"] = ref["class"].where(ref["class"].isin(_CLASS_ORDER), pd.NA)
    ref = ref.dropna(subset=["class"])
    return ref.loc[:, ["u", "g", "r", "i", "redshift", "class"]]

def _make_color_edges(values):
    arr = np.asarray(values, dtype=float)
    finite = np.isfinite(arr)
    vals = arr[finite]
    if vals.size == 0:
        return np.array([-1.0, 1.0], dtype=float)

    quantiles = np.linspace(0.0, 1.0, _COLOR_BINS + 1)
    edges = np.quantile(vals, quantiles).astype(float)
    edges[0] = edges[0] - _EPS
    edges[-1] = edges[-1] + _EPS
    for i in range(1, edges.size):
        if not np.isfinite(edges[i - 1]):
            edges[i - 1] = float(i - 1)
        if edges[i] <= edges[i - 1]:
            edges[i] = edges[i - 1] + _EPS
    return edges

def _i_edges(vals):
    arr = np.asarray(vals, dtype=float)
    finite = np.isfinite(arr)
    vals = arr[finite]
    if vals.size == 0:
        return np.array((0.0, _I_BIN_WIDTH), dtype=float)

    lo = np.floor(np.min(vals) / _I_BIN_WIDTH) * _I_BIN_WIDTH
    hi = np.ceil(np.max(vals) / _I_BIN_WIDTH) * _I_BIN_WIDTH
    if hi <= lo:
        hi = lo + _I_BIN_WIDTH
    return np.arange(lo, hi + _I_BIN_WIDTH, _I_BIN_WIDTH, dtype=float)

def _digitize_fixed(values, bin_edges):
    arr = np.asarray(values, dtype=float)
    out = np.full(arr.shape, -1, dtype=np.int16)
    finite = np.isfinite(arr)
    if not np.any(finite):
        return out

    clipped = arr[finite]
    clipped = np.clip(clipped, bin_edges[0], bin_edges[-1] - _EPS)
    out[finite] = np.digitize(clipped, np.asarray(bin_edges[1:-1], dtype=float), right=False).astype(np.int16)
    return out

def _digitize_with_fixed_bins(values, bins):
    arr = np.asarray(values, dtype=float)
    out = np.full(arr.shape, -1, dtype=np.int16)
    finite = np.isfinite(arr)
    if not np.any(finite):
        return out

    clipped = arr[finite]
    clipped = np.clip(clipped, bins[0], bins[-1] - _EPS)
    out[finite] = np.digitize(clipped, np.asarray(bins[1:-1], dtype=float), right=False).astype(np.int16)
    return out

def _neighbor_sum_3d(hist):
    padded = np.pad(hist, 1, mode="constant", constant_values=0.0)
    summed = np.zeros_like(hist, dtype=float)
    for dz in (-1, 0, 1):
        z0 = 1 + dz
        z1 = z0 + _COLOR_BINS
        for dy in (-1, 0, 1):
            y0 = 1 + dy
            y1 = y0 + _COLOR_BINS
            for dx in (-1, 0, 1):
                x0 = 1 + dx
                x1 = x0 + _COLOR_BINS
                summed += padded[x0:x1, y0:y1, z0:z1]
    return summed

def _build_hist_stats(reference):
    if reference.empty:
        return None

    u = reference["u"].to_numpy(dtype=float)
    g = reference["g"].to_numpy(dtype=float)
    r = reference["r"].to_numpy(dtype=float)
    i = reference["i"].to_numpy(dtype=float)
    z = reference["redshift"].to_numpy(dtype=float)
    cls = reference["class"].map({"GALAXY": 0, "QSO": 1, "STAR": 2}).to_numpy(dtype=np.int8)

    color_ug = u - g
    color_gr = g - r
    color_ri = r - i

    z_bin = _digitize_fixed(z, _REDSHIFT_BINS)
    i_edges = _i_edges(i)
    i_bin = _digitize_with_fixed_bins(i, i_edges)

    edges_ug = _make_color_edges(color_ug)
    edges_gr = _make_color_edges(color_gr)
    edges_ri = _make_color_edges(color_ri)

    cu = _digitize_with_fixed_bins(color_ug, edges_ug)
    cg = _digitize_with_fixed_bins(color_gr, edges_gr)
    cri = _digitize_with_fixed_bins(color_ri, edges_ri)

    valid = (cls >= 0) & (z_bin >= 0) & (i_bin >= 0) & (cu >= 0) & (cg >= 0) & (cri >= 0)
    if not np.any(valid):
        return None

    cls = cls[valid]
    z_bin = z_bin[valid]
    i_bin = i_bin[valid]
    cu = cu[valid]
    cg = cg[valid]
    cri = cri[valid]

    histograms = {}
    class_stratum_totals = {}
    stratum_counts = {}
    class_totals = np.zeros(3, dtype=float)

    for row in range(len(cls)):
        c = int(cls[row])
        zk = int(z_bin[row])
        ik = int(i_bin[row])
        key = (c, zk, ik)
        if key not in histograms:
            histograms[key] = np.zeros((_COLOR_BINS, _COLOR_BINS, _COLOR_BINS), dtype=float)
        histograms[key][int(cu[row]), int(cg[row]), int(cri[row])] += 1.0

        str_key = (zk, ik)
        if str_key not in stratum_counts:
            stratum_counts[str_key] = np.zeros(3, dtype=float)
        stratum_counts[str_key][c] += 1.0

        class_totals[c] += 1.0

    if class_totals.sum() == 0:
        return None

    class_stratum_totals = {k: float(v.sum()) for k, v in histograms.items()}
    stratum_log_priors = {}
    for key, vals in stratum_counts.items():
        total = float(vals.sum())
        if total <= 0:
            continue
        probs = (vals + _BETA) / (total + 3.0 * _BETA)
        stratum_log_priors[key] = np.log(np.maximum(probs, _EPS))

    global_total = float(class_totals.sum())
    global_prior = (class_totals + _BETA) / (global_total + 3.0 * _BETA)
    log_global_prior = np.log(np.maximum(global_prior, _EPS))

    neighbor_histograms = {k: _neighbor_sum_3d(v) for k, v in histograms.items()}

    return {
        "histograms": histograms,
        "neighbor_histograms": neighbor_histograms,
        "class_stratum_totals": class_stratum_totals,
        "stratum_log_priors": stratum_log_priors,
        "log_global_prior": log_global_prior,
        "i_edges": i_edges,
        "edges_ug": edges_ug,
        "edges_gr": edges_gr,
        "edges_ri": edges_ri,
    }

def _baseline_features(index):
    n = len(index)
    zero = np.zeros(n, dtype=float)
    entropy = np.full(n, np.log(3.0), dtype=float)
    return pd.DataFrame(
        {
            "s_galaxy": zero.copy(),
            "s_qso": zero.copy(),
            "s_star": zero.copy(),
            "margin_qso_minus_star": zero.copy(),
            "margin_star_minus_galaxy": zero.copy(),
            "posterior_entropy": entropy,
        },
        index=index,
    )

def add_class_conditional_color_density_posteriors(raw, deps, aux):
    required = ("u", "g", "r", "i", "redshift")
    if not set(required).issubset(set(raw.columns)):
        return _baseline_features(raw.index)

    ref = _build_reference_from(aux)
    if ref.empty:
        ref = _build_reference_from(raw)

    stats = _build_hist_stats(ref)
    if stats is None:
        return _baseline_features(raw.index)

    u = pd.to_numeric(raw["u"], errors="coerce").to_numpy(dtype=float)
    g = pd.to_numeric(raw["g"], errors="coerce").to_numpy(dtype=float)
    r = pd.to_numeric(raw["r"], errors="coerce").to_numpy(dtype=float)
    i = pd.to_numeric(raw["i"], errors="coerce").to_numpy(dtype=float)
    z = pd.to_numeric(raw["redshift"], errors="coerce").to_numpy(dtype=float)

    color_ug = u - g
    color_gr = g - r
    color_ri = r - i

    z_bin = _digitize_fixed(z, _REDSHIFT_BINS)
    i_bin = _digitize_with_fixed_bins(i, stats["i_edges"])
    cu = _digitize_with_fixed_bins(color_ug, stats["edges_ug"])
    cg = _digitize_with_fixed_bins(color_gr, stats["edges_gr"])
    cri = _digitize_with_fixed_bins(color_ri, stats["edges_ri"])

    n = len(raw)
    log_global = stats["log_global_prior"]
    scores = np.tile(log_global, (n, 1))

    valid = (z_bin >= 0) & (i_bin >= 0) & (cu >= 0) & (cg >= 0) & (cri >= 0)
    idxs = np.nonzero(valid)[0]

    histograms = stats["histograms"]
    neighbors = stats["neighbor_histograms"]
    class_stratum_totals = stats["class_stratum_totals"]
    stratum_log_priors = stats["stratum_log_priors"]

    for row in idxs:
        zb = int(z_bin[row])
        ib = int(i_bin[row])
        cu_i = int(cu[row])
        cg_i = int(cg[row])
        ri_i = int(cri[row])

        baseline = stratum_log_priors.get((zb, ib), log_global)
        scores[row, :] = baseline

        for c in (0, 1, 2):
            key = (c, zb, ib)
            hist = histograms.get(key)
            if hist is None:
                continue

            total = class_stratum_totals.get(key, 0.0)
            if total <= 0:
                continue

            cell = float(hist[cu_i, cg_i, ri_i])
            if cell > 0.0:
                p = (cell + _ALPHA) / (total + _ALPHA * _COLOR_CELLS)
                scores[row, c] = baseline[c] + np.log(p)
                continue

            neigh = neighbors.get(key)
            neigh_count = float(neigh[cu_i, cg_i, ri_i]) if neigh is not None else 0.0
            if neigh_count > 0.0:
                p = (neigh_count + _ALPHA) / (total + _ALPHA * _COLOR_CELLS)
                scores[row, c] = baseline[c] + np.log(p)

    probs = np.exp(scores - scores.max(axis=1, keepdims=True))
    probs = probs / (probs.sum(axis=1, keepdims=True) + _EPS)
    entropy = -np.sum(probs * np.log(np.maximum(probs, _EPS)), axis=1)

    return pd.DataFrame(
        {
            "s_galaxy": scores[:, 0],
            "s_qso": scores[:, 1],
            "s_star": scores[:, 2],
            "margin_qso_minus_star": scores[:, 1] - scores[:, 2],
            "margin_star_minus_galaxy": scores[:, 2] - scores[:, 0],
            "posterior_entropy": entropy,
        },
        index=raw.index,
    )

FEATURE_GROUPS = [
    {
        "name": "class_conditional_color_density_posteriors",
        "fn": add_class_conditional_color_density_posteriors,
        "depends_on": [],
        "description": "Builds stratum-conditioned color-histogram Bayesian scores for GALAXY/QSO/STAR with neighbor and prior backoff.",
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