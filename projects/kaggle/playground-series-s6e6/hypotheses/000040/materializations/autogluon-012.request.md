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
- Hypothesis ID: 000040
- Source file: autogluon-011.py
- Failed node: 20260624T062209-cbcf9ca6-348
- Run: 20260622T034843-eca160d1

# Execution Error

```text
failed.yaml error:
Process exited with 1

stderr.log:
Traceback (most recent call last):
  File "/home/xai/DEV/TheML/.worktrees/root-hypothesis-revisions/projects/kaggle/playground-series-s6e6/runs/20260622T034843-eca160d1/artifacts/20260624T062209-cbcf9ca6-348/02-code.py", line 791, in main
    transformed = run_feature_groups(
                  ^^^^^^^^^^^^^^^^^^^
  File "/home/xai/DEV/TheML/.worktrees/root-hypothesis-revisions/src/tml/features/groups.py", line 59, in run_feature_groups
    block = group["fn"](raw.copy(), {key: value.copy() for key, value in deps.items()}, aux.copy())
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/xai/DEV/TheML/.worktrees/root-hypothesis-revisions/projects/kaggle/playground-series-s6e6/runs/20260622T034843-eca160d1/artifacts/20260624T062209-cbcf9ca6-348/02-code.py", line 173, in add_class_conditional_color_density_posteriors
    class_idx[target_clean == cname] = cid
    ~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^
IndexError: only integers, slices (`:`), ellipsis (`...`), numpy.newaxis (`None`) and integer or boolean arrays are valid indices
Traceback (most recent call last):
  File "/home/xai/DEV/TheML/.worktrees/root-hypothesis-revisions/projects/kaggle/playground-series-s6e6/runs/20260622T034843-eca160d1/artifacts/20260624T062209-cbcf9ca6-348/02-code.py", line 915, in <module>
    main()
  File "/home/xai/DEV/TheML/.worktrees/root-hypothesis-revisions/projects/kaggle/playground-series-s6e6/runs/20260622T034843-eca160d1/artifacts/20260624T062209-cbcf9ca6-348/02-code.py", line 791, in main
    transformed = run_feature_groups(
                  ^^^^^^^^^^^^^^^^^^^
  File "/home/xai/DEV/TheML/.worktrees/root-hypothesis-revisions/src/tml/features/groups.py", line 59, in run_feature_groups
    block = group["fn"](raw.copy(), {key: value.copy() for key, value in deps.items()}, aux.copy())
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/xai/DEV/TheML/.worktrees/root-hypothesis-revisions/projects/kaggle/playground-series-s6e6/runs/20260622T034843-eca160d1/artifacts/20260624T062209-cbcf9ca6-348/02-code.py", line 173, in add_class_conditional_color_density_posteriors
    class_idx[target_clean == cname] = cid
    ~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^
IndexError: only integers, slices (`:`), ellipsis (`...`), numpy.newaxis (`None`) and integer or boolean arrays are valid indices

stdout.log:
AutoGluon materialization: loaded aux file star_classification.csv rows=100000 cols=18 passed_to_groups=True
AutoGluon materialization: starting feature groups
TheML feature group: start name=class_conditional_color_density_posteriors
TheML feature group: failed name=class_conditional_color_density_posteriors elapsed_s=0.210 rows=824782 cols=0
```

# Previous Code

```python
import numpy as np
import pandas as pd

_CLASS_ORDER = ("GALAXY", "QSO", "STAR")
_CLASS_TO_INDEX = {"GALAXY": 0, "QSO": 1, "STAR": 2}
_Z_BREAKS = (-0.01, 0.20, 0.60, 1.05, 1.70, 2.45, 3.60, 5.00, 7.01)
_I_BIN_WIDTH = 0.50
_COLOR_BINS = 18
_COLOR_BINS3 = _COLOR_BINS * _COLOR_BINS * _COLOR_BINS
_ALPHA = 1.0
_BETA = 0.5
_EPS = 1e-12

_TARGET_COLUMNS = ("target", "label", "class", "y", "target_label", "Class", "ClassLabel", "target_class")
_TRAIN_HINT_COLUMNS = ("is_train", "is_train_row", "train", "in_train", "is_labeled")

def _to_aux_dataframe(aux):
    if aux is None:
        return None
    if isinstance(aux, pd.DataFrame):
        return aux
    if isinstance(aux, pd.Series):
        return aux.to_frame(name="target")
    return pd.DataFrame(aux)

def _extract_target_and_fit_mask(aux, index):
    aux_df = _to_aux_dataframe(aux)
    if aux_df is None or len(aux_df) == 0:
        return None, np.zeros(len(index), dtype=bool)

    target = None
    for col in _TARGET_COLUMNS:
        if col in aux_df.columns:
            ser = aux_df[col].reindex(index)
            if ser.notna().sum() > 0:
                target = ser
                break

    fit_mask = None
    for col in _TRAIN_HINT_COLUMNS:
        if col in aux_df.columns:
            hint = aux_df[col].reindex(index)
            if hint.notna().sum() > 0:
                hint = hint.fillna(False)
                if hint.dtype == bool or str(hint.dtype) == "boolean":
                    fit_mask = hint.to_numpy(dtype=bool)
                else:
                    fit_mask = pd.to_numeric(hint, errors="coerce").fillna(0).astype(int).to_numpy(dtype=bool)
                break

    if fit_mask is None:
        if target is None:
            return None, np.zeros(len(index), dtype=bool)
        return target, target.notna().to_numpy()

    if target is None:
        return None, np.zeros(len(index), dtype=bool)
    return target, np.logical_and(fit_mask, target.notna().to_numpy())

def _shift_slices(shift, size):
    if shift >= 0:
        return slice(0, size - shift), slice(shift, size)
    return slice(-shift, size), slice(0, size + shift)

def _color_bin_edges(values, fit_mask):
    v = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(v)
    if fit_mask is None:
        ref = v[finite]
    else:
        ref = v[np.asarray(fit_mask, dtype=bool) & finite]
    if ref.size == 0:
        if not finite.any():
            return np.linspace(0.0, 1.0, 17, dtype=np.float64)
        center = float(v[finite][0])
        return np.linspace(center - 8.0, center + 8.0, 17, dtype=np.float64)

    probs = np.linspace(0.005, 0.995, 17)
    edges = np.quantile(ref, probs, method="linear").astype(np.float64)
    for i in range(1, len(edges)):
        if edges[i] <= edges[i - 1]:
            edges[i] = edges[i - 1] + 1e-12
    return edges

def _neighbor_smooth_and_mass(p_ckm_5d, n_ckmb_5d, B):
    p1 = np.array(p_ckm_5d, dtype=np.float32, copy=True)
    p2 = np.array(p_ckm_5d, dtype=np.float32, copy=True)
    m1 = np.array(n_ckmb_5d, dtype=np.float32, copy=True)
    m2 = np.array(n_ckmb_5d, dtype=np.float32, copy=True)
    w1 = np.ones((B, B, B), dtype=np.float32)
    w2 = np.ones((B, B, B), dtype=np.float32)

    for d1 in (-2, -1, 0, 1, 2):
        s1, t1 = _shift_slices(d1, B)
        for d2 in (-2, -1, 0, 1, 2):
            s2, t2 = _shift_slices(d2, B)
            for d3 in (-2, -1, 0, 1, 2):
                d = max(abs(d1), abs(d2), abs(d3))
                if d == 0 or d > 2:
                    continue
                w = 0.5 ** d
                s3, t3 = _shift_slices(d3, B)

                p2[..., t1, t2, t3] += p_ckm_5d[..., s1, s2, s3] * w
                m2[..., t1, t2, t3] += n_ckmb_5d[..., s1, s2, s3]
                w2[t1, t2, t3] += w

                if d <= 1:
                    p1[..., t1, t2, t3] += p_ckm_5d[..., s1, s2, s3] * w
                    m1[..., t1, t2, t3] += n_ckmb_5d[..., s1, s2, s3]
                    w1[t1, t2, t3] += w

    p1 = p1 / np.where(w1 > 0, w1, 1.0)
    p2 = p2 / np.where(w2 > 0, w2, 1.0)
    return p1, m1, p2, m2

def add_class_conditional_color_density_posteriors(raw, deps, aux):
    idx = raw.index
    n_rows = len(raw)
    if n_rows == 0:
        return pd.DataFrame(index=idx)

    target, fit_mask = _extract_target_and_fit_mask(aux, idx)

    z_edges = np.asarray(_Z_BREAKS, dtype=np.float64)
    redshift = raw["redshift"].to_numpy(dtype=np.float64)
    redshift = np.clip(redshift, z_edges[0], z_edges[-1])
    z_bin = np.searchsorted(z_edges, redshift, side="right") - 1
    z_bin = np.clip(z_bin, 0, len(z_edges) - 2).astype(np.int16)

    i_val = raw["i"].to_numpy(dtype=np.float64)

    i_fit_mask = fit_mask if fit_mask.any() else np.ones(n_rows, dtype=bool)
    i_ref = i_val[np.asarray(i_fit_mask, dtype=bool)]
    i_min = np.floor(np.nanmin(i_ref) / _I_BIN_WIDTH) * _I_BIN_WIDTH
    i_max = np.ceil(np.nanmax(i_ref) / _I_BIN_WIDTH) * _I_BIN_WIDTH
    if not np.isfinite(i_min) or not np.isfinite(i_max) or i_max <= i_min:
        i_min = np.floor(np.nanmin(i_val) / _I_BIN_WIDTH) * _I_BIN_WIDTH
        i_max = i_min + _I_BIN_WIDTH

    n_i_bins = int(np.ceil((i_max - i_min) / _I_BIN_WIDTH))
    if n_i_bins < 1:
        n_i_bins = 1
    i_bin = np.floor((i_val - i_min) / _I_BIN_WIDTH).astype(np.int64)
    i_bin = np.clip(i_bin, 0, n_i_bins - 1)

    u = raw["u"].to_numpy(dtype=np.float64)
    g = raw["g"].to_numpy(dtype=np.float64)
    r = raw["r"].to_numpy(dtype=np.float64)
    c1 = u - g
    c2 = g - r
    c3 = r - i_val

    e1 = _color_bin_edges(c1, i_fit_mask)
    e2 = _color_bin_edges(c2, i_fit_mask)
    e3 = _color_bin_edges(c3, i_fit_mask)

    b1 = np.searchsorted(e1, c1, side="right").astype(np.int16)
    b2 = np.searchsorted(e2, c2, side="right").astype(np.int16)
    b3 = np.searchsorted(e3, c3, side="right").astype(np.int16)
    b1 = np.clip(b1, 0, _COLOR_BINS - 1)
    b2 = np.clip(b2, 0, _COLOR_BINS - 1)
    b3 = np.clip(b3, 0, _COLOR_BINS - 1)
    cell = ((b1.astype(np.int64) * _COLOR_BINS) + b2.astype(np.int64)) * _COLOR_BINS + b3.astype(np.int64)

    class_idx = np.full(n_rows, -1, dtype=np.int8)
    if target is not None:
        target_clean = pd.Series(target).astype("string").str.strip().str.upper()
        for cname, cid in _CLASS_TO_INDEX.items():
            class_idx[target_clean == cname] = cid

    fit_label_mask = np.zeros(n_rows, dtype=bool)
    if fit_mask is not None:
        fit_label_mask = np.logical_and(fit_mask, class_idx >= 0)

    if fit_label_mask.any():
        K = len(_Z_BREAKS) - 1
        rows = np.nonzero(fit_label_mask)[0]
        c_fit = class_idx[rows].astype(np.int64)
        z_fit = z_bin[rows].astype(np.int64)
        i_fit = i_bin[rows].astype(np.int64)
        cell_fit = cell[rows].astype(np.int64)

        counts = np.bincount(
            c_fit * (K * n_i_bins * _COLOR_BINS3) + z_fit * (n_i_bins * _COLOR_BINS3) + i_fit * _COLOR_BINS3 + cell_fit,
            minlength=3 * K * n_i_bins * _COLOR_BINS3,
        ).astype(np.float64)
        n_ckmb = counts.reshape(3, K, n_i_bins, _COLOR_BINS3)

        n_ckm = n_ckmb.sum(axis=3)
        n_ck = n_ckm.sum(axis=2)
        n_cm = n_ckmb.sum(axis=1).sum(axis=2)
        n_cb = n_ckmb.sum(axis=(1, 2))
        n_c = n_ck.sum(axis=1)

        N_km = n_ckm.sum(axis=0)
        N_k = n_ck.sum(axis=0)
        N_m = n_cm.sum(axis=0)
        N = n_c.sum()

        p_ckm = (n_ckmb + _ALPHA) / (n_ckm[:, :, :, None] + _ALPHA * _COLOR_BINS3)
        p_ckb = (n_cb + _ALPHA) / (n_c[:, None] + _ALPHA * _COLOR_BINS3)
        p_ck = (n_ckmb.sum(axis=2) + _ALPHA) / (n_ck[:, :, None] + _ALPHA * _COLOR_BINS3)
        p_cm = (n_ckmb.sum(axis=1) + _ALPHA) / (n_cm[:, :, None] + _ALPHA * _COLOR_BINS3)

        pi_ckm = (n_ckm + _BETA) / (N_km[None, :, :] + 3.0 * _BETA)
        pi_ck = (n_ck + _BETA) / (N_k[None, :] + 3.0 * _BETA)
        pi_cm = (n_cm + _BETA) / (N_m[None, :] + 3.0 * _BETA)
        pi_c = (n_c + _BETA) / (N + 3.0 * _BETA)

        p_ckm_5d = p_ckm.reshape(3, K, n_i_bins, _COLOR_BINS, _COLOR_BINS, _COLOR_BINS)
        n_ckmb_5d = n_ckmb.reshape(3, K, n_i_bins, _COLOR_BINS, _COLOR_BINS, _COLOR_BINS)
        p_ckm1_5d, m1_5d, p_ckm2_5d, m2_5d = _neighbor_smooth_and_mass(p_ckm_5d, n_ckmb_5d, _COLOR_BINS)

        p_ckm1 = p_ckm1_5d.reshape(3, K, n_i_bins, _COLOR_BINS3)
        p_ckm2 = p_ckm2_5d.reshape(3, K, n_i_bins, _COLOR_BINS3)
        m1 = m1_5d.reshape(3, K, n_i_bins, _COLOR_BINS3)
        m2 = m2_5d.reshape(3, K, n_i_bins, _COLOR_BINS3)

        k_idx = z_bin.astype(np.int64)
        i_idx = i_bin.astype(np.int64)
        b_idx = cell.astype(np.int64)

        scores = np.zeros((n_rows, 3), dtype=np.float64)

        Nk_row = N_k[k_idx]
        Nm_row = N_m[i_idx]
        Nkm_row = N_km[k_idx, i_idx]

        for c in range(3):
            n_ckmb_row = n_ckmb[c, k_idx, i_idx, b_idx]
            p_ckm_row = p_ckm[c, k_idx, i_idx, b_idx]
            p_ckm1_row = p_ckm1[c, k_idx, i_idx, b_idx]
            p_ckm2_row = p_ckm2[c, k_idx, i_idx, b_idx]
            p_ck_row = p_ck[c, k_idx, b_idx]
            p_cm_row = p_cm[c, i_idx, b_idx]
            p_c_row = p_ckb[c, b_idx]

            m1_row = m1[c, k_idx, i_idx, b_idx]
            m2_row = m2[c, k_idx, i_idx, b_idx]

            chosen_p = np.ones(n_rows, dtype=np.float64)
            chosen_logpi = np.full(n_rows, np.log(np.maximum(pi_c[c], _EPS)), dtype=np.float64)

            cond_a = (Nkm_row >= 300.0) & (n_ckmb_row >= 6.0) & (m1_row >= 6.0)
            if cond_a.any():
                chosen_p[cond_a] = p_ckm_row[cond_a]
                chosen_logpi[cond_a] = np.log(np.maximum(pi_ckm[c, k_idx[cond_a], i_idx[cond_a]], _EPS))

            cond_b = (~cond_a) & (m1_row >= 30.0)
            if cond_b.any():
                chosen_p[cond_b] = p_ckm1_row[cond_b]
                chosen_logpi[cond_b] = np.log(np.maximum(pi_ckm[c, k_idx[cond_b], i_idx[cond_b]], _EPS))

            cond_c = (~cond_a & ~cond_b) & (m2_row >= 80.0)
            if cond_c.any():
                chosen_p[cond_c] = p_ckm2_row[cond_c]
                chosen_logpi[cond_c] = np.log(np.maximum(pi_ckm[c, k_idx[cond_c], i_idx[cond_c]], _EPS))

            cond_d = (~cond_a & ~cond_b & ~cond_c) & (Nk_row >= 250.0)
            if cond_d.any():
                chosen_p[cond_d] = p_ck_row[cond_d]
                chosen_logpi[cond_d] = np.log(np.maximum(pi_ck[c, k_idx[cond_d]], _EPS))

            cond_e = (~cond_a & ~cond_b & ~cond_c & ~cond_d) & (Nm_row >= 250.0)
            if cond_e.any():
                chosen_p[cond_e] = p_cm_row[cond_e]
                chosen_logpi[cond_e] = np.log(np.maximum(pi_cm[c, i_idx[cond_e]], _EPS))

            cond_f = (~cond_a & ~cond_b & ~cond_c & ~cond_d & ~cond_e) & (n_c[c] >= 400.0)
            if cond_f.any():
                chosen_p[cond_f] = p_c_row[cond_f]
                # pi remains pi_c

            scores[:, c] = np.log(np.maximum(chosen_p, _EPS)) + chosen_logpi
    else:
        scores = np.full((n_rows, 3), np.log(1.0 / 3.0), dtype=np.float64)

    margin_qso_star = scores[:, 1] - scores[:, 2]
    margin_star_galaxy = scores[:, 2] - scores[:, 0]

    logits = scores - scores.max(axis=1, keepdims=True)
    exp_scores = np.exp(logits)
    sum_exp = np.where(exp_scores.sum(axis=1, keepdims=True) == 0, 1.0, exp_scores.sum(axis=1, keepdims=True))
    prob = exp_scores / sum_exp
    entropy = -np.sum(prob * np.log(np.maximum(prob, _EPS)), axis=1)

    return pd.DataFrame(
        {
            "s_GALAXY": scores[:, 0],
            "s_QSO": scores[:, 1],
            "s_STAR": scores[:, 2],
            "s_QSO_minus_STAR": margin_qso_star,
            "s_STAR_minus_GALAXY": margin_star_galaxy,
            "entropy_softmax": entropy,
        },
        index=idx,
    )

FEATURE_GROUPS = [
    {
        "name": "class_conditional_color_density_posteriors",
        "fn": add_class_conditional_color_density_posteriors,
        "depends_on": [],
        "description": "Computes class-conditional sparse color-density scores with support-gated Chebyshev neighborhood smoothing and a robust fallback ladder.",
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