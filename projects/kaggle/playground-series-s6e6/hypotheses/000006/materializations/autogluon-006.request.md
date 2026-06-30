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
- Hypothesis ID: 000006
- Source file: autogluon-002.py
- Failed node: 
- Run: 

# Execution Error

```text
Invalid materialization for hypothesis 000006 (autogluon); response=hypotheses/000006/materializations/autogluon-002.response.md: Group materialization must not call functions in top-level assignments
```

# Repair Notes

- The validator rejected a module-level assignment whose right-hand side calls a function, constructor, generator, or comprehension. Keep only literal constants at module level. Move calls such as `np.array(...)`, `tuple(...)`, `range(...)`, `pd.IntervalIndex(...)`, `np.linspace(...)`, and comprehensions into the feature function or a helper that is called from the feature function. For example, replace `_Q12_LEVELS = tuple(i / 12 for i in range(13))` with an explicit literal tuple, or construct it inside the function.

# Previous Code

```python
import numpy as np
import pandas as pd

EPS = 1e-8
FLAT_THRESHOLD = 1e-6
INDEX_WEIGHT = np.array([1.0, 2.0, 3.0, 4.0], dtype=float)

def _frequency_bins_3(values: pd.Series) -> pd.Series:
    """
    Equal-mass low/medium/high bins via rank percentiles.
    Uses only training/test covariate values passed in `values`.
    """
    ranks = values.rank(method="average", pct=True)
    bins = np.where(ranks <= 1.0 / 3.0, "low", np.where(ranks <= 2.0 / 3.0, "medium", "high"))
    return pd.Series(bins, index=values.index, dtype="object")

def add_bandpass_break_localization(raw, deps, aux):
    # u,g,r,i,z bands only
    lu = -0.4 * raw["u"].astype(float)
    lg = -0.4 * raw["g"].astype(float)
    lr = -0.4 * raw["r"].astype(float)
    li = -0.4 * raw["i"].astype(float)
    lz = -0.4 * raw["z"].astype(float)

    d_ug = lu - lg
    d_gr = lg - lr
    d_ri = lr - li
    d_iz = li - lz

    a_ug = d_ug.abs()
    a_gr = d_gr.abs()
    a_ri = d_ri.abs()
    a_iz = d_iz.abs()

    A = a_ug + a_gr + a_ri + a_iz
    flat_mask = A < FLAT_THRESHOLD

    a_matrix = np.column_stack(
        [a_ug.to_numpy(), a_gr.to_numpy(), a_ri.to_numpy(), a_iz.to_numpy()]
    )
    d_matrix = np.column_stack(
        [d_ug.to_numpy(), d_gr.to_numpy(), d_ri.to_numpy(), d_iz.to_numpy()]
    )

    k_idx = np.argmax(a_matrix, axis=1)
    k = k_idx + 1  # 1..4 in ug,gr,ri,iz order (tie broken by smallest j)
    p_matrix = a_matrix / (A.to_numpy()[:, None] + EPS)
    p_matrix = np.where(np.isfinite(p_matrix), p_matrix, 0.0)
    entropy = -(p_matrix * np.log(np.maximum(p_matrix, EPS))).sum(axis=1)
    k_soft = (p_matrix * INDEX_WEIGHT[None, :]).sum(axis=1)

    d_k = d_matrix[np.arange(raw.shape[0]), k_idx]
    s_k = np.sign(d_k)

    a_ug_arr = a_ug.to_numpy()
    a_gr_arr = a_gr.to_numpy()
    a_ri_arr = a_ri.to_numpy()
    a_iz_arr = a_iz.to_numpy()
    d_ug_arr = d_ug.to_numpy()
    d_gr_arr = d_gr.to_numpy()
    d_ri_arr = d_ri.to_numpy()
    d_iz_arr = d_iz.to_numpy()

    blue_mean = np.zeros(raw.shape[0], dtype=float)
    blue_mean = np.where(k == 2, a_ug_arr, blue_mean)
    blue_mean = np.where(k == 3, (a_ug_arr + a_gr_arr) / 2.0, blue_mean)
    blue_mean = np.where(k == 4, (a_ug_arr + a_gr_arr + a_ri_arr) / 3.0, blue_mean)

    red_mean = np.zeros(raw.shape[0], dtype=float)
    red_mean = np.where(k == 1, (a_gr_arr + a_ri_arr + a_iz_arr) / 3.0, red_mean)
    red_mean = np.where(k == 2, (a_ri_arr + a_iz_arr) / 2.0, red_mean)
    red_mean = np.where(k == 3, a_iz_arr, red_mean)

    continuity = d_k - (blue_mean + red_mean) / 2.0
    asymmetry = blue_mean - red_mean
    sharpness = np.abs(d_k) / (A.to_numpy() + EPS)
    sign_prev = np.where(k == 2, d_ug_arr, np.where(k == 3, d_gr_arr, np.where(k == 4, d_ri_arr, 0.0)))
    sign_next = np.where(k == 1, d_gr_arr, np.where(k == 2, d_ri_arr, np.where(k == 3, d_iz_arr, 0.0)))
    turnover_count = ((k > 1) & (np.sign(sign_prev) != s_k)).astype(np.int16) + (
        (k < 4) & (np.sign(sign_next) != s_k)
    ).astype(np.int16)

    sharpness_bin = _frequency_bins_3(pd.Series(sharpness, index=raw.index))
    entropy_bin = _frequency_bins_3(pd.Series(entropy, index=raw.index))

    break_present = (~flat_mask).astype(np.int8)
    break_position = np.where(flat_mask, -1, k).astype(np.int16)
    break_soft_position = np.where(flat_mask, 0.0, k_soft)
    break_is_flat = flat_mask.astype(np.int8)
    break_dominant_delta = np.where(flat_mask, 0.0, d_k)
    break_dominant_abs = np.where(flat_mask, 0.0, np.abs(d_k))
    break_dominant_sign = np.where(
        flat_mask, 0, np.where(s_k > 0, 1, np.where(s_k < 0, -1, 0))
    ).astype(np.int8)
    break_continuity = np.where(flat_mask, 0.0, continuity)
    break_asymmetry = np.where(flat_mask, 0.0, asymmetry)
    break_sharpness = np.where(flat_mask, 0.0, sharpness)
    break_entropy = np.where(flat_mask, 0.0, entropy)
    break_blue_mean = np.where(flat_mask, 0.0, blue_mean)
    break_red_mean = np.where(flat_mask, 0.0, red_mean)
    break_turnover_count = np.where(flat_mask, 0, turnover_count)
    break_soft_ug = np.where(flat_mask, 0.0, p_matrix[:, 0])
    break_soft_gr = np.where(flat_mask, 0.0, p_matrix[:, 1])
    break_soft_ri = np.where(flat_mask, 0.0, p_matrix[:, 2])
    break_soft_iz = np.where(flat_mask, 0.0, p_matrix[:, 3])

    return pd.DataFrame(
        {
            "break_present": break_present,
            "break_position": break_position,
            "break_is_flat": break_is_flat,
            "break_soft_position": break_soft_position,
            "break_dominant_delta": break_dominant_delta,
            "break_dominant_abs": break_dominant_abs,
            "break_dominant_sign": break_dominant_sign,
            "break_continuity": break_continuity,
            "break_asymmetry": break_asymmetry,
            "break_sharpness": break_sharpness,
            "break_entropy": break_entropy,
            "break_turnover_count": break_turnover_count,
            "break_blue_mean": break_blue_mean,
            "break_red_mean": break_red_mean,
            "break_weight_ug": break_soft_ug,
            "break_weight_gr": break_soft_gr,
            "break_weight_ri": break_soft_ri,
            "break_weight_iz": break_soft_iz,
            "break_sharpness_confidence": sharpness_bin,
            "break_entropy_confidence": entropy_bin,
        },
        index=raw.index,
    )

FEATURE_GROUPS = [
    {
        "name": "bandpass_break_localization",
        "fn": add_bandpass_break_localization,
        "depends_on": [],
        "description": "Localizes the strongest broadband break across ugriz and summarizes its strength, position certainty, continuity, asymmetry, and turnover behavior.",
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