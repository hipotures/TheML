autogluon

Write Python code for one feature group.

# Task

## Goal
Predict the stellar class for each test-set object.

## Evaluation
Submissions are evaluated using balanced accuracy between predicted class labels and the true class. Submission file must contain `id,class` with one label per test row, where `class` is one of GALAXY, STAR, or QSO.

## Data description
`train.csv` contains 577347 rows with 10 feature columns plus `id`, `galaxy_population`, `spectral_type`, and the target `class`. `test.csv` contains the same predictors without the target across 247435 rows. `sample_submission.csv` shows the required submission columns `id` and `class`. The target is a 3-class stellar classification problem with labels GALAXY, QSO, and STAR.

# Data Overview

-> sample_submission.csv has 247435 rows and 2 columns.
Here is some information about the columns:
id (int64) has range: 577347.00 - 824781.00, 0 nan values
class (object) has 1 unique values: ['GALAXY'], 0 nan values

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
class (object) has 3 unique values: ['GALAXY', 'QSO', 'STAR'], 0 nan values

# Project Target
- ID column: id
- Target column: class
- Problem type: multiclass
- Evaluation metric: balanced_accuracy
- Submission kind: labels

# Hypothesis 

- hypothesis_id: 000001

- title: Broadband color-shape features

- group_name: broadband_color_shape

- family: photometric_sed

- summary: Transform the five ugriz magnitudes into color slopes and curvature features that describe each object's broadband spectral energy distribution shape.

- depends_on: []

- strategy: Create deterministic photometric-shape columns from magnitude differences: color_u_g = u - g, color_g_r = g - r, color_r_i = r - i, color_i_z = i - z, color_u_r = u - r, color_g_i = g - i, color_r_z = r - z, and color_u_z = u - z. Add second-difference curvature terms to capture bends in the spectrum: curve_ugr = (u - g) - (g - r) = u - 2*g + r, curve_gri = (g - r) - (r - i) = g - 2*r + i, and curve_riz = (r - i) - (i - z) = r - 2*i + z. Do not bin, rank, or target-encode anything. Leave raw subtraction values unchanged, including negatives and extreme values; if any source value is non-finite in future data, return null for the affected derived column only.

- expected_signal: GALAXY, STAR, and QSO classes typically separate more clearly in color-color space than in raw magnitudes because these differences remove overall brightness and isolate continuum slope, spectral breaks, and UV excess patterns that align with stellar temperature sequences, galaxy populations, and quasar spectra.

- risk: This group is partly redundant with the existing raw magnitudes and may overlap conceptually with spectral_type and galaxy_population; very noisy magnitudes or redshift-driven bandpass shifts can also make some color differences unstable at class boundaries.

# Group Code Contract

Return only Python code. Do not use markdown fences.

Define semantic feature-group functions and `FEATURE_GROUPS`.

Each feature function must use this signature:

```python
def add_group_name(raw, deps, aux, ctx):
    ...
    return new_features
```

Rules:
- `raw` is the raw/base train+test covariate frame without target labels. It includes ID columns.
- `deps` is a dict of dependency outputs by logical group name. Use `deps["group_name"]["local_feature"]`; never use final `G001_` prefixes.
- `aux` is an auxiliary DataFrame when available, otherwise empty.
- `ctx` is runtime context.
- Return a pandas DataFrame containing only new local feature columns with `index=raw.index`.
- Preserve row count, row order, and index exactly.
- Do not return raw/input columns.
- Do not create columns starting with `G001_` or any `G\\d+_` prefix.
- Outputs may be numeric, boolean, categorical, or string scalar columns. Do not return nested lists, dicts, tuples, or sets.
- You may compute covariate-only train+test statistics from `raw`; do not use target labels, validation labels, predictions, or leaderboard feedback.
- Do not read train/test/sample_submission files, write files, train models, create `main()`, sort dependency DAGs, assign final prefixes, concatenate final blocks, or implement orchestration.

Register groups like this:

```python
FEATURE_GROUPS = [
    {
        "name": "group_name",
        "fn": add_group_name,
        "depends_on": [],
        "description": "Short description.",
    }
]
```

AutoGluon-specific rule: do not train AutoGluon, instantiate `TabularPredictor`, call `.predict()`, create `submission.csv`, or define `preprocess(df)`. The external TheML AutoGluon wrapper will run the group executor, attach the target, train AutoGluon, ignore raw ID columns during fit, and write the submission.