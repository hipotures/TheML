Write preprocessing feature-group code for an AutoGluon wrapper.

Your output must define one semantic feature group for preprocessing.
The fixed wrapper imports this file, runs `FEATURE_GROUPS`, logs group timing,
renames returned columns, assembles the final DataFrame, and handles all
non-preprocessing work.

Do not write the wrapper.

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

# Hypothesis
- title: Robust auxiliary reference distance
- group_name: aide_aux_reference_distribution_distance
- family: aide_auxiliary_reference
- summary: Compare each object with an external auxiliary reference population to encode how typical or outlying its sky position, photometry, and redshift profile is under a robust survey-like distribution.
- strategy: Load the auxiliary reference table when present and build reference statistics only from overlapping numeric predictors among alpha, delta, u, g, r, i, z, and redshift; exclude target-like, identifier, and categorical fields from this feature group. Apply identical deterministic cleaning to auxiliary, train, and test rows: set photometry values <= -9000 to missing, clip redshift below 0 to 0, wrap alpha into [0, 360), and replace raw alpha in joint geometry with sin_alpha=sin(2*pi*alpha/360) and cos_alpha=cos(2*pi*alpha/360) while retaining delta as linear. For each usable marginal column in {sin_alpha, cos_alpha, delta, u, g, r, i, z, redshift}, compute auxiliary median m_j and robust scale s_j using 1.4826*MAD, falling back to IQR/1.349 and then to the median positive scale across usable columns; drop a column from distance calculations if no positive scale can be determined. For every row, emit per-column robust_z_j=(x_j-m_j)/s_j clipped to [-8,8], abs_robust_z_j, signed_median_delta_j=x_j-m_j, abs_median_delta_j, and empirical_cdf_j based on searchsorted over sorted nonmissing auxiliary values with plotting-position smoothing p=(count_less+0.5*count_equal+0.5)/(n_ref+1), clipped away from exactly 0 and 1. Emit missing flags for each usable column plus missing_count and available_ratio. Aggregate available marginal columns into mean_abs_z, max_abs_z, rms_abs_z, mean_rank_shift=mean(abs(empirical_cdf_j-0.5)), max_rank_shift, and mean_signed_z so that both outlier magnitude and directional offset are represented. For joint distance, form the standardized vector of clipped robust z values over the same usable columns, estimate the auxiliary covariance on rows complete across those columns after clipping standardized values to [-6,6], and regularize C with ridge lambda=max(1e-6*trace(C)/d, 1e-8); if the complete auxiliary sample is too small, covariance rank is unusable, or d<2, fall back to diagonal covariance. Compute a missingness-aware Mahalanobis squared distance using only coordinates available for the row and the corresponding covariance submatrix, then rescale by d_full/d_available to keep rows with fewer coordinates comparable; emit mahalanobis_sq, mahalanobis, log1p_mahalanobis_sq, joint_available_count, and joint_available_ratio. If the auxiliary table is absent or fewer than three usable columns remain, emit neutral constants for all distance aggregates and explicit unavailable indicators. Fit all reference statistics once from auxiliary data only and apply them unchanged to train and test features; do not use class labels or train a classifier inside this hypothesis.
- expected_signal: Objects from different stellar classes should occupy different parts of the joint color, magnitude, redshift, and sky-position manifold, so robust auxiliary-distance features can expose class-specific typicality and outlier structure that a balanced-accuracy model may use especially for minority QSO or STAR separation.
- risk: The auxiliary reference may have a different footprint, calibration, or selection function than the competition data, causing distances to encode distribution shift rather than class signal; covariance estimates can be noisy in sparse complete cases, and these features may be redundant with raw predictors or dominate if the downstream model overweights broad outlier scores.

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

AutoGluon wrapper boundary:
- Do not train AutoGluon.
- Do not import or instantiate `TabularPredictor`.
- Do not call `.fit()`, `.predict()`, `.predict_proba()`, or `.leaderboard()`.
- Do not define `main()`.
- Do not define `preprocess(df)`.
- Do not read project data files or `./input`.
- The fixed AutoGluon wrapper handles all of those steps outside this generated file.