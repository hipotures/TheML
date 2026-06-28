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
- title: Train-fitted redshift-color K-correction residual manifold
- group_name: k_correction_residual_manifold
- family: k_correction_consistency
- summary: Learn a deterministic redshift- and color-conditioned photometric correction manifold from the predictors and encode how each object departs from locally expected pseudo-rest-frame SED geometry.
- strategy: Treat the correction as a training-fitted unsupervised transformer: within validation, fit all surfaces, bin edges, medians, and MADs on the training fold only; for final submission, refit on the full labeled training set and apply unchanged to test. Compute adjacent colors c_u=u-g, c_g=g-r, c_r=r-i, c_i=i-z, and c_z=g-r, plus clipped redshift z_c in the train-supported range. Use strata h=(spectral_type, galaxy_population), but require minimum support before fitting. Define redshift regimes [0,0.05), [0.05,0.3), [0.3,0.8), [0.8,1.8), [1.8,7.1]; merge adjacent regimes within h until each h x regime slice has at least 4000 rows, then fall back to spectral_type-only, then global regime fits if needed. For each band q, estimate a robust expected redshift correction Delta_q(z,c_q) rather than an absolute magnitude: set the response as m_q minus the h-specific low-redshift reference median for that band, where the reference is computed from rows with z<0.03 when available and otherwise from the lowest 2 percent of z in that h. Fit Huber or quantile-robust polynomial surfaces Delta_hat_q=beta1*z+beta2*z^2+beta3*c_q+beta4*c_q^2+beta5*z*c_q, with no intercept after centering so the correction is neutral at the reference redshift. Winsorize c_q and z to fold-training 0.5/99.5 percentiles for fitting and prediction, and emit clipping flags per band. Around regime boundaries, blend neighboring regime predictions with a linear weight over width 0.04*(1+z) when both fits exist; otherwise use the available fallback fit and mark a fallback-level feature. Compute pseudo-rest magnitudes M_q=m_q-Delta_hat_q. Within each effective stratum/regime, create quantile bins on (z_c,c_q) using up to 25 z bins and 10 color bins, reducing bin counts if needed so expected support is at least 150 rows per cell. For each q, store fold-training median and MAD for M_q, adjacent corrected colors D1=M_u-M_g, D2=M_g-M_r, D3=M_r-M_i, D4=M_i-M_z, and curvature terms C1=D1-D2, C2=D2-D3, C3=D3-D4. Produce residuals value-minus-median and scaled residuals residual/(MAD+1e-6). If a cell has fewer than 100 rows, borrow a distance-weighted median/MAD from neighboring quantile cells within radius 2; if still unsupported, fall back to the effective stratum/regime baseline and set low-support flags. Keep only deterministic correction residuals, scaled residuals, corrected-color and curvature residuals, plus compact support, clipping, blend, and fallback indicators.
- expected_signal: GALAXY, QSO, and STAR objects can share raw magnitude ranges but should differ in how consistently their colors and bandwise magnitudes follow a redshift-conditioned SED manifold, so residual geometry should improve balanced accuracy especially for minority or boundary cases whose raw photometry is ambiguous.
- risk: The correction is only a proxy because distance modulus and true rest-frame SEDs are not observed, so fitted residuals may partly encode survey selection effects or redundant magnitude information; sparse high-redshift strata, aggressive fallback pooling, and fold-inconsistent preprocessing can also create unstable features or optimistic validation if statistics are not fit strictly inside each training fold.

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