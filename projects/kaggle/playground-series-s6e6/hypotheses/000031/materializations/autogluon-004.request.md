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
- title: Leakage-safe redshift-local color-tube residuals
- group_name: redshift_adaptive_color_tube_residuals
- family: color_manifold_geometry
- summary: Represent each object by its along-locus and off-locus position relative to robust redshift-local color manifolds, with stable signed residual coordinates and targeted emphasis near the quasar-star color-overlap regime.
- strategy: Compute adjacent colors c1=u-g, c2=g-r, c3=r-i, c4=i-z and z=max(redshift,0). Fit all bin edges, robust scales, PCA bases, clipping thresholds, and fallback parameters only on the training portion used for the current model fold; for final submission, refit the transformer on all training rows and apply it once to test. Build initial equal-frequency bins from training z quantiles, for example 30 bins, then merge adjacent bins until every final bin has at least 8000 training rows; store sorted edges and bin-centre z values, and clamp out-of-range rows to the nearest boundary bin. For each final bin and for both adjacent 3D color cubes Ugri=[c1,c2,c3] and Griz=[c2,c3,c4], compute per-color medians and scaled MADs on training rows, using scale_j=max(1.4826*MAD_j, 0.05*global_scaled_MAD_j, 1e-6). Standardize colors with these bin statistics, winsorize standardized values to [-6,6] only for fitting, and fit a 3-component PCA per bin and cube. Project each non-winsorized standardized row as t=dot(v1,x), q2=dot(v2,x), q3=dot(v3,x), and d=sqrt(q2^2+q3^2). Make coordinates sign-stable by orienting the first valid bin with dot(v1,[1,1,1])>=0 and a deterministic residual-plane anchor, then aligning each later bin basis to the previous valid bin by maximum-dot or orthogonal-Procrustes alignment while preserving a right-handed basis. For each bin and cube, estimate robust train-only coordinate scales for t, q2, q3, and d; emit raw coordinates, standardized t/q2/q3, positive distance d_norm=d/(median(d)+1.4826*MAD(d)+1e-6), log1p(d_norm), signs of q2 and q3, and bounded ratios such as q2_hat/(1+abs(t_hat)+d_norm), q3_hat/(1+abs(t_hat)+d_norm), and d_norm/(1+abs(t_hat)). Clip standardized signed coordinates to their bin-local training 0.5th and 99.5th percentiles, and clip positive distance features to their 99.5th percentile, while retaining a small set of raw unstandardized coordinates for tree models. Add overlap emphasis g(z)=exp(-0.5*((z-2.7)/0.45)^2) and emit gated versions of d_norm, log1p(d_norm), abs(q2_hat), and abs(q3_hat) for both cubes. If a bin has invalid scales, too few usable rows after merging, or numerically unstable PCA, transform its rows with the nearest valid bin by bin-centre z; if no local bin is valid, use global training statistics and emit a fallback indicator.
- expected_signal: Objects close to the dominant stellar-like optical color tube should have small orthogonal residuals, while galaxies and quasars should show class-specific departures in residual magnitude, side, and along-locus position; the redshift-local normalization prevents high-density redshift regions from dominating the scale, and the gated residuals should help balanced accuracy by sharpening QSO-versus-STAR separation in the known color-degeneracy window.
- risk: The unsupervised local tube can be contaminated by the majority class in bins where the dominant color manifold is not actually stellar, residual PCA directions can become unstable when the two orthogonal eigenvalues are similar, and the resulting correlated residual, ratio, and gated features may overfit unless the downstream model is regularized and the transformer is fit inside validation folds.

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