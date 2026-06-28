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
- title: Regularized Redshift Manifold Geometry Residuals
- group_name: photoz_trajectory_geometry
- family: trajectory_consistency
- summary: Represent each object by how its SDSS color vector deviates from the smooth empirical color-redshift manifold and from the manifold local direction and curvature, making class-specific departures from physically plausible redshift trajectories explicit.
- strategy: Using train predictors only, define colors c=[u-g,g-r,r-i,i-z] and fit a class-agnostic trajectory over redshift. Build K=180 redshift knots from train quantiles between the 0.5th and 99.5th percentiles. At each knot q_k, compute Gaussian-kernel weights w_i=exp(-(z_i-q_k)^2/(2h_k^2)) with initial h_k=0.06; increase h_k by 1.5x until the effective sample size (sum w)^2/sum(w^2) is at least 1200, capped at h_k=0.8. Estimate a weighted Huber mean mu_k for the 4 colors and a weighted robust covariance Sigma_k from Huber-clipped residuals. Regularize covariance as Sigma_k=0.9*Sigma_k+0.1*I*trace(Sigma_k)/4, then floor eigenvalues at 1e-5. Smooth each color mean over knots with a cubic smoothing spline or penalized B-spline using q_k as x and effective sample size as weight; derive v(z)=dmu/dz and a(z)=d2mu/dz2 from the smoothed curve. Interpolate covariance by linearly interpolating neighboring regularized covariance matrices and applying the same shrinkage/eigenvalue floor after interpolation. For each row, clip z to the knot range, interpolate mu(z), v(z), a(z), and Sigma(z), then compute r=c-mu(z). Whiten with L=chol(Sigma(z)), y=L^{-1}r, t=L^{-1}v(z), u=t/(||t||+1e-8), and aw=L^{-1}a(z). Emit signed tangent residual T=y dot u, orthogonal squared distance O=max(0,||y||^2-T^2), total whitened distance D=||y||^2, curvature residual C=(y dot aw)/(||aw||+1e-8), and normalized versions T/sigma_T(z), O/sigma_O(z), C/sigma_C(z), where sigmas are robust MAD scales estimated from train residual projections in the same local weighted neighborhoods and smoothed over knots. For z below 0.01, multiply T and C by z/0.01 while leaving O and D unchanged; for clipped tails outside the knot range, set a(z)=0 and damp T and C by exp(-distance_outside/0.05). Clip all emitted features to the train 0.1st and 99.9th percentiles before modeling.
- expected_signal: Balanced accuracy may improve because the features normalize color behavior by redshift and separate objects that are close in raw magnitudes but differ in whether their colors follow the dominant smooth trajectory, move along it, or sit off it; this should especially help recover minority QSO and STAR cases that occupy distinctive residual geometry despite overlap in simple color-redshift space.
- risk: The trajectory is sensitive to redshift quality and smoothing choices: undersmoothing can encode noise and unstable derivatives, oversmoothing can erase useful local QSO curvature, and covariance whitening can amplify artifacts in sparse redshift regions. The features are also computationally heavier than simple color ratios and may be redundant with strong tree models using raw colors and redshift directly.

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