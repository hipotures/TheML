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
- title: Leakage-Safe Adaptive Flux-Branch Posteriors
- group_name: redshift_branch_deconvolved_flux_posteriors
- family: probabilistic_density_geometry
- summary: Estimate uncertainty-smoothed class and quasar-redshift-branch posterior features from relative ugriz flux geometry within brightness-local strata, with explicit sparse-bin backoff and balanced-prior calibration for class-overlap regions.
- strategy: Convert magnitudes to log relative fluxes x = [-0.4*ln(10)*(u-i), -0.4*ln(10)*(g-i), -0.4*ln(10)*(r-i), -0.4*ln(10)*(z-i)], winsorizing each dimension to the training-fold 0.1/99.9 percentiles before fitting or scoring. Define overlapping i-band windows of width 0.8 and step 0.4 over the training-fold i range; rows outside the range use the nearest edge window plus the global model. Fit density groups G = {STAR, GALAXY, QSO_low, QSO_mid, QSO_high}, with QSO_low redshift < 2.2, QSO_mid 2.2 <= redshift <= 3.5, and QSO_high redshift > 3.5. For each window and group, fit a Gaussian mixture on x with K chosen by BIC from {1,2,3,5,8}, capped by floor(n_eff/60), using diagonal covariance when n_eff is small and full covariance only when the stratum has enough samples; impose a covariance eigenvalue floor of 1e-4 after robust scaling. For n_eff < 120, blend local and global group log-likelihoods with lambda = n_eff/(n_eff+120). Estimate a diagonal observation-smoothing matrix V_w from robust within-window MAD scales, capped to a conservative range, subtract V_w from fitted component covariances with positive-semidefinite flooring to approximate intrinsic covariance, then score query rows with Sigma_intrinsic + V_w. Use triangular window weights max(0, 1 - |i-center|/0.4) to average log-likelihoods across active windows. Apply soft redshift gates to QSO branches using sigmoid transitions of width 0.05 around 2.2 and 3.5, renormalized across branches, while STAR and GALAXY remain unbranched. Normalize class evidence twice: once with smoothed empirical window priors tempered by tau=0.5, and once with uniform class priors for balanced-accuracy robustness; QSO branch priors are smoothed within the QSO class and backed off to global branch frequencies with weight n_branch/(n_branch+80). Emit clipped posterior logits for STAR, GALAXY, QSO, and each QSO branch, pairwise class margins, QSO-branch margins against STAR and GALAXY, branch concentration, branch entropy, and the count of branches exceeding half the maximum branch mass, clipping probabilities to [1e-8, 1-1e-8] and log features to [-12,12]. For training rows, generate all statistics and density features out-of-fold only; fit final density objects on the full training data only for test-set feature generation.
- expected_signal: The features should expose smooth class evidence in the color-flux regions where hard cuts confuse stars, quasars, and compact galaxies, while the tempered and uniform-prior normalizations make minority-class evidence less likely to be suppressed under a balanced-accuracy metric.
- risk: The noise proxy is only an approximation because true photometric errors are unavailable, sparse high-redshift QSO strata may still overfit or collapse to global behavior, Gaussian mixtures may be computationally expensive at this scale, and any failure to compute training features out-of-fold would create target leakage.

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