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
- title: Out-of-fold local XDQSO-style flux likelihood margins
- group_name: xdqso_inspired_flux_density_scores
- family: probabilistic_flux_density_scoring
- summary: Build leakage-controlled, redshift- and brightness-conditioned generative scores from normalized photometric flux shape so each object receives class-relative likelihood evidence for GALAXY, STAR, and QSO in the local color-flux manifold.
- strategy: Convert magnitudes to log relative flux coordinates using i-band as the anchor: x = [-0.4*(u-i), -0.4*(g-i), -0.4*(r-i), -0.4*(z-i)], which is equivalent to log10 flux ratios and avoids unstable raw division. Fit all preprocessing parameters on train only: winsorize each x component to the train 0.5% and 99.5% quantiles, and optionally standardize x within the density-fitting pipeline using train-only robust median/MAD statistics. Condition the density models on measured redshift and i magnitude using soft local cells. Use redshift slabs that cover the full observed range including negative values, e.g. [-0.02,0.8), [0.8,2.5), [2.5,4.0), [4.0,7.2], and overlapping i-band bins with width 0.4 mag and 0.2 mag spacing across the train i range extended by one half-width. Assign each object triangular interpolation weights across neighboring i bins and linearly blend across adjacent redshift slab boundaries within a small transition band, so scoring changes smoothly near bin edges. For each class and local cell, fit a regularized 4D Gaussian mixture on class-labeled training rows with effective sample weights from the soft cell assignment. Choose component count deterministically from effective class support: >=8000 use 8 components, 3000-7999 use 5, 1000-2999 use 3, 300-999 use 1 or 2 selected by validation log likelihood, and below 300 back off to the broader redshift slab; if the broader slab has fewer than 300 class-effective rows, use a global class model. Use covariance shrinkage toward a diagonal covariance with a variance floor, cap ill-conditioned eigenvalue ratios, and compute all scores in log space with log-sum-exp. To prevent leakage when these features are used for training, generate train-row scores out-of-fold: for each stratified fold, fit the full set of density models on the other folds only and score the held-out fold; fit final models on all training rows only for test scoring. For each object and class, aggregate local log likelihoods by the soft cell weights, blend with the corresponding global class log likelihood when local support is weak, and form two normalized posterior variants: one using uniform class priors to suit balanced accuracy and one using smoothed local empirical priors pi_c = 0.8*freq_c(cell)+0.2/3 to retain population-frequency information. Emit the three class probabilities from the uniform-prior posterior, the three probabilities from the local-prior posterior, pairwise log-odds margins for QSO-vs-STAR, QSO-vs-GALAXY, and STAR-vs-GALAXY, top probability, top-two probability gap, posterior entropy, and effective local support. If any class likelihood is non-finite, replace only that class score with its backed-off global score; if all are invalid, emit uniform probabilities and zero margins.
- expected_signal: The features summarize nonlinear class overlap in color-flux space while adapting to magnitude-dependent noise and redshift-dependent locus shifts, which should help the final classifier recover STAR and QSO cases that are otherwise diluted by dominant GALAXY patterns and therefore improve balanced accuracy.
- risk: The method is computationally expensive and can overfit sparse local cells, especially for rare high-redshift QSO-like regions; posterior features may also duplicate information already present in colors and redshift, and incorrect covariance regularization or non-OOF train scoring would create unstable or overly optimistic training signals.

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