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
- title: Hierarchical class-prior encoding from spectral, population, redshift, and magnitude context
- group_name: metadata_redshift_magnitude_class_priors
- family: conditional_target_prior
- summary: This feature group injects empirical, context-conditioned class prevalence signals derived from astrophysical metadata so that the classifier can distinguish otherwise overlapping photometric patterns by the expected target mix in each metadata regime.
- strategy: Fit this encoder exclusively from train labels. First estimate global priors π_k for k in {GALAXY, QSO, STAR}. Bin redshift into fixed edges [-0.01, 0.15, 0.40, 1.0, 1.4, 2.0, 2.8, 3.5, 4.5, 7.02] and i-band into fixed edges [-inf, 17, 18, 19, 20, 21, 22, 23, 24, 25, inf]. For each row define a hierarchical key chain: L4=(spectral_type, galaxy_population, redshift_bin, i_bin), L3=(spectral_type, galaxy_population, redshift_bin), L2=(spectral_type, galaxy_population), L1=global. For every observed key at each level compute class counts n_c(k), total n_c, and Dirichlet-smoothed posteriors p_c(k)=(n_c(k)+α·π_k)/(n_c+α) with α=30. Let λ_c=n_c/(n_c+β), β=120. For a row with counts at each level, blend priors by descending backoff: p*=λ_4 p_4 + (1-λ_4)[λ_3 p_3 + (1-λ_3)(λ_2 p_2 + (1-λ_2)π)]. If any key is unseen, skip its level and continue from the next parent; if no counts exist at all, use π. Clamp final p*_k to [1e-6,1-1e-6], then emit: p*_GALAXY, p*_QSO, p*_STAR, class entropy H=-Σ p*log p*, max probability, top-two gap, and one-vs-rest logit-like margins log((p*+eps)/(1-p*+3eps)) for each class. Map unexpected categorical levels at inference time directly to global priors before applying interpolation.
- expected_signal: The priors explicitly encode how class balance changes with redshift and brightness within known astrophysical metadata strata, which helps balanced_accuracy by improving recall in feature-overlap regions where color-only boundaries between stars, galaxies, and quasars are weak or unstable.
- risk: Because the features are target-derived, aggressive binning or too little smoothing can overfit sparse strata and create context-specific noise; in deployment, shifts in class prevalence by redshift, spectral_type, or i-band range would degrade calibration, so the hierarchical backoff and fixed shrinkage are required to avoid unstable outputs, and out-of-fold generation is needed if this is used inside internal validation.

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