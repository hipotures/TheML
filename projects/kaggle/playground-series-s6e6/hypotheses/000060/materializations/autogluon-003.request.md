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
- title: Leakage-safe hierarchical metadata class-prior encoding
- group_name: metadata_redshift_magnitude_class_priors
- family: conditional_target_prior
- summary: Learn smoothed class-prevalence signals conditioned on spectral metadata, galaxy-population tag, redshift regime, and apparent brightness so ambiguous photometric regions receive empirical context about which stellar class is most plausible.
- strategy: Fit the encoder from training labels only, using out-of-fold computation for train-side features during validation or model fitting and a refit on all training data only for test inference. Estimate global class priors π_k for k in {GALAXY, QSO, STAR}. Assign redshift_bin with fixed edges [-inf, 0.0, 0.1, 0.4, 1.0, 1.4, 2.0, 2.8, 3.5, 4.5, inf] and i_bin with fixed edges [-inf, 17, 18, 19, 20, 21, 22, 23, 24, 25, inf], so every observed or future numeric value maps to a bin. For each row define hierarchical keys L4=(spectral_type, galaxy_population, redshift_bin, i_bin), L3=(spectral_type, galaxy_population, redshift_bin), L2=(spectral_type, galaxy_population), and L1=global. At every observed key and level, compute class counts n_l(k), total n_l, and Dirichlet-smoothed posteriors p_l(k)=(n_l(k)+α·π_k)/(n_l+α), with α=30. Blend from specific to general using count reliability λ_l=n_l/(n_l+β_l), with β_4=150, β_3=250, and β_2=400, producing p*=λ_4 p_4+(1-λ_4)[λ_3 p_3+(1-λ_3)(λ_2 p_2+(1-λ_2)π)]. If a key is unseen or contains an unexpected categorical value, set that level's λ to 0 and continue backing off; if all levels are unavailable, use π. Clamp p*_k to [1e-6, 1-1e-6] and renormalize to sum to 1. Emit p*_GALAXY, p*_QSO, p*_STAR, class entropy -Σp*log(p*), max probability, top-two probability gap, one-vs-rest margins log((p*_k+ε)/(1-p*_k+ε)) for each class, and prior-lift features log((p*_k+ε)/(π_k+ε)) to express how much the local context differs from the global class mix.
- expected_signal: The features summarize stable empirical class-rate shifts across metadata, redshift, and brightness regimes, giving the downstream classifier useful context when ugriz colors overlap while the lift and margin forms help balanced_accuracy by exposing minority-class-enriched regions rather than only raw majority prevalence.
- risk: This is a target-derived encoding, so leakage is likely unless train features are generated out-of-fold; sparse cells, hard bin boundaries, or shifted class prevalence between train and test can produce miscalibrated priors, and the features may be redundant with redshift and magnitude splits already learned by strong tree models, making shrinkage, backoff, and validation discipline essential.

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