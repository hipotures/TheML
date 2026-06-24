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
- title: Fold-wise Bayesian color posteriors with hierarchical stratum backoff
- group_name: class_conditional_color_density_posteriors
- family: bayesian_density_scoring
- summary: This group generates class-aware posterior-like compatibility features by estimating how likely each object’s colors are under galaxy, quasar, and stellar color distributions conditioned on redshift and i-band brightness, then emitting calibrated class score and margin signals.
- strategy: For each model-training fold, build all statistics from the training partition only and reuse the same transforms for validation and test rows. Construct colors c=[u−g, g−r, r−i]. Clip redshift to [−0.01, 7.01] and clip each color to foldwise 0.5th/99.5th percentile bounds, keeping two overflow bins so binning is always valid. Use fixed redshift intervals z_breaks=[−0.01, 0.20, 0.60, 1.05, 1.70, 2.45, 3.60, 5.00, 7.01] and fixed 0.50-mag i-band intervals with hard clipping to edge bins. For each color component, use 16 equiprobable bins within the clipped range plus two overflow bins; combine into 3D cells b=(b1,b2,b3) and keep sparse counts n_{c,k,m,b} for class c∈{GALAXY,QSO,STAR}, redshift bin k and i-bin m. Apply Dirichlet smoothing α=1 and class-prior smoothing β=0.5: π_{c,k,m}=(N^c_{k,m}+β)/(N_{k,m}+3β), P_{c,k,m}(b)=(n_{c,k,m,b}+α)/(N^c_{k,m}+α·B^3), where B is number of 1D color bins including overflow bins and N^c_{k,m}=∑_b n_{c,k,m,b}. Precompute marginals P_{c,k}(b) (sum over m), P_c(b) (sum over k,m), and global π_c from the same fold. Score each sample deterministically with a strict backoff chain: if N_{k,m} and the target cell support are above n_min=180, use P_{c,k,m}(b); else smooth within the same stratum with Chebyshev-neighborhood averaging first radius 1 then radius 2 using weights [1, 1/2, 1/4, ...], renormalized; if still sparse use P_{c,k}(b), then P_c(b), and finally π_c. Use s_c=log(p_support+1e-12)+log(π_support+1e-12), where p_support is the chosen likelihood level and π_support is corresponding prior at that level. Output dense features s_GALAXY, s_QSO, s_STAR, margins s_QSO−s_STAR and s_STAR−s_GALAXY, and softmax entropy of [s_GALAXY,s_QSO,s_STAR].
- expected_signal: Conditioning color distributions on both redshift and i-band brightness aligns with known astrophysical shifts in object loci, so score contrasts resolve overlap regions with lower ambiguity and preserve class prevalence in each physical regime, which should improve balanced recall of minority structures such as high-redshift QSOs without forcing a rigid global decision boundary.
- risk: Although staged backoff reduces brittle sparsity effects, the fixed stratum and bin edges can still induce boundary sensitivity and noisy estimates in poorly populated regions if tuned on full data, and maintaining many 3D sparse cubes can increase memory/time unless explicitly implemented with sparse structures and strict fold-based fitting.

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