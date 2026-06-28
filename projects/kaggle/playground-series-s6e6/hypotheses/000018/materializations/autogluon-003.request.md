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
- title: Redshift-Aware Tag Color Concordance
- group_name: catalog_tag_color_concordance
- family: tag_photometry_consistency
- summary: Measure how consistently each object's observed ugriz color profile agrees with the semantic temperature and galaxy-population expectations implied by its catalog tags, while treating redshift-related color displacement and tag contradictions as structured ambiguity rather than plain noise.
- strategy: Compute adjacent colors c_ug=u-g, c_gr=g-r, c_ri=r-i, and c_iz=i-z. Fit all distributional constants on the training fold only, then reuse them for validation/test: for each color cj, store q02_j and q98_j and define normalized redness components rj=clip((cj-q02_j)/(q98_j-q02_j),0,1), with denominator floored at 1e-6. Build observed redness R=0.40*r_ug+0.30*r_gr+0.20*r_ri+0.10*r_iz and also keep the four rj components. Map spectral_type expectations as O/B=0.00, A/F=0.33, G/K=0.67, M=1.00 and galaxy_population expectations as Blue_Cloud=0.00, Red_Sequence=1.00; unseen or missing categories map to 0.50. Produce scalar concordance features for each tag source: signed residuals R-te and R-ge, absolute residuals |R-te| and |R-ge|, squared residuals, agreement scores 1-|R-te| and 1-|R-ge| clipped to [0,1], and the signed/absolute tag-disagreement te-ge. Produce component-level residual summaries separately against te and ge: mean, median, min, max, L1 sum, squared L2 sum, and max absolute residual over {r_ug,r_gr,r_ri,r_iz}. Add ordinal color-shape features that compare local color slope to expectations: mean of early colors (r_ug,r_gr), mean of late colors (r_ri,r_iz), early-minus-late redness, and their residuals versus each tag expectation. Create coarse R bins [0,0.25), [0.25,0.50), [0.50,0.75), [0.75,1.00] plus binary contradiction indicators: hot_tag_red_color=(te<=0.33 and R>=0.67), cool_tag_blue_color=(te>=0.67 and R<=0.33), red_sequence_blue_color=(ge>=0.67 and R<=0.33), blue_cloud_red_color=(ge<=0.33 and R>=0.67), and dual_tag_mismatch=(|R-te|>=0.35 and |R-ge|>=0.35). For redshift handling, fit training-fold redshift quantiles q10_z, q50_z, and q90_z and define a smooth trust weight wz=1-clip((redshift-q50_z)/(q90_z-q50_z),0,1), with non-finite or denominator failures defaulting to wz=0.5. Emit both raw mismatch features and wz-weighted mismatch/contradiction features, leaving the raw signals available for high-redshift classes where color displacement may itself be predictive. Any non-finite color, normalized component, R, residual, or score is replaced by 0.50 for continuous agreement features and 0 for indicator features.
- expected_signal: The raw tags likely encode useful but imperfect astrophysical priors, so explicitly modeling agreement, residual direction, contradiction strength, and redshift-dependent reliability can help the classifier distinguish ordinary stars and galaxies from QSOs or ambiguous objects whose colors violate simple tag expectations, which is especially relevant for balanced accuracy on minority or boundary classes.
- risk: The hand-coded semantic ordering and thresholds may duplicate information already recoverable from raw colors and categorical tags, while train-fold quantiles and redshift weighting can be unstable under distribution shift; overly strong contradiction features may also penalize physically valid high-redshift objects whose observed colors differ from local stellar or galaxy expectations.

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