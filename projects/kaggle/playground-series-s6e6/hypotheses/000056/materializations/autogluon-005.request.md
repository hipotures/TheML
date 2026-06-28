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
- title: Fold-safe sky-context reddening-free locus residuals
- group_name: local_reddening_free_locus_offsets
- family: sky_photometry_normalization
- summary: Compare each object with the robust local color locus expected for nearby sky position and redshift regime, expressing its photometry as field-normalized reddening-resistant residuals rather than absolute calibrated colors.
- strategy: From magnitudes compute colors c_ug=u-g, c_gr=g-r, c_ri=r-i, c_iz=i-z and reddening-resistant axes q_gr_ri=c_gr-1.582*c_ri and q_ri_iz=c_ri-0.987*c_iz. Build the statistic tables only from the current training data used to fit the model; in cross-validation, rebuild them inside each fold, and for final inference fit them on the full labeled train predictors before transforming test. Assign objects to HEALPix cells at nside=32 using alpha and delta, with each lookup context defined as the object cell plus all valid neighboring cells. Use fixed redshift bins [-inf,0.01), [0.01,0.1), [0.1,0.3), [0.3,0.5), [0.5,1.0), [1.0,2.0), [2.0,inf). For each context+redshift bin compute count, median, and MAD for q_gr_ri, q_ri_iz, c_ug, c_gr, c_ri, c_iz, plus broad colors c_ur=u-r and c_gi=g-i. If the context+bin count is below 60, fall back to the same sky context pooled over all redshifts; if that is below 60, fall back to the corresponding global redshift-bin statistics; if that is below 60, use full-training global statistics. Use MAD_scaled=1.4826*MAD with a floor such as max(MAD_scaled,1e-6), compute clipped residuals z_f=clip((f-median_f)/MAD_scaled,-8,8) for all listed color and Q features, and emit q_locus_distance=sqrt(z_q_gr_ri^2+z_q_ri_iz^2). Also emit local density descriptors log1p(count) and a robust z-score of that log-count relative to the global distribution of counts for the same redshift bin and fallback layer. Include compact fallback indicators identifying whether the row used stratified local, unstratified local, global-bin, or full-global statistics.
- expected_signal: Local robust centering should reduce sky-dependent calibration, extinction, and crowding drift that otherwise broadens class color distributions, while the reddening-resistant Q residuals preserve deviations from the stellar locus that help distinguish GALAXY, STAR, and QSO classes under balanced accuracy.
- risk: The features can be noisy in sparse sky or high-redshift contexts, may partially normalize away real spatial population differences, and can be redundant with raw colors and redshift; fold-unsafe statistic fitting would also create validation leakage, so the transformer must be fitted only on the training portion for each validation split.

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