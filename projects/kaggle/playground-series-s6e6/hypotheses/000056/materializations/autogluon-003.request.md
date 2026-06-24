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
- title: Sky-context standardized reddening-free locus residuals
- group_name: local_reddening_free_locus_offsets
- family: sky_photometry_normalization
- summary: Create per-object descriptors that compare each source to the local reddening-insensitive color locus at its sky and redshift context, converting raw colors into field-calibrated residuals and density-normalized offsets that are more stable for class separation.
- strategy: Compute base colors c_ug=u-g, c_gr=g-r, c_ri=r-i, and c_iz=i-z, then derive reddening-free axes q1=c_gr-1.582*c_ri and q2=c_ri-0.987*c_iz. Assign each object to a HEALPix context using nside=32 and expand the context to its 8 neighbors (total of up to 9 cells). Assign redshift strata with fixed edges [-inf,0.01,0.1,0.3,0.5,1.0,2.0,inf] and build robust statistics on the training rows only for every context+stratum: median and MAD of q1,q2,c_ug,c_gr,c_ri,c_iz and robust counts n_ctx. If n_ctx<60 at context+stratum, fallback to the same context aggregated across all redshift strata; if still <60, use global statistics for that feature set; if still <60, use full-train global statistics. Use a MAD floor of 1e-6 and compute z-values z_f=(f-median_f)/MAD_f then clip each to [-8,8]. Emit z-scores for q1, q2, c_ug, c_gr, c_ri, and c_iz, emit locus deviation d=sqrt(z_q1^2+z_q2^2), and emit log-density features z_logdens=(log1p(n_ctx)-global_median_logdens_stratum)/MAD_logdens_stratum using the same chosen fallback layer as the medians; additionally emit binary indicators whether the feature layer came from stratified context, unstratified context, or global fallback.
- expected_signal: By removing local field-dependent photometric and extinction drift before classification, the same object classes are expressed in a common relative color space across sky regions, which should reduce intra-class scatter of GALAXY/STAR/QSO signatures while preserving class-outlier structure in a way that is more aligned with balanced accuracy.
- risk: Sparse high-redshift or masked regions can force global fallbacks that weaken the correction, the context smoothing can remove real astrophysical sky variation tied to class prevalence, and several emitted residuals may be highly correlated with raw colors, increasing redundancy and potential overfitting if the downstream model is insufficiently regularized.

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