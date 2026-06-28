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
- title: Surrogate photometry flag margins
- group_name: surrogate_photometry_flag_margins
- family: measurement_quality
- summary: Approximate missing SDSS-style photometric quality and support-boundary flags from the available predictors so likely corrupted, saturated, edge-case, or single-band-anomalous measurements become explicit classification context.
- strategy: Use only training-predictor fit constants and apply them unchanged to test rows: for alpha, delta, u, g, r, i, z, redshift, adjacent colors, and all pairwise colors, store robust quantiles q0.001, q0.01, q0.99, q0.999 and MAD-scaled centers; create lower_tail=max(0,(q0.001-x)/(q0.01-q0.001+eps)), upper_tail=max(0,(x-q0.999)/(q0.999-q0.99+eps)), two_sided_tail=max(lower_tail,upper_tail), and clipped empirical support percentile min(rank_pct,1-rank_pct). For ugriz only, fit a robust straight continuum per row against log effective wavelength anchors 3543, 4770, 6231, 7625, and 9134 Angstrom, compute leave-one-band residuals standardized by training MAD, then summarize maximum isolated residual, residual dispersion, count of bands above 2 and 4 robust sigma, signed residuals at the most extreme blue and red bands, and one-hot indicators for which band is most pathological. Add aggregate support features such as maximum numeric tail, count of numeric values beyond q0.001/q0.999, maximum color tail, count of extreme colors, alpha wrap-edge distance min(alpha,360-alpha), delta survey-edge distance from training min/max, and redshift cap proximity to -0.01 and 7.01; replace zero scale denominators with eps and clip all margin magnitudes to [0,10]. External basis used: SDSS quasar-selection documentation notes rejection of BLENDED, BRIGHT, SATURATED, EDGE, deblending, interpolation, and other photometry-error flags, and SDSS target-selection documentation discusses bright/faint magnitude limits and imaging-pipeline effects (https://classic.sdss.org/dr7/products/general/edr_html/node54.php, https://classic.sdss.org/dr7/products/general/target_quality.php).
- expected_signal: Objects with bad or boundary photometry can be scattered away from normal color loci, especially causing stars to mimic quasars or compact galaxies, so explicit quality-surrogate margins may help balanced accuracy by separating true astrophysical outliers from measurement or survey-support artifacts.
- risk: The group may be redundant with existing depth, censoring, and continuum-shape features, and because it uses empirical support edges it can overfit synthetic generation or survey-release-specific artifacts rather than stable astrophysical class structure.

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