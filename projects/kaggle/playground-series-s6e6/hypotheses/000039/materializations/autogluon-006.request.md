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
- title: Fold-safe local principal-color residual manifold
- group_name: local_principal_color_residuals
- family: locus_geometry
- summary: Represent each object by robust standardized departures from a locally centered SDSS stellar-locus coordinate system, so the model can use class-specific manifold offsets after controlling for redshift, sky position, and catalog tag context.
- strategy: Compute SDSS principal-color coordinates from observed ugriz for every row: s = -0.249*u + 0.794*g - 0.555*r + 0.234, w = -0.227*g + 0.792*r - 0.567*i + 0.050, x = 0.707*g - 0.707*r - 0.983, y = -0.270*r + 0.800*i - 0.534*z + 0.059, and l = -0.436*u + 1.129*g - 0.119*r - 0.574*i + 0.1984. The s/w/x/y definitions are used for all rows; l is considered valid only when 0.5 < g-r < 0.8, with l_valid=0 and l-derived residual values masked/imputed to 0 outside that range. Definitions match the SDSS principal-color and SEGUE l-color documentation at https://www.astro.princeton.edu/~strauss/DR3/QA.html and https://www.sdss3.org/dr8/algorithms/segueii/segue_target_selection.php. Build deterministic context keys using redshift bins [-inf, 0.02, 0.2, 0.6, 1.2, 2.4, 4.0, inf], alpha_bin = floor((alpha mod 360)/30) for 12 RA cells, delta bins split by [-inf, -5, 10, 25, 40, 55, inf], and tag_pair = (spectral_type, galaxy_population). For each coordinate, fit unsupervised training-only medians m_c, MAD values, and valid counts at hierarchical levels: redshift+sky+tag, redshift+sky, redshift+tag, redshift only, tag only, and global. For cross-validation, fit these statistics inside each training fold and apply to the held-out fold; for final test features, refit once on the full training data. For each object and coordinate, choose the first hierarchy level with at least 300 valid rows for s/w/x/y or at least 150 l-valid rows for l; otherwise use the global coordinate statistics. Define scale_c = max(1.4826*MAD_c, 0.02) for s/w/x/y and max(1.4826*MAD_l, 0.03) for l. Emit signed residuals r_c = c - m_c, clipped robust scores z_c = clip(r_c/scale_c, -10, 10), abs_z_c, coordinate missing/reliability flags, and the selected backoff level per coordinate. Aggregate over available coordinates with max_abs_z, mean_abs_z, l2_z = sqrt(mean(z_c^2) * available_count), tail_count_2 = count(|z_c| > 2), tail_count_35 = count(|z_c| > 3.5), tail_ratio_2, positive_tail_count, negative_tail_count, available_color_count, in_locus = 1 when all available |z_c| < 0.75 and at least four coordinates are available, and a 4-bit sign code for z_s, z_w, z_x, z_y.
- expected_signal: Stars should concentrate near the locally normalized stellar locus, while quasars and galaxies should create larger and directionally structured departures that vary with redshift and catalog context; these residual features can improve balanced accuracy by sharpening minority-class recall without relying only on raw magnitudes or redshift.
- risk: The features may be redundant with raw colors, color ratios, or other locus geometry groups; overly fine context cells can create noisy medians if fallback thresholds are too permissive; conditioning on spectral_type and galaxy_population can reduce incremental signal if those tags already encode most class information; and the SDSS-derived coefficients assume comparable photometric calibration, so any mismatch in the synthetic dataset could weaken the residual interpretation.

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