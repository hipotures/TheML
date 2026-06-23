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

# External Data Description for /home/xai/DEV/aideml/workspaces/2-liberal-celadon-panther/input/star_classification.csv

Original SDSS17 Stellar Classification Dataset.

This is the original real-world dataset that inspired the synthetic Playground
Series S6E6 competition data. It can be used as raw auxiliary data, but it is
not automatically merged with train.csv or test.csv.

Common columns with the competition data:
alpha, delta, u, g, r, i, z, redshift, class.

Columns present in this original dataset but not in the competition files:
obj_ID, run_ID, rerun_ID, cam_col, field_ID, spec_obj_ID, plate, MJD, fiber_ID.

Competition columns not present in this original dataset:
id, spectral_type, galaxy_population.

Generated code should decide whether and how to use this file. Any merge,
filtering, cleaning of sentinel magnitudes, or column mapping must be done
explicitly by the generated solution code.

# Hypothesis
- title: Local principal-color manifold residuals
- group_name: local_principal_color_residuals
- family: locus_geometry
- summary: Measure each object's signed and standardized deviation from local SDSS principal-color planes, conditioned on redshift and coarse sky/catalog context, so class distinctions are expressed as controlled departures from stellar or contamination-leaning manifolds rather than raw color magnitudes.
- strategy: Compute orthogonal stellar-manifold coordinates in each source from fixed SDSS/SEGUE linear combinations: s = -0.249*u + 0.794*g - 0.555*r + 0.234; w = -0.227*g + 0.792*r - 0.567*i + 0.050; x = 0.707*g - 0.707*r - 0.988; y = -0.270*r + 0.800*i - 0.534*z + 0.059; l = -0.436*u + 1.129*g - 0.119*r - 0.574*i + 0.1984. Build context bins from: (1) redshift bin with fixed edges [-0.01, 0.2, 0.6, 1.2, 2.4, 4.0, 7.0], (2) sky cell from alpha/delta using coarse grids (e.g., 12 bins in alpha and 6 bins in delta), and (3) the pair (spectral_type, galaxy_population). For each context, compute median and MAD of each principal-color value using training data only; derive signed residuals c_res = c - median_c and robust scores z_c = (c - median_c)/(1e-6 + 1.4826*MAD_c). Emit per-object features: z_s, z_w, z_x, z_y, z_l, abs(z_*) values, max_abs_z = max(|z_s|,|z_w|,|z_x|,|z_y|,|z_l|), l2_z = sqrt(z_s^2 + z_w^2 + z_x^2 + z_y^2 + z_l^2), an all_close flag where all |z| < 0.8, and sign-code bits for the sign pattern of (z_s, z_w, z_x, z_y). If a context has <300 rows, back off to redshift+tag only, then tag only, then global medians to avoid noisy baselines; clip all z_c to [-10,10]. References informing coefficients: SDSS QA principal colors and SEGUE l-color definitions at https://www.astro.princeton.edu/~strauss/DR3/QA.html and https://www.sdss3.org/dr8/algorithms/segueii/segue_target_selection.php.
- expected_signal: Stars remain close to the principal-color loci while quasars and galaxies more frequently produce structured perpendicular excursions that vary with redshift and tag context, so standardized residual geometry provides a compact, class-aware morphology signal that is less sensitive to absolute photometric level shifts than raw magnitudes alone.
- risk: Feature redundancy is likely with other manifold/locus groups and may offer diminishing returns; sky-cell medians can become unstable in sparse angular/redshift regions, causing noisy residuals unless fallback is robust, and fixed coefficients are tied to SDSS calibration conventions so any photometric-system mismatch could attenuate separability.

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