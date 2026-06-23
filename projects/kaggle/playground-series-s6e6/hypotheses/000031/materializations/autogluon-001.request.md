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
- title: Redshift-adaptive color-manifold tube residuals
- group_name: redshift_adaptive_color_tube_residuals
- family: color_manifold_geometry
- summary: Model the thin stellar color manifold separately in redshift slices and provide signed, scale-normalized manifold-distance features that quantify whether each source lies on, orthogonally off, or near the ambiguity intersection of the quasar and stellar loci.
- strategy: Compute c1=u-g, c2=g-r, c3=r-i, c4=i-z for every row and define z_plus = max(redshift, 0). Create equal-frequency redshift bins on z_plus (for example 20 bins), then merge adjacent bins until each bin has at least 5,000 rows; use the same bin edges for both train and test so transformation is deterministic. For each final bin, fit robust scaling on [c1,c2,c3] (ugri cube) and [c2,c3,c4] (griz cube) using median subtraction and MAD scaling, then fit PCA (centered on the bin medians) to obtain v1 as the primary locus axis and v2,v3 as orthogonal residual axes for each cube. For each row in a bin, compute t_ugri=v1_ugri·(x_ugri−μ_ugri), r_ugri=(I−v1_ugri v1_ugri^T)(x_ugri−μ_ugri), d_ugri=||r_ugri||2, and s_ugri=sign(v2_ugri·(x_ugri−μ_ugri)); compute the analogous t_griz, r_griz, d_griz, s_griz for the griz cube. Normalize d_ugri and d_griz by bin-specific 1.4826*MAD(d), clip normalized residuals to [0,99.5th percentile] per bin, and output both raw residuals, normalized residuals, signs, and axis-position ratios t/(MAD(t)+1e-6). Add ambiguity-aware gating using g(z)=exp(−(z_plus−2.7)^2/(2*0.35^2)), based on the known quasar–stellar color overlap near z≈2.7; emit g(z)*d_norm_ugri and g(z)*d_norm_griz as extra features to emphasize redshift regions where off-manifold distances are most diagnostic. For rows falling in undefined/empty bins (boundary artifacts), use the nearest non-empty bin by bin centroid redshift and reuse that bin’s PCA statistics and scales.
- expected_signal: This group converts implicit manifold geometry into explicit supervised-friendly signals: stars should have small orthogonal residuals in their local color tubes, while galaxies and quasars more often produce larger standardized departures, with the overlap-gated variants giving extra separation in the known problematic quasar-star regime around z 2.7 where raw colors alone are less decisive.
- risk: If redshift bins are too fine, PCA directions can be noisy or contaminated by class mixing, especially at sparse high-redshift extremes, and residual normalization can become unstable for ultra-sparse or highly noisy color regions unless robust clipping and minimum-bin constraints are enforced.

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