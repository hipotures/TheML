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
- title: Quasar-Star Locus Outlier and Degeneracy Geometry
- group_name: quasar_stellar_locus_conflict
- family: locus_conflict_geometry
- summary: Model each object’s color-space geometry against the SDSS stellar manifold using two three-band color cubes while adding redshift-linked quasar-inclusion corridors and ambiguity-buffer penalties, so the classifier sees both strong quasar outlier evidence and the known redshift regions where quasar and stellar populations become confounded.
- strategy: Compute c_ug=u-g, c_gr=g-r, c_ri=r-i, c_iz=i-z for every row. Form two 3D color vectors v1=(c_ug,c_gr,c_ri) and v2=(c_gr,c_ri,c_iz). Fit a robust mean and covariance for each vector set on the full available set (train+test), then extract the first principal axis for each cube; define orthogonal residual vectors by subtracting the axis component for each row and compute Mahalanobis orthogonal distances d1,d2 using residual covariance with a color floor added per component (equivalent to adding 0.03^2 in each color term). Create soft outlier scores s1=max(d1-4,0) and s2=max(d2-4,0), and also keep standardized versions d1/z_scale and d2/z_scale. Add three regime features: (a) UV-like regime score favoring very blue c_ug in a bright point-source band, (b) a mid-z bridge score that peaks inside 0.65<c_ug<1.5 and 0<c_gr<0.2 and decays smoothly outside, and (c) a high-z regime score keyed by strong c_ug redness, outlierness in v2, and i-magnitude gating near i<20.2. Add suppressor penalties for known quasar-mimicking contaminant strips defined by white-dwarf/A-star-like color windows from the SDSS target-selection tables, then combine by subtraction so these regions reduce effective quasar support. All scores are clipped to [0,1], all denominators are clamped to avoid divide-by-zero, and invalid rows are deterministically filled with 0.0.
- expected_signal: Separates many quasars that are true color outliers from the thin stellar manifold while explicitly preserving and down-weighting overlap zones where quasars near z≈2.7–2.8 and known contamination strips resemble stars, which should improve minority-class recall without collapsing balanced accuracy into the dominant class boundary.
- risk: Relies on SDSS-era, color-threshold logic that may be mismatched to this dataset’s calibration and label definitions; fixed inclusion/exclusion heuristics can over-penalize legitimate rare quasars near boundaries and may be redundant with existing manifold/locus groups, so incremental gain can be modest if those groups already capture the same geometry.

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