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
- title: UBVRI Transform Consistency
- group_name: ubvri_transform_consistency
- family: cross_system_photometry
- summary: Represent each object by how consistently its SDSS ugriz colors can be translated into Johnson-Cousins UBVRI color space under published stellar and quasar transformation assumptions, exposing objects whose broadband SED behaves like a normal stellar continuum versus a non-stellar or out-of-domain spectrum.
- strategy: Using fixed SDSS-published ugriz-to-UBVRcIc transformations from https://www.sdss3.org/dr8/algorithms/sdssUBVRITransform.php, compute base colors u-g, g-r, r-i, i-z, g-i, and r-z; derive pseudo UBVRI colors separately under the Jester all-star equations, Jester z<=2.1 quasar equations, Jordi Population I and Population II equations, and Karaali main-sequence B-V equation where available; for each shared physical color such as U-B, B-V, V-I, R-I, and B-g, add signed and absolute disagreements between the stellar, quasar, Population I, Population II, and two-color estimates; add self-consistency residuals where one transformed estimate can be reconstructed from an independent color relation, such as B-g from u-g versus B-g from g-r, R-I inferred from r-i versus r-z, and B-V from g-r only versus B-V from both u-g and g-r; standardize residuals by the published RMS or coefficient uncertainty when provided, otherwise leave them in magnitudes; add validity-margin features for transformation domains including Rc-Ic<1.15, U-B<0, 0.3<B-V<1.1, g-i<=2.1, and redshift<=2.1 for the quasar equations, with negative values outside the valid region; clip only extreme transformed colors and residuals to broad finite bounds based on train-test combined predictor quantiles to avoid numerical domination without using labels.
- expected_signal: Stars should usually satisfy multiple stellar color-system transformations with small internal disagreement, quasars may align better with the quasar-specific transform only in its valid redshift range, and galaxies or unusual high-redshift objects should show larger cross-system inconsistency, adding class-separating evidence beyond raw ugriz colors.
- risk: Most derived quantities are linear or piecewise-linear functions of existing colors, so the group may be partly redundant with prior color-shape features; the published transformations were not designed for all galaxy or high-redshift quasar SEDs, so out-of-domain residuals may be noisy if the synthetic data distribution departs from SDSS calibration assumptions.

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