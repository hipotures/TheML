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
- title: Wavelength-aware ugriz shape diagnostics
- group_name: broadband_color_shape
- family: photometric_sed
- summary: Encode each object’s optical spectral shape with deterministic slope- and curvature-derived descriptors from the ugriz bands so class separation depends on continuum geometry rather than absolute brightness.
- strategy: Use only raw u, g, r, i, z magnitudes and treat the SDSS-like band pivot wavelengths as fixed constants in Å: λu=3562, λg=4686, λr=6165, λi=7481, λz=8931. For each row compute first-order color descriptors on the fixed ordering and selected longer baselines: u−g, g−r, r−i, i−z, u−r, g−i, g−z, and u−z. Convert adjacent pair colors to wavelength-normalized slopes to remove non-uniform band spacing effects: s_ug=(u−g)/(ln λg−ln λu), s_gr=(g−r)/(ln λr−ln λg), s_ri=(r−i)/(ln λi−ln λr), s_iz=(i−z)/(ln λz−ln λi). Compute curvature on magnitudes to capture bends in the SED over successive triples: c_ugr=u−2g+r, c_gri=g−2r+i, c_riz=r−2i+z, and a broader c_uz=u−2r+z. Add corresponding slope-scale curvature terms to avoid aliasing from uneven wavelength spacing: k_ugr=(s_gr−s_ug)/(0.5*(ln λr−ln λu)), k_gri=(s_ri−s_gr)/(0.5*(ln λi−ln λg)), k_riz=(s_iz−s_ri)/(0.5*(ln λz−ln λr)). Keep values deterministic, per-row, and untransformed (including negative and large values); do not bin, rank, target-encode, or combine with labels. If any operand used in a specific derived column is non-finite, emit null for that column only and keep all other derived columns available.
- expected_signal: Color differences remove most overall flux-level effects and are known to separate stellar, galactic, and quasar populations better than raw magnitudes, while explicit slope and curvature terms summarize continuum slope changes and breaks that map to stellar temperature sequences, galaxy populations, and quasar outlier structure, which should help improve balanced multiclass discrimination.
- risk: The engineered descriptors are intentionally redundant with raw bands and with existing categorical context, so marginal gain may be limited; some quasar redshift regions have near-overlap in color space with stars; and formulas relying on u-band and endpoint combinations can amplify photometric noise for faint/high-error objects.

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