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
- title: Rest-frame 4000Å break curvature with smooth redshift gating
- group_name: redshifted_4000a_break_curvature
- family: restframe_break_geometry
- summary: This group encodes how the broadband color trajectory bends near the observed-frame position of the 4000 Angstrom rest-frame continuum break as redshift shifts it through ugriz, providing a redshift-localized shape signature that distinguishes galaxy-like stellar-population continua from quasar and stellar SED patterns.
- strategy: Use SDSS effective wavelengths λu=3551, λg=4686, λr=6165, λi=7481, λz=8931 Å. For each object, compute λ_break = 4000*(1+redshift). Define colors c1=u-g, c2=g-r, c3=r-i, c4=i-z. Define regime flags by intervals of λ_break: G in [3551,4686), GR in [4686,6165), RI in [6165,7481), IZ in [7481,8931), and out_of_band=1 otherwise. Set all regime outputs to 0 for out_of_band. For active regime j, compute t=(λ_break-λ_left)/(λ_right-λ_left), p=clip(t,0,1), and w=4*p*(1-p). Compute regime-specific local jump and asymmetry, then multiply both by w and by the corresponding regime flag. For G: jump_G = c1 - c2 and asym_G = (c1-c2) - (c2-c3). For GR: jump_GR = c2 - 0.5*(c1+c3) and asym_GR = (c2-c1) - (c3-c2). For RI: jump_RI = c3 - 0.5*(c2+c4) and asym_RI = (c3-c2) - (c4-c3). For IZ: jump_IZ = c4 - c3 and asym_IZ = (c4-c3) - (c3-c2). Emit 12 numeric features: weighted break terms w*jump_G, w*asym_G, w*jump_GR, w*asym_GR, w*jump_RI, w*asym_RI, w*jump_IZ, w*asym_IZ, plus w and p, and one-hot regime flags; all inactive regime values stay 0.
- expected_signal: The 4000Å break is a physically interpretable anchor that shifts from u/g to z with increasing redshift, so weighting a redshift-localized second-difference/curvature representation should isolate galaxy-specific continuum structure while suppressing global color drift, which helps resolve class confusion in overlapping color regions and should improve balanced accuracy on a class-imbalanced multiclass split.
- risk: The hypothesis depends on redshift precision and the assumed fixed filter centers, so noisy redshift estimates can misassign regimes; it contributes weakly at very low or high redshift where the break is outside ugriz coverage, and broad emission features, dust, or heavy photometric noise can mimic or mask the expected curvature, creating redundancy with other spectral-shape feature groups and potential overfitting to specific redshift-color patterns.

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