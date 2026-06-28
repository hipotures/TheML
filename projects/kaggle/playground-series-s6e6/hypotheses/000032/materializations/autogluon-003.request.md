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
- title: Redshift-localized 4000A break curvature
- group_name: redshifted_4000a_break_curvature
- family: restframe_break_geometry
- summary: Encode the local broadband curvature around the observed-frame location of the rest-frame 4000 Angstrom continuum break, using redshift to make galaxy-like stellar-population breaks separable from smoother stellar continua and quasar color patterns.
- strategy: Use fixed SDSS effective wavelengths lambda_u=3551, lambda_g=4686, lambda_r=6165, lambda_i=7481, lambda_z=8931 Angstrom. Compute lambda_break=4000*(1+redshift) and adjacent colors c_ug=u-g, c_gr=g-r, c_ri=r-i, c_iz=i-z. Assign exactly one interval flag according to lambda_break: in_ug=[lambda_u,lambda_g), in_gr=[lambda_g,lambda_r), in_ri=[lambda_r,lambda_i), in_iz=[lambda_i,lambda_z), and out_of_band otherwise. For the active interval, compute p=clip((lambda_break-lambda_left)/(lambda_right-lambda_left),0,1) and a centrality weight w=4*p*(1-p); for out_of_band set p=0, w=0, and all interval-specific measurements to 0. Define local break-height residuals against neighboring colors: jump_ug=c_ug-c_gr, jump_gr=c_gr-0.5*(c_ug+c_ri), jump_ri=c_ri-0.5*(c_gr+c_iz), jump_iz=c_iz-c_ri. Define signed curvature/asymmetry terms: curv_ug=c_ug-2*c_gr+c_ri, curv_gr=c_ug-2*c_gr+c_ri, curv_ri=c_gr-2*c_ri+c_iz, curv_iz=c_gr-2*c_ri+c_iz. Emit interval-gated weighted features for each interval, e.g. in_ug*w*jump_ug and in_ug*w*curv_ug through in_iz*w*jump_iz and in_iz*w*curv_iz, plus p, w, the four interval flags, and out_of_band. This keeps inactive intervals at zero while allowing tree or linear models to learn both the break strength and where within the relevant filter pair the expected discontinuity falls.
- expected_signal: A true 4000A break creates a redshift-dependent color discontinuity and second-difference pattern as it moves through ugriz, which should be strongest for many galaxies and less coherent for stars or QSOs; localized weighting reduces noise from unrelated colors and may improve balanced accuracy by clarifying galaxy versus non-galaxy decisions in overlapping color-redshift regions.
- risk: The features rely directly on redshift and fixed filter centers, so redshift errors, unusual emission-line objects, dust reddening, or faint photometric noise can shift or distort the apparent break; the group is weak outside the ugriz-covered break range and may be partly redundant with generic color, redshift, and spectral-shape features, increasing overfitting risk if the validation split does not reflect the test distribution.

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