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
- title: RR Lyrae strip margins
- group_name: rr_lyrae_instability_strip_margins
- family: stellar_contaminant_geometry
- summary: Represent each object's proximity to the SDSS RR Lyrae and blue horizontal-branch instability-strip color locus, with brightness, spectral-type, and near-zero-redshift consistency, so stellar contaminants that overlap quasar color space become explicit.
- strategy: Compute adjacent colors ug=u-g, gr=g-r, ri=r-i, and iz=i-z. Use the published single-epoch SDSS RR Lyrae broad box 0.99<=ug<=1.28, -0.11<=gr<=0.31, -0.13<=ri<=0.20, and -0.19<=iz<=0.23; derive signed normalized margins, positive lower/upper violations, aggregate minimum margin, mean violation, max violation, and squared normalized distance to the box center, scaling by interval half-widths and clipping normalized values to [-10,10]. Compute rotated color distances Dug=ug+0.67*gr-1.07 and Dgr=0.45*ug-gr-0.12; encode complete-strip margins for -0.05<=Dug<=0.35 and 0.06<=Dgr<=0.55 plus restricted-strip margins using Dug>=0.15 and Dgr>=0.23 with the same upper bounds. Add calibrated-regime gates for 14<=r<=20, u<21.1, r<19.7, spectral compatibility mapped as A/F=1.0, O/B=0.4, G/K=0.1, M=0.0, and near-zero-redshift support 1/(1+(abs(redshift)/0.004)^2); multiply these gates with the color-strip scores as within-group interactions. Treat unknown categories as zero compatibility and nonfinite numeric values as worst-margin after clipping. Source basis: https://faculty.washington.edu/ivezic/Publications/203458.web.pdf and https://www.sdss4.org/dr17/algorithms/legacy_special_target/.
- expected_signal: Low-redshift quasars and A/F or horizontal-branch stars can occupy similar blue UV-excess color space, so explicit instability-strip proximity may improve STAR versus QSO separation without using target statistics.
- risk: The cuts were calibrated for bright SDSS point-source samples and may be redundant with raw colors, spectral_type, and existing stellar-locus features; without variability, morphology, or proper motion, quasars inside the same color strip can still receive misleading stellar support.

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