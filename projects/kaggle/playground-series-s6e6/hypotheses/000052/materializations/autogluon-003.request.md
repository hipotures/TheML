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
- title: Calibrated SEGUE subtype envelope margins
- group_name: segue_stellar_subtype_margins
- family: stellar_color_geometry
- summary: Represent each object by robust continuous distances to several canonical SDSS-like stellar subtype envelopes, preserving subtype-specific color geometry as structured evidence for or against stellar classification.
- strategy: Compute colors ug=u-g, gr=g-r, ri=r-i, iz=i-z, gi=g-i and linear colors l_color=-0.436*u+1.129*g-0.119*r-0.574*i+0.1984, s_color=-0.249*u+0.794*g-0.555*r+0.234, p1=0.91*ug+0.415*gr-1.280, v=0.283*ug-0.354*gr+0.455*ri+0.766*iz. For every primitive or derived clause variable, fit training-only robust scale scale=max(MAD,0.01) and training-only clipping limits at the 0.5/99.5 percentiles before applying penalties. Use nonnegative normalized envelope penalties: interval_penalty(x,a,b)=0 inside [a,b] else distance to nearest bound divided by scale_x; upper_penalty(x,k)=max(0,x-k)/scale_x; lower_penalty(x,k)=max(0,k-x)/scale_x. Build prototype penalties as weighted means of normalized clause penalties, with equal weights unless a prototype has alternative branches, in which case take the minimum branch penalty. Prototypes: white_dwarf uses gr in [-1.0,-0.2], ug in [-1.0,0.7], and ug+2*gr<=-0.1; cool_white_dwarf uses r in [14.5,20.5], gi in [-2.0,1.7], and min of blue branch ug in [-0.4,1.0] for gi<0.6 versus red branch ug in [-0.2,1.4] for gi>=0.6, including a small branch-boundary penalty around gi=0.6; A_BHB uses ug in [0.8,1.5], gr in [-0.5,0.2], and v in [-0.15,0.15]; K_giants uses gr in [0.35,0.8], ri in [0.15,0.6], and l_color>=0.07; low_metallicity uses gr in [-0.5,0.75], ug in [0.6,3.0], and l_color>=0.135; M_subdwarf uses ri<=0.9, ri<=0.787*gr-0.356 expressed as upper_penalty(ri-0.787*gr+0.356,0), and gi in [1.8,2.4]; main_sequence_white_dwarf_pairs uses ug<=2.25, gr in [-0.2,1.2], ri in [0.5,2.0], gr>-1.0*ri+1.13 expressed as lower_penalty(gr+ri-1.13,0), gr<0.95*ri+0.50 expressed as upper_penalty(gr-0.95*ri-0.50,0), and min of iz in [-0.5,1.0] for ri<1.0 versus iz in [-0.2,0.7] for ri>=1.0 with a small branch-boundary penalty around ri=1.0. Emit each prototype penalty after clipping to a finite cap such as 20, plus aggregates star_margin_best, star_margin_second, star_margin_gap, star_margin_mean_top3, star_inlier_count using threshold <=0.05, and star_confidence=exp(-star_margin_best). Do not multiply penalties by redshift, because redshift is already a highly discriminative measured feature and a hand gate can suppress real low-redshift contaminants or unusual stars; instead optionally emit redshift-adjusted interaction summaries such as star_margin_best*log1p(max(redshift,0)) if interactions are allowed within this group.
- expected_signal: Continuous subtype-envelope proximity should help balanced accuracy by giving the classifier explicit evidence for STAR-like color geometry and by distinguishing objects that satisfy one broad color cut from objects that consistently match a full stellar subtype region.
- risk: The boundaries are inherited from SDSS-style targeting logic and may be miscalibrated for this synthetic or shifted sample; the features can be redundant with generic color and redshift predictors, and compact quasars or galaxies with stellar-like colors may receive misleadingly low penalties.

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