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
- title: SEGUE stellar atmospheric manifold residual stack
- group_name: segue_stellar_atmospheric_indices
- family: atmospheric_color_diagnostics
- summary: Construct a context-aware set of SDSS color-space residual features that quantify how far each object departs from a robust stellar-atmosphere manifold, so STAR-like continua cluster near the manifold while GALAXY and QSO objects exhibit larger structured offsets.
- strategy: Using only train/test columns u,g,r,i,z,redshift,spectral_type,galaxy_population and target-free preprocessing, compute colors ug=u-g, gr=g-r, ri=r-i, iz=i-z. Compute the canonical indices p1=0.91*ug+0.415*gr-1.280, v=0.283*ug-0.354*gr+0.455*ri+0.766*iz, and l=-0.436*u+1.129*g-0.119*r-0.574*i+0.1984 (all in magnitudes). Build redshift bins from train-only deciles (q0..q100 at 10 equal-mass cuts, including clipped edges), assign each row to zbin, then estimate robust center/scale for each x in {p1,v,l} with hierarchical binning: (zbin,spectral_type,galaxy_population) if n>=800, else (zbin,spectral_type) if n>=800, else spectral_type if n>=800, else global. For each bin use median_x and mad_x, where mad_x = 1.4826*median(abs(x-median_x)); define z_x=(x-median_x)/max(mad_x,1e-6). Clamp mad_x-derived divisors to epsilon and clamp all standardized outputs to [-8,8]. Derive bounded tail and manifold-residual features: p1_violation_left=max(0,-0.70-p1), p1_violation_right=max(0,p1+0.25), p1_violation=p1_violation_left+p1_violation_right; l_tail_low=max(0,0.07-l), l_tail_high=max(0,l-0.135), l_tail=l_tail_low+l_tail_high; v_abs=abs(v); abs_z_mean=(abs(z_p1)+abs(z_v)+abs(z_l))/3; locus_dist=sqrt(z_p1*z_p1+z_v*z_v+z_l*z_l); cross_vp=v_abs*z_p1; cross_vl=v_abs*z_l; residual_sign_balance=sign(z_p1)*sign(z_v)*sign(z_l). Replace missing feature values with cell medians, replace NaN/inf with finite defaults (0 for z-scores, median for raw indices), and keep global fallback statistics fixed from train-derived bins only.
- expected_signal: The hierarchical robust standardization removes global scale shifts from redshift and population context while preserving manifold shape, so STAR objects should yield near-zero residual geometry and lower composite distances, whereas QSO and many GALAXY SEDs should show larger tail and orthogonal departures; this reduces overlap in blue-color regions and is well aligned with improving balanced_accuracy across minority classes.
- risk: Bin boundaries and hard thresholds are piecewise and can overfit if train and future test distributions differ (especially redshift extremes or spectral mix changes), some engineered residuals may be redundant with other color-geometry features, and sparse bin fallback can still collapse signals for rare subtype combinations.

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