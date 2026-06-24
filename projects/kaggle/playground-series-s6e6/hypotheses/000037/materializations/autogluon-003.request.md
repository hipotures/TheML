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
- title: Redshift-Conditioned Galaxy-Manifold Deviation Features
- group_name: population_color_manifold_drifts
- family: tag_conditioned_color_geometry
- summary: Learn redshift-dependent color manifolds separately for the two population tags and represent each object by how strongly it aligns or diverges from its tagged manifold versus the alternative manifold, using robust, bin-local geometry to surface class-consistent offsets in overlap-heavy color regions.
- strategy: Compute a fixed color vector per object using ugri z-band contrasts (u-g, g-r, r-i, i-z, u-r). Bin redshift using training data only into B=24 equal-frequency bins on t=log1p(redshift), with all bin edges taken from training and clipped so every test object maps deterministically. For every (bin b, population p) cell, estimate robust location mu_{b,p,f} and robust scale sigma_{b,p,f}=1.4826*MAD for each color f from training rows. If a cell has n_{b,p}<N_min (N_min=300) or sigma_{b,p,f} is zero/near-zero, shrink toward global training statistics: w=min(1,n_{b,p}/N_min); mũ=w*mu_{b,p,f}+(1-w)*mu_global_f, sigmã=w*sigma_{b,p,f}+(1-w)*sigma_global_f. If both nearby bins also lack mass, fallback to adjacent-bin averaging with count-weighted interpolation before global fallback. For each object in bin b with tag p, compute standardized residuals z_{b,p,f}=clip((c_f-mũ_{b,p,f})/sigmã_{b,p,f},-8,8), and D_assigned=mean_f(z_{b,p,f}^2). Compute D_opposite from the opposite population statistics in the same bin using the same clipping and scaling. Add cross-manifold discrimination features: D_diff=log1p(D_opposite)-log1p(D_assigned) and D_ratio=D_opposite/(D_assigned+1e-6). Also add per-color relative manifold features Delta_f=(c_f-mũ_{b,Red,f})/sigmã_{b,Red,f} - (c_f-mũ_{b,Blue,f})/sigmã_{b,Blue,f}. Build red/blue divider features per color: mid_f=0.5*(mũ_{b,Red,f}+mũ_{b,Blue,f}), gap_f=mũ_{b,Red,f}-mũ_{b,Blue,f}, and m_f=(c_f-mid_f)*sign(gap_f), where sign indicates side of the bin-wise separator and magnitude indicates margin-to-boundary. For any unavailable color contribution after shrink/fallback, set its term to zero and emit a missing contribution indicator.
- expected_signal: The resulting features make the model sensitive to the expected redshift evolution of red-sequence versus blue-cloud trajectories while letting stars and quasars be penalized for inconsistent manifold fit, which should improve boundary quality where raw magnitudes overlap and help balanced accuracy by giving clearer minority-class separation.
- risk: If population tags are noisy for non-galactic objects, manifold-distance penalties can introduce structured errors, sparse bin estimation can still be unstable even with shrinkage, hard bin edges may create discontinuities near cut points, and diagonalized robust distances can miss useful color covariance unless compensated by downstream learners.

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