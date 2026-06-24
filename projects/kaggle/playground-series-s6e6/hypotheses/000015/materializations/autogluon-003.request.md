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
- title: Redshift Feasibility Margins Across SDSS Template Domains
- group_name: redshift_template_domain_margins
- family: redshift_geometry
- summary: Model redshift as a compact class-geometry signal by encoding how closely each object sits to the operational STAR, GALAXY, and QSO redshift feasibility ranges used by SDSS-style spectral fitting.
- strategy: From raw redshift z, construct deterministic domain-feasibility features for three class templates: STAR domain S=[-v_max/c, +v_max/c] where v_max=1200 km/s and c=299792.458 km/s, GALAXY domain G=[-0.01, 1.00], QSO domain Q=[0.0333, 7.00]. For each domain d in {S,G,Q}, compute signed margin m_d(z)= (z-low_d) if z<low_d else (z-high_d) if z>high_d else 0, and absolute gap g_d=|m_d| (for in-domain values g_d=0). Add domain flags f_d=1 if low_d<=z<=high_d else 0, preserving exact in-domain equality. Add global sign feature s=sign(z) and star_velocity_proxy = z / (1200/299792.458) to expose star-like Doppler scaling. Add ordered regime-distance features for sorted breakpoints T=[-0.01, 0, 0.0333, 1.0, 2.2, 3.0, 3.5, 4.5, 5.0, 7.0]: for each t in T, d_t_low=z-t, d_t_high=next_t-z, then replace missing from end segments with a large finite constant and take clipped values log1p(|d|) and also raw signed distances where useful; clip all distance/margin features to a finite symmetric cap (e.g., ±10 after log1p transform space) to avoid leverage. For impossible-probability control, create hard penalties p_G = max(0, -m_G) and p_Q = max(0, -(z-0.0333)) for z<0.0333 and p_S = max(0, |m_S|-0.00401) to reinforce the near-zero star domain. All operations are deterministic and vectorizable, with no target leakage and no floating point inf/NaN outputs.
- expected_signal: Converting z into class-domain geometry and explicit boundary-gap features should increase separability under balanced accuracy because STAR cases are tightly concentrated around |z|≈1200 km/s, galaxies concentrate in a low-to-moderate positive band, and quasars occupy a broader high-z regime with characteristic low-z and intermediate-z ambiguity windows; this gives tree and linear models a prior-compatible, interpretable structure that softens boundary flips and reduces reliance on raw redshift alone.
- risk: The feature group is highly correlated with raw redshift, so gains may be small for high-capacity models and can overfit if domain boundaries are treated as hard physics in a distribution-shifted synthetic test set; fixed breakpoints can become brittle under dataset or simulator redshift drift, and clipped margin transforms can hide but not remove edge-case miscalibration near overlap regions.

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