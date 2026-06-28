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
- title: AIDE broadband flux-ratio photometric expansion
- group_name: aide_broadband_flux_ratios
- family: aide_photometry
- summary: Add deterministic nonlinear photometric transforms that recast ugriz magnitudes and redshift into broadband ratio, gap, flux-proxy, curvature, and compact redshift-interaction views while preserving the original AIDE-style feature idea.
- strategy: Use only u, g, r, i, z, and redshift, with no target labels, fitted statistics, row-order information, or model predictions. Let eps=1e-6 and k=-0.4*ln(10). For each magnitude band m in {u,g,r,i,z}, compute log_flux_proxy=k*m and flux_proxy=exp(clip(log_flux_proxy,-30,10)); emit both, allowing downstream duplicate removal to discard affine-equivalent copies if desired. For ordered band pairs P={(u,g),(g,r),(r,i),(i,z),(u,r),(u,i),(u,z),(g,z),(r,z)}, emit abs_gap=abs(ma-mb), signed_norm_gap=(ma-mb)/(abs(mb)+eps), safe_mag_ratio=ma/(sign(mb)*max(abs(mb),eps)), and flux_ratio=flux_proxy_a/(flux_proxy_b+eps). Add curvature terms u-2*r+i, g-2*i+z, and the adjacent color-curvature contrasts (u-g)-(g-r), (g-r)-(r-i), and (r-i)-(i-z). For redshift, set zr=max(redshift,0), emit zr, log1p(zr), zr^2, and zr^3 after clipping zr to [0,8], then add a small fixed set of interactions: log1p(zr) times each band flux_proxy, zr*(u-g), zr*(g-r), zr^2*(r-i), and zr^2*(g-z). Replace any non-finite result with a bounded sentinel derived from the same clipping rule, and keep all feature names deterministic and identical for train and test.
- expected_signal: Balanced accuracy may improve because the added features expose class-relevant broadband spectral slope, flux-scale, color-curvature, and redshift-dependent separation patterns that are only implicit in raw magnitudes and simple colors, helping minority classes such as QSO and STAR without using label-derived information.
- risk: Many outputs are correlated with raw magnitudes or existing color features, so they can add noise and training cost; ratio and exponential terms may also exaggerate measurement artifacts or rare high-redshift tails if clipping, denominator protection, and downstream regularization are not handled consistently.

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