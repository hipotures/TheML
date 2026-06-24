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
- title: Regime-Weighted Dual Rest-Frame Break Residuals
- group_name: dual_restframe_break_alignment
- family: break_competition_geometry
- summary: This feature family encodes how strongly each object exhibits a 1216 Å and 4000 Å rest-frame discontinuity when mapped into ugriz, by measuring break jump residuals against a local smooth-continuum expectation so class differences from Lyman-dropout-like quasars versus 4000 Å-dominated galaxies versus weak-break stellar SEDs are expressed in a single geometric signal.
- strategy: Let ugriz magnitudes be m=[m_u,m_g,m_r,m_i,m_z] with effective wavelengths λ=[3551,4670,6170,7480,8930] Å and z̃=clip(redshift, 0, 7). For each target rest-frame break λ_b in {1216,4000}, compute λ_b^obs=λ_b*(1+z̃) and locate the unique interval j ∈ {0,1,2,3} where λ_j ≤ λ_b^obs < λ_{j+1}; if none exists (below u band or above z band), emit all break-related features as 0 and set miss_b=1 and skip residual computation for that break. For valid intervals, define fractional position Δt_b=(ln λ_b^obs−ln λ_j)/(ln λ_{j+1}−ln λ_j), observed jump J_obs,b = m_j−m_{j+1}, and observed slope S_obs,b=(m_j−m_{j+1})/(ln λ_j−ln λ_{j+1}). Estimate local smooth continuum slope S_cont,b via linear fit in ln λ: if j=1 or 2, fit through points (ln λ_{j-1},m_{j-1}) and (ln λ_{j+2},m_{j+2}); if j=0, fit through (ln λ_1,m_1) and (ln λ_2,m_2); if j=3, fit through (ln λ_2,m_2) and (ln λ_4,m_4). Then J_exp,b = S_cont,b·(ln λ_j−ln λ_{j+1}) and residual R_b=(J_obs,b−J_exp,b)/S_loc,b, where S_loc,b is the median absolute local color scale over available adjacent colors around the interval, specifically S_loc,b=median(|m_0−m_1|, |m_1−m_2|, |m_2−m_3|, |m_3−m_4|)+1e-6; clip R_b to [−8,8]. Add interval-confidence C_b=clip(1−2·|Δt_b−0.5|, 0, 1) so C_b downweights placements near band boundaries; generate miss_b, boundary_b=(C_b<0.2), and regime flags lya_regime=(z̃∈[1.9,6.5]), balmer_regime=(z̃∈[0,1.3]). Output deterministic features per break: raw jump J_obs,b, normalized residual R_b, confidence-weighted jump J_obs,b·C_b, confidence-weighted residual R_b·C_b, and miss/border/regime flags. Build competition features: Break_balance=((R_1216·C_1216)−(R_4000·C_4000)), Break_abs_diff=|R_1216·C_1216|−|R_4000·C_4000|, and sign-consistent variants that suppress invalid breaks by multiplying with (1−miss_b).
- expected_signal: Because quasars at suitable redshift should show strong positive 1216 Å-break residuals with weak competing 4000 Å structure while low-redshift galaxies usually show opposite balance and stellar SEDs remain smoother across both windows, these confidence-weighted cross-break residual contrasts should separate the three classes in a regime-aware, redshift-consistent manner and reduce dependence on raw color magnitude alone.
- risk: The feature signal is sensitive to redshift-dependent regime assignment, so objects near band-edge transitions or with systematic redshift offsets can misplace breaks and distort residuals; broad-band filters and emission-line features can mimic continuum jumps, and the scheme may be less stable for intrinsically smooth or noisy SEDs where local slopes are poorly defined despite clipping.

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