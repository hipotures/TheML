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
- title: Redshift-aligned absorption trough residuals
- group_name: redshifted_absorption_trough_residuals
- family: absorption_line_geometry
- summary: Measure whether ugriz fluxes show physically aligned, redshift-dependent broadband depressions near major optical absorption complexes, using local continuum residuals to add line-blanketing geometry beyond ordinary colors.
- strategy: Convert magnitudes to log-flux proxies directly as logf_b = -0.4*ln(10)*m_b for bands b={u,g,r,i,z}, with centers c=[3551,4686,6165,7481,8931] Å and approximate band edges e=[3000,4100,5500,7000,8200,9200] Å. Use rest-frame absorption features CaII_K=3933.7, CaII_H=3968.5, Hdelta=4102, Gband=4304, Hgamma=4341, Hbeta=4861, Mgb=5175, and NaD=5893 Å. For wavelength projection use z_eff=max(redshift,0) and add a negative_redshift_flag=1 when redshift<0. For each line compute lambda_obs=lambda_rest*(1+z_eff), visibility v=1 only if 3000<=lambda_obs<=9200, else set all residual features for that line to 0. For visible lines, find containing band j where e_j<=lambda_obs<e_{j+1}. Estimate continuum in log-flux/log-wavelength space without using the containing band: if bands j-1 and j+1 both exist, linearly interpolate between their log-flux values at log(lambda_obs); otherwise fit an ordinary least-squares line to the remaining four non-containing bands and predict at log(lambda_obs). Compute r_line=(exp(logf_cont)-exp(logf_j))/(exp(logf_cont)+1e-12), then deficit d_line=clip(max(r_line,0),0,3) and excess x_line=clip(max(-r_line,0),0,3). Define a band-position reliability weight q_line=clip(1 - 0.5*abs(lambda_obs-c_j)/max(c_j-e_j,e_{j+1}-c_j), 0.5, 1.0). Build visibility-weighted means, not raw sums, to avoid feature counts dominating: blue_abs_deficit=weighted_mean(d_line,q_line) over CaII_K,CaII_H,Hdelta,Gband,Hgamma; red_abs_deficit=weighted_mean(d_line,q_line) over Hbeta,Mgb,NaD; total_abs_deficit=weighted_mean(d_line,q_line) over all lines; total_abs_excess=weighted_mean(x_line,q_line) over all lines. Add contrasts metal_blanketing_skew=(blue_abs_deficit-red_abs_deficit)/(blue_abs_deficit+red_abs_deficit+1e-6), absorption_excess_ratio=total_abs_excess/(total_abs_deficit+1e-6), CaHK_to_Balmer=(weighted_mean CaII_K/CaII_H deficits)/(weighted_mean Hdelta/Hgamma deficits + 1e-6), and visible_line_weight=sum(v*q_line). Also add per-line visibility indicators and redshift-regime gated copies of the main aggregates for z_eff in [0,0.45), [0.45,1.1), [1.1,2.0), and [2.0,7.1], setting gated features to 0 outside their regime.
- expected_signal: These features may improve balanced accuracy by exposing class-specific absorption-blanketing structure that is aligned in rest-frame wavelength rather than fixed observed color space, helping separate galaxies, stars, and QSOs in regions where their raw ugriz colors and redshift values overlap.
- risk: Broadband filters dilute narrow absorption features, so residuals can be noisy and highly correlated with ordinary color, continuum-slope, and break features; the approximate passbands and continuum extrapolation at u/z edges can introduce systematic artifacts, and regime-gated variants increase dimensionality enough to require regularization or feature selection.

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