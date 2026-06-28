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
- title: Asinh-Jacobian weighted SED shape
- group_name: asinh_jacobian_weighted_shape
- family: flux_uncertainty_geometry
- summary: Transform ugriz luptitudes into signed relative fluxes and encode five-band continuum geometry with weights that down-weight bands whose asinh response indicates low effective signal, preserving reliable SED shape while limiting noisy color excursions.
- strategy: For each band b in {u,g,r,i,z}, invert the SDSS asinh magnitude relation with softening constants b_soft={u:1.4e-10,g:0.9e-10,r:1.2e-10,i:1.8e-10,z:7.4e-10}: a_b=-(ln(10)/2.5)*m_b-ln(b_soft_b), clipped to [-35,35], and q_b=2*b_soft_b*sinh(a_b), where q_b is signed flux in maggies relative to the zero point. Define the asinh response index x_b=q_b/(2*b_soft_b) and reliability weight w_b=abs(x_b)/sqrt(1+x_b^2), clipped to [0,1]; this is near zero at the luptitude softening floor and near one in the high-SNR logarithmic regime. Compute n_eff=sum(w_b). If any q_b is non-finite or n_eff < 1e-4, replace the five q_b values with training-only medians from the matching redshift-bin x spectral_type x galaxy_population cell, using redshift bins [-inf,0.02,0.1,0.4,0.8,1.5,3.0,inf] and falling back to redshift-bin medians and then global medians; emit low_confidence=1, otherwise low_confidence=0. For stable scale separation, compute q_level=sum(w_b*q_b)/max(n_eff,1e-6), q_scale=max(weighted median of abs(q_b-q_level), median(abs(q_train_global)), 1e-12), and normalized shape y_b=(q_b-q_level)/q_scale. On centered band coordinates t={-2,-1,0,1,2}, fit y_b=s0+s1*t+s2*(t^2-2) by weighted ridge least squares with diagonal penalty 1e-6 on coefficients and weights w_b, emitting s0,s1,s2 plus weighted residual RMS and weighted residual MAE. Emit signed adjacent contrasts c_ab=sqrt(w_a*w_b/max(w_a+w_b,1e-6))*(q_a-q_b)/q_scale for ug, gr, ri, iz; second-difference curvature contrasts d_ugr, d_gri, d_riz using the same endpoint-weight scale sqrt(w_left*w_right/max(w_left+w_right,1e-6)); and reliability descriptors n_eff, frac_hi=count(w_b>0.6)/5, frac_mid=count(w_b>0.2)/5, min_w, max_w, and mean reliable adjacent absolute contrast over pairs with both weights >0.1, defaulting to 0 when no pair qualifies.
- expected_signal: Balanced accuracy may improve because faint or partially non-detected bands, especially u and z, can distort ordinary color differences, while the Jacobian-weighted flux shape keeps reliable continuum slope and curvature information useful for separating stellar loci, galaxy continua, and quasar UV or emission-line affected SEDs.
- risk: The features are partly redundant with raw magnitudes, ordinary colors, and other flux transforms, so gains may be incremental; fixed SDSS softening constants can be miscalibrated if the source photometry differs from standard luptitudes, train-only fallback medians must be handled carefully to avoid validation leakage, and scale normalization may compress rare extreme colors that are genuinely informative for QSO or outlier STAR cases.

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