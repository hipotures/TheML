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
- title: Reliability-weighted luptitude flux geometry
- group_name: luptitude_regime_flux_features
- family: asinh_flux_domain
- summary: Re-express the five SDSS-like asinh magnitudes as calibrated flux-shape descriptors with explicit reliability gates for bands near the luptitude softening floor, so color, slope, curvature, and concentration signals remain usable for faint or noisy objects.
- strategy: Use only per-row photometric predictors u,g,r,i,z, with no target-derived statistics. Create corrected magnitudes m_corr using fixed SDSS-style offsets u_corr=u-0.04, z_corr=z+0.02, and g_corr=g, r_corr=r, i_corr=i; if provenance validation later shows offsets are inappropriate, this group should be generated from the same formulas with zero offsets as an ablation, not mixed row-wise. Use softening constants b={u:1.4e-10,g:0.9e-10,r:1.2e-10,i:1.8e-10,z:7.4e-10} and compute maggy-normalized inverse-luptitude flux f_b=2*b_b*sinh(-(ln(10)/2.5)*m_corr,b-ln(b_b)). For numerical stability clip the sinh argument to a finite implementation-safe range, set eps=1e-12, and define absf_b=abs(f_b). Build reliability variables q_b=absf_b/(absf_b+2*b_b+eps), floor_b=1(absf_b<=2*b_b), ultra_floor_b=1(absf_b<=0.2*b_b), neg_b=1(f_b<0), soft_count=sum(floor_b), ultra_floor_count=sum(ultra_floor_b), floor_frac=soft_count/5, neg_flux_count=sum(neg_b), min_q, mean_q, and reliability_weight=1-floor_frac. Define signed soft-clipped flux ftilde_b=sign(f_b)*max(absf_b,b_b), with sign(0)=1, and log_abs_flux v_b=log10(abs(ftilde_b)+eps). Create adjacent flux-color coordinates flux_color_ab=-2.5*(v_a-v_b) for ug,gr,ri,iz, matching the magnitude-color sign convention; also compute raw magnitude colors mag_color_ab=m_corr,a-m_corr,b and mismatch_ab=mag_color_ab-flux_color_ab. Add second differences over adjacent flux colors, including curvature_ugr=flux_color_ug-flux_color_gr, curvature_gri=flux_color_gr-flux_color_ri, curvature_riz=flux_color_ri-flux_color_iz, and a broad curvature contrast curvature_blue_red=curvature_ugr-curvature_riz. Fit log_abs_flux against centered log wavelength x_b=log10(lambda_b)-mean(log10(lambda)) for lambda=[3551,4686,6165,7481,8931] using both unweighted least squares and reliability-weighted least squares with weights max(q_b,0.05), extracting intercept, slope, and quadratic curvature from X=[1,x,x^2]. Compute residual summaries from the weighted fit: max_abs_residual, mean_abs_residual, and blue_vs_red_residual=resid_u+resid_g-resid_i-resid_z. From absolute soft-clipped flux shares p_b=abs(ftilde_b)/(sum_abs_ftilde+eps), create max_share, min_share, l2_share=sum(p_b^2), entropy_share=-sum(p_b*log(p_b+eps)), blue_share=p_u+p_g, red_share=p_i+p_z, and center_share=p_r. Add sign-pattern summaries neg_flux_count, sign_changes_across_bands over u-g-r-i-z, and sign_entropy from positive versus negative band counts with eps smoothing. Finally create regime-gated variants for the main shape descriptors, multiplying weighted slope, weighted curvature, each mismatch_ab, entropy_share, and max_share by reliability_weight and by floor_frac, so the model can learn separate high-reliability and floor-dominated behavior.
- expected_signal: Balanced accuracy should benefit because this group separates astrophysical spectral-shape information from luptitude floor artifacts: galaxies, quasars, and stars can have similar raw colors in parts of color-redshift space, but reliability-weighted flux slopes, curvature, residual structure, and floor/negative-flux flags expose whether those colors are physically coherent or dominated by low-signal measurement behavior.
- risk: The constants and AB offsets assume SDSS-like asinh photometry, so incorrect provenance can bias every derived feature; log, sinh, and ratio-style operations require strict clipping and eps handling; the descriptors are correlated with ordinary colors and may overfit without regularized models or validation; and weighted polynomial fits add compute cost across the full train and test tables.

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