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
- title: Regime-gated luptitude flux geometry descriptors
- group_name: luptitude_regime_flux_features
- family: asinh_flux_domain
- summary: This feature family reconstructs each object's five-band spectral shape in luptitude-derived flux space while explicitly modeling the asinh softening regime so that slope and curvature cues used for class separation are preserved when measurements sit near the photometric floor.
- strategy: For each row use u, g, r, i, z and apply SDSS asinh photometric offsets before inversion with fixed defaults u_corr = u - 0.04 and z_corr = z + 0.02, while g_corr = g, r_corr = r, i_corr = i (all offsets are constants to be toggled by data provenance). Use b = {u:1.4e-10, g:0.9e-10, r:1.2e-10, i:1.8e-10, z:7.4e-10} and compute inverse-luptitude flux proxies in maggy-normalized units as f_b = 2*b_b*sinh(-(ln(10)/2.5)*m_corr,b - ln(b_b)). Use eps=1e-12 and define absf_b=|f_b|. Define floor regime flags floor1_b = 1(absf_b <= 2*b_b) and floor2_b = 1(absf_b <= 0.2*b_b), then soft_count = Σ floor1_b, floor_frac = soft_count/5, ultra_floor_count = Σ floor2_b, neg_flux_count = Σ(f_b<0), and soft_regime_weight = floor_frac. Define soft-clipped flux f̃_b = sign(f_b)*max(absf_b, b_b). Build safe log flux coordinates v_b = log10(abs(f̃_b) + eps) and adjacent luptitude-flux colors d̂_ug = v_u-v_g, d̂_gr = v_g-v_r, d̂_ri = v_r-v_i, d̂_iz = v_i-v_z. Compute weighted spectral geometry by fitting v_b against log10(λ_b) with X=[1,t,t^2], t_b in Å = [3551,4686,6165,7481,8931], weights w_b = 1 + 0.5*floor1_b: fit v = β0 + β1*t + β2*t^2 to obtain slope=β1 and curvature=β2. Also compute a pure finite-difference curvature proxy κ = d̂_ug - 2*d̂_gr + d̂_ri and second-order finite-difference on adjacent bands analogously. For each adjacent pair (u,g),(g,r),(r,i),(i,z), define mag_color_ab = m_corr,a - m_corr,b and flux_color_ab = 2.5*(log10(abs(f_a)+eps)-log10(abs(f_b)+eps)); mismatch_ab = mag_color_ab - flux_color_ab, and mismatch̃_ab computed using f̃_b similarly. Add shape-summary descriptors from the signed-safe shares p̃_b = abs(f̃_b)/(Σ abs(f̃)+eps): max_share, entropy_sh = -Σ p̃_b log(p̃_b+eps), l2_share = Σ p̃_b^2, and gini-like concentration = 1 - l2_share; include sign entropy from the binary sign histogram across bands with eps smoothing. Create regime-interacted variants slope_hi = slope*(1-floor_frac), slope_lo = slope*floor_frac, curvature_hi, curvature_lo, and mismatch_hi/mismatch_lo with the same gating so models can separate high-SNR and floor-dominated behavior.
- expected_signal: Replacing raw magnitude-space colors with luptitude-flux shape features and explicitly gating them by floor regime should reduce sensitivity to low-SNR nonlinearity and negative-flux noise, giving more stable SED slope, curvature, and mismatch indicators for borderline objects where STAR, QSO, and GALAXY occupy overlapping photometric regions, which is exactly where balanced_accuracy on rare-class boundaries is usually lost.
- risk: The inversion constants assume SDSS-style asinh calibration; if the dataset uses a different photometric zero-point or nonstandard preprocessing, flux reconstruction can be biased and degrade all dependent features, and many engineered descriptors overlap existing color-based variables so overfitting or collinearity is possible without strong regularization and validation; ratio and log operations also need strict epsilon/clipping to avoid instability when flux values are extremely close to zero.

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