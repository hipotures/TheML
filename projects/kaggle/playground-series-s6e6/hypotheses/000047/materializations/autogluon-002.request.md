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
- title: Rest-frame-aware Flux Allocation and Concentration
- group_name: flux_allocation_entropy
- family: spectral_energy_allocation
- summary: Represent each object as a normalized five-band energy-share profile and summarize how compact, uneven, and wavelength-biased that profile is, with and without approximate rest-frame correction, to capture class-specific SED shape signatures that are insensitive to total brightness scale.
- strategy: For each row, start from magnitudes m_u,m_g,m_r,m_i,m_z and build a stable share vector p by log-domain softmax: l_k=-0.4*m_k, l_k:=l_k-max(l), q_k=exp(ln(10)*l_k), and p_k=(q_k+α)/(Σ q_j+5α) with α=1e-12. If any row contains non-finite values or if Σ q_k<=0, force p=[0.2,0.2,0.2,0.2,0.2] and set an invalid-share flag for downstream audit. Sortability, concentration, and contrast features are then computed from p only: entropy H=-Σ p_k ln p_k, normalized entropy Hn=H/ln(5), Simpson concentration Sc=Σ p_k^2, Gini-style concentration G=1-Sc, effective band count Neff=1/(Sc+1e-12), largest-share ratio R12=p_(1)/ (p_(2)+1e-12), top-share gap p_(1)-p_(2), cumulative buckets B=p_u+p_g, M=p_g+p_r, R=p_i+p_z, and differences ΔBR=B-R, ΔUI=p_u-p_i, ΔGR=p_g-p_r, ΔZI=p_z-p_g. Add wavelength-distribution shape statistics using canonical SDSS centers λ=[355.1,468.6,616.5,748.1,893.1] nm and x=ln λ: μ=Σ p_k x_k, v=Σ p_k (x_k-μ)^2, σ=max(√v,1e-12), skew=Σ p_k (x_k-μ)^3/σ^3, kurtosis=kurt=Σ p_k (x_k-μ)^4/σ^4-3, plus centroid ratio μ/μ_max_band and scale ratio v/( (max(x)-min(x))^2 ). For redshift-aware descriptors, compute λ_rf_k=λ_k/(1+z_clamp), x_rf=ln λ_rf_k with z_clamp=clip(z,-0.99,20), then μ_rf,v_rf,skew_rf,kurt_rf analogously, and include Δμ=μ-μ_rf and Δv=v-v_rf. Replace any NaN/inf from arithmetic with 0 and clip skew and kurtosis to [-8,8] before use.
- expected_signal: A flux-share formulation removes overall magnitude scaling while preserving the relative spectral-shape distribution, which can distinguish galaxy/star/QSO populations that overlap in raw magnitudes but differ in how optical energy is concentrated across bands; coupling these concentration/asymmetry descriptors with coarse rest-frame shift-aware moments adds discriminatory structure where redshifted features blur absolute-color rules and therefore should improve balanced accuracy.
- risk: These engineered descriptors are likely correlated with existing color and slope-based groups, can become unstable in extreme low-SNR or near-degenerate-band regimes despite the numeric guards, and the rest-frame moment terms can partially encode redshift-class priors, so overfitting and weak generalization are possible without validation controls.

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