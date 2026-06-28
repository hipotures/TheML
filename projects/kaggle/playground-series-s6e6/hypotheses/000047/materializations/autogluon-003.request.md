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
- title: Stable Flux-Share Entropy and Wavelength Concentration
- group_name: flux_allocation_entropy
- family: spectral_energy_allocation
- summary: Represent each object by the normalized distribution of its optical flux across ugriz bands and summarize the profile's concentration, asymmetry, wavelength location, and redshift-shifted placement without depending on total brightness.
- strategy: For each row, use magnitudes m_u,m_g,m_r,m_i,m_z to form numerically stable flux shares with a log-softmax transform: a_k=-0.4*ln(10)*m_k, a_k:=a_k-max(a), q_k=exp(a_k), and p_k=(q_k+alpha)/(sum_j q_j+5*alpha) with alpha=1e-12. If any input magnitude is non-finite or the normalized vector is invalid, set p_k=0.2 for all five bands and add an invalid_flux_share flag. From p compute entropy and concentration descriptors: H=-sum p_k*ln(p_k), Hn=H/ln(5), Simpson concentration Sc=sum p_k^2, impurity Gi=1-Sc, effective band count Neff=1/(Sc+1e-12), maximum share p_max, minimum share p_min, max-min spread, top-two ratio p_(1)/(p_(2)+1e-12), and top-two gap p_(1)-p_(2). Add ordered allocation contrasts: blue_share=p_u+p_g, central_share=p_r, red_share=p_i+p_z, blue_minus_red, blue_fraction=blue_share/(blue_share+red_share+1e-12), p_u-p_i, p_g-p_r, p_r-p_z, and p_z-p_g. Use SDSS effective wavelengths lambda=[355.1,468.6,616.5,748.1,893.1] nm with x=ln(lambda) and compute p-weighted wavelength statistics: centroid mu=sum p_k*x_k, variance v=sum p_k*(x_k-mu)^2, sd=sqrt(max(v,0)), standardized skew=sum p_k*(x_k-mu)^3/(sd^3+1e-12), excess kurtosis=sum p_k*(x_k-mu)^4/(sd^4+1e-12)-3, and normalized location/spread mu_norm=(mu-min(x))/(max(x)-min(x)) and v_norm=v/((max(x)-min(x))^2). For redshift-aware placement, clamp z to zc=clip(redshift,-0.95,20) and compute rest-frame band centers x_rf_k=ln(lambda_k/(1+zc)); because central moments are translation-invariant under this shift, include only non-redundant rest-frame location features: mu_rf=sum p_k*x_rf_k, mu_rf_norm relative to the fixed log-wavelength range, delta_mu=mu-mu_rf, and p-weighted distances to broad rest-frame anchors centered at 400, 550, and 700 nm, d_anchor=sum p_k*abs(x_rf_k-ln(anchor)). Replace NaN or inf values with 0, clip ratio features to [0,1e6], and clip skew/kurtosis to [-8,8].
- expected_signal: The normalized flux-share profile removes overall brightness and distance scale while preserving whether energy is flat, sharply concentrated, blue-weighted, red-weighted, or centered in specific optical regions, which can separate GALAXY, STAR, and QSO cases that overlap in raw magnitudes or simple colors and can improve balanced accuracy for minority classes.
- risk: Many descriptors are deterministic transforms of colors and redshift, so they may be redundant with existing color-slope features or encode class priors through redshift; extreme or erroneous photometry can still produce unstable ratios, and anchor-distance terms could overfit validation folds if redshift distributions differ between train and test.

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