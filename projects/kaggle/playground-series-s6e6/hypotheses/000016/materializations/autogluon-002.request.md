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
- title: Galactic- and SDSS-frame sky geometry
- group_name: sky_frame_position_geometry
- family: spatial_geometry
- summary: Represent celestial position in continuous Galactic and SDSS survey-aligned angular manifolds that remove RA/Dec discontinuities and expose real astrophysical and observational-footprint priors linked to object class imbalance.
- strategy: Standardize coordinates first: wrap alpha into [0,360) with a = ((alpha % 360) + 360) % 360, convert to radians ar = a*pi/180, and clip delta to [-90,90] before delta_rad = delta*pi/180; build base Cartesian direction p = (cos(delta_rad)*cos(ar), cos(delta_rad)*sin(ar), sin(delta_rad)) and unit-length guard by renormalizing with max(||p||, eps). Create wrap-safe continuity features sin(alpha), cos(alpha), sin(delta), cos(delta), and the raw unit vector components. Convert to Galactic frame using fixed IAU 1958 J2000 matrix M_eq_to_gal = [[-0.0548755604, 0.4941094279, -0.8676661490], [-0.8734370902, -0.4448296300, -0.1980763734], [-0.4838350155, 0.7469822445, 0.4559837762]], then g = M_eq_to_gal p, clamp each component to [-1,1], and compute l = atan2(g_y, g_x) mapped to [0, 2pi), b = atan2(g_z, sqrt(g_x^2+g_y^2)); add sin/cos l, sin/cos b, |b|, distance-to-Galactic-plane features abs(b) and pi/2-abs(b), and great-circle distances to Galactic center/anti-center: d_gc = acos(cos(b)*cos(l)) and d_anticenter = acos(cos(b)*cos(l-pi)). For SDSS-style survey geometry, define x_hat = unit(r_2018(185,32.5)), v = unit(r_2018(275,0)), z_hat = normalize(cross(x_hat, v)), y_hat = cross(z_hat, x_hat); then eta = asin(dot(p, z_hat)), lambda = atan2(dot(p, y_hat), dot(p, x_hat)), with lambda wrapped to [-pi,pi]. Add sin/cos eta and lambda plus Stripe-aware periodic terms in eta using stripe width 2.5 degrees: eta_deg = eta*180/pi, eta_phase = (eta_deg % 2.5), eta_phase_centered = eta_phase - 1.25, and sin(2*pi*eta_phase_centered/2.5), cos(2*pi*eta_phase_centered/2.5). Optionally include integer stripe-band bins floor((eta_deg+90)/2.5). All trig inputs should be numerically clipped and any NaN/Inf mapped deterministically to finite defaults.
- expected_signal: The revised geometry stack gives the model an orthogonal positional prior: stars should remain concentrated near low Galactic latitude while redshifted and extragalactic classes follow SDSS footprint and stripe-like anisotropy, so balanced accuracy can improve by reducing confusions where photometric features overlap and minority classes are spatially structured.
- risk: If train and test survey footprints or SDSS stripe conventions differ, these features can transfer poorly and encode footprint artifacts as spurious signal; eta periodic terms can overfit high-frequency scan pattern noise, and angular wrapping/deg-to-rad edge handling must be strict to avoid discontinuity-induced instability at poles or boundaries.

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