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
- title: Rest-frame passband landmark alignment
- group_name: rest_frame_filter_landmarks
- family: spectral_redshift_geometry
- summary: Represent each object's ugriz photometry by the physical rest-frame wavelength regions sampled at its redshift and by how key spectral landmarks fall within, between, or outside those shifted passbands.
- strategy: Use fixed SDSS effective wavelengths for bands u,g,r,i,z as lambda_obs = [3543, 4770, 6231, 7625, 9134] Angstrom. For each row set z_safe = max(redshift, 0), add neg_redshift_flag = I(redshift < 0), and compute ordered rest wavelengths lambda_rest_j = lambda_obs_j / (1 + z_safe). Derive span_min, span_max, span_width, log_span_width = log1p(span_width), and the rest-frame center of each adjacent color interval. Add occupancy counts for non-overlapping rest-frame bins: below_912, 912_1216, 1216_2500, 2500_3646, 3646_4000, 4000_7000, and above_7000, plus normalized count fractions and count entropy over the seven bins. For each landmark L in {912, 1216, 2500, 3646, 4000, 7000}, compute in_span = I(span_min <= L <= span_max), side = -1 if L < span_min, 0 if inside, and 1 if L > span_max, outside_log_distance = 0 when inside else log1p(distance from L to the nearest span edge), nearest_band_idx = argmin_j abs(log(lambda_rest_j / L)) with ties resolved to the lower-index band, nearest_band_mag, nearest_band_rest_lambda, signed_log_ratio = log(lambda_rest_nearest / L), and abs_log_ratio. When L is inside the passband span, find the unique bracketing adjacent pair b,b+1 such that lambda_rest_b <= L <= lambda_rest_{b+1}; compute bracket_left_mag, bracket_right_mag, bracket_color = mag_b - mag_{b+1}, fractional_position = (log(L) - log(lambda_rest_b)) / (log(lambda_rest_{b+1}) - log(lambda_rest_b)), interpolated_mag_at_L by linear interpolation in log wavelength, and landmark_curvature_proxy as interpolated_mag_at_L minus the mean of the two endpoint magnitudes. When L is outside the span, set bracketing index features to the nearest endpoint pair, set fractional_position and interpolation-derived residual features to 0, and rely on side and outside_log_distance to identify extrapolation. Also add, for each adjacent observed color u-g, g-r, r-i, and i-z, the signed log distance from that color interval center to each landmark and a flag for whether the landmark lies inside that color interval, so the model can connect color breaks to their physical rest-frame location.
- expected_signal: Balanced accuracy may improve because redshift changes which physical continuum regions the fixed observed bands sample; explicit landmark-position and bracketing features can separate stars with near-zero-redshift smooth continua from galaxies with 4000 Angstrom or Balmer-region structure and QSOs whose ultraviolet and Lyman-region behavior moves through the optical filters.
- risk: The features are strongly coupled to redshift quality and hardcoded broad spectral landmarks, so noisy redshifts, broad filters, and only five photometric points can make boundary flags brittle; many derived distances may also be redundant with raw redshift, magnitudes, and colors, creating overfitting risk if validation is not stratified by class and redshift regime.

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