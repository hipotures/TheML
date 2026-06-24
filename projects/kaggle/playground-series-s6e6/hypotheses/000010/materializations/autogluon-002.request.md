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
- title: Rest-frame landmark alignment with boundary-aware passband context
- group_name: rest_frame_filter_landmarks
- family: spectral_redshift_geometry
- summary: Construct a compact representation that expresses how each object’s ugriz measurements shift into rest-frame wavelength space and where key spectral landmarks fall relative to that shifted coverage, turning raw observed colors into physically aligned redshift-aware descriptors.
- strategy: Use fixed SDSS effective wavelengths λobs = [3543, 4770, 6231, 7625, 9134] Å for u,g,r,i,z. For each row define z_safe = max(redshift, 0), z_neg_flag = I(redshift < 0), and compute λrest_j = λobs_j / (1 + z_safe) for j=1..5 (with band order preserved). Compute span_min = min(λrest), span_max = max(λrest), and span_width = span_max - span_min. Build ordered occupancy counts over deterministic rest-frame bins that partition the domain: n_lt_912, n_912_1216, n_1216_2500, n_2500_3646, n_3646_4000, n_4000_7000, n_gt_7000, plus a consistency feature total_bands = 5; add a normalized entropy over these bin counts (only nonzero bins) to capture how broadly a spectrum probes rest-frame regions. For each landmark L in {912, 1216, 2500, 3646, 4000, 7000} compute: in_span = I(span_min <= L <= span_max), below_span = I(L < span_min), above_span = I(L > span_max), and nearest_band_idx k = argmin_j |λrest_j - L| (tie resolved to smaller j). Define signed_log_dist_L = sign(λrest_k - L) * log1p(|λrest_k - L|), and abs_log_dist_L = log1p(|λrest_k - L|). Return nearest_band_mag_L = observed magnitude at k and nearest_band_id_L = k. If k is interior and both adjacent bands straddle L, return left_right_delta_L = (mag_k - mag_{k-1}) and (mag_{k+1} - mag_k); if exactly one neighbor exists inside the sampled set, return only the available one. If L is strictly between λrest_k and λrest_{k+1}, compute linear_predict_at_L from mags k and k+1 and return landmark_residual_L = mag_k - linear_predict_at_L (same formula with k and k-1 when interpolation from below is used), else set landmark_residual_L = 0 when outside span. For outside-span landmarks, clamp endpoint_band = 1 or 5 (nearest endpoint), set outside_distance_L = log1p(|L - clamp(L, span_min, span_max)|), and expose outside_side_L = I(L < span_min) else 1 for above, enabling explicit handling of hard saturation at edges.
- expected_signal: By explicitly encoding whether physically meaningful breaks and UV discontinuities are entering, inside, or missed by the observed passbands, the model can discriminate classes that share coarse colors but differ by redshifted spectral content, especially separating low-redshift stellar continua from galaxies with a 4000 Å/Balmer region signature and quasars with high-redshift Lyman-region behavior.
- risk: This family depends on redshift precision and fixed landmark positions, so noisy or systematically biased redshifts, wide broad-band photometry, and coarse 5-point sampling can produce unstable hard-threshold decisions and duplicate information already present in raw colors and redshift, increasing overfitting risk on minority classes without strong additional signal.

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