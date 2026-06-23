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

# External Data Description for /home/xai/DEV/aideml/workspaces/2-liberal-celadon-panther/input/star_classification.csv

Original SDSS17 Stellar Classification Dataset.

This is the original real-world dataset that inspired the synthetic Playground
Series S6E6 competition data. It can be used as raw auxiliary data, but it is
not automatically merged with train.csv or test.csv.

Common columns with the competition data:
alpha, delta, u, g, r, i, z, redshift, class.

Columns present in this original dataset but not in the competition files:
obj_ID, run_ID, rerun_ID, cam_col, field_ID, spec_obj_ID, plate, MJD, fiber_ID.

Competition columns not present in this original dataset:
id, spectral_type, galaxy_population.

Generated code should decide whether and how to use this file. Any merge,
filtering, cleaning of sentinel magnitudes, or column mapping must be done
explicitly by the generated solution code.

# Hypothesis
- title: Asinh-band censoring and dropout regime geometry
- group_name: asinh_censoring_regime_geometry
- family: detection_regime_geometry
- summary: Represent each object by whether each ugriz measurement is in a reliable detection regime or in the SDSS low-signal asinh regime, and encode the resulting band-visibility and dropout geometry so class-separating patterns tied to faint-band suppression are made explicit.
- strategy: For each band b in {u,g,r,i,z}, invert the SDSS asinh magnitude m_b using the published relation m = -2.5/ln(10) * [asinh((f/f0)/(2 b_b)) + ln b_b], where b_u=1.4e-10, b_g=0.9e-10, b_r=1.2e-10, b_i=1.8e-10, b_z=7.4e-10, then compute f_b/f0 = 2 b_b sinh(-(m_b * ln(10)/2.5) - ln b_b) and q_b = (f_b/f0)/(2 b_b). Create per-band regime features: detect_10b_b = 1[f_b/f0 > 10 b_b] (equiv m_b < m10b_b), neg_b = 1[f_b/f0 < 0], and delta_b = clip((m_b - m10b_b)/(m0_b - m10b_b), 0, 1), with m10b_b={22.12,22.60,22.29,21.85,20.32} and m0_b={24.63,25.11,24.80,24.36,22.83} for u,g,r,i,z respectively. Add joint descriptors: count of detected bands, count of non-detected redder-than-blue bands, longest contiguous missing-band run in wavelength order, blue-dropout pattern flags (e.g., u-drop, g-drop, u+g-drop while r/i/z detected), detected-band-only color slopes and curvatures from f_b/f0, and mass-center entropy of abs(f_b/f0) after zeroing not-detected bands; clip f_b/f0 and delta_b to training quantile tails to avoid single-object blow-up and keep edge handling stable when very bright or extreme values appear.
- expected_signal: Because asinh magnitudes compress faint flux, raw magnitude differences can hide true non-detections; regime-aware features expose low-SNR structure and dropout signatures (especially blue-band suppression at specific redshifts and low-confidence detections) that are known to be relevant for distinguishing stars from galaxies and quasars, which should improve class separation in the ambiguous boundary regions measured by balanced accuracy.
- risk: Reliance on published softening constants and threshold magnitudes creates calibration sensitivity if the competition data were reduced with different photometric conventions, and regime flags are intrinsically noisy near boundaries, so they can inject instability or duplicate existing flux-asinh or depth-based features unless regularized.

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