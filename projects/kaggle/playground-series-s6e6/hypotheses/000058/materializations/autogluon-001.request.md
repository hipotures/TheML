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
- title: Deconvolved Flux-Branch Posteriors
- group_name: redshift_branch_deconvolved_flux_posteriors
- family: probabilistic_density_geometry
- summary: Model each object’s relative ugriz flux pattern with uncertainty-smoothed, class- and redshift-branch-conditioned generative densities and use the resulting posterior mass, margins, and ambiguity signals as compact features that resolve quasar-star overlap while preserving galaxy cues.
- strategy: Convert ugriz magnitudes to linear fluxes F_b = 10^{-0.4 m_b} and form a 4D relative-flux vector x = [log(F_u/F_i), log(F_g/F_i), log(F_r/F_i), log(F_z/F_i)] plus optional additive color fallback features [u-g, g-r, r-i, i-z] for robustness. Assign each row to overlapping i-band windows (width 0.8 mag, step 0.4); for quasars also split a redshift branch variable r in {z<2.2, 2.2<=z<=3.5, z>3.5}, while galaxies and stars use a shared non-branch mode. In every populated (i-window, class, branch) stratum, fit a fixed-K Gaussian mixture on x using only training rows of that stratum (for example K=10-20, diagonal floor on covariance, equalized component weight initialization). Estimate a per-window empirical noise matrix V from robust MAD of colors in that i-window, and deconvolve-noise likelihoods by evaluating each component with Sigma_tilde = Sigma_k + V. For each query row in stratum s, compute deconvolved class likelihood L_c,s = Σ_k π_ck N(x|μ_ck, Σ_tilde_ck), branch likelihoods for QSO_low/QSO_mid/QSO_high likewise, then normalize to posterior masses P_c and P_qso,* using smoothed class priors from train counts in the same i-window (backoff to nearest populated window if empty). Emit compact features: posterior logits log((P_c+1e-8)/(1-P_c+1e-8)) for STAR, GALAXY, QSO and for each QSO branch, pairwise margins log((P_qso+1e-8)/(P_star+1e-8)) and log((P_qso+1e-8)/(P_galaxy+1e-8)), branch concentration max(P_qso_branch)/sum(P_qso_*), branch entropy over the three QSO branches, and branch-count indicator (number of branches with P_qso_branch > 0.5*max_branch), with all probabilities clipped to [1e-8, 1-1e-8] and all log features truncated to finite range.
- expected_signal: This approach converts the physical separation logic behind SDSS quasar-stellar-galaxy color behavior into smooth likelihoods, which should better capture medium-z quasar ambiguity near the stellar-overlap region, preserve galaxy probability structure, and provide calibrated uncertainty-sensitive signals that are more informative for balanced accuracy than hard color cuts.
- risk: High-dimensional mixture fitting can overfit in sparse bins (especially high-redshift quasars), branch priors may drift if the test distribution differs from training, and using magnitude-derived noise proxies instead of true photometric errors may miscalibrate likelihood sharpness near faint limits.

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