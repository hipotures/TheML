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
- title: Relative-flux likelihood margins for class discrimination
- group_name: xdqso_inspired_flux_density_scores
- family: probabilistic_flux_density_scoring
- summary: Create a compact probabilistic representation of each object by modeling its i-band-normalized flux shape in redshift-aware, magnitude-binned flux space and encoding class membership strength through class-marginal likelihood margins.
- strategy: Convert magnitudes to linear fluxes with f_b = 10^{-0.4 m_b}. Build the 4D relative-flux vector r = (f_u/f_i, f_g/f_i, f_r/f_i, f_z/f_i) (use f_i as reference; this avoids ratio blowups from u-band nonlinearity). Bin objects into overlapping i-band bins of width 0.2 mag with 0.1 mag step across [17.7, 22.5], and coarse redshift slabs e.g. z<0.8, 0.8<=z<2.5, 2.5<=z<4.0, z>=4.0. For each class c in {GALAXY, QSO, STAR} and each (i-bin, z-slab), fit a 4D Gaussian mixture on r from train only (for instance 20 components, matching the classical XDQSO setup) using class-labeled points. During scoring, evaluate class-conditional densities L_c(r | i,z) = prior_c(i,z)*sum_k pi_ck N(r; mu_ck, Sigma_ck + Sigma_proxy), where Sigma_proxy is a small data-driven positive-definite proxy error covariance (for example, robust diagonal variance of each ratio within the same i-bin; add a floor to avoid singularity). Smoothly handle empty/low-count bins by backing off to adjacent i-bins and then to global class models. Convert to pseudo-posteriors p_c = L_c / (L_GALAXY + L_QSO + L_STAR + eps) and emit: p_GALAXY, p_STAR, p_QSO, pairwise log-odds features (log p_QSO - log p_STAR, log p_QSO - log p_GALAXY, log p_STAR - log p_GALAXY), top-class confidence, and posterior entropy.
- expected_signal: This directly captures class overlap structure in a noise-aware generative manifold rather than relying on handcrafted color cuts, which is especially valuable where quasar and stellar loci intersect at medium redshift, so it should improve recall balance for the minority and ambiguous classes under balanced accuracy.
- risk: Density fitting is compute-heavy and sensitive to sparse bins (especially for rare high-redshift objects), and synthetic uncertainty proxies may bias likelihood calibration if noise is misestimated, requiring aggressive regularization and binning fallback to control instability.

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