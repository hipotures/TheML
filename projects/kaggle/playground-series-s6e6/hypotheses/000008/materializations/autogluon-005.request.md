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
- title: Hierarchical Survey-Manifold Rarity
- group_name: survey_manifold_rarity
- family: density_rarity
- summary: Quantify how common each object is within a coarse survey manifold that combines redshift regime, catalog tags, and broad photometric energy allocation, so rarity and tag-photometry mismatch become explicit class cues.
- strategy: Fit all bin edges and count tables using train.csv predictors only, with no target labels. Convert ugriz magnitudes to pseudo-fluxes f_b = 10^(-0.4*m_b), total_flux = sum_b f_b, and normalized shares s_b = f_b/(total_flux+eps), eps = 1e-12. Derive blue_balance = log((s_u+s_g+eps)/(s_i+s_z+eps)), concentration = max(s_u,s_g,s_r,s_i,s_z), and brightness = -2.5*log10(total_flux+eps). Bin redshift with fixed edges [-inf,0.002,0.01,0.05,0.2,0.6,1.2,2.5,inf]. For blue_balance, concentration, and brightness, compute train quantile edges at deciles 0.1 through 0.9, drop duplicate edges, and if fewer than 3 bins remain for a coordinate, recompute with tertile edges 0.33 and 0.67; if ties still collapse, use the remaining valid ordered edges. Apply the train-derived numeric edges to both train and test, clipping transform-time values to the train min/max for the coordinate before bin assignment. Encode spectral_type and galaxy_population with train-known levels plus an explicit UNK bucket for unseen levels. Define A = (redshift_bin, spectral_type_level, galaxy_population_level), B = (blue_balance_bin, concentration_bin, brightness_bin), and J = (A,B). Count N_A, N_B, and N_J on train rows, and use cartesian support sizes K_A, K_B, and K_J implied by the fitted bins and category levels including UNK. With alpha = 1 smoothing, compute p_A=(N_A+alpha)/(N+alpha*K_A), p_B=(N_B+alpha)/(N+alpha*K_B), and p_J=(N_J+alpha)/(N+alpha*K_J). Emit rarity_score = -log(p_J), marginal_rarity = -0.5*(log(p_A)+log(p_B)), and interaction_surprise = log((p_A*p_B+eps)/(p_J+eps)); positive interaction_surprise means the combined cell is rarer than expected from its tag-redshift and photometric marginals. In cross-validation, refit all bins and count tables inside each training fold before transforming the held-out fold; for final submission, refit once on the full training set and transform test.csv.
- expected_signal: QSO and other minority or boundary cases should more often appear in sparse or cross-inconsistent manifold cells, while common STAR and GALAXY populations occupy dense redshift-tag-photometry regions, so these features can improve balanced accuracy by exposing class-atypical density structure rather than only raw color and redshift values.
- risk: The high-dimensional joint grid may still be sparse, making smoothed densities noisy in tail cells and somewhat sensitive to bin choices; if fitted outside the validation fold it can also leak validation distribution information into model selection, and the resulting rarity signals may be redundant with strong color-redshift features or unstable under train-test density shift.

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