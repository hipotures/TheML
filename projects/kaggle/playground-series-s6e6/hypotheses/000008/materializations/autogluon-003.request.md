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
- title: Smoothed Survey-Manifold Rarity with Hierarchical Surprise
- group_name: survey_manifold_rarity
- family: density_rarity
- summary: Represent each object on a coarse astrophysical manifold defined by redshift regime, catalog tags, and broad-band flux pattern, then quantify how uncommon that manifold location is and how much its tag and photometric components disagree from independence.
- strategy: Fit all transforms/statistics using train.csv only. First compute pseudo-fluxes for every row: f_b = 10^(-0.4 * m_b) for b in {u,g,r,i,z}, total_flux = f_u+f_g+f_r+f_i+f_z, and normalized shares s_b = f_b / total_flux. From these derive three continuous manifold coordinates: blue_balance = log((s_u+s_g+eps)/(s_i+s_z+eps)), concentration = max(s_u,s_g,s_r,s_i,s_z), brightness = -2.5*log10(total_flux+eps), where eps = 1e-12. Bin redshift with fixed edges R = [-inf, 0.002, 0.01, 0.05, 0.2, 0.6, 1.2, 2.5, inf], assigning below/above limits to boundary bins. For each of blue_balance, concentration, and brightness, compute train quantile edges at 0.1,0.2,...,0.9, drop duplicate edges; if any dimension collapses below 3 bins, fall back to tertiles at 0.33 and 0.67 for that feature to avoid degenerate binning, then apply train-derived edges to test by clipping to train min/max before binning. Encode categorical fields with an explicit UNK level for any unseen category at transform time: spectral_type_bin in {M,O/B,G/K,A/F,UNK}, galaxy_population_bin in {Red_Sequence,Blue_Cloud,UNK}. Build indices: A = (redshift_bin, spectral_type_bin, galaxy_population_bin), B = (blue_balance_bin, concentration_bin, brightness_bin), and joint J = (A, B). From train counts, compute N_A(a), N_B(b), N_J(j), total N=|train|, and support sizes K_A, K_B, K_J over all reachable bins including UNK and end-clipped numeric bins. Use α=1 Laplace smoothing: p_A(a)=(N_A(a)+α)/(N+α*K_A), p_B(b)=(N_B(b)+α)/(N+α*K_B), p_J(j)=(N_J(j)+α)/(N+α*K_J). Emit three numeric features per row: rarity_score = -log(p_J(j)); interaction_surprise = log((p_A(a)*p_B(b)+eps)/(p_J(j)+eps)); and local_dominance = log(p_A(a)*p_B(b)) (or equivalently logit-like baseline strength). All lookups use train-fitted bin edges and count tables, and unseen/overflow numeric positions use clipped bins so no feature becomes undefined.
- expected_signal: Rare manifold cells are often populated by class-atypical physics regimes (for example compact high-redshift or spectrally inconsistent objects), so explicit rarity and interaction-surprise signals should improve minority-class separation and raise balanced accuracy beyond relying on marginal colors and redshift alone.
- risk: Even with smoothing, the joint grid can remain sparse and noisy in long-tail regions, making rarity estimates unstable and potentially overfitting idiosyncratic training structure; this can be aggravated by a train-test domain shift in the manifold occupancy, and the features may partly duplicate information already available from redshift/cutoff color patterns while adding some preprocessing overhead.

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