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
- title: Night-sky line redshift interference
- group_name: spectral_skyline_interference
- family: spectroscopic_quality_geometry
- summary: Encode whether class-diagnostic rest-frame spectral features are shifted onto strong observed-frame night-sky residual regions or out of the spectrograph window, exposing redshift regimes where galaxy, quasar, and stellar template evidence may be selectively degraded.
- strategy: Use fixed rest-frame line families for galaxy emission lines ([O II] 3727, Hbeta 4861, [O III] 4959/5007, Halpha 6563), quasar lines (Lyalpha 1216, C IV 1549, C III] 1909, Mg II 2798, Hbeta 4861), and stellar absorption anchors (Ca K/H 3934/3969 plus Balmer 4102/4341/4861/6563). For each object compute observed wavelength as rest_wavelength * (1 + redshift), using max(1 + redshift, 1e-6) for safety. Against the SDSS spectral window 3800-9200 Angstrom and sky-residual anchors 5577, 6300, and 6363 Angstrom plus a broad OH-forest zone above 7000 Angstrom, derive per-family fractions inside the window, fractions near the blue/red window edge within 50 Angstrom, minimum clipped distance to a sky anchor, soft contamination scores exp(-(distance/8)^2) and exp(-(distance/25)^2), counts within 10 and 25 Angstrom of sky anchors, OH-zone exposure, and pairwise galaxy-minus-quasar and stellar-minus-extragalactic contamination contrasts. Cap wavelength distances at 500 Angstrom, set missing or invalid calculations to neutral zero-valued scores, and use only deterministic constants without target statistics. Source grounding: SDSS documents the 3800-9200 Angstrom spectral coverage and strong residuals at 5577, 6300, 6363 and red OH sky lines (https://classic.sdss.org/dr5/products/spectra/index.php), while SDSS redshift classification is based on shifted galaxy, quasar, and stellar templates (https://www.sdss4.org/dr17/algorithms/redshifts/).
- expected_signal: Raw redshift is smooth, but spectral classification can change sharply when decisive lines land on sky residuals or leave the usable wavelength range; these localized redshift-interference descriptors may clarify QSO versus GALAXY confusion and identify STAR-like cases where extragalactic line evidence is unavailable or unreliable.
- risk: The labels may already come from high-confidence spectra where sky-line artifacts were resolved, so the features can be redundant with redshift; hardcoded line lists and narrow wavelength margins may add sparse, brittle thresholds or overfit survey-specific spectral-processing quirks.

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