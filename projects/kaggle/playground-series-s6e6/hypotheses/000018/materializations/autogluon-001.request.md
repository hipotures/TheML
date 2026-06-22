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
- title: Catalog Tag Color Concordance
- group_name: catalog_tag_color_concordance
- family: tag_photometry_consistency
- summary: Measure whether each object's broadband color temperature agrees with the semantic ordering implied by its provided spectral-type and galaxy-population catalog tags, treating contradictions as informative astrophysical ambiguity.
- strategy: Compute adjacent SDSS colors u-g, g-r, r-i, and i-z, then form a deterministic observed redness score as the mean of clipped color components: clip((u-g+0.5)/3.5,0,1), clip((g-r+0.5)/2.0,0,1), clip((r-i+0.5)/1.5,0,1), and clip((i-z+0.5)/1.5,0,1). Map spectral_type along the hot-to-cool stellar sequence O/B=0.00, A/F=0.33, G/K=0.67, M=1.00, and map galaxy_population as Blue_Cloud=0.00 and Red_Sequence=1.00. Derive signed and absolute differences between observed redness and each mapped expectation, the difference between the two tag expectations, coarse redness bins [0,0.33), [0.33,0.67), [0.67,1], and binary contradiction indicators for hot-tag-red-color, cool-tag-blue-color, red-sequence-blue-color, and blue-cloud-red-color. Clip all color-derived values before scoring; if an unexpected category appears, use neutral 0.5 for its expectation and leave contradiction indicators off. Background used: SDSS documents that ugriz colors are used for stellar and quasar photometric transformations and classification context (https://classic.sdss.org/dr5/algorithms/sdssUBVRITransform.php, https://www.sdss3.org/dr10/algorithms/redshifts.php), while astronomy references describe O/B/A/F/G/K/M as a temperature-color sequence and galaxies as red-sequence versus blue-cloud populations (https://lweb.cfa.harvard.edu/~pberlind/atlas/htmls/note.html, https://academic.oup.com/mnras/article/481/1/1183/5078380).
- expected_signal: The categorical tags are likely strong but noisy summaries; converting them into agreement and contradiction geometry can help separate ordinary stars and galaxies from QSOs or mislabeled-borderline objects whose colors conflict with the simple stellar-temperature or galaxy-population expectation, improving balanced accuracy on minority and ambiguous classes.
- risk: This may be partly redundant with raw categorical features and existing color-derived groups, and fixed color cutoffs could be brittle for unusual redshifted objects whose observed colors violate local stellar or galaxy expectations for legitimate physical reasons.

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