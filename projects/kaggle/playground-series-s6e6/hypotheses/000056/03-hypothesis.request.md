# Project Context

## Goal
Predict the stellar class for each test-set object.

## Evaluation
Submissions are evaluated using balanced accuracy between predicted class labels and the true class. Submission file must contain `id,class` with one label per test row, where `class` is one of GALAXY, STAR, or QSO.

## Data description
`train.csv` contains 577347 rows with 10 feature columns plus `id`, `galaxy_population`, `spectral_type`, and the target `class`. `test.csv` contains the same predictors without the target across 247435 rows. `sample_submission.csv` shows the required submission columns `id` and `class`. The target is a 3-class stellar classification problem with labels GALAXY, QSO, and STAR.

# Data Overview

-> sample_submission.csv has 247435 rows and 2 columns.
Here is some information about the columns:
id (int64) has range: 577347.00 - 824781.00, 0 nan values
class (object) has 1 unique values: ['GALAXY'], 0 nan values

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
class (object) has 3 unique values: ['GALAXY', 'QSO', 'STAR'], 0 nan values

# Project Target
- ID column: id
- Target column: class
- Problem type: multiclass
- Evaluation metric: balanced_accuracy
- Submission kind: labels

# Instruction

Analyze the selected ROOT hypothesis, the project context, and the previous
revisions.

Create the next revision by expanding or improving the hypothesis while
preserving its substantive feature-group idea. You are not writing code. You are
not creating a new feature group.

The revision should make the hypothesis more precise, internally consistent,
implementable, or better justified. Improve the hypothesis text, especially
`strategy`, when there is a concrete useful improvement. Do not force arbitrary
changes.

Use previous revisions only as context. If they contain useful details, use that
context, but do not mechanically merge them. The output must be one coherent
revised hypothesis using exactly the same JSON schema as a newly generated
hypothesis.

# Web Search

Live web search is enabled. If internet verification is useful, use the
`web_search` tool before writing the revised hypothesis. Check at most 3 sources.
Use web search only to verify domain facts, feature-engineering background,
validation risks, or implementation constraints relevant to the selected
hypothesis. Do not use web search to invent a different feature-group idea, copy
a public solution, or broaden the scope. If web search is not useful, continue
without inventing sources.

# Selected Hypothesis

Hypothesis ID: 000056

Previous versions:

- rev 1: Field-calibrated reddening-free locus offsets
  group_name=local_reddening_free_locus_offsets
  family=sky_photometry_normalization
  summary=Estimate local sky-cell photometric zero-point/locus drift in reddening-free SDSS color space and represent each object by its field-normalized deviation from nearby-field stellar color behavior so class boundaries are set on corrected shapes rather than raw calibrated magnitudes.
  strategy=For every object compute reddening-free coordinates Qgri=(g-r)−1.582(r−i) and Qriz=(r−i)−0.987(i−z). Tessellate sky into HEALPix cells at nside=32; for robustness, define a context as the cell plus its 8 adjacent cells. Within each context, and within coarse redshift strata [<0.01, 0.01–0.1, 0.1–0.5, 0.5–1, 1–2, >2], compute robust medians and MADs for Qgri, Qriz, and supporting colors (u−g, g−r, r−i, i−z). Emit standardized offsets: (Qgri−median)/MAD and (Qriz−median)/MAD, the paired whitened locus distance sqrt(z1^2+z2^2), and color residuals of the object versus local median in u−r and g−i similarly standardized; also emit local context density log1p(n/(area of context)) and its z-score against global context density. If context count in a redshift stratum is below threshold (e.g., <40), shrink to context without stratification, then to global medians; clip all standardized values to a safe range such as [−8,8].
  expected_signal=This group removes field-level photometric shifts caused by calibration, crowding, and local extinction-systematics before class separation, so stars remain aligned to a stable local locus while galaxies and quasars stay as outliers in a normalized basis, improving class separability under balanced accuracy.
  risk=Sparse high-redshift or very small sky contexts can make local medians noisy and inject unstable features, the same correction can attenuate real astrophysical sky gradients if the context is too aggressive, and the approach adds implementation complexity because robust map-based statistics must be fit once on the available rows and then applied consistently.

- rev 2: Sky-context standardized reddening-free locus residuals
  group_name=local_reddening_free_locus_offsets
  family=sky_photometry_normalization
  summary=Create per-object descriptors that compare each source to the local reddening-insensitive color locus at its sky and redshift context, converting raw colors into field-calibrated residuals and density-normalized offsets that are more stable for class separation.
  strategy=Compute base colors c_ug=u-g, c_gr=g-r, c_ri=r-i, and c_iz=i-z, then derive reddening-free axes q1=c_gr-1.582*c_ri and q2=c_ri-0.987*c_iz. Assign each object to a HEALPix context using nside=32 and expand the context to its 8 neighbors (total of up to 9 cells). Assign redshift strata with fixed edges [-inf,0.01,0.1,0.3,0.5,1.0,2.0,inf] and build robust statistics on the training rows only for every context+stratum: median and MAD of q1,q2,c_ug,c_gr,c_ri,c_iz and robust counts n_ctx. If n_ctx<60 at context+stratum, fallback to the same context aggregated across all redshift strata; if still <60, use global statistics for that feature set; if still <60, use full-train global statistics. Use a MAD floor of 1e-6 and compute z-values z_f=(f-median_f)/MAD_f then clip each to [-8,8]. Emit z-scores for q1, q2, c_ug, c_gr, c_ri, and c_iz, emit locus deviation d=sqrt(z_q1^2+z_q2^2), and emit log-density features z_logdens=(log1p(n_ctx)-global_median_logdens_stratum)/MAD_logdens_stratum using the same chosen fallback layer as the medians; additionally emit binary indicators whether the feature layer came from stratified context, unstratified context, or global fallback.
  expected_signal=By removing local field-dependent photometric and extinction drift before classification, the same object classes are expressed in a common relative color space across sky regions, which should reduce intra-class scatter of GALAXY/STAR/QSO signatures while preserving class-outlier structure in a way that is more aligned with balanced accuracy.
  risk=Sparse high-redshift or masked regions can force global fallbacks that weaken the correction, the context smoothing can remove real astrophysical sky variation tied to class prevalence, and several emitted residuals may be highly correlated with raw colors, increasing redundancy and potential overfitting if the downstream model is insufficiently regularized.

Return valid JSON only. Do not use markdown fences.

Return exactly this JSON object:

{
  "title": "short descriptive title",
  "group_name": "same concise_snake_case_group_name as the previous version",
  "family": "compact feature-family label",
  "summary": "one comprehensive sentence describing the semantic feature-set idea; do not list formulas, exact feature names, or implementation details here",
  "depends_on": [],
  "strategy": "concrete deterministic feature logic, formulas, bins, statistics, and edge handling",
  "expected_signal": "why this group may improve or clarify the metric",
  "risk": "specific overfitting, leakage, cost, redundancy, or instability risk"
}