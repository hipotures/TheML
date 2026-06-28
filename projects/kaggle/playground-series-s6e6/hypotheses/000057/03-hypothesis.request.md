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

Hypothesis ID: 000057

Previous versions:

- rev 1: Quasar-Star Locus Outlier and Degeneracy Geometry
  group_name=quasar_stellar_locus_conflict
  family=locus_conflict_geometry
  summary=Model each object’s color-space geometry against the SDSS stellar manifold using two three-band color cubes while adding redshift-linked quasar-inclusion corridors and ambiguity-buffer penalties, so the classifier sees both strong quasar outlier evidence and the known redshift regions where quasar and stellar populations become confounded.
  strategy=Compute c_ug=u-g, c_gr=g-r, c_ri=r-i, c_iz=i-z for every row. Form two 3D color vectors v1=(c_ug,c_gr,c_ri) and v2=(c_gr,c_ri,c_iz). Fit a robust mean and covariance for each vector set on the full available set (train+test), then extract the first principal axis for each cube; define orthogonal residual vectors by subtracting the axis component for each row and compute Mahalanobis orthogonal distances d1,d2 using residual covariance with a color floor added per component (equivalent to adding 0.03^2 in each color term). Create soft outlier scores s1=max(d1-4,0) and s2=max(d2-4,0), and also keep standardized versions d1/z_scale and d2/z_scale. Add three regime features: (a) UV-like regime score favoring very blue c_ug in a bright point-source band, (b) a mid-z bridge score that peaks inside 0.65<c_ug<1.5 and 0<c_gr<0.2 and decays smoothly outside, and (c) a high-z regime score keyed by strong c_ug redness, outlierness in v2, and i-magnitude gating near i<20.2. Add suppressor penalties for known quasar-mimicking contaminant strips defined by white-dwarf/A-star-like color windows from the SDSS target-selection tables, then combine by subtraction so these regions reduce effective quasar support. All scores are clipped to [0,1], all denominators are clamped to avoid divide-by-zero, and invalid rows are deterministically filled with 0.0.
  expected_signal=Separates many quasars that are true color outliers from the thin stellar manifold while explicitly preserving and down-weighting overlap zones where quasars near z≈2.7–2.8 and known contamination strips resemble stars, which should improve minority-class recall without collapsing balanced accuracy into the dominant class boundary.
  risk=Relies on SDSS-era, color-threshold logic that may be mismatched to this dataset’s calibration and label definitions; fixed inclusion/exclusion heuristics can over-penalize legitimate rare quasars near boundaries and may be redundant with existing manifold/locus groups, so incremental gain can be modest if those groups already capture the same geometry.

- rev 2: Redshift-aware quasar outlier geometry with calibrated contamination suppression
  group_name=quasar_stellar_locus_conflict
  family=locus_conflict_geometry
  summary=Create a compact quasar-support signal from orthogonal color-manifold outlier distance and redshift-conditioned color corridors, then explicitly dampen that signal in narrow stellar-mimic strips to improve separation in known confusion regions.
  strategy=Fit all geometry statistics on train only. Compute colors c_ug=u-g, c_gr=g-r, c_ri=r-i, c_iz=i-z. Form two vectors v1=(c_ug,c_gr,c_ri) and v2=(c_gr,c_ri,c_iz). For each vector set, estimate mu as median per component and covariance S from 5% winsorized train values; extract first principal axis p1,p2. For each row, residuals are r1=v1-mu1 - p1*(p1^T(v1-mu1)) and r2=v2-mu2 - p2*(p2^T(v2-mu2)). Build orthogonal covariance Co1,Co2 from train residuals with ridge and shrinkage: C* = (1-0.1)Cov(r)+0.1*diag(diag(Cov(r))) + 0.03^2*I, then add eps=1e-6*I before inversion. Compute Mahalanobis orthogonal distances d1=sqrt(r1^T C*1^{-1} r1), d2 likewise, then clip to [0,30]. Normalize with train medians and MAD: z1=(d1-p50_1)/(1.4826*MAD_1+1e-6), z2=(d2-p50_2)/(1.4826*MAD_2+1e-6). Define outlier soft features o1=clip((d1-t1)/4,0,1), o2=clip((d2-t2)/4,0,1), with t1,t2=p95 of d1,d2 from train. Define corridor gates with explicit bins: blue low-z gate b = sigmoid((−c_ug−0.05)/0.20)*sigmoid((0.45-c_gr)/0.18)*sigmoid((20.5-i)/0.9); mid-z bridge m = exp(−0.5*((c_ug-0.85)^2/0.18^2 + (c_gr-0.20)^2/0.12^2))*sigmoid((1.8 - redshift)/0.8)*sigmoid((redshift-0.4)/0.35); high-z gate h = exp(−0.5*((c_ug-1.55)^2/0.30^2 + (c_ri-0.25)^2/0.22^2))*sigmoid((redshift-2.2)/0.4)*sigmoid((3.9-redshift)/1.0)*sigmoid((21.0-i)/1.0). Build white-dwarf/A-star suppressors: q1=exp(−0.5*((c_ug-0.55)^2/0.12^2 + (c_gr-0.10)^2/0.08^2 + (c_ri-0.05)^2/0.08^2)), q2=exp(−0.5*((c_ug-1.05)^2/0.16^2 + (c_gr-0.30)^2/0.10^2 + (c_ri-0.12)^2/0.10^2)); penalty q = (q1+q2)*sigmoid((0.2-abs(redshift-2.7))/0.4)*sigmoid((20.0-i)/0.6). Keep generated features bounded: geom_signal=clip(0.45*o1+0.35*o2+0.1*b+0.1*h,0,1), mid_bridge=clip(m,0,1), conflict=clip(geom_signal-0.6*q+0.2*m,-1,1). Replace any NaN/inf with 0 and keep all intermediate calculations deterministic.
  expected_signal=This revision preserves the locus-conflict core idea while adding train-safe fitting, stable regularization, and explicit redshift-conditioned quasar corridors, which should better separate true quasar outliers from the stellar manifold and reduce false positives from narrow contaminant strips, improving balanced accuracy through stronger minority-class discrimination.
  risk=Thresholds and strip centers are still heuristic and can be miscalibrated if this dataset’s photometric zero-points differ from SDSS-style behavior, and the added geometry + corridor signals can be redundant with other locus-based groups, so gains may be modest without careful orthogonal integration.

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