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

Hypothesis ID: 000027

Previous versions:

- rev 1: AIDE id sequence scan context
  group_name=aide_id_sequence_scan_context
  family=aide_sequence_context
  summary=Evaluate the AIDE id-sequence feature path directly in TheML by encoding row/order structure through id rank, normalized rank, coarse blocks, modulo cycles, parity, log id, and neighboring id gaps.

  strategy=If an id column is present, coerce it to numeric and derive deterministic sequence features: missing-id indicator, ordinal rank, rank normalized to [0,1], coarse blocks at 1k and 100k scales, modulo residues for 2, 11, and 97, even-id indicator, log1p clipped id, and previous/next id gaps after sorting by id. If id is absent, use the row order as a neutral fallback only so the materialization remains valid. Do not use labels or any model outputs.
  expected_signal=Survey or synthetic generation order can leak weak scan, batching, or sampling structure. The AIDE code contained this feature path, and evaluating it as a separate hypothesis makes the potential id/order signal explicit rather than mixing it into other photometric groups.
  risk=Id-derived features can be dataset-specific and may not generalize; if ids are arbitrary or shuffled, the group may add pure noise or encourage leaderboard-only artifacts.

- rev 2: Robust id-order context encoding
  group_name=aide_id_sequence_scan_context
  family=aide_sequence_context
  summary=This feature group models the row identifier as a proxy for acquisition order and batching structure by encoding each object’s global and local position in identifier space to capture systematic non-astronomical effects that may align with class priors.
  strategy=Parse `id` to integer deterministically; set `id_valid` and `id_missing` flags, and when invalid assign neutral numeric values derived from 0 while preserving the missing indicator. Sort rows by numeric id and define zero-based `id_rank` in that ordered index space, then compute `id_rank_norm = id_rank / (N-1)` clipped to [0,1]. Compute global span features from the split-min/max IDs: `id_min`, `id_max`, `id_span = max(1, id_max-id_min)`, `id_rel = (id - id_min) / id_span`, and `log_id = log1p(max(0,id))`; z-score `log_id` using training mean/std. Add block context: `block_1k = floor((id-id_min)/1000)`, `block_100k = floor((id-id_min)/100000)`, and `within_1k = ((id-id_min) mod 1000) / 1000`. Add periodic signatures with bounded trig features for residues: for m in {2,11,97}, `sin(2π*(id mod m)/m)` and `cos(2π*(id mod m)/m)`. Add neighborhood features from id-order using lag/lead: `prev_gap = id - lag(id)` and `next_gap = lead(id) - id`, with boundary rows filled with 0 and boolean `is_first`, `is_last`; clip gaps at training 99th percentile to cap outliers. Add continuity indicators: `id_even = id mod 2`, `noncontiguous = 1 if prev_gap>1 or next_gap>1 else 0` (or next only at edges), and `adjacent_step = 1 if prev_gap==1 and next_gap==1 else 0`. Keep all transformations strictly within identifiers and ordering, with no target or model-output dependence.
  expected_signal=Because train/test IDs are contiguous across split boundaries, id-based sequence context can expose latent survey or ingestion structure (scan strips, batches, chunk boundaries, periodic routing artefacts) that may carry weak but consistent class-dependent shifts not represented by photometric features, improving multiclass balanced accuracy.
  risk=If the ID assignment is random, remapped, or changes across future deployments, these features can become pure noise or encode dataset-specific artifacts; clipped gaps and explicit boundary handling reduce instability, but do not eliminate overfitting risk.

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