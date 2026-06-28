You are designing feature-group hypothesis.

# Task

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

# Prior Root Groups

Use this only to avoid obvious repetition. Do not imitate the best previous group.
- raw_autogluon_baseline (baseline): Train AutoGluon on the raw project columns without generated feature groups.
- score below baseline; broadband_color_shape (photometric_sed): Represent each object’s observed optical spectral-energy-distribution shape through deterministic broadband color, slope, and curvature descriptors that emphasize continuum geometry over absolute brightness.
- score above baseline; redshift_regime_catalog_consistency (astrophysical_consistency): Represent whether an object's redshift-implied distance regime is consistent with its catalog spectral and population tags, emphasizing stellar-like near-zero cases, galaxy-like intermediate cases, and quasar-like high-redshift cases.
- score above baseline; galactic_sightline_context (sky_position): Represent each object's sky position as a physically meaningful Galactic line-of-sight context so the model can learn foreground-star density and extragalactic visibility patterns tied to the Milky Way disk and poles.
- score above baseline; distance_corrected_luminosity_signature (luminosity_distance): Re-express the five observed magnitudes as a redshift-adjusted intrinsic brightness profile and summarize its central level, brightest/faintest inferred bands, and dispersion to expose class-specific luminosity regimes beyond apparent brightness.
- score below baseline; catalog_template_residuals (template_mismatch): Measure how atypical each object's ugriz color pattern is relative to the robust catalog-template locus implied by its spectral and population tags, turning tag-conditioned photometric mismatch into class-separation features.
- score below baseline; bandpass_break_localization (spectral_break): Encode the observed ugriz spectrum by the position, direction, concentration, and local context of its strongest adjacent-band discontinuity, distinguishing decisive spectral breaks from diffuse or noisy broadband curvature.
- score above baseline; band_order_topology (ordinal_sed): Encode the shape of each object's ugriz brightness ordering and slope topology so the model can use relative spectral-continuum structure without depending on absolute apparent brightness.
- score above baseline; survey_manifold_rarity (density_rarity): Quantify how common each object is within a coarse survey manifold that combines redshift regime, catalog tags, and broad photometric energy allocation, so rarity and tag-photometry mismatch become explicit class cues.
- score below baseline; canonical_locus_coordinates (locus_projection): Represent each object by its signed and normalized displacement from canonical SDSS stellar and red-galaxy color manifolds, with validity-aware summaries that distinguish tight stellar-locus membership from galaxy-track behavior and broad photometri…
- score above baseline; rest_frame_filter_landmarks (spectral_redshift_geometry): Represent each object's ugriz photometry by the physical rest-frame wavelength regions sampled at its redshift and by how key spectral landmarks fall within, between, or outside those shifted passbands.
- score below baseline; lrg_target_cut_margins (target_selection_geometry): Convert the documented SDSS luminous-red-galaxy color-magnitude selection geometry into continuous proximity and hard-membership signals that describe how strongly each object resembles or violates the red-galaxy target region.
- score below baseline; faint_blue_galaxy_wedge_margins (blue_galaxy_selection): Encode graded membership in a faint blue galaxy color-magnitude selection envelope by measuring normalized interior depth, boundary proximity, and directional violations against galaxy-targeting cuts.
- score below baseline; emission_line_bandpass_resonance (line_filter_geometry): Encode how physically important rest-frame emission lines for quasars and galaxies redshift into SDSS ugriz passbands and whether the matched bands show local broadband excesses relative to a smooth adjacent continuum.
- score below baseline; photoz_color_consistency (photometric_redshift): Measure how unusual an object's observed redshift is relative to the redshift distribution expected for nearby objects in broadband color-magnitude space, using local disagreement and ambiguity as class-separating evidence.
- score above baseline; redshift_template_domain_margins (redshift_geometry): Encode redshift as class-feasibility geometry by measuring each object against near-zero stellar, low-redshift galaxy, and broad quasar template domains rather than relying on redshift only as an unconstrained scalar.
- score above baseline; sky_frame_position_geometry (spatial_geometry): Represent sky location in physically aligned Galactic and survey-coordinate frames so the classifier can use continuous celestial geometry, Milky Way structure, and SDSS footprint context instead of discontinuous raw equatorial angles.
- score above baseline; redshift_luminosity_plausibility (absolute_photometry): Convert apparent ugriz photometry into redshift-scaled intrinsic-brightness plausibility descriptors that expose whether each object’s observed flux is consistent with a stellar, galactic, or quasar-like luminosity regime.
- score below baseline; catalog_tag_color_concordance (tag_photometry_consistency): Measure how consistently each object's observed ugriz color profile agrees with the semantic temperature and galaxy-population expectations implied by its catalog tags, while treating redshift-related color displacement and tag contradictions as str…
- score below baseline; observed_sed_continuum_moments (spectral_shape): Represent the five observed ugriz magnitudes as a brightness-normalized broadband continuum shape that captures smooth spectral tilt, broad curvature, and localized band departures from a low-order spectral trend.
- score below baseline; survey_depth_limit_margins (selection_depth_geometry): Encode apparent brightness as proximity to canonical SDSS-style galaxy and quasar spectroscopic depth boundaries so the model can learn class changes at bright galaxy limits, quasar targeting depths, and faint survey edges.
- score below baseline; aide_broadband_flux_ratios (aide_photometry): Add deterministic nonlinear photometric transforms that recast ugriz magnitudes and redshift into broadband ratio, gap, flux-proxy, curvature, and compact redshift-interaction views while preserving the original AIDE-style feature idea.
- score above baseline; aide_sky_cell_local_residuals (aide_spatial_context): Encode each object by how dense and photometrically unusual its local sky neighborhood is, using smoothed sky-cell context to capture spatial survey effects without using labels.
- score below baseline; aide_redshift_bin_color_residuals (aide_photometric_redshift): This feature group measures how unusually each object’s color and color-curvature profile behaves relative to the typical profile of training objects at similar redshift, exposing local photometric deviations that raw colors and redshift alone may n…
- score below baseline; aide_catalog_rank_frequency_context (aide_catalog_context): Create unsupervised catalog-context features that describe how common each object's metadata tags are and how unusual its photometric and redshift values are globally and within comparable catalog groups.
- score above baseline; aide_aux_reference_distribution_distance (aide_auxiliary_reference): Compare each object with an external auxiliary reference population to encode how typical or outlying its sky position, photometry, and redshift profile is under a robust survey-like distribution.
- score above baseline; aide_smooth_spline_sed_interactions (aide_smooth_basis): Represent smooth nonlinear structure across sky position, photometric SED shape, redshift, and a small set of color-redshift manifolds using robust spline bases plus compact tensor interactions.
- score below baseline; aide_id_sequence_scan_context (aide_sequence_context): Encode each object's identifier as deterministic position, block, periodic, and local-neighborhood context so the model can test whether row-id order captures acquisition or batching structure related to class prevalence.
- score above baseline; foreground_reddening_geometry (reddening_invariant): This feature group decomposes broadband photometry into foreground-dust amplitude, dereddened color geometry, and sky-extinction context so classification relies more on intrinsic SED differences than on line-of-sight reddening.
- score above baseline; redshifted_lyman_discontinuity (restframe_discontinuity): Create rest-frame broadband features that describe whether the Lyman-limit and Lyman-alpha discontinuities fall inside the observed ugriz window and how strongly the SED changes across the corresponding blue and red passbands.
- score below baseline; sdss_quasar_targeting_surface (astrophysical_selection_geometry): Encode SDSS-inspired quasar target selection as smooth, fold-fitted distances from robust stellar-locus tubes in ugriz color space plus soft inclusion and rejection surfaces for the redshift and magnitude regimes where quasars overlap stellar contam…
- score below baseline; redshift_adaptive_color_tube_residuals (color_manifold_geometry): Represent each object by its along-locus and off-locus position relative to robust redshift-local color manifolds, with stable signed residual coordinates and targeted emphasis near the quasar-star color-overlap regime.
- score above baseline; redshifted_4000a_break_curvature (restframe_break_geometry): Encode the local broadband curvature around the observed-frame location of the rest-frame 4000 Angstrom continuum break, using redshift to make galaxy-like stellar-population breaks separable from smoother stellar continua and quasar color patterns.
- score below baseline; restframe_sed_family_fit (spectral_family): Characterize each object by the relative adequacy of power-law-like and smoothly curved rest-frame ugriz continuum families, with absorption-aware variants that preserve shape information while reducing sensitivity to absolute brightness.
- score above baseline; blackbody_continuum_distance (physically_informed_shape_model): This feature group represents each object's normalized ugriz continuum by its closest match to a redshift-corrected thermal blackbody shape and by the structured residual pattern left after that fit.
- score below baseline; eboss_ptf_c1_c3_geometry (color_plane_selection_geometry): Encode the SDSS-style rotated ugri color-plane geometry used for quasar selection as robust boundary-margin and bright-contaminant-pocket proximity features that summarize how far each object lies from quasar-like versus stellar or low-redshift colo…
- score below baseline; segue_stellar_atmospheric_indices (atmospheric_color_diagnostics): Construct SDSS principal-color residual features that measure how tightly each object's broadband photometry follows a robust stellar-atmosphere color locus after accounting for redshift, spectral type, and population context.
- score below baseline; population_color_manifold_drifts (tag_conditioned_color_geometry): Represent each object by how well its broadband color geometry agrees with the redshift-local color manifold implied by its supplied galaxy-population tag compared with the alternative population manifold.
- score below baseline; photoz_trajectory_geometry (trajectory_consistency): Represent each object by how its SDSS color vector deviates from the smooth empirical color-redshift manifold and from the manifold local direction and curvature, making class-specific departures from physically plausible redshift trajectories expli…
- score below baseline; local_principal_color_residuals (locus_geometry): Represent each object by robust standardized departures from a locally centered SDSS stellar-locus coordinate system, so the model can use class-specific manifold offsets after controlling for redshift, sky position, and catalog tag context.
- score below baseline; class_conditional_color_density_posteriors (bayesian_density_scoring): This feature group converts raw photometric measurements into class evidence by comparing each object’s local color pattern against class-specific empirical color distributions conditioned on redshift and brightness, then progressively smoothing and…
- score below baseline; xdqso_inspired_flux_density_scores (probabilistic_flux_density_scoring): Build leakage-controlled, redshift- and brightness-conditioned generative scores from normalized photometric flux shape so each object receives class-relative likelihood evidence for GALAXY, STAR, and QSO in the local color-flux manifold.
- score below baseline; error_deconvolved_locus_tube_residuals (error_aware_locus_geometry): Fit robust local color-locus tubes conditioned on redshift and apparent brightness, then encode noise-corrected orthogonal offsets and side-of-locus information so objects that genuinely depart from the narrow photometric manifold are separated from…
- score below baseline; redshift_partitioned_sed_pca_residuals (manifold_projection): Represent each object's normalized ugriz spectral-energy shape by how well it lies on a redshift-local low-dimensional photometric manifold, exposing class-specific continuum mismatches and band-profile anomalies.
- score below baseline; redshifted_absorption_trough_residuals (absorption_line_geometry): Measure whether ugriz fluxes show physically aligned, redshift-dependent broadband depressions near major optical absorption complexes, using local continuum residuals to add line-blanketing geometry beyond ordinary colors.
- score below baseline; pairwise_color_lattice_residuals (pairwise_color_geometry): Characterize each object by redshift-local residual geometry of the complete ugriz pairwise color lattice, using robust centering and intrinsic low-rank orthogonal directions to expose non-adjacent color-manifold departures relevant to class separat…
- score below baseline; local_manifold_codimension (manifold_geometry): Characterize each object by whether its photometry and redshift lie on a dense, locally low-dimensional neighborhood structure or in a sparse extrapolative region whose geometry is likely to differ across stellar classes.
- score below baseline; flux_allocation_entropy (spectral_energy_allocation): Represent each object by the normalized distribution of its optical flux across ugriz bands and summarize the profile's concentration, asymmetry, wavelength location, and redshift-shifted placement without depending on total brightness.
- score below baseline; luptitude_regime_flux_features (asinh_flux_domain): Re-express the five SDSS-like asinh magnitudes as calibrated flux-shape descriptors with explicit reliability gates for bands near the luptitude softening floor, so color, slope, curvature, and concentration signals remain usable for faint or noisy…
- score below baseline; asinh_censoring_regime_geometry (detection_regime_geometry): Describe each source by the wavelength-ordered pattern of reliable, low-signal, and effectively censored asinh-photometric bands, then summarize how that censoring pattern changes the observable SED shape.
- score below baseline; main_sequence_parallax_plausibility (photometric_parallax): Represent each object by how physically plausible its ugriz photometry is under a main-sequence stellar distance interpretation, using deviations from a stellar absolute-magnitude manifold as evidence against or for the STAR class.
- score below baseline; k_correction_residual_manifold (k_correction_consistency): Learn a deterministic redshift- and color-conditioned photometric correction manifold from the predictors and encode how each object departs from locally expected pseudo-rest-frame SED geometry.
- score below baseline; segue_stellar_subtype_margins (stellar_color_geometry): Represent each object by robust continuous distances to several canonical SDSS-like stellar subtype envelopes, preserving subtype-specific color geometry as structured evidence for or against stellar classification.
- score above baseline; dual_restframe_break_alignment (break_competition_geometry): Encode whether each object’s redshift places a strong continuum discontinuity at the observed ugriz location expected for either the Lyman 1216 Å break or the 4000 Å break, then compare the two signals to distinguish quasar-like dropout structure, g…
- score below baseline; restframe_anchor_sed_shape (restframe_shape): Create rest-frame descriptors of broadband spectral shape on fixed physical wavelength anchors so continuum slope, curvature, and stellar-break structure are compared consistently across redshift rather than through drifting observed-frame filters.
- score below baseline; asinh_jacobian_weighted_shape (flux_uncertainty_geometry): Transform ugriz luptitudes into signed relative fluxes and encode five-band continuum geometry with weights that down-weight bands whose asinh response indicates low effective signal, preserving reliable SED shape while limiting noisy color excursio…
- score below baseline; local_reddening_free_locus_offsets (sky_photometry_normalization): Compare each object with the robust local color locus expected for nearby sky position and redshift regime, expressing its photometry as field-normalized reddening-resistant residuals rather than absolute calibrated colors.
- score below baseline; quasar_stellar_locus_conflict (locus_conflict_geometry): Measure how strongly each object departs from the stellar color locus while adding redshift-aware quasar corridor support and dampening support in compact stellar-mimic regions where quasar and star colors are intrinsically ambiguous.
- score below baseline; redshift_branch_deconvolved_flux_posteriors (probabilistic_density_geometry): Estimate uncertainty-smoothed class and quasar-redshift-branch posterior features from relative ugriz flux geometry within brightness-local strata, with explicit sparse-bin backoff and balanced-prior calibration for class-overlap regions.
- score below baseline; tag_redshift_compatibility_residuals (tag_manifold_consistency): Quantify whether an object's broadband color pattern is typical for its catalog tag combination at comparable redshift, and whether competing tag regimes provide a more plausible explanation.
- score below baseline; metadata_redshift_magnitude_class_priors (conditional_target_prior): Learn smoothed class-prevalence signals conditioned on spectral metadata, galaxy-population tag, redshift regime, and apparent brightness so ambiguous photometric regions receive empirical context about which stellar class is most plausible.
- redshift_slice_angular_environment (spatial_environment): Represent each object by whether it lies in a sky-local overdensity of objects at comparable redshift and coherent broadband color, exposing galaxy-cluster and large-scale-structure context distinct from foreground stars and isolated quasar candidat…
- score below baseline; noise_scale_color_decisiveness (photometric_uncertainty_geometry): Represent whether each object's ugriz color pattern is physically decisive or noise-ambiguous relative to canonical SDSS photometric precision, making borderline star, galaxy, and quasar color overlaps explicit without using labels.
- score below baseline; ubvri_transform_consistency (cross_system_photometry): Represent each object by how consistently its SDSS ugriz colors can be translated into Johnson-Cousins UBVRI color space under published stellar and quasar transformation assumptions, exposing objects whose broadband SED behaves like a normal stella…
- score below baseline; quasar_small_bump_bandpass_contrast (quasar_continuum_geometry): Represent whether the redshifted SDSS bands sample the broad quasar Fe II and Balmer-continuum small-blue-bump region and whether the observed ugriz flux shape shows the expected localized pseudo-continuum excess.
- fiber_collision_crowding_context (spectroscopic_selection_geometry): Encode whether an object lies in a close angular crowding regime where SDSS fiber-placement and deblending constraints can make galaxies, blended stars, and compact quasars appear with different catalog-neighborhood signatures.
- spectroscopic_redshift_grid_imprint (spectroscopic_pipeline_geometry): Represent each object by how its reported redshift sits within the numerical search windows and velocity-grid structure used by SDSS-style spectral template classification, capturing pipeline-imprinted evidence separate from raw redshift magnitude.
- surrogate_photometry_flag_margins (measurement_quality): Approximate missing SDSS-style photometric quality and support-boundary flags from the available predictors so likely corrupted, saturated, edge-case, or single-band-anomalous measurements become explicit classification context.
- seeing_limited_resolvability_proxy (morphology_proxy): Estimate whether each object's redshift, apparent brightness, and catalog population imply an angularly resolved galaxy-like source or an unresolved point-source-like object under SDSS seeing, adding a surrogate for missing morphology information.
- sdss_scan_order_incoherence (temporal_photometry): Represent whether an object's five-band photometry is smoother in physical wavelength order than in the SDSS camera scan-time order, exposing non-simultaneous, moving-source, variability, or band-quality effects that ordinary color features can miss.
- virtual_infrared_tail_extrapolation (missing_wavelength_surrogate): Extrapolate each object's optical ugriz continuum beyond the z band into near/mid-infrared anchor wavelengths to expose whether its red-side spectral tail behaves more like a stellar photosphere, a galaxy stellar bump, or a quasar hot-dust/power-law…
- spectral_skyline_interference (spectroscopic_quality_geometry): Encode whether class-diagnostic rest-frame spectral features are shifted onto strong observed-frame night-sky residual regions or out of the spectrograph window, exposing redshift regimes where galaxy, quasar, and stellar template evidence may be se…
- wdms_color_bridge_signature (stellar_binary_contaminants): Encode the two-component SDSS optical color signature of white-dwarf/main-sequence and related hot-star contaminants that occupy quasar-like color space but are physically stellar.
- stellar_radial_velocity_frame_margins (kinematic_redshift_geometry): Reinterpret near-zero spectroscopic redshift as line-of-sight stellar radial velocity in sky-dependent Galactic reference frames, separating velocity-plausible Milky Way stars from cosmological galaxy and quasar redshifts.
- u_band_continuum_anomaly (uv_photometry): Compare each object's observed near-ultraviolet brightness with the smooth optical continuum implied by its redder SDSS bands to expose UV excesses and dropouts that separate normal stellar loci, galaxies, and quasars.
- uv_blanketing_metallicity_plausibility (stellar_photometric_parameters): Represent whether an object's ugr colors are physically plausible as an F/G main-sequence star by translating u-band line blanketing at fixed optical color into stellar temperature and metallicity support, so quasar or galaxy impostors with impossib…

# Web search

Live web search is available and should be used before proposing the hypothesis.
Use the `web_search` tool for both stages:
- Search queries: use `web_search` to find feature engineering ideas,
  preprocessing directions, data representations, validation traps, simple
  baseline patterns, public Kaggle discussions/notebooks, package
  documentation, and domain or method background relevant to this task.
- Page retrieval: use `web_search` again to open/fetch the most relevant result
  pages or direct URLs and read their page content.

Analyze the retrieved information before using it. Do not rely only on search
result titles, snippets, or URLs. Convert externally found techniques into one
generalized, leakage-safe, budget-compatible semantic feature-group idea. Do not
copy public solutions directly. If `web_search` is not useful or not available,
continue normally without inventing sources.

# Rules

- Create exactly one new semantic feature group.
- Do not create a full preprocessing pipeline.
- Do not modify existing groups.
- Do not discuss model families, ensembling, validation folds, training time, hyperparameters, or AutoGluon configuration.
- The hypothesis must later generate exactly one feature-group function.
- The group may contain multiple derived columns only when they belong to one coherent semantic idea.
- A hypothesis must represent one homogeneous group-level change. Multiple implementation details are allowed only when they are necessary parts of that same semantic change.
- Use prior root groups only to avoid obvious repetition.
- must use `depends_on: []`.
- Do not output code.

Return valid JSON only. Do not use markdown fences.

Return exactly this JSON object:

{
  "title": "short descriptive title",
  "group_name": "concise_snake_case_group_name",
  "family": "compact feature-family label",
  "summary": "one comprehensive sentence describing the semantic feature-group idea; do not list formulas, exact feature names, or implementation details here",
  "depends_on": [],
  "strategy": "concrete deterministic feature logic, formulas, bins, statistics, and edge handling",
  "expected_signal": "why this group may improve or clarify the metric",
  "risk": "specific overfitting, leakage, cost, redundancy, or instability risk"
}