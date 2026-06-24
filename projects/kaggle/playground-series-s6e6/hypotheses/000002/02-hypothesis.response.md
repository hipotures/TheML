{
  "title": "Refined redshift-regime catalog consistency stratification",
  "group_name": "redshift_regime_catalog_consistency",
  "family": "astrophysical_consistency",
  "summary": "Capture how well each object’s distance regime inferred from redshift aligns with the available spectral and population metadata so the model can distinguish physically plausible class combinations from inconsistent edge cases.",
  "depends_on": [],
  "strategy": "Use only redshift, spectral_type, and galaxy_population. Compute abs_redshift = abs(redshift), redshift_sign_flag = 1[redshift < 0], and signed_log_redshift = sign(redshift) * log1p(abs_redshift). Discretize abs_redshift into deterministic bins: R0 if abs_redshift < 0.003, R1 if 0.003 ≤ abs_redshift < 0.0333, R2 if 0.0333 ≤ abs_redshift < 0.25, R3 if 0.25 ≤ abs_redshift < 1.0, R4 if 1.0 ≤ abs_redshift < 3.0, and R5 if abs_redshift ≥ 3.0. Boundary rule: bins are [lower, upper) with each exact-boundary value assigned to the next higher bin (except abs_redshift = 0 remains R0). Output redshift_regime as a categorical feature and add two crossed features: redshift_regime × spectral_type and redshift_regime × galaxy_population. Preserve all rows and apply no clipping before log1p except the absolute-value transform.",
  "expected_signal": "This adds a physics-aware partition where very low redshift values can isolate nearby stellar-like behavior, mid bins represent extragalactic galaxies, and high bins emphasize quasar-like objects, while interactions with the catalog tags help recover distinctions in overlapping regimes and reduce confusion around ambiguous blue/low-redshift examples.",
  "risk": "Hard-edged bins can be brittle if the redshift scale or preprocessing shifts between datasets, rare tag-regime combinations may be noisy, and regime×tag crosses can overfit when combined with strong nonlinear learners unless regularized."
}
