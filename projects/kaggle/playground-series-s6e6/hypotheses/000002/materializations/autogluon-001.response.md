import numpy as np
import pandas as pd


def add_redshift_regime_catalog_consistency(raw, deps, aux):
    redshift = raw["redshift"]

    abs_redshift = redshift.abs()
    signed_log_redshift = np.sign(redshift) * np.log1p(abs_redshift)
    negative_redshift_flag = (redshift < 0).astype("uint8")

    redshift_regime = pd.Series(
        pd.Categorical(
            np.select(
                [
                    abs_redshift < 0.003,      # near_zero
                    abs_redshift < 0.08,       # low (includes 0.003 exactly)
                    abs_redshift < 0.8,        # mid (includes 0.08 exactly)
                    abs_redshift >= 0.8,       # high (includes 0.8 exactly)
                ],
                [
                    "near_zero",
                    "low",
                    "mid",
                    "high",
                ],
                default="high",
            ),
            categories=["near_zero", "low", "mid", "high"],
            ordered=False,
        ),
        index=raw.index,
    )

    spectral_cross = raw["spectral_type"].astype("string") + " x " + redshift_regime.astype("string")
    population_cross = raw["galaxy_population"].astype("string") + " x " + redshift_regime.astype("string")

    new_features = pd.DataFrame(
        {
            "abs_redshift": abs_redshift,
            "signed_log_redshift": signed_log_redshift,
            "negative_redshift_flag": negative_redshift_flag,
            "redshift_regime": redshift_regime,
            "spectral_type_x_redshift_regime": pd.Categorical(spectral_cross),
            "galaxy_population_x_redshift_regime": pd.Categorical(population_cross),
        },
        index=raw.index,
    )
    return new_features


FEATURE_GROUPS = [
    {
        "name": "redshift_regime_catalog_consistency",
        "fn": add_redshift_regime_catalog_consistency,
        "depends_on": [],
        "description": "Build redshift regime bins and cross them with spectral_type and galaxy_population.",
    }
]