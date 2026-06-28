import numpy as np
import pandas as pd


REDSHIFT_REGIME_EDGES = (0.0, 0.003, 0.0333, 0.25, 1.0, 3.0, np.inf)
REDSHIFT_REGIME_LABELS = (
    "R0_zero_like",
    "R1_very_low",
    "R2_low",
    "R3_moderate",
    "R4_high",
    "R5_extreme",
)
SPECTRAL_TYPE_LEVELS = ("G/K", "M", "O/B", "A/F")
GALAXY_POPULATION_LEVELS = ("Red_Sequence", "Blue_Cloud")


def add_redshift_regime_catalog_consistency(raw, deps, aux):
    redshift = raw["redshift"].astype("float64")
    abs_redshift = redshift.abs()

    regime = pd.cut(
        abs_redshift,
        bins=list(REDSHIFT_REGIME_EDGES),
        labels=list(REDSHIFT_REGIME_LABELS),
        right=False,
        include_lowest=True,
        ordered=True,
    )

    spectral = pd.Series(
        pd.Categorical(
            raw["spectral_type"].astype("string"),
            categories=list(SPECTRAL_TYPE_LEVELS),
        ),
        index=raw.index,
    ).astype("string")

    population = pd.Series(
        pd.Categorical(
            raw["galaxy_population"].astype("string"),
            categories=list(GALAXY_POPULATION_LEVELS),
        ),
        index=raw.index,
    ).astype("string")

    regime_str = regime.astype("string")

    new_features = pd.DataFrame(index=raw.index)
    new_features["abs_redshift"] = abs_redshift
    new_features["redshift_negative_flag"] = (redshift < 0).astype("int8")
    new_features["redshift_zero_like_flag"] = (abs_redshift < 0.003).astype("int8")
    new_features["signed_log_redshift"] = np.sign(redshift) * np.log1p(abs_redshift)
    new_features["redshift_regime"] = regime
    new_features["redshift_regime_code"] = regime.cat.codes.astype("int8")
    new_features["redshift_regime_x_spectral_type"] = (
        regime_str + "__" + spectral
    ).astype("category")
    new_features["redshift_regime_x_galaxy_population"] = (
        regime_str + "__" + population
    ).astype("category")
    new_features["redshift_regime_x_spectral_type_x_galaxy_population"] = (
        regime_str + "__" + spectral + "__" + population
    ).astype("category")

    return new_features


FEATURE_GROUPS = [
    {
        "name": "redshift_regime_catalog_consistency",
        "fn": add_redshift_regime_catalog_consistency,
        "depends_on": [],
        "description": "Encodes redshift regimes and their consistency with spectral type and galaxy population tags.",
    }
]