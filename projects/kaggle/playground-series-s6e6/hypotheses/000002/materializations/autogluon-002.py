import numpy as np
import pandas as pd

REDSHIFT_REGIME_BOUNDS = (0.0, 0.003, 0.0333, 0.25, 1.0, 3.0)
REDSHIFT_REGIME_LABELS = ("R0", "R1", "R2", "R3", "R4", "R5")


def add_redshift_regime_catalog_consistency(raw, deps, aux):
    redshift = raw["redshift"]
    abs_redshift = redshift.abs()
    signed_log_redshift = np.sign(redshift) * np.log1p(abs_redshift)
    redshift_sign_flag = (redshift < 0).astype("uint8")

    regime_values = np.select(
        [
            abs_redshift.eq(0),
            abs_redshift.lt(REDSHIFT_REGIME_BOUNDS[1]),
            abs_redshift.lt(REDSHIFT_REGIME_BOUNDS[2]),
            abs_redshift.lt(REDSHIFT_REGIME_BOUNDS[3]),
            abs_redshift.lt(REDSHIFT_REGIME_BOUNDS[4]),
            abs_redshift.lt(REDSHIFT_REGIME_BOUNDS[5]),
        ],
        ["R0", "R0", "R1", "R2", "R3", "R4"],
        default="R5",
    )
    redshift_regime = pd.Series(
        pd.Categorical(regime_values, categories=REDSHIFT_REGIME_LABELS, ordered=True),
        index=raw.index,
    )
    redshift_regime_str = redshift_regime.astype("string")

    return pd.DataFrame(
        {
            "abs_redshift": abs_redshift,
            "redshift_sign_flag": redshift_sign_flag,
            "signed_log_redshift": signed_log_redshift,
            "redshift_regime": redshift_regime,
            "redshift_regime_x_spectral_type": redshift_regime_str + "_" + raw["spectral_type"].astype("string"),
            "redshift_regime_x_galaxy_population": redshift_regime_str + "_" + raw["galaxy_population"].astype("string"),
        },
        index=raw.index,
    )


FEATURE_GROUPS = [
    {
        "name": "redshift_regime_catalog_consistency",
        "fn": add_redshift_regime_catalog_consistency,
        "depends_on": [],
        "description": "Build redshift regime, signed/log-transformed redshift attributes, and regime interactions with spectral type and galaxy population.",
    },
]