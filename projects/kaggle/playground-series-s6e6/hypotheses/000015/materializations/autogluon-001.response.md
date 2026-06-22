import numpy as np
import pandas as pd


def add_redshift_template_domain_margins(raw, deps, aux):
    redshift = pd.to_numeric(raw["redshift"], errors="coerce")
    index = redshift.index
    z = redshift.to_numpy(dtype=float)
    finite = np.isfinite(z)

    velocity_width = 1200.0 / 299792.458
    galaxy_low, galaxy_high = -0.01, 1.0
    qso_low, qso_high = 0.0333, 7.0
    regime_boundaries = (0.0, 0.0333, 1.0, 2.2, 3.0, 3.5, 4.5, 5.0)
    regime_names = ("0_0", "0_0333", "1_0", "2_2", "3_0", "3_5", "4_5", "5_0")

    features = pd.DataFrame(index=index)

    abs_z = np.abs(z)
    features["redshift_abs"] = abs_z
    features["redshift_sign"] = np.sign(z)

    stellar_margin = np.where(
        finite,
        np.where(abs_z > velocity_width, abs_z - velocity_width, 0.0),
        np.nan,
    )
    features["redshift_stellar_velocity_margin"] = stellar_margin

    features["redshift_stellar_velocity_signed_margin"] = np.where(
        finite,
        np.where(
            abs_z > velocity_width,
            (abs_z - velocity_width) * np.sign(z),
            0.0,
        ),
        np.nan,
    )
    features["redshift_in_stellar_velocity_domain"] = pd.Series(
        np.where(finite, abs_z <= velocity_width, pd.NA),
        index=index,
        dtype="boolean",
    )

    galaxy_lower_margin = np.where(finite, z - galaxy_low, np.nan)
    galaxy_upper_margin = np.where(finite, z - galaxy_high, np.nan)
    galaxy_in_domain = finite & (z >= galaxy_low) & (z <= galaxy_high)
    galaxy_distance = np.where(
        finite,
        np.where(
            galaxy_in_domain,
            0.0,
            np.where(
                z < galaxy_low,
                galaxy_low - z,
                z - galaxy_high,
            ),
        ),
        np.nan,
    )
    features["redshift_galaxy_lower_margin"] = galaxy_lower_margin
    features["redshift_galaxy_upper_margin"] = galaxy_upper_margin
    features["redshift_in_galaxy_domain"] = pd.Series(
        np.where(galaxy_in_domain, True, pd.NA),
        index=index,
        dtype="boolean",
    )
    features["redshift_galaxy_distance_to_domain"] = galaxy_distance

    qso_lower_margin = np.where(finite, z - qso_low, np.nan)
    qso_upper_margin = np.where(finite, z - qso_high, np.nan)
    qso_in_domain = finite & (z >= qso_low) & (z <= qso_high)
    qso_distance = np.where(
        finite,
        np.where(
            qso_in_domain,
            0.0,
            np.where(
                z < qso_low,
                qso_low - z,
                z - qso_high,
            ),
        ),
        np.nan,
    )
    features["redshift_qso_lower_margin"] = qso_lower_margin
    features["redshift_qso_upper_margin"] = qso_upper_margin
    features["redshift_in_qso_domain"] = pd.Series(
        np.where(qso_in_domain, True, pd.NA),
        index=index,
        dtype="boolean",
    )
    features["redshift_qso_distance_to_domain"] = qso_distance

    regime_bin = np.full(z.shape[0], -1, dtype=np.int16)
    finite_vals = z[finite]
    if finite_vals.size > 0:
        regime_bin[finite] = np.searchsorted(
            np.array(regime_boundaries, dtype=float),
            finite_vals,
            side="right",
        )
    features["redshift_regime_bin"] = regime_bin

    for boundary, name in zip(regime_boundaries, regime_names):
        safe_label = name.replace(".", "p")
        features[f"redshift_minus_bnd_{safe_label}"] = np.where(finite, z - boundary, np.nan)
        features[f"redshift_ge_bnd_{safe_label}"] = pd.Series(
            np.where(finite, z >= boundary, pd.NA),
            index=index,
            dtype="boolean",
        )

    return features


FEATURE_GROUPS = [
    {
        "name": "redshift_template_domain_margins",
        "fn": add_redshift_template_domain_margins,
        "depends_on": [],
        "description": "Builds redshift-domain geometry features for stellar, galaxy, and quasar template feasibility with signed margins, in-domain flags, and ordered boundary regime indicators.",
    },
]