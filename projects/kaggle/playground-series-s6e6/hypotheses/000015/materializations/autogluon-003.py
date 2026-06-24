import numpy as np
import pandas as pd


def _format_breakpoint(bp):
    token = f"{bp}"
    return token.replace("-", "m").replace(".", "p")


def _compute_domain_margin(z_values, low, high):
    margin = np.where(
        z_values < low,
        z_values - low,
        np.where(z_values > high, z_values - high, 0.0),
    )
    gap = np.abs(margin)
    inside = np.where((z_values >= low) & (z_values <= high), 1, 0).astype(np.uint8)
    return margin, gap, inside


def add_redshift_template_domain_margins(raw, deps, aux):
    z_values = raw["redshift"].to_numpy(dtype=np.float64)
    index = raw.index

    v_max = 1200.0
    c_kms = 299792.458

    star_low = -v_max / c_kms
    star_high = v_max / c_kms

    galaxy_low = -0.01
    galaxy_high = 1.0

    qso_low = 0.0333
    qso_high = 7.0

    distance_cap = 10.0
    log1p_cap = 10.0

    star_velocity_scale = c_kms / v_max
    star_penalty_cut = 0.00401
    sentinel_next = 1000.0

    m_star, g_star, f_star = _compute_domain_margin(z_values, star_low, star_high)
    m_galaxy, g_galaxy, f_galaxy = _compute_domain_margin(z_values, galaxy_low, galaxy_high)
    m_qso, g_qso, f_qso = _compute_domain_margin(z_values, qso_low, qso_high)

    features = {
        "redshift_star_margin": np.clip(m_star, -distance_cap, distance_cap),
        "redshift_star_gap": np.clip(g_star, 0.0, distance_cap),
        "redshift_star_in_domain": f_star,
        "redshift_galaxy_margin": np.clip(m_galaxy, -distance_cap, distance_cap),
        "redshift_galaxy_gap": np.clip(g_galaxy, 0.0, distance_cap),
        "redshift_galaxy_in_domain": f_galaxy,
        "redshift_qso_margin": np.clip(m_qso, -distance_cap, distance_cap),
        "redshift_qso_gap": np.clip(g_qso, 0.0, distance_cap),
        "redshift_qso_in_domain": f_qso,
        "redshift_sign": np.sign(z_values),
        "redshift_star_velocity_proxy": z_values * star_velocity_scale,
        "redshift_penalty_star_domain": np.clip(np.abs(m_star) - star_penalty_cut, 0.0, distance_cap),
        "redshift_penalty_galaxy_low": np.clip(-m_galaxy, 0.0, distance_cap),
        "redshift_penalty_qso_low": np.clip(-(z_values - qso_low), 0.0, distance_cap),
    }

    breakpoints = (-0.01, 0.0, 0.0333, 1.0, 2.2, 3.0, 3.5, 4.5, 5.0, 7.0)
    n_breakpoints = len(breakpoints)

    for i, t in enumerate(breakpoints):
        token = _format_breakpoint(t)

        d_low = np.clip(z_values - t, -distance_cap, distance_cap)
        features[f"redshift_regime_dlow_{token}"] = d_low
        features[f"redshift_regime_dlow_abslog1p_{token}"] = np.clip(
            np.log1p(np.abs(d_low)), 0.0, log1p_cap
        )

        if i < n_breakpoints - 1:
            raw_d_high = breakpoints[i + 1] - z_values
        else:
            raw_d_high = sentinel_next - z_values

        d_high = np.clip(raw_d_high, -distance_cap, distance_cap)
        features[f"redshift_regime_dhigh_{token}"] = d_high
        features[f"redshift_regime_dhigh_abslog1p_{token}"] = np.clip(
            np.log1p(np.abs(d_high)), 0.0, log1p_cap
        )

    return pd.DataFrame(features, index=index)


FEATURE_GROUPS = [
    {
        "name": "redshift_template_domain_margins",
        "fn": add_redshift_template_domain_margins,
        "depends_on": [],
        "description": "Build redshift geometry and domain-feasibility features from STAR/GALAXY/QSO templates with boundary gaps, signed distances, penalties, and ordered breakpoint regime distances.",
    }
]