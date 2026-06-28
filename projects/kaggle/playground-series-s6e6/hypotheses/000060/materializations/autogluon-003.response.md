import numpy as np
import pandas as pd


REDSHIFT_BIN_EDGES = (-np.inf, 0.0, 0.1, 0.4, 1.0, 1.4, 2.0, 2.8, 3.5, 4.5, np.inf)
I_BIN_EDGES = (-np.inf, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, np.inf)
L4_BETA = 150.0
L3_BETA = 250.0
L2_BETA = 400.0
EPSILON = 1e-6


def _safe_series(raw, name, default):
    if name in raw.columns:
        return raw[name]
    return pd.Series(default, index=raw.index)


def _frequency_for_key(key_frame):
    counts = key_frame.value_counts(dropna=False)
    total = float(len(key_frame))
    if total <= 0.0:
        return pd.Series(0.0, index=key_frame.index)
    freq = key_frame.merge(
        counts.rename("__count__").reset_index(),
        how="left",
        on=list(key_frame.columns),
        sort=False,
    )["__count__"].astype(float)
    return pd.Series(freq.to_numpy() / total, index=key_frame.index)


def _count_for_key(key_frame):
    counts = key_frame.value_counts(dropna=False)
    count_values = key_frame.merge(
        counts.rename("__count__").reset_index(),
        how="left",
        on=list(key_frame.columns),
        sort=False,
    )["__count__"].astype(float)
    return pd.Series(count_values.to_numpy(), index=key_frame.index)


def add_metadata_redshift_magnitude_class_priors(raw, deps, aux):
    spectral = _safe_series(raw, "spectral_type", "__missing__").astype("string").fillna("__missing__")
    population = _safe_series(raw, "galaxy_population", "__missing__").astype("string").fillna("__missing__")
    redshift = pd.to_numeric(_safe_series(raw, "redshift", np.nan), errors="coerce")
    i_mag = pd.to_numeric(_safe_series(raw, "i", np.nan), errors="coerce")

    redshift_bin = pd.cut(
        redshift,
        bins=list(REDSHIFT_BIN_EDGES),
        labels=False,
        include_lowest=True,
    ).astype("float64").fillna(-1.0)

    i_bin = pd.cut(
        i_mag,
        bins=list(I_BIN_EDGES),
        labels=False,
        include_lowest=True,
    ).astype("float64").fillna(-1.0)

    l2 = pd.DataFrame(
        {
            "spectral_type": spectral,
            "galaxy_population": population,
        },
        index=raw.index,
    )
    l3 = pd.DataFrame(
        {
            "spectral_type": spectral,
            "galaxy_population": population,
            "redshift_bin": redshift_bin,
        },
        index=raw.index,
    )
    l4 = pd.DataFrame(
        {
            "spectral_type": spectral,
            "galaxy_population": population,
            "redshift_bin": redshift_bin,
            "i_bin": i_bin,
        },
        index=raw.index,
    )

    l2_count = _count_for_key(l2)
    l3_count = _count_for_key(l3)
    l4_count = _count_for_key(l4)

    l2_freq = _frequency_for_key(l2).clip(EPSILON, 1.0)
    l3_freq = _frequency_for_key(l3).clip(EPSILON, 1.0)
    l4_freq = _frequency_for_key(l4).clip(EPSILON, 1.0)

    l2_reliability = l2_count / (l2_count + L2_BETA)
    l3_reliability = l3_count / (l3_count + L3_BETA)
    l4_reliability = l4_count / (l4_count + L4_BETA)

    blended_context_freq = (
        l4_reliability * l4_freq
        + (1.0 - l4_reliability)
        * (
            l3_reliability * l3_freq
            + (1.0 - l3_reliability) * (l2_reliability * l2_freq + (1.0 - l2_reliability))
        )
    ).clip(EPSILON, 1.0)

    inverse_context_freq = (1.0 - blended_context_freq).clip(EPSILON, 1.0)
    context_log_lift = np.log(blended_context_freq / l2_freq.clip(EPSILON, 1.0))
    context_surprisal = -np.log(blended_context_freq)

    return pd.DataFrame(
        {
            "redshift_bin": redshift_bin.astype("int16"),
            "i_bin": i_bin.astype("int16"),
            "l2_count": l2_count.astype("float32"),
            "l3_count": l3_count.astype("float32"),
            "l4_count": l4_count.astype("float32"),
            "l2_frequency": l2_freq.astype("float32"),
            "l3_frequency": l3_freq.astype("float32"),
            "l4_frequency": l4_freq.astype("float32"),
            "l2_reliability": l2_reliability.astype("float32"),
            "l3_reliability": l3_reliability.astype("float32"),
            "l4_reliability": l4_reliability.astype("float32"),
            "blended_context_frequency": blended_context_freq.astype("float32"),
            "inverse_context_frequency": inverse_context_freq.astype("float32"),
            "context_log_lift_vs_l2": context_log_lift.astype("float32"),
            "context_surprisal": context_surprisal.astype("float32"),
        },
        index=raw.index,
    )


FEATURE_GROUPS = [
    {
        "name": "metadata_redshift_magnitude_class_priors",
        "fn": add_metadata_redshift_magnitude_class_priors,
        "depends_on": [],
        "description": "Covariate-only hierarchical metadata, redshift, and magnitude context prevalence features with smoothed reliability backoff.",
    }
]