import numpy as np
import pandas as pd

NUMERIC_BANDS = ("u", "g", "r", "i", "z")
QUANTILE_BINS = 12
CLIP_QUANTILE = 0.999
RANK_METHOD = "average"


def add_aide_catalog_rank_frequency_context(raw, deps, aux):
    spectrum = raw["spectral_type"].astype("string")
    population = raw["galaxy_population"].astype("string")
    spectrum_population = spectrum.str.cat(population, sep="|")
    row_count = float(len(raw))

    features = pd.DataFrame(index=raw.index)

    spectrum_freq = spectrum.value_counts(dropna=False) / row_count
    population_freq = population.value_counts(dropna=False) / row_count
    pair_freq = spectrum_population.value_counts(dropna=False) / row_count

    features["spectral_type_freq"] = spectrum.map(spectrum_freq).astype(np.float64)
    features["galaxy_population_freq"] = population.map(population_freq).astype(np.float64)
    features["spectral_population_pair_freq"] = spectrum_population.map(pair_freq).astype(np.float64)

    redshift_nonneg = raw["redshift"].clip(lower=0.0)
    redshift_clipped = redshift_nonneg.clip(upper=float(redshift_nonneg.quantile(CLIP_QUANTILE))
    )
    redshift_log1p = np.log1p(redshift_clipped)
    redshift_log1p_clipped = redshift_log1p.clip(
        upper=float(redshift_log1p.quantile(CLIP_QUANTILE))
    )

    numeric_series = {
        "u": raw["u"],
        "g": raw["g"],
        "r": raw["r"],
        "i": raw["i"],
        "z": raw["z"],
        "redshift_clipped": redshift_clipped,
        "redshift_log1p_clipped": redshift_log1p_clipped,
    }

    global_rank_cols = []
    within_spectrum_rank_cols = []
    within_population_rank_cols = []

    for name, values in numeric_series.items():
        global_rank = values.rank(method=RANK_METHOD, pct=True)
        qbin = pd.qcut(
            values,
            q=QUANTILE_BINS,
            labels=False,
            duplicates="drop",
        )
        qbin_freq = qbin.value_counts(normalize=True).sort_index()

        features[f"{name}_global_percentile"] = global_rank.astype(np.float64)
        features[f"{name}_quantile_bin12"] = qbin.astype("float64")
        features[f"{name}_quantile_bin_freq12"] = qbin.map(qbin_freq).astype(np.float64)

        within_spectrum_rank = values.groupby(spectrum, sort=False).rank(method=RANK_METHOD, pct=True)
        within_population_rank = values.groupby(population, sort=False).rank(method=RANK_METHOD, pct=True)

        features[f"{name}_spectrum_within_percentile"] = within_spectrum_rank.astype(np.float64)
        features[f"{name}_population_within_percentile"] = within_population_rank.astype(np.float64)

        global_rank_cols.append(f"{name}_global_percentile")
        within_spectrum_rank_cols.append(f"{name}_spectrum_within_percentile")
        within_population_rank_cols.append(f"{name}_population_within_percentile")

    features["global_rank_mean"] = features[global_rank_cols].mean(axis=1)
    features["global_rank_std"] = features[global_rank_cols].std(axis=1, ddof=0)
    features["spectrum_within_rank_mean"] = features[within_spectrum_rank_cols].mean(axis=1)
    features["spectrum_within_rank_std"] = features[within_spectrum_rank_cols].std(axis=1, ddof=0)
    features["population_within_rank_mean"] = features[within_population_rank_cols].mean(axis=1)
    features["population_within_rank_std"] = features[within_population_rank_cols].std(axis=1, ddof=0)

    return features


FEATURE_GROUPS = [
    {
        "name": "aide_catalog_rank_frequency_context",
        "fn": add_aide_catalog_rank_frequency_context,
        "depends_on": [],
        "description": "Adds AIDE-style catalog-frequency and percentile/rank context using spectral/generic categories and photometric/redshift quantiles.",
    }
]