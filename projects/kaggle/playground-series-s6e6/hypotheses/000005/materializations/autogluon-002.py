import numpy as np
import pandas as pd


def _mad_series(values: pd.Series) -> float:
    s = values.dropna()
    if s.empty:
        return 0.0
    med = s.median()
    return (s - med).abs().median()


def add_catalog_template_residuals(raw, deps, aux):
    # Colors in adjacent-color space
    colors = pd.DataFrame(
        {
            "c1": raw["u"] - raw["g"],
            "c2": raw["g"] - raw["r"],
            "c3": raw["r"] - raw["i"],
            "c4": raw["i"] - raw["z"],
        },
        index=raw.index,
    )
    color_cols = ["c1", "c2", "c3", "c4"]

    spectral_type = raw["spectral_type"].astype(object)
    galaxy_population = raw["galaxy_population"].astype(object)

    pair_key = pd.DataFrame(
        {
            "spectral_type": spectral_type,
            "galaxy_population": galaxy_population,
        },
        index=raw.index,
    )
    pair_data = pd.concat([pair_key, colors], axis=1)

    pair_counts = pair_data.groupby(["spectral_type", "galaxy_population"]).size()
    pair_medians = pair_data.groupby(["spectral_type", "galaxy_population"])[color_cols].median()
    pair_mads = pair_data.groupby(["spectral_type", "galaxy_population"])[color_cols].agg(_mad_series)

    pair_medians = pair_medians.rename(columns={c: f"{c}_median" for c in color_cols})
    pair_mads = pair_mads.rename(columns={c: f"{c}_mad" for c in color_cols})

    type_key = pd.DataFrame({"spectral_type": spectral_type}, index=raw.index)
    type_data = pd.concat([type_key, colors], axis=1)

    type_counts = type_data.groupby("spectral_type").size()
    type_medians = type_data.groupby("spectral_type")[color_cols].median().rename(
        columns={c: f"{c}_median" for c in color_cols}
    )
    type_mads = type_data.groupby("spectral_type")[color_cols].agg(_mad_series).rename(
        columns={c: f"{c}_mad" for c in color_cols}
    )

    global_medians = colors.median()
    global_mads = colors.apply(_mad_series)

    idx_pairs = pd.MultiIndex.from_arrays([spectral_type, galaxy_population])

    pair_count = pair_counts.reindex(idx_pairs).fillna(0).astype(int)
    type_count = type_counts.reindex(spectral_type).fillna(0).astype(int)

    pair_med_lookup = pair_medians.reindex(idx_pairs)
    pair_mad_lookup = pair_mads.reindex(idx_pairs)
    type_med_lookup = type_medians.reindex(spectral_type)
    type_mad_lookup = type_mads.reindex(spectral_type)

    pair_ok = pair_count.ge(1000)
    type_ok = spectral_type.map(type_counts).fillna(0).astype(int).ge(1000)

    chosen_median = pd.DataFrame(
        {c: pd.Series(float(global_medians[c]), index=raw.index) for c in color_cols},
        index=raw.index,
    )
    chosen_mad = pd.DataFrame(
        {c: pd.Series(float(global_mads[c]), index=raw.index) for c in color_cols},
        index=raw.index,
    )

    for c in color_cols:
        chosen_median[c] = pair_med_lookup[f"{c}_median"].where(pair_ok, chosen_median[c])
        chosen_median[c] = type_med_lookup[f"{c}_median"].where((~pair_ok) & type_ok, chosen_median[c])

        chosen_mad[c] = pair_mad_lookup[f"{c}_mad"].where(pair_ok, chosen_mad[c])
        chosen_mad[c] = type_mad_lookup[f"{c}_mad"].where((~pair_ok) & type_ok, chosen_mad[c])

    mad_safe = chosen_mad.clip(lower=0.02)
    residuals = (colors - chosen_median) / mad_safe
    residuals = residuals.clip(lower=-10.0, upper=10.0)

    residuals = residuals.rename(columns={c: f"template_residual_{c}" for c in color_cols})

    abs_residual = residuals.abs()
    out = pd.concat(
        [
            residuals,
            pd.DataFrame(
                {
                    "template_residual_mean_abs": abs_residual.mean(axis=1),
                    "template_residual_max_abs": abs_residual.max(axis=1),
                    "template_residual_rss": (residuals**2).sum(axis=1),
                    "template_residual_count_exceeds_2": (abs_residual > 2.0).sum(axis=1),
                },
                index=raw.index,
            ),
        ],
        axis=1,
    )

    return out


FEATURE_GROUPS = [
    {
        "name": "catalog_template_residuals",
        "fn": add_catalog_template_residuals,
        "depends_on": [],
        "description": "Build template-mismatch residual features by comparing object colors to tag-conditioned color centroids with MAD normalization and fallback priors.",
    }
]