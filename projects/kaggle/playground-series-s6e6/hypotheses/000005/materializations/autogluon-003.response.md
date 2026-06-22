import numpy as np
import pandas as pd


def _mad_series(values: pd.Series) -> float:
    s = values.dropna()
    if s.empty:
        return 0.0
    med = s.median()
    return (s - med).abs().median()


def add_catalog_template_residuals(raw, deps, aux):
    colors = pd.DataFrame(
        {
            "c1": raw["u"] - raw["g"],
            "c2": raw["g"] - raw["r"],
            "c3": raw["r"] - raw["i"],
            "c4": raw["i"] - raw["z"],
        },
        index=raw.index,
    )
    color_cols = ("c1", "c2", "c3", "c4")

    spectral_type = raw["spectral_type"].astype(object)
    galaxy_population = raw["galaxy_population"].astype(object)

    pair_data = pd.concat(
        [
            spectral_type.rename("spectral_type"),
            galaxy_population.rename("galaxy_population"),
            colors,
        ],
        axis=1,
    )
    pair_counts = pair_data.groupby(["spectral_type", "galaxy_population"]).size()
    pair_medians = (
        pair_data.groupby(["spectral_type", "galaxy_population"])[list(color_cols)]
        .median()
        .rename(columns={c: f"{c}_median" for c in color_cols})
    )
    pair_mads = (
        pair_data.groupby(["spectral_type", "galaxy_population"])[list(color_cols)]
        .agg(_mad_series)
        .rename(columns={c: f"{c}_mad" for c in color_cols})
    )

    type_data = pd.concat([spectral_type.rename("spectral_type"), colors], axis=1)
    type_counts = type_data.groupby("spectral_type").size()
    type_medians = (
        type_data.groupby("spectral_type")[list(color_cols)]
        .median()
        .rename(columns={c: f"{c}_median" for c in color_cols})
    )
    type_mads = (
        type_data.groupby("spectral_type")[list(color_cols)]
        .agg(_mad_series)
        .rename(columns={c: f"{c}_mad" for c in color_cols})
    )

    global_medians = colors.median()
    global_mads = colors.apply(_mad_series)

    pair_key = pd.Series(
        list(zip(spectral_type.to_numpy(), galaxy_population.to_numpy())),
        index=raw.index,
        dtype=object,
    )
    pair_count = pair_key.map(pair_counts).fillna(0).astype(int)
    type_count = spectral_type.map(type_counts).fillna(0).astype(int)

    pair_ok = pair_count.ge(1000)
    type_ok = type_count.ge(1000)

    chosen_median = pd.DataFrame(
        {c: float(global_medians[c]) for c in color_cols},
        index=raw.index,
    )
    chosen_mad = pd.DataFrame(
        {c: float(global_mads[c]) for c in color_cols},
        index=raw.index,
    )

    for c in color_cols:
        pair_med_col = pair_key.map(pair_medians[f"{c}_median"])
        type_med_col = spectral_type.map(type_medians[f"{c}_median"])
        pair_mad_col = pair_key.map(pair_mads[f"{c}_mad"])
        type_mad_col = spectral_type.map(type_mads[f"{c}_mad"])

        chosen_median[c] = pair_med_col.where(pair_ok, chosen_median[c])
        chosen_median[c] = type_med_col.where((~pair_ok) & type_ok, chosen_median[c])

        chosen_mad[c] = pair_mad_col.where(pair_ok, chosen_mad[c])
        chosen_mad[c] = type_mad_col.where((~pair_ok) & type_ok, chosen_mad[c])

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