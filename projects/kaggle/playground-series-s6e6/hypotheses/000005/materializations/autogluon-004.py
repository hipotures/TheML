import numpy as np
import pandas as pd

_COLOR_COMPONENTS = (
    ("template_z1", "u", "g"),
    ("template_z2", "g", "r"),
    ("template_z3", "r", "i"),
    ("template_z4", "i", "z"),
)

_TEMPLATE_COUNT_THRESHOLD = 600
_TEMPLATE_SCALE_FLOOR = 0.02
_TEMPLATE_SHRINK_DENOM = 900.0


def _mad(series):
    median_val = series.median()
    return (series - median_val).abs().median()


def add_catalog_template_residuals(raw, deps, aux):
    spectral = raw["spectral_type"]
    population = raw["galaxy_population"]

    index = raw.index
    key_index = pd.MultiIndex.from_arrays([spectral.to_numpy(), population.to_numpy()])

    cell_counts = raw.groupby(["spectral_type", "galaxy_population"]).size()
    spectral_counts = raw.groupby("spectral_type").size()

    n_cell = cell_counts.reindex(key_index, fill_value=0).to_numpy(dtype="float64")
    n_spectral = spectral_counts.reindex(spectral, fill_value=0).to_numpy(dtype="float64")
    use_global_spectral = n_spectral < _TEMPLATE_COUNT_THRESHOLD

    z_cols = []
    abs_z_cols = []

    for out_name, left_col, right_col in _COLOR_COMPONENTS:
        color = pd.Series(
            raw[left_col].to_numpy(dtype="float64") - raw[right_col].to_numpy(dtype="float64"),
            index=index,
        )

        cell_median = color.groupby([spectral, population]).median()
        cell_mad = color.groupby([spectral, population]).apply(_mad)

        spectral_median = color.groupby(spectral).median()
        spectral_mad = color.groupby(spectral).apply(_mad)

        global_median = float(color.median())
        global_mad = float(_mad(color))

        mu_cell = cell_median.reindex(key_index, fill_value=0.0).to_numpy(dtype="float64")
        mad_cell = cell_mad.reindex(key_index, fill_value=0.0).to_numpy(dtype="float64")

        mu_spec = spectral_median.reindex(spectral, fill_value=global_median).to_numpy(dtype="float64")
        mad_spec = spectral_mad.reindex(spectral, fill_value=global_mad).to_numpy(dtype="float64")

        if use_global_spectral.any():
            mu_spec = np.where(use_global_spectral, global_median, mu_spec)
            mad_spec = np.where(use_global_spectral, global_mad, mad_spec)

        w = np.clip((n_cell - _TEMPLATE_COUNT_THRESHOLD) / _TEMPLATE_SHRINK_DENOM, 0.0, 1.0)

        mu = w * mu_cell + (1.0 - w) * mu_spec
        scale = w * mad_cell + (1.0 - w) * mad_spec
        scale = np.where(scale < _TEMPLATE_SCALE_FLOOR, _TEMPLATE_SCALE_FLOOR, scale)

        z = (color.to_numpy(dtype="float64") - mu) / scale
        z = np.clip(z, -10.0, 10.0)
        abs_z = np.abs(z)

        z_cols.append(z)
        abs_z_cols.append(abs_z)

    z_matrix = np.column_stack(z_cols)
    abs_z_matrix = np.column_stack(abs_z_cols)

    return pd.DataFrame(
        {
            "template_z1": z_matrix[:, 0],
            "template_z2": z_matrix[:, 1],
            "template_z3": z_matrix[:, 2],
            "template_z4": z_matrix[:, 3],
            "template_abs_z_mean": abs_z_matrix.mean(axis=1),
            "template_abs_z_median": np.median(abs_z_matrix, axis=1),
            "template_abs_z_max": abs_z_matrix.max(axis=1),
            "template_abs_z_l2": np.sqrt((z_matrix ** 2).sum(axis=1)),
            "template_abs_z_gt2_count": (abs_z_matrix > 2.0).sum(axis=1),
            "template_abs_z_gt3_count": (abs_z_matrix > 3.0).sum(axis=1),
        },
        index=index,
    )


FEATURE_GROUPS = [
    {
        "name": "catalog_template_residuals",
        "fn": add_catalog_template_residuals,
        "depends_on": [],
        "description": "Creates shrinkage-blended template residual features from adjacent color indices with robust location/scale statistics and compact anomaly descriptors.",
    }
]