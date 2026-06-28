import numpy as np
import pandas as pd


SPECTRAL_EXPECTATION = {
    "O/B": 0.00,
    "A/F": 0.33,
    "G/K": 0.67,
    "M": 1.00,
}

GALAXY_EXPECTATION = {
    "Blue_Cloud": 0.00,
    "Red_Sequence": 1.00,
}

COLOR_WEIGHTS = {
    "ug": 0.40,
    "gr": 0.30,
    "ri": 0.20,
    "iz": 0.10,
}


def _numeric_series(raw, column, default=0.0):
    if column not in raw.columns:
        return pd.Series(default, index=raw.index, dtype="float64")
    return pd.to_numeric(raw[column], errors="coerce").astype("float64")


def _finite_fill(series, fill_value):
    series = pd.Series(series, index=series.index if hasattr(series, "index") else None)
    return series.replace([np.inf, -np.inf], np.nan).fillna(fill_value)


def _training_mask(raw):
    if "id" not in raw.columns:
        return pd.Series(True, index=raw.index)

    ids = pd.to_numeric(raw["id"], errors="coerce")
    finite_ids = ids[np.isfinite(ids)]
    if finite_ids.empty:
        return pd.Series(True, index=raw.index)

    train_like = ids <= 577346
    if int(train_like.sum()) >= 1000 and int((~train_like).sum()) >= 1000:
        return train_like.fillna(False)

    return pd.Series(True, index=raw.index)


def _safe_quantile(series, mask, q, fallback):
    values = series[mask]
    values = values[np.isfinite(values)]
    if values.empty:
        return fallback
    result = values.quantile(q)
    if not np.isfinite(result):
        return fallback
    return float(result)


def _normalized_redness(color, q02, q98):
    denom = q98 - q02
    if not np.isfinite(denom) or abs(denom) < 1e-6:
        denom = 1e-6
    normalized = ((color - q02) / denom).clip(lower=0.0, upper=1.0)
    return _finite_fill(normalized, 0.50)


def _expectation(raw, column, mapping):
    if column not in raw.columns:
        return pd.Series(0.50, index=raw.index, dtype="float64")
    mapped = raw[column].map(mapping)
    return pd.to_numeric(mapped, errors="coerce").fillna(0.50).astype("float64")


def _residual_summary(frame, expectation):
    residuals = frame.sub(expectation, axis=0)
    abs_residuals = residuals.abs()
    return {
        "mean": _finite_fill(residuals.mean(axis=1), 0.50),
        "median": _finite_fill(residuals.median(axis=1), 0.50),
        "min": _finite_fill(residuals.min(axis=1), 0.50),
        "max": _finite_fill(residuals.max(axis=1), 0.50),
        "l1_sum": _finite_fill(abs_residuals.sum(axis=1), 0.50),
        "l2_sq_sum": _finite_fill((residuals * residuals).sum(axis=1), 0.50),
        "max_abs": _finite_fill(abs_residuals.max(axis=1), 0.50),
    }


def add_catalog_tag_color_concordance(raw, deps, aux):
    index = raw.index
    features = pd.DataFrame(index=index)

    u = _numeric_series(raw, "u", 0.0)
    g = _numeric_series(raw, "g", 0.0)
    r = _numeric_series(raw, "r", 0.0)
    i = _numeric_series(raw, "i", 0.0)
    z = _numeric_series(raw, "z", 0.0)
    redshift = _numeric_series(raw, "redshift", 0.0)

    colors = {
        "ug": _finite_fill(u - g, 0.50),
        "gr": _finite_fill(g - r, 0.50),
        "ri": _finite_fill(r - i, 0.50),
        "iz": _finite_fill(i - z, 0.50),
    }

    train_mask = _training_mask(raw)

    redness_components = {}
    for color_name, color_values in colors.items():
        q02 = _safe_quantile(color_values, train_mask, 0.02, 0.0)
        q98 = _safe_quantile(color_values, train_mask, 0.98, 1.0)
        component = _normalized_redness(color_values, q02, q98)
        redness_components[color_name] = component
        features["redness_" + color_name] = component

    observed_redness = (
        COLOR_WEIGHTS["ug"] * redness_components["ug"]
        + COLOR_WEIGHTS["gr"] * redness_components["gr"]
        + COLOR_WEIGHTS["ri"] * redness_components["ri"]
        + COLOR_WEIGHTS["iz"] * redness_components["iz"]
    )
    observed_redness = _finite_fill(observed_redness, 0.50).clip(lower=0.0, upper=1.0)
    features["observed_redness"] = observed_redness

    spectral_expectation = _expectation(raw, "spectral_type", SPECTRAL_EXPECTATION)
    galaxy_expectation = _expectation(raw, "galaxy_population", GALAXY_EXPECTATION)
    features["spectral_expected_redness"] = spectral_expectation
    features["population_expected_redness"] = galaxy_expectation

    for prefix, expectation in (
        ("spectral", spectral_expectation),
        ("population", galaxy_expectation),
    ):
        residual = _finite_fill(observed_redness - expectation, 0.50)
        abs_residual = _finite_fill(residual.abs(), 0.50)
        sq_residual = _finite_fill(residual * residual, 0.50)
        agreement = _finite_fill(1.0 - abs_residual, 0.50).clip(lower=0.0, upper=1.0)

        features[prefix + "_redness_residual"] = residual
        features[prefix + "_redness_abs_residual"] = abs_residual
        features[prefix + "_redness_sq_residual"] = sq_residual
        features[prefix + "_redness_agreement"] = agreement

    tag_disagreement = _finite_fill(spectral_expectation - galaxy_expectation, 0.50)
    features["tag_disagreement_signed"] = tag_disagreement
    features["tag_disagreement_abs"] = _finite_fill(tag_disagreement.abs(), 0.50)

    component_frame = pd.DataFrame(
        {
            "ug": redness_components["ug"],
            "gr": redness_components["gr"],
            "ri": redness_components["ri"],
            "iz": redness_components["iz"],
        },
        index=index,
    )

    for prefix, expectation in (
        ("spectral_component", spectral_expectation),
        ("population_component", galaxy_expectation),
    ):
        summaries = _residual_summary(component_frame, expectation)
        for stat_name, stat_values in summaries.items():
            features[prefix + "_" + stat_name + "_residual"] = stat_values

    early_redness = _finite_fill((redness_components["ug"] + redness_components["gr"]) / 2.0, 0.50)
    late_redness = _finite_fill((redness_components["ri"] + redness_components["iz"]) / 2.0, 0.50)
    early_minus_late = _finite_fill(early_redness - late_redness, 0.50)

    features["early_redness_mean"] = early_redness
    features["late_redness_mean"] = late_redness
    features["early_minus_late_redness"] = early_minus_late

    for prefix, expectation in (
        ("spectral", spectral_expectation),
        ("population", galaxy_expectation),
    ):
        features[prefix + "_early_redness_residual"] = _finite_fill(early_redness - expectation, 0.50)
        features[prefix + "_late_redness_residual"] = _finite_fill(late_redness - expectation, 0.50)
        features[prefix + "_early_minus_late_residual"] = _finite_fill(early_minus_late - expectation, 0.50)

    features["redness_bin_0_025"] = ((observed_redness >= 0.00) & (observed_redness < 0.25)).astype("int8")
    features["redness_bin_025_050"] = ((observed_redness >= 0.25) & (observed_redness < 0.50)).astype("int8")
    features["redness_bin_050_075"] = ((observed_redness >= 0.50) & (observed_redness < 0.75)).astype("int8")
    features["redness_bin_075_100"] = ((observed_redness >= 0.75) & (observed_redness <= 1.00)).astype("int8")

    contradiction_flags = {
        "hot_tag_red_color": (spectral_expectation <= 0.33) & (observed_redness >= 0.67),
        "cool_tag_blue_color": (spectral_expectation >= 0.67) & (observed_redness <= 0.33),
        "red_sequence_blue_color": (galaxy_expectation >= 0.67) & (observed_redness <= 0.33),
        "blue_cloud_red_color": (galaxy_expectation <= 0.33) & (observed_redness >= 0.67),
        "dual_tag_mismatch": (
            (features["spectral_redness_abs_residual"] >= 0.35)
            & (features["population_redness_abs_residual"] >= 0.35)
        ),
    }

    z_q10 = _safe_quantile(redshift, train_mask, 0.10, 0.0)
    z_q50 = _safe_quantile(redshift, train_mask, 0.50, 0.5)
    z_q90 = _safe_quantile(redshift, train_mask, 0.90, 1.0)
    z_denom = z_q90 - z_q50
    if not np.isfinite(z_denom) or abs(z_denom) < 1e-6:
        trust_weight = pd.Series(0.50, index=index, dtype="float64")
    else:
        trust_weight = 1.0 - ((redshift - z_q50) / z_denom).clip(lower=0.0, upper=1.0)
        trust_weight = _finite_fill(trust_weight, 0.50)

    features["redshift_q10_centered"] = _finite_fill(redshift - z_q10, 0.50)
    features["redshift_q50_centered"] = _finite_fill(redshift - z_q50, 0.50)
    features["redshift_q90_centered"] = _finite_fill(redshift - z_q90, 0.50)
    features["redshift_color_trust_weight"] = trust_weight

    weighted_columns = [
        "spectral_redness_abs_residual",
        "spectral_redness_sq_residual",
        "population_redness_abs_residual",
        "population_redness_sq_residual",
        "tag_disagreement_abs",
        "spectral_component_l1_sum_residual",
        "spectral_component_l2_sq_sum_residual",
        "spectral_component_max_abs_residual",
        "population_component_l1_sum_residual",
        "population_component_l2_sq_sum_residual",
        "population_component_max_abs_residual",
    ]
    for column in weighted_columns:
        features["wz_weighted_" + column] = _finite_fill(features[column] * trust_weight, 0.50)

    for flag_name, flag_values in contradiction_flags.items():
        clean_flag = pd.Series(flag_values, index=index).fillna(False).astype("int8")
        features[flag_name] = clean_flag
        features["wz_weighted_" + flag_name] = _finite_fill(clean_flag.astype("float64") * trust_weight, 0.0)

    for column in features.columns:
        if pd.api.types.is_bool_dtype(features[column]):
            features[column] = features[column].fillna(False)
        elif pd.api.types.is_numeric_dtype(features[column]):
            fill_value = 0.0 if column.startswith("redness_bin_") or column in contradiction_flags else 0.50
            features[column] = features[column].replace([np.inf, -np.inf], np.nan).fillna(fill_value)

    return features


FEATURE_GROUPS = [
    {
        "name": "catalog_tag_color_concordance",
        "fn": add_catalog_tag_color_concordance,
        "depends_on": [],
        "description": "Redshift-aware color and catalog tag concordance features from ugriz colors, spectral type, and galaxy population.",
    }
]