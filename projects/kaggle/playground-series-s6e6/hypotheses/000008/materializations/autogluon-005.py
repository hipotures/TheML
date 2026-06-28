import numpy as np
import pandas as pd

EPS = 1e-12
SMOOTHING_ALPHA = 1.0
TRAIN_MAX_ID = 577346
UNKNOWN_LEVEL = "__UNK__"

BAND_COLUMNS = ("u", "g", "r", "i", "z")
REDSHIFT_INTERNAL_EDGES = (0.002, 0.01, 0.05, 0.2, 0.6, 1.2, 2.5)
QUANTILE_LEVELS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
TERTILE_LEVELS = (0.33, 0.67)
FIT_MASK_COLUMNS = ("is_fit", "_is_fit", "fit_mask", "_fit_mask", "is_train", "_is_train")


def _fit_mask(raw, aux):
    if aux is not None and len(aux) == len(raw):
        for col in FIT_MASK_COLUMNS:
            if col in aux.columns:
                mask = aux[col]
                if pd.api.types.is_bool_dtype(mask) or set(pd.Series(mask).dropna().unique()).issubset({0, 1, False, True}):
                    return pd.Series(mask, index=raw.index).fillna(False).astype(bool).to_numpy()

    if "id" in raw.columns:
        ids = pd.to_numeric(raw["id"], errors="coerce")
        mask = ids.le(TRAIN_MAX_ID).fillna(False)
        if mask.any():
            return mask.to_numpy()

    return np.ones(len(raw), dtype=bool)


def _ordered_unique(values):
    out = []
    previous = None
    for value in values:
        if np.isfinite(value) and (previous is None or value > previous):
            out.append(float(value))
            previous = float(value)
    return np.asarray(out, dtype="float64")


def _quantile_edges(values, fit_mask):
    fit_values = values[fit_mask]
    fit_values = fit_values[np.isfinite(fit_values)]

    if fit_values.size == 0:
        return np.asarray([], dtype="float64"), 0.0, 0.0

    fit_min = float(np.min(fit_values))
    fit_max = float(np.max(fit_values))

    edges = _ordered_unique(np.quantile(fit_values, np.asarray(QUANTILE_LEVELS, dtype="float64")))
    if edges.size + 1 < 3:
        edges = _ordered_unique(np.quantile(fit_values, np.asarray(TERTILE_LEVELS, dtype="float64")))

    return edges, fit_min, fit_max


def _numeric_bin(values, edges, fit_min, fit_max):
    clean = np.asarray(values, dtype="float64")
    clean = np.where(np.isfinite(clean), clean, fit_min)
    clean = np.clip(clean, fit_min, fit_max)

    if edges.size == 0:
        return np.zeros(clean.shape[0], dtype="int64")

    return np.searchsorted(edges, clean, side="right").astype("int64")


def _category_codes(raw, column, fit_mask):
    train_values = raw.loc[fit_mask, column].astype("string").fillna(UNKNOWN_LEVEL)
    levels = sorted(value for value in train_values.unique().tolist() if value != UNKNOWN_LEVEL)
    levels.append(UNKNOWN_LEVEL)

    level_to_code = {value: idx for idx, value in enumerate(levels)}
    values = raw[column].astype("string").fillna(UNKNOWN_LEVEL)
    codes = values.map(level_to_code).fillna(level_to_code[UNKNOWN_LEVEL]).astype("int64").to_numpy()

    return codes, len(levels)


def add_survey_manifold_rarity(raw, deps, aux):
    fit_mask = _fit_mask(raw, aux)
    n_fit = int(np.sum(fit_mask))
    if n_fit <= 0:
        fit_mask = np.ones(len(raw), dtype=bool)
        n_fit = len(raw)

    magnitudes = raw.loc[:, BAND_COLUMNS].apply(pd.to_numeric, errors="coerce").to_numpy(dtype="float64")
    flux = np.power(10.0, -0.4 * magnitudes)
    total_flux = np.sum(flux, axis=1)
    shares = flux / (total_flux[:, None] + EPS)

    blue_balance = np.log((shares[:, 0] + shares[:, 1] + EPS) / (shares[:, 3] + shares[:, 4] + EPS))
    concentration = np.max(shares, axis=1)
    brightness = -2.5 * np.log10(total_flux + EPS)

    redshift = pd.to_numeric(raw["redshift"], errors="coerce").to_numpy(dtype="float64")
    redshift_edges = np.asarray(REDSHIFT_INTERNAL_EDGES, dtype="float64")
    redshift_clean = np.where(np.isfinite(redshift), redshift, 0.0)
    redshift_bin = np.searchsorted(redshift_edges, redshift_clean, side="right").astype("int64")
    redshift_bin_count = len(REDSHIFT_INTERNAL_EDGES) + 1

    blue_edges, blue_min, blue_max = _quantile_edges(blue_balance, fit_mask)
    concentration_edges, concentration_min, concentration_max = _quantile_edges(concentration, fit_mask)
    brightness_edges, brightness_min, brightness_max = _quantile_edges(brightness, fit_mask)

    blue_bin = _numeric_bin(blue_balance, blue_edges, blue_min, blue_max)
    concentration_bin = _numeric_bin(concentration, concentration_edges, concentration_min, concentration_max)
    brightness_bin = _numeric_bin(brightness, brightness_edges, brightness_min, brightness_max)

    blue_bin_count = int(blue_edges.size + 1)
    concentration_bin_count = int(concentration_edges.size + 1)
    brightness_bin_count = int(brightness_edges.size + 1)

    spectral_code, spectral_count = _category_codes(raw, "spectral_type", fit_mask)
    population_code, population_count = _category_codes(raw, "galaxy_population", fit_mask)

    a_code = ((redshift_bin * spectral_count + spectral_code) * population_count + population_code).astype("int64")
    b_code = ((blue_bin * concentration_bin_count + concentration_bin) * brightness_bin_count + brightness_bin).astype("int64")

    k_a = int(redshift_bin_count * spectral_count * population_count)
    k_b = int(blue_bin_count * concentration_bin_count * brightness_bin_count)
    k_j = int(k_a * k_b)

    j_code = (a_code * k_b + b_code).astype("int64")

    counts_a = np.bincount(a_code[fit_mask], minlength=k_a).astype("float64")
    counts_b = np.bincount(b_code[fit_mask], minlength=k_b).astype("float64")
    counts_j = np.bincount(j_code[fit_mask], minlength=k_j).astype("float64")

    p_a = (counts_a[a_code] + SMOOTHING_ALPHA) / (n_fit + SMOOTHING_ALPHA * k_a)
    p_b = (counts_b[b_code] + SMOOTHING_ALPHA) / (n_fit + SMOOTHING_ALPHA * k_b)
    p_j = (counts_j[j_code] + SMOOTHING_ALPHA) / (n_fit + SMOOTHING_ALPHA * k_j)

    return pd.DataFrame(
        {
            "rarity_score": -np.log(p_j),
            "marginal_rarity": -0.5 * (np.log(p_a) + np.log(p_b)),
            "interaction_surprise": np.log((p_a * p_b + EPS) / (p_j + EPS)),
        },
        index=raw.index,
    )


FEATURE_GROUPS = [
    {
        "name": "survey_manifold_rarity",
        "fn": add_survey_manifold_rarity,
        "depends_on": [],
        "description": "Smoothed rarity and interaction-surprise features over redshift, catalog tags, and broad ugriz flux-share manifold cells.",
    }
]