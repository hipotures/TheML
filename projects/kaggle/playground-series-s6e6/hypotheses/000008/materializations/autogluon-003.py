import numpy as np
import pandas as pd

_REDSHIFT_BINS = (-np.inf, 0.002, 0.01, 0.05, 0.2, 0.6, 1.2, 2.5, np.inf)
_QUANTILE_POINTS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
_TERTILE_POINTS = (1.0 / 3.0, 2.0 / 3.0)
_MAGNITUDE_COLUMNS = ("u", "g", "r", "i", "z")
_SPECTRAL_LEVELS = ("M", "O/B", "G/K", "A/F", "UNK")
_GALAXY_LEVELS = ("Red_Sequence", "Blue_Cloud", "UNK")
_ALPHA = 1.0
_EPS = 1e-12


def _to_float(series):
    return pd.to_numeric(series, errors="coerce").astype(float)


def _manifold_coordinates(raw):
    fluxes = {
        "u": 10.0 ** (-0.4 * _to_float(raw["u"])),
        "g": 10.0 ** (-0.4 * _to_float(raw["g"])),
        "r": 10.0 ** (-0.4 * _to_float(raw["r"])),
        "i": 10.0 ** (-0.4 * _to_float(raw["i"])),
        "z": 10.0 ** (-0.4 * _to_float(raw["z"])),
    }

    total_flux = fluxes["u"] + fluxes["g"] + fluxes["r"] + fluxes["i"] + fluxes["z"]
    norm = total_flux + _EPS

    shares = {
        "u": fluxes["u"] / norm,
        "g": fluxes["g"] / norm,
        "r": fluxes["r"] / norm,
        "i": fluxes["i"] / norm,
        "z": fluxes["z"] / norm,
    }

    blue_balance = np.log((shares["u"] + shares["g"] + _EPS) / (shares["i"] + shares["z"] + _EPS))
    concentration = np.maximum.reduce([shares["u"], shares["g"], shares["r"], shares["i"], shares["z"]])
    brightness = -2.5 * np.log10(total_flux + _EPS)

    return blue_balance, concentration, brightness


def _resolve_train_frame(raw, aux):
    if not isinstance(aux, pd.DataFrame) or aux.empty or len(aux) != len(raw):
        return raw

    for key in ("is_train", "train", "train_mask"):
        if key not in aux.columns:
            continue

        vals = aux[key]
        if pd.api.types.is_bool_dtype(vals):
            mask = vals.fillna(False).astype(bool)
        elif pd.api.types.is_integer_dtype(vals):
            mask = vals.fillna(0).astype(int).astype(bool)
        else:
            mask = vals.astype("string", errors="ignore").str.lower().isin(("1", "true", "train", "yes", "y"))

        if mask.any():
            train_frame = raw.loc[mask.to_numpy()]
            if len(train_frame) > 0:
                return train_frame
            return raw

    return raw


def _build_bin_edges(values):
    arr = np.asarray(_to_float(values), dtype=float)
    arr = arr[np.isfinite(arr)]

    if arr.size == 0:
        arr = np.array([0.0, 1.0], dtype=float)

    train_min = float(np.min(arr))
    train_max = float(np.max(arr))

    if train_min == train_max:
        eps = 1.0 if train_min == 0.0 else abs(train_min) * 1e-6 + 1e-6
        edges = np.array([train_min - eps, train_min, train_max + eps], dtype=float)
        return edges, train_min, train_max

    edges = np.unique(np.quantile(arr, _QUANTILE_POINTS))
    if edges.size < 2:
        edges = np.unique(np.quantile(arr, _TERTILE_POINTS))

    bins = np.unique(np.concatenate(([train_min], edges, [train_max])))

    if bins.size < 3:
        bins = np.array([train_min, (train_min + train_max) * 0.5, train_max], dtype=float)

    bins = np.array(sorted(bins), dtype=float)
    cleaned = [bins[0]]
    for val in bins[1:]:
        prev = cleaned[-1]
        if val <= prev:
            nxt = np.nextafter(prev, np.inf)
            if not np.isfinite(nxt) or nxt <= prev:
                nxt = prev + 1e-12
            cleaned.append(float(nxt))
        else:
            cleaned.append(float(val))

    if len(cleaned) < 3:
        lo = cleaned[0]
        hi = cleaned[-1]
        cleaned = [lo, lo + (hi - lo) * 0.5, hi]

    return np.array(cleaned, dtype=float), train_min, train_max


def _apply_bin(values, bins, lower, upper):
    clipped = _to_float(values).clip(lower=lower, upper=upper)
    clipped = clipped.fillna(lower)

    try:
        codes = pd.cut(
            clipped,
            bins=bins,
            labels=False,
            include_lowest=True,
            right=True,
        )
    except ValueError:
        n_bins = max(3, len(bins) - 1)
        lo = float(np.min(clipped.to_numpy()))
        hi = float(np.max(clipped.to_numpy()))
        if lo == hi:
            hi = lo + 1.0
        fallback = np.linspace(lo, hi, n_bins + 1)
        codes = pd.cut(
            clipped,
            bins=fallback,
            labels=False,
            include_lowest=True,
            right=True,
        )

    return codes.fillna(0).astype("int64")


def _categorical_with_unk(values, allowed):
    vals = values.astype("string").fillna("UNK")
    allowed_set = set(allowed)
    return vals.where(vals.isin(allowed_set), "UNK").astype("object")


def add_survey_manifold_rarity(raw, deps, aux):
    _ = deps

    redshift_all = _to_float(raw["redshift"])
    spectral_all = _categorical_with_unk(raw["spectral_type"], _SPECTRAL_LEVELS)
    galaxy_all = _categorical_with_unk(raw["galaxy_population"], _GALAXY_LEVELS)

    blue_balance, concentration, brightness = _manifold_coordinates(raw)
    train_frame = _resolve_train_frame(raw, aux)

    blue_edges, blue_min, blue_max = _build_bin_edges(_manifold_coordinates(train_frame)[0])
    con_edges, con_min, con_max = _build_bin_edges(_manifold_coordinates(train_frame)[1])
    bright_edges, bright_min, bright_max = _build_bin_edges(_manifold_coordinates(train_frame)[2])

    blue_bin_train = _apply_bin(_manifold_coordinates(train_frame)[0], blue_edges, blue_min, blue_max)
    con_bin_train = _apply_bin(_manifold_coordinates(train_frame)[1], con_edges, con_min, con_max)
    bright_bin_train = _apply_bin(_manifold_coordinates(train_frame)[2], bright_edges, bright_min, bright_max)

    spectral_train = _categorical_with_unk(train_frame["spectral_type"], _SPECTRAL_LEVELS)
    galaxy_train = _categorical_with_unk(train_frame["galaxy_population"], _GALAXY_LEVELS)
    redshift_train = pd.cut(
        _to_float(train_frame["redshift"]),
        bins=_REDSHIFT_BINS,
        labels=False,
        include_lowest=True,
        right=False,
    ).astype("int64")

    blue_bin_all = _apply_bin(blue_balance, blue_edges, blue_min, blue_max)
    con_bin_all = _apply_bin(concentration, con_edges, con_min, con_max)
    bright_bin_all = _apply_bin(brightness, bright_edges, bright_min, bright_max)
    redshift_all_bin = pd.cut(
        redshift_all,
        bins=_REDSHIFT_BINS,
        labels=False,
        include_lowest=True,
        right=False,
    ).astype("int64")

    a_train = pd.DataFrame(
        {
            "redshift_bin": redshift_train.to_numpy(),
            "spectral_type_bin": spectral_train.to_numpy(),
            "galaxy_population_bin": galaxy_train.to_numpy(),
        },
        index=train_frame.index,
    )
    b_train = pd.DataFrame(
        {
            "blue_balance_bin": blue_bin_train.to_numpy(),
            "concentration_bin": con_bin_train.to_numpy(),
            "brightness_bin": bright_bin_train.to_numpy(),
        },
        index=train_frame.index,
    )
    j_train = pd.concat([a_train, b_train], axis=1)

    n_a = a_train.value_counts(dropna=False).astype(float)
    n_b = b_train.value_counts(dropna=False).astype(float)
    n_j = j_train.value_counts(dropna=False).astype(float)

    k_a = (len(_REDSHIFT_BINS) - 1) * len(_SPECTRAL_LEVELS) * len(_GALAXY_LEVELS)
    k_b = (len(blue_edges) - 1) * (len(con_edges) - 1) * (len(bright_edges) - 1)
    k_j = k_a * k_b

    a_all = pd.DataFrame(
        {
            "redshift_bin": redshift_all_bin.to_numpy(),
            "spectral_type_bin": spectral_all.to_numpy(),
            "galaxy_population_bin": galaxy_all.to_numpy(),
        },
        index=raw.index,
    )
    b_all = pd.DataFrame(
        {
            "blue_balance_bin": blue_bin_all.to_numpy(),
            "concentration_bin": con_bin_all.to_numpy(),
            "brightness_bin": bright_bin_all.to_numpy(),
        },
        index=raw.index,
    )
    j_all = pd.concat([a_all, b_all], axis=1)

    a_idx = pd.MultiIndex.from_frame(a_all.astype(object))
    b_idx = pd.MultiIndex.from_frame(b_all.astype(object))
    j_idx = pd.MultiIndex.from_frame(j_all.astype(object))

    n = float(len(train_frame))
    n_a_lookup = n_a.reindex(a_idx, fill_value=0.0).to_numpy(dtype=float)
    n_b_lookup = n_b.reindex(b_idx, fill_value=0.0).to_numpy(dtype=float)
    n_j_lookup = n_j.reindex(j_idx, fill_value=0.0).to_numpy(dtype=float)

    p_a = (n_a_lookup + _ALPHA) / (n + _ALPHA * k_a)
    p_b = (n_b_lookup + _ALPHA) / (n + _ALPHA * k_b)
    p_j = (n_j_lookup + _ALPHA) / (n + _ALPHA * k_j)

    rarity_score = -np.log(p_j)
    interaction_surprise = np.log((p_a * p_b + _EPS) / (p_j + _EPS))
    local_dominance = np.log(p_a * p_b)

    return pd.DataFrame(
        {
            "rarity_score": rarity_score,
            "interaction_surprise": interaction_surprise,
            "local_dominance": local_dominance,
        },
        index=raw.index,
    )


FEATURE_GROUPS = [
    {
        "name": "survey_manifold_rarity",
        "fn": add_survey_manifold_rarity,
        "depends_on": [],
        "description": "Build manifold rarity, interaction surprise, and local dominance features from redshift-regime/categorical bins and quantile-binned photometric manifold cells.",
    },
]