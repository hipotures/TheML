import numpy as np
import pandas as pd

_DECILE_QUANTILES = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
_REDSHIFT_BINS = (-np.inf, 0.002, 0.01, 0.05, 0.2, 0.6, 1.2, 2.5, np.inf)
_UNK = "__UNK__"
_TRAIN_ID_CUTOFF = 577347


def _infer_train_mask(raw, aux):
    n = len(raw)
    if n == 0:
        return np.array([], dtype=bool)

    if aux is not None and hasattr(aux, "empty") and not aux.empty and len(aux) == n:
        for col in ("is_train", "is_train_row", "is_train_mask", "train_mask"):
            if col in aux.columns:
                return aux[col].fillna(False).astype(bool).to_numpy()

    if "id" not in raw.columns:
        return np.ones(n, dtype=bool)

    ids = pd.to_numeric(raw["id"], errors="coerce")
    if ids.notna().sum() == 0:
        return np.ones(n, dtype=bool)

    ids_arr = ids.to_numpy()
    min_id = float(np.nanmin(ids_arr))
    max_id = float(np.nanmax(ids_arr))

    if min_id == 0.0:
        if max_id >= _TRAIN_ID_CUTOFF and n > _TRAIN_ID_CUTOFF:
            return ids_arr < _TRAIN_ID_CUTOFF

    if min_id >= _TRAIN_ID_CUTOFF:
        return np.zeros(n, dtype=bool)

    return np.ones(n, dtype=bool)


def _compute_photometric_coords(frame):
    u = pd.to_numeric(frame["u"], errors="coerce").to_numpy(dtype=float)
    g = pd.to_numeric(frame["g"], errors="coerce").to_numpy(dtype=float)
    r = pd.to_numeric(frame["r"], errors="coerce").to_numpy(dtype=float)
    i = pd.to_numeric(frame["i"], errors="coerce").to_numpy(dtype=float)
    z = pd.to_numeric(frame["z"], errors="coerce").to_numpy(dtype=float)

    flux = np.column_stack(
        (
            np.power(10.0, -0.4 * u),
            np.power(10.0, -0.4 * g),
            np.power(10.0, -0.4 * r),
            np.power(10.0, -0.4 * i),
            np.power(10.0, -0.4 * z),
        )
    )

    total_flux = flux.sum(axis=1)
    shares = np.zeros_like(flux, dtype=float)
    valid = total_flux > 0
    np.divide(flux, total_flux[:, None], out=shares, where=valid[:, None])

    blue_balance = (shares[:, 0] + shares[:, 1]) - (shares[:, 3] + shares[:, 4])
    flux_concentration = shares.max(axis=1)
    total_brightness = np.full_like(total_flux, np.nan, dtype=float)
    total_brightness[valid] = -2.5 * np.log10(total_flux[valid])

    return (
        pd.Series(blue_balance, index=frame.index),
        pd.Series(flux_concentration, index=frame.index),
        pd.Series(total_brightness, index=frame.index),
    )


def _compute_decile_edges(values):
    arr = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.array((-np.inf, np.inf), dtype=float)

    edges = np.quantile(arr, _DECILE_QUANTILES)
    edges = np.unique(np.asarray(edges, dtype=float))

    if edges.size <= 1:
        return np.array((-np.inf, np.inf), dtype=float)

    return np.r_[-np.inf, edges[1:-1], np.inf]


def _to_bin_code(values, edges):
    codes = pd.cut(
        pd.to_numeric(values, errors="coerce"),
        bins=edges,
        labels=False,
        include_lowest=True,
        right=True,
        duplicates="drop",
    )
    return codes.fillna(-1).astype(int).to_numpy()


def _normalize_category(series, train_levels):
    allowed = set(train_levels)
    s = series.astype("string").fillna(_UNK).astype(str)
    return s.where(s.isin(allowed), _UNK).astype(str)


def _lookup_counts(keys, columns, count_dict):
    key_frame = keys[columns]
    if key_frame.shape[1] == 1:
        lookup_keys = key_frame.iloc[:, 0].to_numpy()
    else:
        lookup_keys = pd.MultiIndex.from_frame(key_frame).to_flat_index()

    return pd.Series(lookup_keys).map(count_dict).fillna(0.0).to_numpy(dtype=float)


def add_survey_manifold_rarity(raw, deps, aux):
    idx = raw.index
    n_total = len(raw)

    if n_total == 0:
        return pd.DataFrame(
            {
                "joint_log_density": np.array([], dtype=float),
                "rarity_score": np.array([], dtype=float),
                "interaction_surprise": np.array([], dtype=float),
            },
            index=idx,
        )

    train_mask = _infer_train_mask(raw, aux)
    if len(train_mask) != n_total:
        train_mask = np.ones(n_total, dtype=bool)

    train_df = raw.loc[train_mask]
    if train_df.empty:
        train_df = raw
        train_mask = np.ones(n_total, dtype=bool)

    train_blue, train_conc, train_bright = _compute_photometric_coords(train_df)
    blue_edges = _compute_decile_edges(train_blue)
    conc_edges = _compute_decile_edges(train_conc)
    bright_edges = _compute_decile_edges(train_bright)

    raw_blue, raw_conc, raw_bright = _compute_photometric_coords(raw)
    raw_blue_bin = _to_bin_code(raw_blue, blue_edges)
    raw_conc_bin = _to_bin_code(raw_conc, conc_edges)
    raw_bright_bin = _to_bin_code(raw_bright, bright_edges)

    train_blue_bin = _to_bin_code(train_blue, blue_edges)
    train_conc_bin = _to_bin_code(train_conc, conc_edges)
    train_bright_bin = _to_bin_code(train_bright, bright_edges)

    raw_redshift_bin = _to_bin_code(raw["redshift"], _REDSHIFT_BINS)
    train_redshift_bin = _to_bin_code(train_df["redshift"], _REDSHIFT_BINS)

    train_spec_levels = sorted(_normalize_category(train_df["spectral_type"], []).unique())
    train_gal_levels = sorted(_normalize_category(train_df["galaxy_population"], []).unique())

    raw_spectral = _normalize_category(raw["spectral_type"], train_spec_levels)
    raw_galaxy = _normalize_category(raw["galaxy_population"], train_gal_levels)
    train_spectral = _normalize_category(train_df["spectral_type"], train_spec_levels)
    train_galaxy = _normalize_category(train_df["galaxy_population"], train_gal_levels)

    train_keys = pd.DataFrame(
        {
            "redshift_bin": train_redshift_bin,
            "spectral_type": train_spectral,
            "galaxy_population": train_galaxy,
            "blue_bin": train_blue_bin,
            "flux_concentration_bin": train_conc_bin,
            "total_brightness_bin": train_bright_bin,
        },
        index=train_df.index,
    )

    all_keys = pd.DataFrame(
        {
            "redshift_bin": raw_redshift_bin,
            "spectral_type": raw_spectral,
            "galaxy_population": raw_galaxy,
            "blue_bin": raw_blue_bin,
            "flux_concentration_bin": raw_conc_bin,
            "total_brightness_bin": raw_bright_bin,
        },
        index=idx,
    )

    joint_cols = [
        "redshift_bin",
        "spectral_type",
        "galaxy_population",
        "blue_bin",
        "flux_concentration_bin",
        "total_brightness_bin",
    ]
    tag_cols = ["redshift_bin", "spectral_type", "galaxy_population"]
    photo_cols = ["blue_bin", "flux_concentration_bin", "total_brightness_bin"]

    joint_counts = train_keys[joint_cols].value_counts().to_dict()
    tag_counts = train_keys[tag_cols].value_counts().to_dict()
    photo_counts = train_keys[photo_cols].value_counts().to_dict()

    joint_train_count = _lookup_counts(all_keys, joint_cols, joint_counts)
    tag_count = _lookup_counts(all_keys, tag_cols, tag_counts)
    photo_count = _lookup_counts(all_keys, photo_cols, photo_counts)

    n_train = float(len(train_df))

    redshift_bins = len(_REDSHIFT_BINS) - 1
    blue_bins = max(len(blue_edges) - 1, 1)
    conc_bins = max(len(conc_edges) - 1, 1)
    bright_bins = max(len(bright_edges) - 1, 1)
    spectral_cells = len(train_spec_levels) + 1
    galaxy_cells = len(train_gal_levels) + 1

    k_joint = redshift_bins * spectral_cells * galaxy_cells * blue_bins * conc_bins * bright_bins

    joint_log_density = np.log((joint_train_count + 1.0) / (n_train + k_joint))
    rarity_score = -joint_log_density
    interaction_surprise = np.log((tag_count + 1.0) * (photo_count + 1.0)) - np.log((joint_train_count + 1.0) * n_train)

    return pd.DataFrame(
        {
            "joint_log_density": joint_log_density,
            "rarity_score": rarity_score,
            "interaction_surprise": interaction_surprise,
        },
        index=idx,
    )


FEATURE_GROUPS = [
    {
        "name": "survey_manifold_rarity",
        "fn": add_survey_manifold_rarity,
        "depends_on": [],
        "description": "Encodes manifold-based rarity and surprise from redshift-tag-photo joint frequency, with Laplace-smoothed density estimates.",
    }
]