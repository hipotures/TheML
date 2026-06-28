import numpy as np
import pandas as pd


COLOR_CHANNELS = (
    ("ug", "u", "g", 1.0, -1.0, None, 0.0),
    ("gr", "g", "r", 1.0, -1.0, None, 0.0),
    ("ri", "r", "i", 1.0, -1.0, None, 0.0),
    ("iz", "i", "z", 1.0, -1.0, None, 0.0),
    ("ui", "u", "i", 1.0, -1.0, None, 0.0),
    ("ur", "u", "r", 1.0, -1.0, None, 0.0),
    ("gz", "g", "z", 1.0, -1.0, None, 0.0),
    ("rz", "r", "z", 1.0, -1.0, None, 0.0),
    ("u_2r_i", "u", "r", 1.0, -2.0, "i", 1.0),
    ("g_2i_z", "g", "i", 1.0, -2.0, "z", 1.0),
)

N_REDSHIFT_BINS = 24
MIN_USABLE_BINS = 12
SPARSE_BIN_SHRINK_N = 400.0
MIN_STD = 1e-6
STD_CLIP = 8.0
TRAIN_ID_MAX = 577346


def _training_mask(raw, aux):
    if aux is not None and len(aux) == len(raw):
        for name in (
            "fit_mask",
            "train_mask",
            "is_train",
            "is_training",
            "is_fit",
            "fold_train_mask",
        ):
            if name in aux.columns:
                mask = aux[name].to_numpy()
                if mask.dtype == bool:
                    return mask.copy()
                if np.issubdtype(mask.dtype, np.number):
                    return mask.astype(float) > 0.5
                lowered = pd.Series(mask, index=raw.index).astype(str).str.lower()
                return lowered.isin(("1", "true", "train", "fit", "training")).to_numpy()

        for name in ("split", "dataset", "source", "row_role", "role"):
            if name in aux.columns:
                lowered = aux[name].astype(str).str.lower()
                mask = lowered.isin(("train", "fit", "training"))
                if mask.any():
                    return mask.to_numpy()

    if "id" in raw.columns:
        ids = pd.to_numeric(raw["id"], errors="coerce").to_numpy()
        mask = np.isfinite(ids) & (ids <= TRAIN_ID_MAX)
        if mask.any() and not mask.all():
            return mask

    return np.ones(len(raw), dtype=bool)


def _color_matrix(raw):
    u = pd.to_numeric(raw["u"], errors="coerce").to_numpy(dtype=float)
    g = pd.to_numeric(raw["g"], errors="coerce").to_numpy(dtype=float)
    r = pd.to_numeric(raw["r"], errors="coerce").to_numpy(dtype=float)
    i = pd.to_numeric(raw["i"], errors="coerce").to_numpy(dtype=float)
    z = pd.to_numeric(raw["z"], errors="coerce").to_numpy(dtype=float)

    return np.column_stack(
        (
            u - g,
            g - r,
            r - i,
            i - z,
            u - i,
            u - r,
            g - z,
            r - z,
            u - (2.0 * r) + i,
            g - (2.0 * i) + z,
        )
    )


def _redshift_edges(z_train):
    quantiles = np.linspace(0.0, 1.0, N_REDSHIFT_BINS + 1)
    edges = np.unique(np.quantile(z_train, quantiles))

    if len(edges) - 1 < MIN_USABLE_BINS:
        z_min = float(np.nanmin(z_train))
        z_max = float(np.nanmax(z_train))
        if z_max <= z_min:
            z_max = z_min + 1.0
        edges = np.linspace(z_min, z_max, N_REDSHIFT_BINS + 1)

    if len(edges) < 2:
        edges = np.array((0.0, 7.01), dtype=float)

    edges[0] = min(edges[0], 0.0)
    edges[-1] = max(edges[-1], 7.01)
    return edges


def add_aide_redshift_bin_color_residuals(raw, deps, aux):
    z0 = np.clip(pd.to_numeric(raw["redshift"], errors="coerce").to_numpy(dtype=float), 0.0, 7.01)
    colors = _color_matrix(raw)
    train_mask = _training_mask(raw, aux)
    usable_train = train_mask & np.isfinite(z0) & np.all(np.isfinite(colors), axis=1)

    if not usable_train.any():
        usable_train = np.isfinite(z0) & np.all(np.isfinite(colors), axis=1)
    if not usable_train.any():
        usable_train = np.ones(len(raw), dtype=bool)

    z_train = z0[usable_train]
    color_train = colors[usable_train]
    edges = _redshift_edges(z_train)
    n_bins = len(edges) - 1

    bins_all = np.searchsorted(edges, z0, side="right") - 1
    bins_all = np.clip(bins_all, 0, n_bins - 1)

    bins_train = bins_all[usable_train]
    counts = np.bincount(bins_train, minlength=n_bins).astype(float)

    global_mean = np.nanmean(color_train, axis=0)
    global_std = np.nanstd(color_train, axis=0, ddof=1)
    global_std = np.where(np.isfinite(global_std) & (global_std >= MIN_STD), global_std, MIN_STD)

    bin_mean = np.tile(global_mean, (n_bins, 1))
    bin_std = np.tile(global_std, (n_bins, 1))

    for bin_idx in range(n_bins):
        in_bin = bins_train == bin_idx
        if not np.any(in_bin):
            continue
        values = color_train[in_bin]
        bin_mean[bin_idx] = np.nanmean(values, axis=0)
        if values.shape[0] >= 2:
            local_std = np.nanstd(values, axis=0, ddof=1)
            bin_std[bin_idx] = np.where(np.isfinite(local_std) & (local_std >= MIN_STD), local_std, global_std)

    weights = np.minimum(1.0, counts / SPARSE_BIN_SHRINK_N)[:, None]
    shrunk_mean = (weights * bin_mean) + ((1.0 - weights) * global_mean[None, :])
    shrunk_std = (weights * bin_std) + ((1.0 - weights) * global_std[None, :])
    shrunk_std = np.maximum(shrunk_std, MIN_STD)

    local_mean = shrunk_mean[bins_all]
    local_std = shrunk_std[bins_all]
    residual = colors - local_mean
    standardized = np.clip(residual / local_std, -STD_CLIP, STD_CLIP)

    out = pd.DataFrame(index=raw.index)

    for idx, channel in enumerate(COLOR_CHANNELS):
        name = channel[0]
        out[f"{name}_redshift_bin_residual"] = residual[:, idx]
        out[f"{name}_redshift_bin_zresid"] = standardized[:, idx]

    out["color_residual_l2"] = np.sqrt(np.sum(residual * residual, axis=1))
    out["color_residual_abs_mean"] = np.mean(np.abs(residual), axis=1)
    out["color_residual_mean"] = np.mean(residual, axis=1)
    out["color_residual_std"] = np.std(residual, axis=1)
    out["color_zresid_mean"] = np.mean(standardized, axis=1)
    out["color_zresid_std"] = np.std(standardized, axis=1)
    out["color_zresid_abs_max"] = np.max(np.abs(standardized), axis=1)

    return out


FEATURE_GROUPS = [
    {
        "name": "aide_redshift_bin_color_residuals",
        "fn": add_aide_redshift_bin_color_residuals,
        "depends_on": [],
        "description": "Redshift-bin shrunk color and color-curvature residuals against local training photometric profiles.",
    }
]