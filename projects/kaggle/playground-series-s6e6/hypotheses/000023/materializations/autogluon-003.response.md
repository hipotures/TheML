import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype, is_object_dtype, is_string_dtype

EPS = 1e-6
CHANNEL_NAMES = (
    "u_minus_g",
    "g_minus_r",
    "r_minus_i",
    "i_minus_z",
    "u_minus_i",
    "u_minus_r",
    "g_minus_z",
    "r_minus_z",
    "u_minus_2r_plus_i",
    "g_minus_2i_plus_z",
)
CHANNEL_COUNT = 10


def _coerce_bool_mask(values, n_rows):
    if values is None or len(values) != n_rows:
        return None

    ser = pd.Series(values)

    if is_bool_dtype(ser):
        if ser.isna().all():
            return None
        return ser.fillna(False).to_numpy(dtype=bool)

    if is_numeric_dtype(ser):
        vals = pd.to_numeric(ser, errors="coerce")
        unique = pd.Index(vals.dropna().unique())
        if unique.empty:
            return None
        if unique.isin([0, 1]).all():
            return vals.fillna(0).astype(np.int8).astype(bool).to_numpy()

    if is_object_dtype(ser) or is_string_dtype(ser):
        lowered = ser.dropna().astype(str).str.strip().str.lower()
        unique = set(lowered.unique())
        if not unique:
            return None
        if {"train", "val", "validation", "test"}.intersection(unique):
            if "train" in unique:
                return lowered.eq("train").to_numpy(dtype=bool)
            if "validation" in unique or "val" in unique:
                return lowered.isin({"train", "validation", "val"}).to_numpy(dtype=bool)
        if {"0", "1"}.issubset(unique) and unique.issubset({"0", "1", "true", "false", "t", "f", "yes", "no", "y", "n"}):
            return lowered.isin(["1", "true", "t", "yes", "y"]).to_numpy(dtype=bool)

    return None


def _infer_train_mask(raw, aux):
    n_rows = len(raw)
    if aux is not None and isinstance(aux, pd.DataFrame) and not aux.empty and len(aux) == n_rows:
        preferred_cols = ("is_train", "is_train_mask", "train_mask", "train", "train_indicator", "train_flag")
        for col in preferred_cols:
            if col in aux.columns:
                mask = _coerce_bool_mask(aux[col], n_rows)
                if mask is not None and mask.ndim == 1 and mask.size == n_rows and mask.any():
                    return mask

        for col in aux.columns:
            mask = _coerce_bool_mask(aux[col], n_rows)
            if mask is not None and mask.ndim == 1 and mask.size == n_rows and mask.any():
                return mask

    if "is_train" in raw.columns:
        mask = _coerce_bool_mask(raw["is_train"], n_rows)
        if mask is not None and mask.any():
            return mask

    fallback = np.ones(n_rows, dtype=bool)
    if "id" in raw.columns:
        ids = pd.to_numeric(raw["id"], errors="coerce")
        if ids.notna().all():
            if np.isfinite(ids.min()) and np.isfinite(ids.max()):
                # For datasets with contiguous train ids first, keep the largest likely training block.
                n_unique = ids.nunique(dropna=False)
                if ids.min() >= 0 and float(n_unique) == float(ids.max() - ids.min() + 1):
                    return np.asarray(ids <= ids.quantile(0.7), dtype=bool)

    return fallback


def _build_bin_edges(redshift_train):
    edges = np.quantile(redshift_train, np.linspace(0.0, 1.0, 25))
    edges = np.unique(np.asarray(edges, dtype=np.float64))
    if edges.size < 13:
        zmin = float(np.nanmin(redshift_train))
        zmax = float(np.nanmax(redshift_train))
        if zmin == zmax:
            eps = 1e-6 if zmin == 0.0 else abs(zmin) * 1e-6
            if eps == 0.0:
                eps = 1e-6
            zmin -= eps
            zmax += eps
        edges = np.linspace(zmin, zmax, 25)
        edges = np.unique(edges)

    if edges.size < 2:
        zmin = float(np.nanmin(redshift_train))
        zmax = float(np.nanmax(redshift_train))
        if zmin == zmax:
            eps = 1e-6
            edges = np.array([zmin - eps, zmax + eps], dtype=np.float64)
        else:
            edges = np.array([zmin, zmax], dtype=np.float64)

    return edges


def add_aide_redshift_bin_color_residuals(raw, deps, aux):
    redshift = raw["redshift"].to_numpy(dtype=np.float64, copy=False)
    z0 = np.clip(redshift, 0.0, 7.01)

    train_mask = _infer_train_mask(raw, aux)
    if not np.any(train_mask):
        train_mask = np.ones(len(raw), dtype=bool)

    redshift_train = z0[train_mask]
    if redshift_train.size == 0:
        redshift_train = z0

    edges = _build_bin_edges(redshift_train)
    num_bins = edges.size - 1
    if num_bins < 1:
        edges = np.array([0.0, 7.01], dtype=np.float64)
        num_bins = 1

    bins = np.searchsorted(edges, z0, side="right") - 1
    bins = np.clip(bins, 0, num_bins - 1).astype(np.int32)

    u = raw["u"].to_numpy(dtype=np.float64, copy=False)
    g = raw["g"].to_numpy(dtype=np.float64, copy=False)
    r = raw["r"].to_numpy(dtype=np.float64, copy=False)
    i = raw["i"].to_numpy(dtype=np.float64, copy=False)
    z = raw["z"].to_numpy(dtype=np.float64, copy=False)

    channels = np.column_stack(
        (
            u - g,
            g - r,
            r - i,
            i - z,
            u - i,
            u - r,
            g - z,
            r - z,
            u - 2.0 * r + i,
            g - 2.0 * i + z,
        )
    )

    n_rows = channels.shape[0]
    residual_sum_sq = np.zeros(n_rows, dtype=np.float64)
    residual_sum_abs = np.zeros(n_rows, dtype=np.float64)
    residual_sum = np.zeros(n_rows, dtype=np.float64)
    residual_sum_sq_raw = np.zeros(n_rows, dtype=np.float64)
    standardized_sum = np.zeros(n_rows, dtype=np.float64)
    standardized_maxabs = np.zeros(n_rows, dtype=np.float64)

    train_bins = bins[train_mask]
    for c in range(CHANNEL_COUNT):
        ch = channels[:, c]
        ch_train = ch[train_mask]

        mu_b = np.zeros(num_bins, dtype=np.float64)
        sigma_b = np.zeros(num_bins, dtype=np.float64)
        n_b = np.zeros(num_bins, dtype=np.float64)

        for b in range(num_bins):
            sel = train_bins == b
            if np.any(sel):
                vals = ch_train[sel]
                n = vals.shape[0]
                n_b[b] = n
                mu_b[b] = np.mean(vals)
                sigma_b[b] = np.std(vals, ddof=1) if n > 1 else 0.0
            else:
                mu_b[b] = 0.0
                sigma_b[b] = 0.0

        if ch_train.size > 0:
            mu_all = float(np.mean(ch_train))
            sigma_all = float(np.std(ch_train, ddof=1)) if ch_train.size > 1 else 0.0
        else:
            mu_all = 0.0
            sigma_all = 0.0

        w = np.minimum(1.0, n_b / 400.0)
        mu_tilde = w * mu_b + (1.0 - w) * mu_all
        sigma_tilde = w * sigma_b + (1.0 - w) * sigma_all
        sigma_tilde = np.maximum(sigma_tilde, EPS)

        d = ch - mu_tilde[bins]
        s = d / sigma_tilde[bins]

        residual_sum_sq += d * d
        residual_sum_abs += np.abs(d)
        residual_sum += d
        residual_sum_sq_raw += d * d
        standardized_sum += s
        standardized_maxabs = np.maximum(standardized_maxabs, np.abs(s))

    mean_d = residual_sum / float(CHANNEL_COUNT)
    mean_abs_d = residual_sum_abs / float(CHANNEL_COUNT)
    var_d = residual_sum_sq_raw / float(CHANNEL_COUNT) - mean_d * mean_d
    var_d = np.maximum(var_d, 0.0)
    std_d = np.sqrt(var_d)

    features = pd.DataFrame(
        {
            "aide_redshift_resid_root_sum_sq": np.sqrt(residual_sum_sq),
            "aide_redshift_resid_mean_abs": mean_abs_d,
            "aide_redshift_resid_mean": mean_d,
            "aide_redshift_resid_std": std_d,
            "aide_redshift_resid_std_mean": standardized_sum / float(CHANNEL_COUNT),
            "aide_redshift_resid_max_abs_std": standardized_maxabs,
        },
        index=raw.index,
    )

    return features


FEATURE_GROUPS = [
    {
        "name": "aide_redshift_bin_color_residuals",
        "fn": add_aide_redshift_bin_color_residuals,
        "depends_on": [],
        "description": "Build redshift-conditioned residual and standardized-residual summaries over fixed AIDE color/curvature channels.",
    }
]