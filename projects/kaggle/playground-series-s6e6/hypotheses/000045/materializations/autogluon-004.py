import numpy as np
import pandas as pd


MAG_COLUMNS = ("u", "g", "r", "i", "z")
COLOR_PAIRS = (
    ("u", "g"),
    ("u", "r"),
    ("u", "i"),
    ("u", "z"),
    ("g", "r"),
    ("g", "i"),
    ("g", "z"),
    ("r", "i"),
    ("r", "z"),
    ("i", "z"),
)
COLOR_NAMES = (
    "u_minus_g",
    "u_minus_r",
    "u_minus_i",
    "u_minus_z",
    "g_minus_r",
    "g_minus_i",
    "g_minus_z",
    "r_minus_i",
    "r_minus_z",
    "i_minus_z",
)
N_QUANTILE_BINS = 32
MIN_BIN_ABSOLUTE = 1500
MIN_BIN_FRACTION = 0.001
MIN_SCALE = 1e-6
RESIDUAL_CLIP = 8.0
MAX_PCA_COMPONENTS = 4
N_SCORE_FEATURES = 3
EIGEN_ABSOLUTE_TOL = 1e-8
EIGEN_RELATIVE_TOL = 1e-6


def _training_mask(raw, aux):
    n_rows = len(raw)

    if isinstance(aux, pd.DataFrame) and len(aux) == n_rows:
        for col in ("is_train", "train", "fit_mask", "is_fit", "fold_train"):
            if col in aux.columns:
                values = aux[col]
                if values.notna().any():
                    return values.astype(bool).to_numpy()

        for col in ("split", "dataset", "source", "row_type"):
            if col in aux.columns:
                lowered = aux[col].astype(str).str.lower()
                mask = lowered.isin(("train", "fit", "training"))
                if mask.any():
                    return mask.to_numpy()

    if "id" in raw.columns:
        ids = pd.to_numeric(raw["id"], errors="coerce")
        finite_ids = ids[np.isfinite(ids)]
        if len(finite_ids) and finite_ids.min() >= 0:
            sorted_ids = finite_ids.sort_values()
            gaps = sorted_ids.diff()
            large_gaps = gaps[gaps > 1]
            if len(large_gaps):
                first_test_id = sorted_ids.loc[large_gaps.index[0]]
                mask = ids < first_test_id
                if mask.any() and (~mask).any():
                    return mask.fillna(False).to_numpy()

    return np.ones(n_rows, dtype=bool)


def _make_color_matrix(frame):
    cols = []
    for left, right in COLOR_PAIRS:
        cols.append(
            pd.to_numeric(frame[left], errors="coerce").to_numpy(dtype=np.float64)
            - pd.to_numeric(frame[right], errors="coerce").to_numpy(dtype=np.float64)
        )
    return np.column_stack(cols)


def _redshift_values(frame):
    return pd.to_numeric(frame["redshift"], errors="coerce").to_numpy(dtype=np.float64)


def _initial_redshift_edges(train_redshift):
    quantiles = np.linspace(0.0, 1.0, N_QUANTILE_BINS + 1)
    edges = np.nanquantile(train_redshift, quantiles)
    edges = np.unique(edges[np.isfinite(edges)])

    if len(edges) < 2:
        center = np.nanmedian(train_redshift)
        if not np.isfinite(center):
            center = 0.0
        edges = np.array([center - 0.5, center + 0.5], dtype=np.float64)

    edges = edges.astype(np.float64)
    edges[0] = -np.inf
    edges[-1] = np.inf
    return edges


def _merge_sparse_bins(train_redshift, edges, min_count):
    finite_edges = list(edges)
    changed = True

    while changed and len(finite_edges) > 2:
        changed = False
        counts = []
        for idx in range(len(finite_edges) - 1):
            left = finite_edges[idx]
            right = finite_edges[idx + 1]
            if idx == len(finite_edges) - 2:
                mask = (train_redshift >= left) & (train_redshift <= right)
            else:
                mask = (train_redshift >= left) & (train_redshift < right)
            counts.append(int(np.sum(mask)))

        sparse = [idx for idx, count in enumerate(counts) if count < min_count]
        if not sparse:
            break

        idx = sparse[0]
        if idx == 0:
            del finite_edges[1]
        elif idx == len(counts) - 1:
            del finite_edges[-2]
        elif counts[idx - 1] <= counts[idx + 1]:
            del finite_edges[idx]
        else:
            del finite_edges[idx + 1]
        changed = True

    return np.asarray(finite_edges, dtype=np.float64)


def _assign_bins(redshift, edges):
    clean = np.asarray(redshift, dtype=np.float64)
    finite_train_mid = np.nanmedian(clean[np.isfinite(clean)]) if np.isfinite(clean).any() else 0.0
    clean = np.where(np.isfinite(clean), clean, finite_train_mid)
    bins = np.searchsorted(edges[1:-1], clean, side="right")
    return np.clip(bins, 0, len(edges) - 2)


def _fit_bin_stats(train_colors, train_redshift, train_bins, n_bins):
    global_mu = np.nanmedian(train_colors, axis=0)
    global_sigma = np.nanmedian(np.abs(train_colors - global_mu), axis=0)
    global_sigma = np.where(np.isfinite(global_sigma) & (global_sigma >= MIN_SCALE), global_sigma, MIN_SCALE)

    stats = []
    populated = []

    for bin_idx in range(n_bins):
        mask = train_bins == bin_idx
        block = train_colors[mask]

        if len(block) == 0:
            stats.append(None)
            continue

        mu = np.nanmedian(block, axis=0)
        sigma = np.nanmedian(np.abs(block - mu), axis=0)
        sigma = np.where(np.isfinite(sigma) & (sigma >= MIN_SCALE), sigma, MIN_SCALE)

        standardized = np.clip((block - mu) / sigma, -RESIDUAL_CLIP, RESIDUAL_CLIP)
        standardized = np.where(np.isfinite(standardized), standardized, 0.0)

        if len(standardized) >= 2:
            _, singular_values, vh = np.linalg.svd(standardized, full_matrices=False)
            eigenvalues = (singular_values * singular_values) / max(len(standardized) - 1, 1)
            if len(eigenvalues):
                threshold = max(EIGEN_ABSOLUTE_TOL, EIGEN_RELATIVE_TOL * float(eigenvalues[0]))
                keep = np.flatnonzero(eigenvalues > threshold)[:MAX_PCA_COMPONENTS]
            else:
                keep = np.array([], dtype=np.int64)
        else:
            eigenvalues = np.array([], dtype=np.float64)
            vh = np.empty((0, train_colors.shape[1]), dtype=np.float64)
            keep = np.array([], dtype=np.int64)

        components = vh[keep].copy() if len(keep) else np.empty((0, train_colors.shape[1]), dtype=np.float64)
        kept_eigenvalues = eigenvalues[keep].copy() if len(keep) else np.empty(0, dtype=np.float64)

        for comp_idx in range(len(components)):
            row = components[comp_idx]
            anchor = int(np.argmax(np.abs(row)))
            if row[anchor] < 0:
                components[comp_idx] = -row

        if len(kept_eigenvalues):
            ridge = EIGEN_RELATIVE_TOL * float(np.mean(kept_eigenvalues))
        else:
            ridge = 0.0

        finite_z = train_redshift[mask]
        finite_z = finite_z[np.isfinite(finite_z)]
        midpoint = float(np.median(finite_z)) if len(finite_z) else float(bin_idx)

        stats.append(
            {
                "mu": mu,
                "sigma": sigma,
                "components": components,
                "eigenvalues": kept_eigenvalues,
                "ridge": ridge,
                "midpoint": midpoint,
            }
        )
        populated.append(bin_idx)

    if not populated:
        stats = [
            {
                "mu": global_mu,
                "sigma": global_sigma,
                "components": np.empty((0, train_colors.shape[1]), dtype=np.float64),
                "eigenvalues": np.empty(0, dtype=np.float64),
                "ridge": 0.0,
                "midpoint": 0.0,
            }
        ]
        populated = [0]

    return stats, populated, global_mu, global_sigma


def _nearest_populated_bin(bin_idx, stats, populated):
    if stats[bin_idx] is not None and len(stats[bin_idx]["components"]) > 0:
        return bin_idx

    populated_with_axes = [idx for idx in populated if stats[idx] is not None and len(stats[idx]["components"]) > 0]
    candidates = populated_with_axes if populated_with_axes else populated
    if not candidates:
        return bin_idx

    if stats[bin_idx] is not None:
        midpoint = stats[bin_idx]["midpoint"]
    else:
        midpoint = float(bin_idx)

    return min(candidates, key=lambda idx: abs(stats[idx]["midpoint"] - midpoint))


def add_pairwise_color_lattice_residuals(raw, deps, aux):
    out_index = raw.index
    required = set(MAG_COLUMNS + ("redshift",))
    missing = [col for col in required if col not in raw.columns]
    if missing:
        return pd.DataFrame(index=out_index)

    colors = _make_color_matrix(raw)
    redshift = _redshift_values(raw)
    fit_mask = _training_mask(raw, aux)

    if not np.any(fit_mask):
        fit_mask = np.ones(len(raw), dtype=bool)

    train_colors = colors[fit_mask]
    train_redshift = redshift[fit_mask]

    finite_train_redshift = train_redshift[np.isfinite(train_redshift)]
    if len(finite_train_redshift) == 0:
        finite_train_redshift = np.array([0.0], dtype=np.float64)
        train_redshift = np.where(np.isfinite(train_redshift), train_redshift, 0.0)

    min_count = max(MIN_BIN_ABSOLUTE, int(np.ceil(MIN_BIN_FRACTION * len(train_colors))))
    edges = _initial_redshift_edges(finite_train_redshift)
    edges = _merge_sparse_bins(train_redshift, edges, min_count)

    train_bins = _assign_bins(train_redshift, edges)
    all_bins = _assign_bins(redshift, edges)
    n_bins = len(edges) - 1

    stats, populated, global_mu, global_sigma = _fit_bin_stats(train_colors, train_redshift, train_bins, n_bins)

    residuals = np.zeros((len(raw), len(COLOR_PAIRS)), dtype=np.float64)
    scores = np.zeros((len(raw), N_SCORE_FEATURES), dtype=np.float64)
    mahalanobis = np.zeros(len(raw), dtype=np.float64)

    fallback_stat = {
        "mu": global_mu,
        "sigma": global_sigma,
        "components": np.empty((0, len(COLOR_PAIRS)), dtype=np.float64),
        "eigenvalues": np.empty(0, dtype=np.float64),
        "ridge": 0.0,
        "midpoint": 0.0,
    }

    for bin_idx in range(n_bins):
        row_mask = all_bins == bin_idx
        if not np.any(row_mask):
            continue

        stat_idx = _nearest_populated_bin(bin_idx, stats, populated)
        stat = stats[stat_idx] if 0 <= stat_idx < len(stats) and stats[stat_idx] is not None else fallback_stat

        block = colors[row_mask]
        standardized = np.clip((block - stat["mu"]) / stat["sigma"], -RESIDUAL_CLIP, RESIDUAL_CLIP)
        standardized = np.where(np.isfinite(standardized), standardized, 0.0)
        residuals[row_mask] = standardized

        components = stat["components"]
        eigenvalues = stat["eigenvalues"]
        if len(components):
            block_scores = standardized @ components.T
            score_width = min(N_SCORE_FEATURES, block_scores.shape[1])
            scores[row_mask, :score_width] = block_scores[:, :score_width]
            mahalanobis[row_mask] = np.sqrt(
                np.sum((block_scores * block_scores) / (eigenvalues + stat["ridge"] + MIN_SCALE), axis=1)
            )

    data = {}
    for idx, name in enumerate(COLOR_NAMES):
        data["resid_" + name] = residuals[:, idx]

    for idx in range(N_SCORE_FEATURES):
        data["pca_resid_score_" + str(idx + 1)] = scores[:, idx]

    data["pca_resid_mahalanobis"] = mahalanobis

    return pd.DataFrame(data, index=out_index)


FEATURE_GROUPS = [
    {
        "name": "pairwise_color_lattice_residuals",
        "fn": add_pairwise_color_lattice_residuals,
        "depends_on": [],
        "description": "Redshift-local robust residuals and intrinsic PCA geometry for the complete ugriz pairwise color lattice.",
    }
]