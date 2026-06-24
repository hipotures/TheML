from __future__ import annotations

import numpy as np
import pandas as pd

_CLASS_LABELS = ("GALAXY", "QSO", "STAR")
_CLASS_TO_INDEX = {"GALAXY": 0, "QSO": 1, "STAR": 2}
_SPECTRAL_CLASS_MAP = {"M": "STAR", "A/F": "STAR", "G/K": "GALAXY", "O/B": "QSO"}
_REDSHIFT_BINS = (-0.01, 0.20, 0.60, 1.05, 1.70, 2.45, 3.60, 5.00, 7.01)
_COLOR_Q_LOW = 0.005
_COLOR_Q_HIGH = 0.995
_COLOR_INTERIOR_BINS = 16
_COLOR_BINS = _COLOR_INTERIOR_BINS + 2
_COLOR_CELLS = _COLOR_BINS ** 3
_ALPHA = 1.0
_BETA = 0.5
_N_MIN = 180
_EPS = 1e-12
_CHUNK_SIZE = 256
_FOLD_CANDIDATES = (
    "fold",
    "fold_id",
    "fold_idx",
    "cv_fold",
    "oof_fold",
    "validation_fold",
    "validation_group",
    "split_id",
)

def _to_float(raw: pd.DataFrame, col: str) -> np.ndarray:
    return pd.to_numeric(raw[col], errors="coerce").to_numpy(dtype=float)

def _infer_class_indices(raw: pd.DataFrame) -> np.ndarray:
    n = len(raw)
    z = _to_float(raw, "redshift")
    class_idx = np.full(n, -1, dtype=np.int8)

    if "spectral_type" in raw.columns:
        spectral = raw["spectral_type"].astype("string").str.strip().str.upper()
        mapped = spectral.map(_SPECTRAL_CLASS_MAP)
        for name, idx in _CLASS_TO_INDEX.items():
            class_idx[mapped == name] = idx

    finite_z = np.isfinite(z)
    unknown = class_idx < 0
    class_idx[unknown & finite_z & (z < 0.20)] = _CLASS_TO_INDEX["STAR"]
    unknown = class_idx < 0
    class_idx[unknown & finite_z & (z >= 2.0)] = _CLASS_TO_INDEX["QSO"]
    unknown = class_idx < 0
    class_idx[unknown] = _CLASS_TO_INDEX["GALAXY"]
    return class_idx

def _extract_fold_series(raw: pd.DataFrame, aux: pd.DataFrame | None) -> pd.Series:
    n = len(raw)
    default = pd.Series([pd.NA] * n, index=raw.index, dtype=object)

    if not isinstance(aux, pd.DataFrame) or aux.empty:
        return default

    raw_index = raw.index
    raw_ids = raw["id"] if "id" in raw.columns else None

    for fold_col in _FOLD_CANDIDATES:
        if fold_col not in aux.columns:
            continue

        candidate: pd.Series | None = None

        if raw_ids is not None and "id" in aux.columns:
            pair = aux[[ "id", fold_col ]].copy()
            pair = pair.dropna(subset=["id", fold_col])
            if not pair.empty:
                mapping = dict(zip(pair["id"].to_numpy(), pair[fold_col].to_numpy(), strict=False))
                candidate = pd.Series(raw_ids.to_numpy(), index=raw_index, dtype=object).map(mapping)

        if candidate is None and len(aux) == n:
            candidate = pd.Series(aux[fold_col].to_numpy(), index=raw_index, dtype=object)

        if candidate is not None:
            mapped = candidate
            if mapped.notna().sum() >= 2 and mapped.dropna().nunique() >= 2:
                return mapped

    return default

def _assign_redshift_bins(z_vals: np.ndarray) -> np.ndarray:
    z_vals = np.asarray(z_vals, dtype=float)
    idx = np.full(len(z_vals), -1, dtype=np.int16)
    finite = np.isfinite(z_vals)
    if not np.any(finite):
        return idx

    clipped = np.clip(z_vals[finite], _REDSHIFT_BINS[0], _REDSHIFT_BINS[-1])
    bin_idx = np.searchsorted(_REDSHIFT_BINS, clipped, side="right") - 1
    bin_idx = np.clip(bin_idx, 0, len(_REDSHIFT_BINS) - 2)
    idx[finite] = bin_idx.astype(np.int16)
    return idx

def _build_i_edges(i_vals: np.ndarray, train_mask: np.ndarray) -> np.ndarray:
    i_vals = np.asarray(i_vals, dtype=float)
    train_mask = np.asarray(train_mask, dtype=bool)

    valid = np.isfinite(i_vals) & train_mask
    if valid.sum() < 16:
        valid = np.isfinite(i_vals)

    selected = i_vals[valid]
    if selected.size == 0:
        lo = -0.5
        hi = 0.5
    else:
        lo = np.floor(np.nanmin(selected) * 2.0) / 2.0
        hi = np.ceil(np.nanmax(selected) * 2.0) / 2.0
        if not (np.isfinite(lo) and np.isfinite(hi)) or lo == hi:
            lo = np.floor(np.nanmin(selected))
            hi = np.ceil(np.nanmax(selected))
            if lo == hi:
                lo -= 0.5
                hi += 0.5

    edges = np.arange(lo, hi + 0.5, 0.5, dtype=float)
    if edges.size < 2:
        edges = np.array((lo, lo + 0.5), dtype=float)
    if edges[-1] <= edges[0]:
        edges = np.array((lo, lo + 0.5), dtype=float)
    return edges

def _assign_i_bins(i_vals: np.ndarray, i_edges: np.ndarray) -> np.ndarray:
    i_vals = np.asarray(i_vals, dtype=float)
    idx = np.full(len(i_vals), -1, dtype=np.int16)
    finite = np.isfinite(i_vals)
    if finite.any() and len(i_edges) >= 2:
        clipped = np.clip(
            i_vals[finite],
            i_edges[0],
            np.nextafter(i_edges[-1], -np.inf),
        )
        idx[finite] = np.searchsorted(i_edges, clipped, side="right") - 1
    return idx

def _build_color_edges(values: np.ndarray, train_mask: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    train_mask = np.asarray(train_mask, dtype=bool)

    valid = np.isfinite(values) & train_mask
    if valid.sum() < 16:
        valid = np.isfinite(values)

    selected = values[valid]
    if selected.size == 0:
        edges = np.linspace(-1.0, 1.0, _COLOR_BINS, dtype=float)
        return edges

    lo = float(np.quantile(selected, _COLOR_Q_LOW))
    hi = float(np.quantile(selected, _COLOR_Q_HIGH))
    if not (np.isfinite(lo) and np.isfinite(hi)) or lo >= hi:
        lo = float(np.nanmin(selected))
        hi = float(np.nanmax(selected))
        if not (np.isfinite(lo) and np.isfinite(hi)) or lo == hi:
            lo = -1.0
            hi = 1.0

    clipped = np.clip(selected, lo, hi)
    probs = np.linspace(0.0, 1.0, _COLOR_BINS, dtype=float)
    edges = np.quantile(clipped, probs, method="linear").astype(float)

    if edges.size != _COLOR_BINS or not np.all(np.isfinite(edges)) or not np.all(np.diff(edges) > 0.0):
        edges = np.linspace(lo, hi, _COLOR_BINS)
    else:
        edges[0] = lo
        edges[-1] = hi

    return edges

def _assign_color_bins(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    idx = np.full(len(values), -1, dtype=np.int16)
    finite = np.isfinite(values)
    if finite.any() and edges.size >= 2:
        clipped = np.clip(values[finite], edges[0], np.nextafter(edges[-1], -np.inf))
        bins = np.searchsorted(edges, clipped, side="right") - 1
        bins = np.clip(bins, 0, _COLOR_BINS - 1)
        idx[finite] = bins.astype(np.int16)
    return idx

def _build_chebyshev_neighbors(radius: int) -> tuple[np.ndarray, np.ndarray]:
    offsets: list[tuple[int, int, int]] = []
    weights: list[float] = []
    for dz in range(-radius, radius + 1):
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                d = max(abs(dx), abs(dy), abs(dz))
                if d <= radius:
                    offsets.append((dz, dy, dx))
                    weights.append(1.0 / (2.0 ** d))

    offset_array = np.array(offsets, dtype=np.int16)
    w = np.asarray(weights, dtype=float)

    neighbor = np.full((_COLOR_CELLS, len(offsets)), -1, dtype=np.int16)
    plane = _COLOR_BINS * _COLOR_BINS

    for cell in range(neighbor.shape[0]):
        c0 = cell // plane
        rem0 = cell - c0 * plane
        c1 = rem0 // _COLOR_BINS
        c2 = rem0 - c1 * _COLOR_BINS

        dzv = c0 + offset_array[:, 0]
        dyv = c1 + offset_array[:, 1]
        dxv = c2 + offset_array[:, 2]

        valid = (
            (dzv >= 0) & (dzv < _COLOR_BINS) &
            (dyv >= 0) & (dyv < _COLOR_BINS) &
            (dxv >= 0) & (dxv < _COLOR_BINS)
        )
        nb = dzv * plane + dyv * _COLOR_BINS + dxv
        nb = np.where(valid, nb, -1).astype(np.int16)
        neighbor[cell] = nb

    return neighbor, w

def _neighbor_average(
    prob_flat: np.ndarray,
    k_idx: np.ndarray,
    i_idx: np.ndarray,
    cell_idx: np.ndarray,
    num_i_bins: int,
    neighbor_idx: np.ndarray,
    neighbor_w: np.ndarray,
) -> np.ndarray:
    n = len(cell_idx)
    if n == 0:
        return np.empty(0, dtype=float)

    out = np.empty(n, dtype=float)
    km = (np.asarray(k_idx, dtype=np.int64) * num_i_bins) + np.asarray(i_idx, dtype=np.int64)
    cell_idx = np.asarray(cell_idx, dtype=np.int64)

    for start in range(0, n, _CHUNK_SIZE):
        end = min(n, start + _CHUNK_SIZE)
        km_sub = km[start:end]
        cell_sub = cell_idx[start:end]
        neigh = neighbor_idx[cell_sub]
        acc = np.zeros(end - start, dtype=float)
        den = np.zeros(end - start, dtype=float)
        rows = km_sub[:, None]

        for j in range(neigh.shape[1]):
            cands = neigh[:, j]
            valid = cands >= 0
            if not np.any(valid):
                continue

            safe = np.where(valid, cands, 0).astype(np.int64)
            probs = prob_flat[rows, safe]
            probs = np.where(valid[:, None], probs, 0.0)
            w = neighbor_w[j]
            acc += probs * w
            den += valid.astype(float) * w

        out[start:end] = np.divide(acc, den, out=np.zeros_like(acc), where=den > 0.0)

    return out

def _score_fold(
    z_vals: np.ndarray,
    u_vals: np.ndarray,
    g_vals: np.ndarray,
    r_vals: np.ndarray,
    i_vals: np.ndarray,
    class_idx: np.ndarray,
    train_mask: np.ndarray,
    target_mask: np.ndarray,
    neigh1: np.ndarray,
    neigh1_w: np.ndarray,
    neigh2: np.ndarray,
    neigh2_w: np.ndarray,
) -> np.ndarray:
    train_mask = np.asarray(train_mask, dtype=bool)
    target_mask = np.asarray(target_mask, dtype=bool)

    target_idx = np.flatnonzero(target_mask)
    n_target = target_idx.size
    if n_target == 0:
        return np.empty((0, len(_CLASS_LABELS)), dtype=float)

    z_idx = _assign_redshift_bins(z_vals)
    i_edges = _build_i_edges(i_vals, train_mask)
    i_idx = _assign_i_bins(i_vals, i_edges)

    c1 = u_vals - g_vals
    c2 = g_vals - r_vals
    c3 = r_vals - i_vals

    e1 = _build_color_edges(c1, train_mask)
    e2 = _build_color_edges(c2, train_mask)
    e3 = _build_color_edges(c3, train_mask)

    b1 = _assign_color_bins(c1, e1)
    b2 = _assign_color_bins(c2, e2)
    b3 = _assign_color_bins(c3, e3)

    b_flat = (b1.astype(np.int64) * (_COLOR_BINS * _COLOR_BINS) + b2.astype(np.int64) * _COLOR_BINS + b3.astype(np.int64))

    num_z_bins = len(_REDSHIFT_BINS) - 1
    num_i_bins = len(i_edges) - 1
    counts = np.zeros((len(_CLASS_LABELS), num_z_bins, num_i_bins, _COLOR_CELLS), dtype=float)

    train_valid = (
        train_mask
        & (z_idx >= 0)
        & (i_idx >= 0)
        & (b_flat >= 0)
    )
    if np.any(train_valid):
        c_train = class_idx[train_valid].astype(np.int64)
        z_train = z_idx[train_valid].astype(np.int64)
        i_train = i_idx[train_valid].astype(np.int64)
        b_train = b_flat[train_valid].astype(np.int64)
        flat = (
            (c_train * num_z_bins + z_train) * num_i_bins + i_train
        ) * _COLOR_CELLS + b_train
        np.add.at(counts.reshape(-1), flat, 1.0)

    n_ckm = counts.sum(axis=3)
    n_km = n_ckm.sum(axis=0)

    n_ck = counts.sum(axis=(2, 3))
    n_k = counts.sum(axis=(0, 2, 3))
    n_cb = n_ck.sum(axis=1)
    n_c = counts.sum(axis=(1, 2, 3))
    n_total = float(counts.sum())

    pi_ckm = (n_ckm + _BETA) / (n_km[None, :, :] + len(_CLASS_LABELS) * _BETA)
    pi_ck = (n_ck + _BETA) / (n_k[None, :] + len(_CLASS_LABELS) * _BETA)
    pi_c = (n_c + _BETA) / (n_total + len(_CLASS_LABELS) * _BETA)

    p_ckmb = (counts + _ALPHA) / (n_ckm[:, :, :, None] + _ALPHA * _COLOR_CELLS)
    p_ckb = (counts.sum(axis=2) + _ALPHA) / (n_ck[:, :, None] + _ALPHA * _COLOR_CELLS)
    p_cb = (counts.sum(axis=(0, 2, 3)) + _ALPHA) / (n_c[:, None] + _ALPHA * _COLOR_CELLS)

    z_t = z_idx[target_idx]
    i_t = i_idx[target_idx]
    b_t = b_flat[target_idx]

    scores = np.empty((n_target, len(_CLASS_LABELS)), dtype=float)

    for c in range(len(_CLASS_LABELS)):
        row_prob = np.full(n_target, 1.0 / _COLOR_CELLS, dtype=float)
        row_pi = np.full(n_target, pi_c[c], dtype=float)

        valid = (z_t >= 0) & (i_t >= 0) & (b_t >= 0)
        if np.any(valid):
            valid_pos = np.flatnonzero(valid)
            z_v = z_t[valid]
            i_v = i_t[valid]
            b_v = b_t[valid]

            nkm_v = n_km[z_v, i_v]
            nckm_v = n_ckm[c, z_v, i_v]
            cell_v = counts[c, z_v, i_v, b_v]
            p_cell = p_ckmb[c, z_v, i_v, b_v]
            use0 = (nkm_v >= _N_MIN) & (cell_v >= _N_MIN)

            if np.any(use0):
                pos0 = valid_pos[use0]
                row_prob[pos0] = p_cell[use0]
                row_pi[pos0] = pi_ckm[c, z_v[use0], i_v[use0]]

            rem_mask = ~use0
            if np.any(rem_mask):
                z_r = z_v[rem_mask]
                i_r = i_v[rem_mask]
                b_r = b_v[rem_mask]
                pos_r = valid_pos[rem_mask]
                nckm_r = nckm_v[rem_mask]

                p1 = _neighbor_average(
                    p_ckmb[c].reshape(num_z_bins * num_i_bins, _COLOR_CELLS),
                    z_r,
                    i_r,
                    b_r,
                    num_i_bins,
                    neigh1,
                    neigh1_w,
                )
                use1 = nckm_r >= _N_MIN
                if np.any(use1):
                    p_idx = pos_r[use1]
                    row_prob[p_idx] = p1[use1]
                    row_pi[p_idx] = pi_ckm[c, z_r[use1], i_r[use1]]

                rem1_mask = ~use1
                if np.any(rem1_mask):
                    z_r2 = z_r[rem1_mask]
                    i_r2 = i_r[rem1_mask]
                    b_r2 = b_r[rem1_mask]
                    pos_r2 = pos_r[rem1_mask]
                    nckm_r2 = nckm_r[rem1_mask]

                    p2 = _neighbor_average(
                        p_ckmb[c].reshape(num_z_bins * num_i_bins, _COLOR_CELLS),
                        z_r2,
                        i_r2,
                        b_r2,
                        num_i_bins,
                        neigh2,
                        neigh2_w,
                    )
                    use2 = nckm_r2 >= _N_MIN
                    if np.any(use2):
                        p_idx2 = pos_r2[use2]
                        row_prob[p_idx2] = p2[use2]
                        row_pi[p_idx2] = pi_ckm[c, z_r2[use2], i_r2[use2]]

                    rem2_mask = ~use2
                    if np.any(rem2_mask):
                        z_r3 = z_r2[rem2_mask]
                        b_r3 = b_r2[rem2_mask]
                        pos_r3 = pos_r2[rem2_mask]
                        nk3 = n_ck[c, z_r3]
                        use3 = nk3 >= _N_MIN

                        if np.any(use3):
                            p_idx3 = pos_r3[use3]
                            row_prob[p_idx3] = p_ckb[c, z_r3[use3], b_r3[use3]]
                            row_pi[p_idx3] = pi_ck[c, z_r3[use3]]

                        rem3_mask = ~use3
                        if np.any(rem3_mask) and n_c[c] >= _N_MIN:
                            p_idx4 = pos_r3[rem3_mask]
                            row_prob[p_idx4] = p_cb[c, b_t[p_idx4]]
                            row_pi[p_idx4] = pi_c[c]

        row_prob = np.clip(row_prob, _EPS, 1.0)
        row_pi = np.clip(row_pi, _EPS, 1.0)
        scores[:, c] = np.log(row_prob) + np.log(row_pi)

    return scores

def add_class_conditional_color_density_posteriors(raw: pd.DataFrame, deps: dict | None, aux: pd.DataFrame | None) -> pd.DataFrame:
    _ = deps
    required = ("u", "g", "r", "i", "redshift")
    missing = [col for col in required if col not in raw.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    index = raw.index
    n = len(raw)

    z = _to_float(raw, "redshift")
    u = _to_float(raw, "u")
    g = _to_float(raw, "g")
    r = _to_float(raw, "r")
    i = _to_float(raw, "i")

    class_idx = _infer_class_indices(raw)

    neigh1, neigh1_w = _build_chebyshev_neighbors(1)
    neigh2, neigh2_w = _build_chebyshev_neighbors(2)

    fold_series = _extract_fold_series(raw, aux)
    unique_folds = [x for x in fold_series.dropna().unique() if pd.notna(x)]

    all_scores = np.full((n, len(_CLASS_LABELS)), np.nan, dtype=float)

    if len(unique_folds) < 2:
        all_scores[:] = _score_fold(
            z,
            u,
            g,
            r,
            i,
            class_idx,
            np.ones(n, dtype=bool),
            np.ones(n, dtype=bool),
            neigh1,
            neigh1_w,
            neigh2,
            neigh2_w,
        )
    else:
        for fold_value in unique_folds:
            target_mask = (fold_series == fold_value).to_numpy(dtype=bool)
            if not target_mask.any():
                continue
            train_mask = ~target_mask
            all_scores[target_mask] = _score_fold(
                z,
                u,
                g,
                r,
                i,
                class_idx,
                train_mask,
                target_mask,
                neigh1,
                neigh1_w,
                neigh2,
                neigh2_w,
            )

        unknown_mask = fold_series.isna().to_numpy(dtype=bool)
        if unknown_mask.any():
            unknown_train = ~unknown_mask
            if not unknown_train.any():
                unknown_train = np.ones(n, dtype=bool)
            all_scores[unknown_mask] = _score_fold(
                z,
                u,
                g,
                r,
                i,
                class_idx,
                unknown_train,
                unknown_mask,
                neigh1,
                neigh1_w,
                neigh2,
                neigh2_w,
            )

    nan_rows = ~np.isfinite(all_scores).all(axis=1)
    if np.any(nan_rows):
        fallback = np.log(1.0 / _COLOR_CELLS) + np.log(1.0 / len(_CLASS_LABELS))
        all_scores[nan_rows, :] = fallback

    s_galaxy = all_scores[:, 0]
    s_qso = all_scores[:, 1]
    s_star = all_scores[:, 2]

    logits = np.column_stack((s_galaxy, s_qso, s_star))
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    ex = np.exp(shifted)
    prob = ex / np.sum(ex, axis=1, keepdims=True)
    entropy = -np.sum(np.clip(prob, _EPS, 1.0) * np.log(np.clip(prob, _EPS, 1.0)), axis=1)

    return pd.DataFrame(
        {
            "s_GALAXY": s_galaxy,
            "s_QSO": s_qso,
            "s_STAR": s_star,
            "s_QSO_minus_STAR": s_qso - s_star,
            "s_STAR_minus_GALAXY": s_star - s_galaxy,
            "s_softmax_entropy": entropy,
        },
        index=index,
    )

FEATURE_GROUPS = [
    {
        "name": "class_conditional_color_density_posteriors",
        "fn": add_class_conditional_color_density_posteriors,
        "depends_on": [],
        "description": "Builds fold-wise, color-space Bayesian class score features using redshift–i stratified Dirichlet-smoothed 3D color-cell posteriors with hierarchical Chebyshev backoff.",
    },
]