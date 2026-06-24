import numpy as np
import pandas as pd
from functools import lru_cache

REDSHIFT_BINS = (-0.01, 0.20, 0.60, 1.05, 1.70, 2.45, 3.60, 5.00, 7.01)
COLOR_BINS = 16
COLOR_BIN_COUNT = 18
I_BIN_WIDTH = 0.50
ALPHA = 1.0
BETA = 0.5
EPS = 1e-12
TOTAL_COLOR_CELLS = COLOR_BIN_COUNT ** 3
_CLASS_ORDER = ("GALAXY", "QSO", "STAR")
_LABEL_CANDIDATES = (
    "target",
    "label",
    "y",
    "class",
    "class_name",
    "classlabel",
    "truth",
    "truth_label",
    "galaxy_type",
    "objclass",
    "target_class",
)
_FOLD_CANDIDATES = (
    "fold",
    "fold_id",
    "cv_fold",
    "split",
    "split_id",
    "oof_fold",
    "fold_idx",
    "partition",
)


def _normalize_name(value):
    return str(value).strip().lower().replace("_", "").replace(" ", "")


def _find_aux_column(aux, candidates, index):
    if aux is None or not isinstance(aux, pd.DataFrame) or aux.empty:
        return None

    aux_cols = {_normalize_name(c): c for c in aux.columns}
    for cand in candidates:
        key = _normalize_name(cand)
        if key in aux_cols:
            return aux[aux_cols[key]].reindex(index)

    return None


def _parse_class_value(value):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return -1

    if isinstance(value, (np.integer, int, np.floating, float)):
        if np.isfinite(float(value)):
            iv = int(np.round(float(value)))
            if float(iv) == float(value) and iv in (0, 1, 2):
                return iv

    text = str(value).strip().lower()
    if not text:
        return -1

    if text in {"0", "0.0", "0.00", "galaxy", "gal", "gala"}:
        return 0
    if text in {"1", "1.0", "1.00", "qso", "quasar", "qsoid", "agn"}:
        return 1
    if text in {"2", "2.0", "2.00", "star", "stellar", "st"}:
        return 2

    if "galaxy" in text:
        return 0
    if "qso" in text or "quasar" in text or "agn" in text:
        return 1
    if "star" in text:
        return 2

    return -1


def _resolve_class_series(aux, index):
    col = _find_aux_column(aux, _LABEL_CANDIDATES, index)
    if col is None:
        return None

    arr = col.to_numpy(dtype=object, copy=False)
    mapped = np.fromiter((_parse_class_value(v) for v in arr), dtype=np.int16, count=len(arr))
    if np.all(mapped < 0):
        return None

    return pd.Series(mapped, index=index, dtype=np.int16)


def _resolve_fold_series(aux, index):
    col = _find_aux_column(aux, _FOLD_CANDIDATES, index)
    if col is None:
        return pd.Series(np.zeros(len(index), dtype=np.int64), index=index)

    vals = col.reindex(index)
    numeric = pd.to_numeric(vals, errors="coerce")
    if numeric.notna().any():
        num = numeric.astype("Float64").fillna(0.0)
        return num.astype(np.int64).astype(np.int64)

    codes, _ = pd.factorize(vals.astype(object), sort=False)
    codes = np.asarray(codes, dtype=np.int64)
    codes[codes < 0] = 0
    return pd.Series(codes, index=index, dtype=np.int64)


def _assign_redshift_strata(redshift):
    arr = np.asarray(redshift, dtype=np.float64)
    bins = np.array(REDSHIFT_BINS, dtype=np.float64)
    clipped = np.clip(arr, bins[0], bins[-1])
    k = np.searchsorted(bins, clipped, side="right") - 1
    return np.clip(k, 0, len(bins) - 2).astype(np.int16)


def _compute_quantile_edges(values):
    arr = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(arr)
    if finite.sum() < 2:
        return None

    probs = np.linspace(0.005, 0.995, COLOR_BINS + 1)
    return np.quantile(arr[finite], probs)


def _assign_color_bins(values, edges):
    arr = np.asarray(values, dtype=np.float64)
    bins = np.full(arr.shape[0], COLOR_BIN_COUNT // 2, dtype=np.int16)
    finite = np.isfinite(arr)

    if not finite.any() or edges is None or len(edges) != COLOR_BINS + 1:
        bins[~np.isfinite(edges).all()] = 8 if COLOR_BIN_COUNT % 2 == 0 else 0
        bins[~finite] = 8
        return bins

    low = edges[0]
    high = edges[-1]
    if not np.isfinite(low) or not np.isfinite(high) or low == high:
        bins[finite] = COLOR_BIN_COUNT // 2
        bins[arr < low] = 0
        bins[arr > high] = COLOR_BIN_COUNT - 1
        bins[~finite] = COLOR_BIN_COUNT // 2
        return bins

    internal = edges[1:-1]
    x = arr[finite]
    idx = np.searchsorted(internal, x, side="right") + 1
    idx = np.asarray(idx, dtype=np.int16)
    idx[x < low] = 0
    idx[x > high] = COLOR_BIN_COUNT - 1
    bins[finite] = idx
    return bins


@lru_cache(maxsize=1)
def _neighbor_lookup():
    total = TOTAL_COLOR_CELLS
    idx1 = []
    w1 = []
    idx2 = []
    w2 = []

    bb = COLOR_BIN_COUNT
    layer = bb * bb

    for cell in range(total):
        c0 = cell // layer
        rem = cell % layer
        c1 = rem // bb
        c2 = rem % bb

        n1 = []
        ww1 = []
        n2 = []
        ww2 = []

        for d0 in (-2, -1, 0, 1, 2):
            nc0 = c0 + d0
            if nc0 < 0 or nc0 >= bb:
                continue
            for d1 in (-2, -1, 0, 1, 2):
                nc1 = c1 + d1
                if nc1 < 0 or nc1 >= bb:
                    continue
                for d2 in (-2, -1, 0, 1, 2):
                    nc2 = c2 + d2
                    if nc2 < 0 or nc2 >= bb:
                        continue

                    dist = max(abs(d0), abs(d1), abs(d2))
                    if dist > 2:
                        continue

                    nbr = (nc0 * bb + nc1) * bb + nc2
                    weight = 2.0 ** (-dist)
                    if dist <= 1:
                        n1.append(np.int16(nbr))
                        ww1.append(weight)
                    n2.append(np.int16(nbr))
                    ww2.append(weight)

        idx1.append(np.array(n1, dtype=np.int16))
        w1.append(np.array(ww1, dtype=np.float64))
        idx2.append(np.array(n2, dtype=np.int16))
        w2.append(np.array(ww2, dtype=np.float64))

    return tuple(idx1), tuple(w1), tuple(idx2), tuple(w2)


def _assemble_output(index, score_matrix, support_level):
    max_row = np.max(score_matrix, axis=1)
    exp_scores = np.exp(score_matrix - max_row[:, None])
    probs = exp_scores / np.maximum(exp_scores.sum(axis=1, keepdims=True), EPS)
    entropy = -np.sum(probs * np.log(np.maximum(probs, EPS)), axis=1) / np.log(3.0)

    return pd.DataFrame(
        {
            "ccdp_score_galaxy": score_matrix[:, 0],
            "ccdp_score_qso": score_matrix[:, 1],
            "ccdp_score_star": score_matrix[:, 2],
            "ccdp_margin_qso_star": score_matrix[:, 1] - score_matrix[:, 2],
            "ccdp_margin_star_galaxy": score_matrix[:, 2] - score_matrix[:, 0],
            "ccdp_softmax_entropy": entropy,
            "ccdp_support_level": support_level.astype(np.int16),
        },
        index=index,
    )


def add_class_conditional_color_density_posteriors(raw, deps, aux):
    index = raw.index
    n = len(raw)
    if n == 0:
        return pd.DataFrame(index=index)

    required = ("u", "g", "r", "i", "z", "redshift")
    if not all(col in raw.columns for col in required):
        return pd.DataFrame(index=index)

    u = pd.to_numeric(raw["u"], errors="coerce").to_numpy(dtype=np.float64, copy=False)
    g = pd.to_numeric(raw["g"], errors="coerce").to_numpy(dtype=np.float64, copy=False)
    r = pd.to_numeric(raw["r"], errors="coerce").to_numpy(dtype=np.float64, copy=False)
    i = pd.to_numeric(raw["i"], errors="coerce").to_numpy(dtype=np.float64, copy=False)
    redshift = pd.to_numeric(raw["redshift"], errors="coerce").to_numpy(dtype=np.float64, copy=False)

    z_bin = _assign_redshift_strata(redshift)

    folds = _resolve_fold_series(aux, index).to_numpy()
    classes = _resolve_class_series(aux, index)

    if classes is None:
        p_uniform = np.log(1.0 / 3.0)
        scores = np.column_stack((p_uniform, p_uniform, p_uniform)).astype(np.float64)
        support = np.full(n, -1, dtype=np.int16)
        return _assemble_output(index, scores, support)

    class_arr = classes.to_numpy(dtype=np.int16)
    labeled = class_arr >= 0
    if not np.any(labeled):
        p_uniform = np.log(1.0 / 3.0)
        scores = np.column_stack((p_uniform, p_uniform, p_uniform)).astype(np.float64)
        support = np.full(n, -1, dtype=np.int16)
        return _assemble_output(index, scores, support)

    global_counts = np.array(
        [int(np.sum(class_arr[labeled] == c)) for c in range(3)],
        dtype=np.float64,
    )
    global_n = float(np.sum(global_counts))
    global_pi = (global_counts + BETA) / (global_n + 3.0 * BETA)
    base_log_scores = np.log(global_pi + EPS)

    scores = np.tile(base_log_scores, (n, 1))
    support_level = np.full(n, -1, dtype=np.int16)

    neigh1_idx, neigh1_w, neigh2_idx, neigh2_w = _neighbor_lookup()
    unique_folds = np.unique(folds)

    c1_edges_global = _compute_quantile_edges(u - g)
    c2_edges_global = _compute_quantile_edges(g - r)
    c3_edges_global = _compute_quantile_edges(r - i)

    i_global_low = np.nanmin(i)
    i_global_high = np.nanmax(i)
    if not np.isfinite(i_global_low) or not np.isfinite(i_global_high) or i_global_low == i_global_high:
        i_global_low = float(np.nanmin(i[np.isfinite(i)]))
        i_global_high = i_global_low + I_BIN_WIDTH
    n_i_global = int(max(1, int(np.ceil((i_global_high - i_global_low) / I_BIN_WIDTH))))

    for fold_id in unique_folds:
        fold_mask = folds == fold_id
        fold_pos = np.flatnonzero(fold_mask)
        if fold_pos.size == 0:
            continue

        fold_train_pos = fold_pos[labeled[fold_pos]]
        if fold_train_pos.size == 0:
            continue

        i_train = i[fold_train_pos]
        finite_i_train = i_train[np.isfinite(i_train)]
        if finite_i_train.size == 0:
            train_i_low = i_global_low
            train_i_high = i_global_high
        else:
            train_i_low = float(np.nanmin(finite_i_train))
            train_i_high = float(np.nanmax(finite_i_train))
            if not np.isfinite(train_i_low) or not np.isfinite(train_i_high):
                train_i_low = i_global_low
                train_i_high = i_global_high
        if train_i_high <= train_i_low:
            train_i_high = train_i_low + I_BIN_WIDTH
        n_i_bins = int(max(1, int(np.ceil((train_i_high - train_i_low) / I_BIN_WIDTH))))

        c1_edges = _compute_quantile_edges(u[fold_train_pos] - g[fold_train_pos])
        c2_edges = _compute_quantile_edges(g[fold_train_pos] - r[fold_train_pos])
        c3_edges = _compute_quantile_edges(r[fold_train_pos] - i[fold_train_pos])

        if c1_edges is None:
            c1_edges = c1_edges_global
        if c2_edges is None:
            c2_edges = c2_edges_global
        if c3_edges is None:
            c3_edges = c3_edges_global

        i_fold = i[fold_pos]
        i_fold_clip = np.clip(i_fold, train_i_low, train_i_high)
        m_fold = np.floor((i_fold_clip - train_i_low) / I_BIN_WIDTH).astype(np.int32)
        if n_i_bins > 1:
            m_fold = np.clip(m_fold, 0, n_i_bins - 1).astype(np.int16)
        else:
            m_fold = np.zeros_like(m_fold, dtype=np.int16)

        k_fold = z_bin[fold_pos]
        c1_fold = _assign_color_bins(u[fold_pos] - g[fold_pos], c1_edges)
        c2_fold = _assign_color_bins(g[fold_pos] - r[fold_pos], c2_edges)
        c3_fold = _assign_color_bins(r[fold_pos] - i[fold_pos], c3_edges)
        b_fold = ((c1_fold.astype(np.int16) * COLOR_BIN_COUNT + c2_fold.astype(np.int16)) * COLOR_BIN_COUNT + c3_fold.astype(np.int16)).astype(
            np.int16
        )

        i_train = i[fold_train_pos]
        i_train_clip = np.clip(i_train, train_i_low, train_i_high)
        m_train = np.floor((i_train_clip - train_i_low) / I_BIN_WIDTH).astype(np.int32)
        if n_i_bins > 1:
            m_train = np.clip(m_train, 0, n_i_bins - 1).astype(np.int16)
        else:
            m_train = np.zeros_like(m_train, dtype=np.int16)

        k_train = z_bin[fold_train_pos]
        c1_train = _assign_color_bins(u[fold_train_pos] - g[fold_train_pos], c1_edges)
        c2_train = _assign_color_bins(g[fold_train_pos] - r[fold_train_pos], c2_edges)
        c3_train = _assign_color_bins(r[fold_train_pos] - i[fold_train_pos], c3_edges)
        b_train = ((c1_train.astype(np.int16) * COLOR_BIN_COUNT + c2_train.astype(np.int16)) * COLOR_BIN_COUNT + c3_train.astype(np.int16)).astype(
            np.int16
        )

        train_df = pd.DataFrame(
            {
                "c": class_arr[fold_train_pos].astype(np.int16),
                "k": k_train.astype(np.int16),
                "m": m_train.astype(np.int16),
                "b": b_train.astype(np.int16),
            }
        )

        counts_ckmb = train_df.groupby(["c", "k", "m", "b"]).size().to_dict()
        counts_ckm = train_df.groupby(["c", "k", "m"]).size().to_dict()
        counts_km_b = train_df.groupby(["k", "m", "b"]).size().to_dict()
        counts_km = train_df.groupby(["k", "m"]).size().to_dict()
        counts_ck = train_df.groupby(["c", "k"]).size().to_dict()
        counts_cm = train_df.groupby(["c", "m"]).size().to_dict()
        counts_ckb = train_df.groupby(["c", "k", "b"]).size().to_dict()
        counts_cmb = train_df.groupby(["c", "m", "b"]).size().to_dict()
        counts_cb = train_df.groupby(["c", "b"]).size().to_dict()
        counts_c = train_df.groupby("c").size().to_dict()
        counts_k = train_df.groupby("k").size().to_dict()
        counts_m = train_df.groupby("m").size().to_dict()

        n_fold = float(len(train_df))
        if n_fold <= 0:
            continue
        pi_c_fold = np.array(
            [
                (counts_c.get(0, 0) + BETA) / (n_fold + 3.0 * BETA),
                (counts_c.get(1, 0) + BETA) / (n_fold + 3.0 * BETA),
                (counts_c.get(2, 0) + BETA) / (n_fold + 3.0 * BETA),
            ],
            dtype=np.float64,
        )

        for loc, pos in enumerate(fold_pos):
            kk = int(k_fold[loc])
            mm = int(m_fold[loc])
            bb = int(b_fold[loc])

            n_km = int(counts_km.get((kk, mm), 0))
            n_km_b = int(counts_km_b.get((kk, mm, bb), 0))
            n_k = int(counts_k.get(kk, 0))
            n_m = int(counts_m.get(mm, 0))

            mode = -1
            if n_km >= 300 and n_km_b >= 5:
                mode = 0
            else:
                mass1 = 0.0
                for bn in neigh1_idx[bb]:
                    mass1 += float(counts_km_b.get((kk, mm, int(bn)), 0))
                if mass1 >= 30.0:
                    mode = 1
                else:
                    mass2 = 0.0
                    for bn in neigh2_idx[bb]:
                        mass2 += float(counts_km_b.get((kk, mm, int(bn)), 0))
                    if mass2 >= 80.0:
                        mode = 2
                    elif n_k >= 300:
                        mode = 3
                    elif n_m >= 300:
                        mode = 4
                    elif n_fold > 0.0:
                        mode = 5
                    else:
                        mode = -1

            support_level[pos] = np.int16(mode)

            if mode == -1:
                continue

            if mode == 0:
                for c in range(3):
                    n_ckm = float(counts_ckm.get((c, kk, mm), 0))
                    n_c_km_b = float(counts_ckmb.get((c, kk, mm, bb), 0))
                    p_local = (n_c_km_b + ALPHA) / (n_ckm + ALPHA * TOTAL_COLOR_CELLS)
                    pi_support = (n_ckm + BETA) / (n_km + 3.0 * BETA)
                    scores[pos, c] = np.log(np.maximum(p_local, EPS)) + np.log(np.maximum(pi_support, EPS))

            elif mode == 1 or mode == 2:
                if mode == 1:
                    nbs = neigh1_idx[bb]
                    wts = neigh1_w[bb]
                else:
                    nbs = neigh2_idx[bb]
                    wts = neigh2_w[bb]

                num = np.zeros(3, dtype=np.float64)
                den = 0.0
                for nb, wt in zip(nbs, wts):
                    wt = float(wt)
                    den += wt * float(counts_km_b.get((kk, mm, int(nb)), 0))
                    for c in range(3):
                        num[c] += wt * float(counts_ckmb.get((c, kk, mm, int(nb)), 0))
                p_smooth = (num + ALPHA) / (den + ALPHA * TOTAL_COLOR_CELLS)
                for c in range(3):
                    n_ckm = float(counts_ckm.get((c, kk, mm), 0))
                    pi_support = (n_ckm + BETA) / (n_km + 3.0 * BETA)
                    scores[pos, c] = np.log(np.maximum(p_smooth[c], EPS)) + np.log(np.maximum(pi_support, EPS))

            elif mode == 3:
                for c in range(3):
                    n_c_k = float(counts_ck.get((c, kk), 0))
                    n_c_k_b = float(counts_ckb.get((c, kk, bb), 0))
                    p_k = (n_c_k_b + ALPHA) / (n_c_k + ALPHA * TOTAL_COLOR_CELLS)
                    pi_support = (n_c_k + BETA) / (n_k + 3.0 * BETA)
                    scores[pos, c] = np.log(np.maximum(p_k, EPS)) + np.log(np.maximum(pi_support, EPS))

            elif mode == 4:
                for c in range(3):
                    n_c_m = float(counts_cm.get((c, mm), 0))
                    n_c_m_b = float(counts_cmb.get((c, mm, bb), 0))
                    p_m = (n_c_m_b + ALPHA) / (n_c_m + ALPHA * TOTAL_COLOR_CELLS)
                    pi_support = (n_c_m + BETA) / (n_m + 3.0 * BETA)
                    scores[pos, c] = np.log(np.maximum(p_m, EPS)) + np.log(np.maximum(pi_support, EPS))

            else:
                for c in range(3):
                    n_c = float(counts_c.get(c, 0))
                    n_c_b = float(counts_cb.get((c, bb), 0))
                    p_glob = (n_c_b + ALPHA) / (n_c + ALPHA * TOTAL_COLOR_CELLS)
                    pi_support = pi_c_fold[c]
                    scores[pos, c] = np.log(np.maximum(p_glob, EPS)) + np.log(np.maximum(pi_support, EPS))

    return _assemble_output(index, scores, support_level)


FEATURE_GROUPS = [
    {
        "name": "class_conditional_color_density_posteriors",
        "fn": add_class_conditional_color_density_posteriors,
        "depends_on": [],
        "description": "Fold-aware Bayesian class posterior features from 3D color-cell histograms with support-aware fallback across redshift–i strata.",
    },
]