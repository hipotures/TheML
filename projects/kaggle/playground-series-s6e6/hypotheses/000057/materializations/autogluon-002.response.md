import numpy as np
import pandas as pd


def _sigmoid(x):
    x = np.asarray(x, dtype=float)
    return 1.0 / (1.0 + np.exp(-np.clip(x, -80.0, 80.0)))


def _safe_percentile(values, q, default):
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float(default)
    return float(np.nanpercentile(arr, q))


def _safe_median(values, default):
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float(default)
    return float(np.nanmedian(arr))


def _safe_mad(values, center, default):
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float(default)
    return float(np.nanmedian(np.abs(arr - center)))


def _winsorized_matrix(matrix):
    arr = np.asarray(matrix, dtype=float)
    if arr.ndim != 2 or arr.size == 0:
        return arr.astype(float)
    out = np.array(arr, copy=True)
    for j in range(out.shape[1]):
        col = out[:, j]
        lo = _safe_percentile(col, 5.0, np.nan)
        hi = _safe_percentile(col, 95.0, np.nan)
        if np.isfinite(lo) and np.isfinite(hi) and lo < hi:
            out[:, j] = np.clip(out[:, j], lo, hi)
    return out


def _safe_covariance(matrix):
    arr = np.asarray(matrix, dtype=float)
    if arr.ndim != 2 or arr.shape[1] == 0 or arr.size == 0:
        dim = max(1, int(arr.shape[1])) if arr.ndim == 2 else 1
        return np.eye(dim)
    finite = np.all(np.isfinite(arr), axis=1)
    arr = arr[finite]
    if arr.shape[0] <= 1:
        return np.eye(arr.shape[1])
    centered = arr - np.mean(arr, axis=0, keepdims=True)
    cov = np.dot(centered.T, centered) / float(max(arr.shape[0] - 1, 1))
    if not np.all(np.isfinite(cov)):
        return np.eye(arr.shape[1])
    return cov


def _first_principal_axis(cov):
    c = np.asarray(cov, dtype=float)
    if c.ndim != 2 or c.shape[0] != c.shape[1] or c.size == 0:
        return np.array([1.0, 0.0, 0.0], dtype=float)
    c = 0.5 * (c + c.T)
    if not np.all(np.isfinite(c)):
        return np.array([1.0, 0.0, 0.0], dtype=float)
    try:
        vals, vecs = np.linalg.eigh(c)
    except Exception:
        return np.array([1.0, 0.0, 0.0], dtype=float)
    if vals.size == 0:
        return np.array([1.0, 0.0, 0.0], dtype=float)
    axis = vecs[:, int(np.nanargmax(vals))]
    norm = float(np.linalg.norm(axis))
    if not np.isfinite(norm) or norm <= 0.0:
        return np.array([1.0, 0.0, 0.0], dtype=float)
    return axis / norm


def _mahalanobis_distance(rows, inv_cov):
    x = np.asarray(rows, dtype=float)
    if x.ndim != 2 or x.size == 0:
        return np.zeros(0, dtype=float)
    inv_c = np.asarray(inv_cov, dtype=float)
    out = np.zeros(x.shape[0], dtype=float)
    if not np.all(np.isfinite(inv_c)):
        return out
    finite = np.all(np.isfinite(x), axis=1)
    if not np.any(finite):
        return out
    xm = x[finite]
    d2 = np.einsum("ij,jk,ik->i", xm, inv_c, xm)
    out[finite] = np.sqrt(np.maximum(d2, 0.0))
    return out


def _infer_train_mask(raw, aux):
    n = len(raw)
    if aux is None or not isinstance(aux, pd.DataFrame) or len(aux) != n or aux.empty:
        return np.ones(n, dtype=bool)

    for col in ("is_train", "in_train", "train_mask", "is_training", "train_indicator"):
        if col in aux.columns:
            s = aux[col]
            if s.dtype == bool:
                return np.asarray(s.fillna(False), dtype=bool)
            if pd.api.types.is_numeric_dtype(s):
                vals = pd.to_numeric(s, errors="coerce")
                if vals.isna().all():
                    continue
                uniq = np.sort(vals.dropna().unique())
                if set(uniq.tolist()).issubset({0.0, 1.0}):
                    return np.asarray((vals > 0.5).to_numpy(), dtype=bool)

    for col in ("split", "partition", "dataset"):
        if col in aux.columns:
            s = aux[col]
            if pd.api.types.is_string_dtype(s) or pd.api.types.is_object_dtype(s):
                low = s.astype(str).str.lower()
                return np.asarray(low == "train", dtype=bool)
            vals = pd.to_numeric(s, errors="coerce")
            if vals.isna().all():
                continue
            uniq = np.sort(vals.dropna().unique())
            if set(uniq.tolist()).issubset({0.0, 1.0}):
                return np.asarray((vals > 0.5).to_numpy(), dtype=bool)

    return np.ones(n, dtype=bool)


def add_quasar_stellar_locus_conflict(raw, deps, aux):
    idx = raw.index

    u = pd.to_numeric(raw["u"], errors="coerce").astype(float).to_numpy()
    g = pd.to_numeric(raw["g"], errors="coerce").astype(float).to_numpy()
    r = pd.to_numeric(raw["r"], errors="coerce").astype(float).to_numpy()
    i = pd.to_numeric(raw["i"], errors="coerce").astype(float).to_numpy()
    z = pd.to_numeric(raw["z"], errors="coerce").astype(float).to_numpy()
    redshift = pd.to_numeric(raw["redshift"], errors="coerce").astype(float).to_numpy()

    c_ug = u - g
    c_gr = g - r
    c_ri = r - i
    c_iz = i - z

    train_mask = _infer_train_mask(raw, aux)

    v1 = np.column_stack([c_ug, c_gr, c_ri])
    v2 = np.column_stack([c_gr, c_ri, c_iz])

    finite_v1 = np.isfinite(v1).all(axis=1)
    finite_v2 = np.isfinite(v2).all(axis=1)

    v1_train_rows = finite_v1 & train_mask
    v2_train_rows = finite_v2 & train_mask
    if not np.any(v1_train_rows):
        v1_train_rows = finite_v1
    if not np.any(v2_train_rows):
        v2_train_rows = finite_v2

    v1_train = v1[v1_train_rows]
    v2_train = v2[v2_train_rows]

    v1_train_win = _winsorized_matrix(v1_train)
    v2_train_win = _winsorized_matrix(v2_train)

    mu1 = np.nanmedian(v1_train_win, axis=0)
    mu2 = np.nanmedian(v2_train_win, axis=0)
    if not np.all(np.isfinite(mu1)):
        mu1 = np.array([_safe_median(v1[:, j], 0.0) for j in range(v1.shape[1])], dtype=float)
    if not np.all(np.isfinite(mu2)):
        mu2 = np.array([_safe_median(v2[:, j], 0.0) for j in range(v2.shape[1])], dtype=float)

    p1 = _first_principal_axis(_safe_covariance(v1_train_win))
    p2 = _first_principal_axis(_safe_covariance(v2_train_win))

    v1_center = v1 - mu1
    v2_center = v2 - mu2

    r1 = v1_center - np.outer(np.einsum("ij,j->i", v1_center, p1), p1)
    r2 = v2_center - np.outer(np.einsum("ij,j->i", v2_center, p2), p2)

    c1 = _safe_covariance(r1[v1_train_rows])
    c2 = _safe_covariance(r2[v2_train_rows])

    c1_star = (0.9 * c1 + 0.1 * np.diag(np.diag(c1)) + (0.03 ** 2) * np.eye(3) + 1e-6 * np.eye(3))
    c2_star = (0.9 * c2 + 0.1 * np.diag(np.diag(c2)) + (0.03 ** 2) * np.eye(3) + 1e-6 * np.eye(3))

    try:
        inv_c1 = np.linalg.inv(c1_star)
    except Exception:
        inv_c1 = np.linalg.pinv(c1_star)

    try:
        inv_c2 = np.linalg.inv(c2_star)
    except Exception:
        inv_c2 = np.linalg.pinv(c2_star)

    d1 = np.clip(_mahalanobis_distance(r1, inv_c1), 0.0, 30.0)
    d2 = np.clip(_mahalanobis_distance(r2, inv_c2), 0.0, 30.0)

    train_d1 = d1[np.isfinite(d1) & (train_mask & finite_v1)]
    if train_d1.size == 0:
        train_d1 = d1[np.isfinite(d1)]
    train_d2 = d2[np.isfinite(d2) & (train_mask & finite_v2)]
    if train_d2.size == 0:
        train_d2 = d2[np.isfinite(d2)]

    p50_1 = _safe_median(train_d1, 0.0)
    p50_2 = _safe_median(train_d2, 0.0)
    mad1 = _safe_mad(train_d1, p50_1, 0.0)
    mad2 = _safe_mad(train_d2, p50_2, 0.0)
    t1 = _safe_percentile(train_d1, 95.0, p50_1)
    t2 = _safe_percentile(train_d2, 95.0, p50_2)

    z1 = (d1 - p50_1) / (1.4826 * (mad1 if mad1 > 0 else 1e-6) + 1e-6)
    z2 = (d2 - p50_2) / (1.4826 * (mad2 if mad2 > 0 else 1e-6) + 1e-6)

    o1 = np.clip((d1 - t1) / 4.0, 0.0, 1.0)
    o2 = np.clip((d2 - t2) / 4.0, 0.0, 1.0)

    blue_gate = _sigmoid((-c_ug - 0.05) / 0.20) * _sigmoid((0.45 - c_gr) / 0.18) * _sigmoid((20.5 - i) / 0.9)
    mid_gate = np.exp(-0.5 * (((c_ug - 0.85) ** 2) / (0.18 ** 2) + ((c_gr - 0.20) ** 2) / (0.12 ** 2))) * _sigmoid(
        (1.8 - redshift) / 0.8
    ) * _sigmoid((redshift - 0.4) / 0.35)
    high_gate = np.exp(-0.5 * (((c_ug - 1.55) ** 2) / (0.30 ** 2) + ((c_ri - 0.25) ** 2) / (0.22 ** 2))) * _sigmoid(
        (redshift - 2.2) / 0.4
    ) * _sigmoid((3.9 - redshift) / 1.0) * _sigmoid((21.0 - i) / 1.0)

    q1 = np.exp(
        -0.5
        * (((c_ug - 0.55) ** 2) / (0.12 ** 2) + ((c_gr - 0.10) ** 2) / (0.08 ** 2) + ((c_ri - 0.05) ** 2) / (0.08 ** 2))
    )
    q2 = np.exp(
        -0.5
        * (((c_ug - 1.05) ** 2) / (0.16 ** 2) + ((c_gr - 0.30) ** 2) / (0.10 ** 2) + ((c_ri - 0.12) ** 2) / (0.10 ** 2))
    )
    suppressor = (q1 + q2) * _sigmoid((0.2 - np.abs(redshift - 2.7)) / 0.4) * _sigmoid((20.0 - i) / 0.6)

    blue_gate = np.clip(blue_gate, 0.0, 1.0)
    mid_gate = np.clip(mid_gate, 0.0, 1.0)
    high_gate = np.clip(high_gate, 0.0, 1.0)

    geom_signal = np.clip(0.45 * o1 + 0.35 * o2 + 0.1 * blue_gate + 0.1 * high_gate, 0.0, 1.0)
    mid_bridge = mid_gate
    conflict = np.clip(geom_signal - 0.6 * suppressor + 0.2 * mid_gate, -1.0, 1.0)

    new_features = pd.DataFrame(
        {
            "quasar_locus_d1": d1,
            "quasar_locus_d2": d2,
            "quasar_locus_z1": z1,
            "quasar_locus_z2": z2,
            "quasar_locus_o1": o1,
            "quasar_locus_o2": o2,
            "quasar_locus_blue_gate": blue_gate,
            "quasar_locus_mid_bridge": mid_bridge,
            "quasar_locus_high_gate": high_gate,
            "quasar_locus_suppressor": suppressor,
            "quasar_locus_geom_signal": geom_signal,
            "quasar_locus_conflict": conflict,
        },
        index=idx,
    )

    return new_features.replace([np.inf, -np.inf], np.nan).fillna(0.0)


FEATURE_GROUPS = [
    {
        "name": "quasar_stellar_locus_conflict",
        "fn": add_quasar_stellar_locus_conflict,
        "depends_on": [],
        "description": "Computes redshift-aware quasar locus-conflict geometry and corridor features with orthogonal Mahalanobis outlier scoring and contaminant suppression.",
    }
]