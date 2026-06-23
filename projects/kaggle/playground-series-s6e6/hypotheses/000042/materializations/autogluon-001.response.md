import numpy as np
import pandas as pd

REDSHIFT_BIN_EDGES = (-0.01, 0.2, 0.5, 1.0, 1.5, 2.2, 3.0, 4.5, 5.5, 7.05)
REQUIRED_FEATURE_COLUMNS = ("u", "g", "r", "i", "z", "redshift")
I_BIN_WIDTH = 0.4
MIN_CELL_ROWS = 3
NOISE_MIN_ROWS = 120
K_NEIGHBORS = 5
MAX_CELL_KNN_SAMPLE = 1800
MAX_GLOBAL_KNN_SAMPLE = 5000
DECONV_MAX = 25.0
DEFAULT_SPREAD = 1.0
EPS = 1e-9
DEFAULT_V1_A = (1.0, 0.0, 0.0)
DEFAULT_V2_A = (0.0, 1.0, 0.0)
DEFAULT_V1_B = (1.0, 0.0, 0.0)
DEFAULT_V2_B = (0.0, 1.0, 0.0)


def _normalize_vector(vec, fallback):
    v = np.asarray(vec, dtype=float)
    norm = np.linalg.norm(v)
    if not np.isfinite(norm) or norm <= 0:
        return np.array(fallback, dtype=float)
    return v / norm


def _extract_columns(frame, columns):
    if frame is None or not isinstance(frame, pd.DataFrame):
        return None
    out = {}
    for col in columns:
        if col not in frame.columns:
            return None
        out[col] = pd.to_numeric(frame[col], errors="coerce").to_numpy(dtype=float)
    return out


def _build_color_matrices(feature_map):
    if feature_map is None:
        return None
    u = feature_map["u"]
    g = feature_map["g"]
    r = feature_map["r"]
    i = feature_map["i"]
    z = feature_map["z"]
    redshift = feature_map["redshift"]

    c1 = u - g
    c2 = g - r
    c3 = r - i
    c4 = i - z
    valid = np.isfinite(redshift) & np.isfinite(i) & np.isfinite(c1) & np.isfinite(c2) & np.isfinite(c3) & np.isfinite(c4)

    cube_a = np.column_stack((c1, c2, c3))
    cube_b = np.column_stack((c2, c3, c4))

    return {
        "u": u,
        "g": g,
        "r": r,
        "i": i,
        "z": z,
        "redshift": redshift,
        "c1": c1,
        "c2": c2,
        "c3": c3,
        "c4": c4,
        "valid": valid,
        "cube_a": cube_a,
        "cube_b": cube_b,
    }


def _fit_line_basis(points, fallback_v1, fallback_v2):
    pts = np.asarray(points, dtype=float)
    pts = pts[np.isfinite(pts).all(axis=1)]
    if pts.shape[0] == 0:
        return None
    center = np.mean(pts, axis=0)
    if pts.shape[0] < 2:
        return center.astype(float), _normalize_vector(fallback_v1, fallback_v1), _normalize_vector(fallback_v2, fallback_v2)
    centered = pts - center
    try:
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        v1 = _normalize_vector(vt[0], fallback_v1)
        v2 = _normalize_vector(vt[1], fallback_v2) if vt.shape[0] >= 2 else _normalize_vector(fallback_v2, fallback_v2)
        return center.astype(float), v1, v2
    except Exception:
        return center.astype(float), _normalize_vector(fallback_v1, fallback_v1), _normalize_vector(fallback_v2, fallback_v2)


def _line_residual(points, center, v1):
    shifted = np.asarray(points, dtype=float) - center
    proj = shifted @ v1
    residual_vec = shifted - np.outer(proj, v1)
    return np.linalg.norm(residual_vec, axis=1)


def _robust_scale(values, fallback):
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return float(fallback)
    med = np.median(vals)
    mad = np.median(np.abs(vals - med))
    scale = 1.4826 * mad
    if not np.isfinite(scale) or scale <= 0:
        return float(fallback)
    return float(scale)


def _knn_scale(points, k=K_NEIGHBORS, sample_cap=MAX_CELL_KNN_SAMPLE):
    pts = np.asarray(points, dtype=float)
    pts = pts[np.isfinite(pts).all(axis=1)]
    n = pts.shape[0]
    if n < k + 1:
        return np.nan

    if n > sample_cap:
        step = int(np.ceil(float(n) / float(sample_cap)))
        if step < 1:
            step = 1
        pts = pts[::step]
        n = pts.shape[0]
    if n < k + 1:
        return np.nan

    pts = pts - np.median(pts, axis=0)
    diff = pts[:, None, :] - pts[None, :, :]
    dist = np.sqrt(np.einsum("ijk,ijk->ij", diff, diff))
    np.fill_diagonal(dist, np.inf)

    k_here = min(k, n - 1)
    nn = np.partition(dist, k_here, axis=1)[:, 1 : k_here + 1]
    nn_med = np.median(nn, axis=1)

    center = np.median(nn_med)
    spread = np.median(np.abs(nn_med - center))
    if not np.isfinite(spread) or spread <= 0:
        spread = np.median(nn_med)
    return float(spread)


def _collect_cell_rows(rz_bin, i_bin, mask):
    pos = np.flatnonzero(mask)
    if pos.size == 0:
        return {}
    pairs = np.column_stack((rz_bin[mask], i_bin[mask]))
    unique, inv = np.unique(pairs, axis=0, return_inverse=True)
    grouped = {}
    for idx, pair in enumerate(unique):
        grouped[(int(pair[0]), int(pair[1]))] = pos[inv == idx]
    return grouped


def _nearest_populated_cell(target, populated):
    if populated.size == 0:
        return None
    dist = np.abs(populated[:, 0] - target[0]) + np.abs(populated[:, 1] - target[1])
    best = int(np.argmin(dist))
    return (int(populated[best, 0]), int(populated[best, 1]))


def _deconvolved_distance(residual, spread, noise):
    if not (np.isfinite(spread) and spread > 0):
        spread = float(DEFAULT_SPREAD)
    if not (np.isfinite(noise) and noise >= 0):
        noise = 0.0
    return np.maximum(np.asarray(residual, dtype=float) - noise, 0.0) / spread


def add_error_deconvolved_locus_tube_residuals(raw, deps, aux):
    del deps

    raw_cols = _extract_columns(raw, REQUIRED_FEATURE_COLUMNS)
    if raw_cols is None:
        return pd.DataFrame(index=raw.index)

    raw_data = _build_color_matrices(raw_cols)
    if raw_data is None:
        return pd.DataFrame(index=raw.index)

    n_rows = len(raw)
    redshift = raw_data["redshift"]
    i_band = raw_data["i"]
    c1 = raw_data["c1"]
    c2 = raw_data["c2"]
    c3 = raw_data["c3"]
    c4 = raw_data["c4"]

    valid = raw_data["valid"]
    if np.any(valid):
        i_valid = i_band[valid]
        i_min = float(np.floor(np.min(i_valid) / I_BIN_WIDTH) * I_BIN_WIDTH)
        i_max = float(np.ceil(np.max(i_valid) / I_BIN_WIDTH) * I_BIN_WIDTH)
        if i_max <= i_min:
            i_max = i_min + I_BIN_WIDTH
        i_edges = np.arange(i_min, i_max + I_BIN_WIDTH * 1.0001, I_BIN_WIDTH).tolist()
    else:
        i_edges = (0.0, 1.0)

    default_v1_a = np.array(DEFAULT_V1_A, dtype=float)
    default_v2_a = np.array(DEFAULT_V2_A, dtype=float)
    default_v1_b = np.array(DEFAULT_V1_B, dtype=float)
    default_v2_b = np.array(DEFAULT_V2_B, dtype=float)

    global_v1_a = default_v1_a.copy()
    global_v2_a = default_v2_a.copy()
    global_mu_a = np.zeros(3, dtype=float)
    global_v1_b = default_v1_b.copy()
    global_v2_b = default_v2_b.copy()
    global_mu_b = np.zeros(3, dtype=float)
    global_spread_a = float(DEFAULT_SPREAD)
    global_spread_b = float(DEFAULT_SPREAD)
    global_noise_a = float(max(DEFAULT_SPREAD * 0.2, EPS))
    global_noise_b = float(max(DEFAULT_SPREAD * 0.2, EPS))

    source_parts = [raw_data]
    aux_cols = _extract_columns(aux, REQUIRED_FEATURE_COLUMNS) if isinstance(aux, pd.DataFrame) and len(aux) else None
    if aux_cols is not None:
        aux_data = _build_color_matrices(aux_cols)
        if aux_data is not None:
            source_parts.append(aux_data)

    source_u = np.concatenate([part["u"] for part in source_parts], axis=0)
    source_g = np.concatenate([part["g"] for part in source_parts], axis=0)
    source_r = np.concatenate([part["r"] for part in source_parts], axis=0)
    source_i = np.concatenate([part["i"] for part in source_parts], axis=0)
    source_z = np.concatenate([part["z"] for part in source_parts], axis=0)
    source_red = np.concatenate([part["redshift"] for part in source_parts], axis=0)

    source_valid = (
        np.isfinite(source_u)
        & np.isfinite(source_g)
        & np.isfinite(source_r)
        & np.isfinite(source_i)
        & np.isfinite(source_z)
        & np.isfinite(source_red)
    )
    if np.any(source_valid):
        sx1 = source_u[source_valid] - source_g[source_valid]
        sx2 = source_g[source_valid] - source_r[source_valid]
        sx3 = source_r[source_valid] - source_i[source_valid]
        sx4 = source_i[source_valid] - source_z[source_valid]
        cube_a = np.column_stack((sx1, sx2, sx3))
        cube_b = np.column_stack((sx2, sx3, sx4))

        gfit_a = _fit_line_basis(cube_a, default_v1_a, default_v2_a)
        gfit_b = _fit_line_basis(cube_b, default_v1_b, default_v2_b)
        if gfit_a is not None:
            global_mu_a, global_v1_a, global_v2_a = gfit_a
            da = _line_residual(cube_a, global_mu_a, global_v1_a)
            global_spread_a = _robust_scale(da, DEFAULT_SPREAD)
            n_noise_a = _knn_scale(cube_a, sample_cap=MAX_GLOBAL_KNN_SAMPLE)
            if np.isfinite(n_noise_a) and n_noise_a > 0:
                global_noise_a = float(n_noise_a)
        if gfit_b is not None:
            global_mu_b, global_v1_b, global_v2_b = gfit_b
            db = _line_residual(cube_b, global_mu_b, global_v1_b)
            global_spread_b = _robust_scale(db, DEFAULT_SPREAD)
            n_noise_b = _knn_scale(cube_b, sample_cap=MAX_GLOBAL_KNN_SAMPLE)
            if np.isfinite(n_noise_b) and n_noise_b > 0:
                global_noise_b = float(n_noise_b)

    if not np.any(valid):
        zero = np.full(n_rows, np.nan, dtype=float)
        return pd.DataFrame(
            {
                "tubeA_deconvolved_distance": zero,
                "tubeA_signed_deconvolved_distance": zero,
                "tubeB_deconvolved_distance": zero,
                "tubeB_signed_deconvolved_distance": zero,
                "tubeA_deconvolved_distance_2p4_3p0": zero,
                "tubeA_signed_deconvolved_distance_2p4_3p0": zero,
                "tubeB_deconvolved_distance_2p4_3p0": zero,
                "tubeB_signed_deconvolved_distance_2p4_3p0": zero,
                "tubeA_deconvolved_distance_z_gt_3p5": zero,
                "tubeA_signed_deconvolved_distance_z_gt_3p5": zero,
                "tubeB_deconvolved_distance_z_gt_3p5": zero,
                "tubeB_signed_deconvolved_distance_z_gt_3p5": zero,
            },
            index=raw.index,
        )

    rz_codes = pd.cut(redshift, bins=REDSHIFT_BIN_EDGES, include_lowest=True, right=False).cat.codes.to_numpy(dtype=int)
    i_codes = pd.cut(i_band, bins=i_edges, include_lowest=True, right=False).cat.codes.to_numpy(dtype=int)

    in_grid = valid & (rz_codes >= 0) & (i_codes >= 0)
    cell_rows = _collect_cell_rows(rz_codes, i_codes, in_grid)

    cell_stats = {}
    for key, rows in cell_rows.items():
        rows = np.asarray(rows, dtype=int)
        if rows.size < MIN_CELL_ROWS:
            continue

        pts_a = np.column_stack((c1[rows], c2[rows], c3[rows]))
        pts_b = np.column_stack((c2[rows], c3[rows], c4[rows]))

        fit_a = _fit_line_basis(pts_a, global_v1_a, global_v2_a)
        fit_b = _fit_line_basis(pts_b, global_v1_b, global_v2_b)
        if fit_a is None or fit_b is None:
            continue

        mu_a, v1_a, v2_a = fit_a
        mu_b, v1_b, v2_b = fit_b

        if np.dot(v2_a, global_v2_a) < 0:
            v2_a = -v2_a
        if np.dot(v2_b, global_v2_b) < 0:
            v2_b = -v2_b

        da = _line_residual(pts_a, mu_a, v1_a)
        db = _line_residual(pts_b, mu_b, v1_b)

        spread_a = _robust_scale(da, global_spread_a)
        spread_b = _robust_scale(db, global_spread_b)

        if rows.size >= NOISE_MIN_ROWS:
            noise_a = _knn_scale(pts_a, sample_cap=MAX_CELL_KNN_SAMPLE)
            noise_b = _knn_scale(pts_b, sample_cap=MAX_CELL_KNN_SAMPLE)
            if not (np.isfinite(noise_a) and noise_a > 0):
                noise_a = global_noise_a
            if not (np.isfinite(noise_b) and noise_b > 0):
                noise_b = global_noise_b
        else:
            noise_a = global_noise_a
            noise_b = global_noise_b

        cell_stats[key] = {
            "mu_a": mu_a,
            "v1_a": v1_a,
            "v2_a": v2_a,
            "spread_a": float(max(spread_a, EPS)),
            "noise_a": float(max(noise_a, 0.0)),
            "mu_b": mu_b,
            "v1_b": v1_b,
            "v2_b": v2_b,
            "spread_b": float(max(spread_b, EPS)),
            "noise_b": float(max(noise_b, 0.0)),
        }

    occupied = np.array(list(cell_stats.keys()), dtype=int)
    assigned = {}
    for key, rows in cell_rows.items():
        final_key = key
        if key not in cell_stats and occupied.size > 0:
            nearest = _nearest_populated_cell(key, occupied)
            if nearest is not None:
                final_key = nearest
        if final_key in assigned:
            assigned[final_key].append(rows)
        else:
            assigned[final_key] = [rows]

    tubeA = np.full(n_rows, np.nan, dtype=float)
    tubeA_signed = np.full(n_rows, np.nan, dtype=float)
    tubeB = np.full(n_rows, np.nan, dtype=float)
    tubeB_signed = np.full(n_rows, np.nan, dtype=float)

    for key, chunks in assigned.items():
        rows = np.concatenate(chunks)
        rows = np.asarray(rows, dtype=int)

        stats = cell_stats.get(key)
        if stats is None:
            mu_a = global_mu_a
            v1_a = global_v1_a
            v2_a = global_v2_a
            spread_a = global_spread_a
            noise_a = global_noise_a
            mu_b = global_mu_b
            v1_b = global_v1_b
            v2_b = global_v2_b
            spread_b = global_spread_b
            noise_b = global_noise_b
        else:
            mu_a = stats["mu_a"]
            v1_a = stats["v1_a"]
            v2_a = stats["v2_a"]
            spread_a = stats["spread_a"]
            noise_a = stats["noise_a"]
            mu_b = stats["mu_b"]
            v1_b = stats["v1_b"]
            v2_b = stats["v2_b"]
            spread_b = stats["spread_b"]
            noise_b = stats["noise_b"]

        pts_a = np.column_stack((c1[rows], c2[rows], c3[rows]))
        pts_b = np.column_stack((c2[rows], c3[rows], c4[rows]))

        ra = _line_residual(pts_a, mu_a, v1_a)
        rb = _line_residual(pts_b, mu_b, v1_b)

        da = _deconvolved_distance(ra, spread_a, noise_a)
        db = _deconvolved_distance(rb, spread_b, noise_b)

        sign_a = (pts_a - mu_a) @ v2_a
        sign_b = (pts_b - mu_b) @ v2_b
        sign_a[sign_a == 0] = 1.0
        sign_b[sign_b == 0] = 1.0

        tubeA[rows] = da
        tubeB[rows] = db
        tubeA_signed[rows] = da * np.sign(sign_a)
        tubeB_signed[rows] = db * np.sign(sign_b)

    tubeA = np.clip(tubeA, 0.0, DECONV_MAX)
    tubeB = np.clip(tubeB, 0.0, DECONV_MAX)
    tubeA_signed = np.clip(tubeA_signed, -DECONV_MAX, DECONV_MAX)
    tubeB_signed = np.clip(tubeB_signed, -DECONV_MAX, DECONV_MAX)

    in_window_low = np.isfinite(redshift) & (redshift >= 2.4) & (redshift <= 3.0)
    in_window_high = np.isfinite(redshift) & (redshift > 3.5)
    low_mask = in_window_low.astype(float)
    high_mask = in_window_high.astype(float)

    out = pd.DataFrame(
        {
            "tubeA_deconvolved_distance": tubeA,
            "tubeA_signed_deconvolved_distance": tubeA_signed,
            "tubeB_deconvolved_distance": tubeB,
            "tubeB_signed_deconvolved_distance": tubeB_signed,
            "tubeA_deconvolved_distance_2p4_3p0": tubeA * low_mask,
            "tubeA_signed_deconvolved_distance_2p4_3p0": tubeA_signed * low_mask,
            "tubeB_deconvolved_distance_2p4_3p0": tubeB * low_mask,
            "tubeB_signed_deconvolved_distance_2p4_3p0": tubeB_signed * low_mask,
            "tubeA_deconvolved_distance_z_gt_3p5": tubeA * high_mask,
            "tubeA_signed_deconvolved_distance_z_gt_3p5": tubeA_signed * high_mask,
            "tubeB_deconvolved_distance_z_gt_3p5": tubeB * high_mask,
            "tubeB_signed_deconvolved_distance_z_gt_3p5": tubeB_signed * high_mask,
        },
        index=raw.index,
    )

    out["tubeA_deconvolved_distance"] = out["tubeA_deconvolved_distance"].clip(0.0, DECONV_MAX)
    out["tubeB_deconvolved_distance"] = out["tubeB_deconvolved_distance"].clip(0.0, DECONV_MAX)
    out["tubeA_deconvolved_distance_2p4_3p0"] = out["tubeA_deconvolved_distance_2p4_3p0"].clip(0.0, DECONV_MAX)
    out["tubeB_deconvolved_distance_2p4_3p0"] = out["tubeB_deconvolved_distance_2p4_3p0"].clip(0.0, DECONV_MAX)
    out["tubeA_deconvolved_distance_z_gt_3p5"] = out["tubeA_deconvolved_distance_z_gt_3p5"].clip(0.0, DECONV_MAX)
    out["tubeB_deconvolved_distance_z_gt_3p5"] = out["tubeB_deconvolved_distance_z_gt_3p5"].clip(0.0, DECONV_MAX)
    out["tubeA_signed_deconvolved_distance"] = out["tubeA_signed_deconvolved_distance"].clip(-DECONV_MAX, DECONV_MAX)
    out["tubeB_signed_deconvolved_distance"] = out["tubeB_signed_deconvolved_distance"].clip(-DECONV_MAX, DECONV_MAX)
    out["tubeA_signed_deconvolved_distance_2p4_3p0"] = out["tubeA_signed_deconvolved_distance_2p4_3p0"].clip(-DECONV_MAX, DECONV_MAX)
    out["tubeB_signed_deconvolved_distance_2p4_3p0"] = out["tubeB_signed_deconvolved_distance_2p4_3p0"].clip(-DECONV_MAX, DECONV_MAX)
    out["tubeA_signed_deconvolved_distance_z_gt_3p5"] = out["tubeA_signed_deconvolved_distance_z_gt_3p5"].clip(-DECONV_MAX, DECONV_MAX)
    out["tubeB_signed_deconvolved_distance_z_gt_3p5"] = out["tubeB_signed_deconvolved_distance_z_gt_3p5"].clip(-DECONV_MAX, DECONV_MAX)

    return out


FEATURE_GROUPS = [
    {
        "name": "error_deconvolved_locus_tube_residuals",
        "fn": add_error_deconvolved_locus_tube_residuals,
        "depends_on": [],
        "description": "Computes redshift- and i-band-conditioned deconvolved stellar-locus tube residuals in ugri and griz spaces with signed geometry context.",
    }
]