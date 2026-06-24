import numpy as np
import pandas as pd
from collections import defaultdict

_Z_BIN_EDGES = (-0.01, 0.15, 0.30, 0.50, 0.80, 1.20, 1.80, 2.40, 3.00, 3.80, 4.70, 5.60, 7.05)
_I_BIN_START = 10.0
_I_BIN_STOP = 26.0
_I_BIN_WIDTH = 0.5
_MAD_SCALE = 1.4826
_SPREAD_EPS = 1e-6
_SCORE_CLIP = 10.0
_GLOBAL_KEY = ("__global__",)
_COUNT_FULL = 180
_COUNT_MIN = 120
_NEIGHBOR_FLOOR_RATIO = 0.15
_EPS = 1e-12


def _to_float_array(values):
    return pd.to_numeric(values, errors="coerce").to_numpy(dtype=np.float64)


def _collect_rows_for_cells(offsets, row_order, cell_ids):
    if not cell_ids:
        return np.array([], dtype=np.int64)

    chunks = []
    for cell in cell_ids:
        start = offsets[int(cell)]
        end = offsets[int(cell) + 1]
        if end > start:
            chunks.append(row_order[start:end])

    if not chunks:
        return np.array([], dtype=np.int64)
    if len(chunks) == 1:
        return chunks[0]
    return np.concatenate(chunks)


def _collect_rows_for_region(offsets, row_order, zi_list, i_start, i_stop, i_bins, num_i_bins):
    if i_stop < i_start or not zi_list:
        return np.array([], dtype=np.int64)

    chunks = []
    for zi in zi_list:
        base = int(zi) * int(num_i_bins)
        for ib in range(int(i_start), int(i_stop) + 1):
            start = offsets[base + ib]
            end = offsets[base + ib + 1]
            if end > start:
                chunks.append(row_order[start:end])

    if not chunks:
        return np.array([], dtype=np.int64)
    if len(chunks) == 1:
        return chunks[0]
    return np.concatenate(chunks)


def _principal_vectors(x_whitened):
    if x_whitened.shape[0] < 2 or np.linalg.norm(x_whitened) <= _EPS:
        return np.array([1.0, 0.0, 0.0], dtype=np.float64), np.array([0.0, 1.0, 0.0], dtype=np.float64)

    centered = x_whitened - np.mean(x_whitened, axis=0)
    try:
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
    except np.linalg.LinAlgError:
        return np.array([1.0, 0.0, 0.0], dtype=np.float64), np.array([0.0, 1.0, 0.0], dtype=np.float64)

    if vh.shape[0] < 2:
        return np.array([1.0, 0.0, 0.0], dtype=np.float64), np.array([0.0, 1.0, 0.0], dtype=np.float64)

    v1 = np.array(vh[0], dtype=np.float64)
    v2 = np.array(vh[1], dtype=np.float64)

    v1_norm = np.linalg.norm(v1)
    if v1_norm <= _EPS:
        v1 = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        v1_norm = 1.0
    v1 = v1 / v1_norm

    v2 = v2 - np.dot(v2, v1) * v1
    v2_norm = np.linalg.norm(v2)
    if v2_norm <= _EPS:
        basis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        if abs(v1[0]) > 0.8:
            basis = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        v2 = basis - np.dot(basis, v1) * v1
        v2_norm = np.linalg.norm(v2)
        if v2_norm <= _EPS:
            v2 = np.array([0.0, 1.0, 0.0], dtype=np.float64)
            v2_norm = 1.0
    v2 = v2 / v2_norm

    if v1[np.argmax(np.abs(v1))] < 0.0:
        v1 = -v1
        v2 = -v2

    return v1, v2


def _five_nn_floor(whitened_region, spread):
    n = whitened_region.shape[0]
    default_floor = _NEIGHBOR_FLOOR_RATIO * spread
    if n < 6:
        return max(default_floor, _SPREAD_EPS)

    try:
        from sklearn.neighbors import NearestNeighbors
    except Exception:
        return max(default_floor, _SPREAD_EPS)

    k = min(6, n)
    nn = NearestNeighbors(n_neighbors=k, metric="euclidean")
    nn.fit(whitened_region)
    distances, _ = nn.kneighbors(whitened_region, n_neighbors=k, return_distance=True)
    if distances.shape[1] < 2:
        return max(default_floor, _SPREAD_EPS)

    neighbor_medians = np.median(distances[:, 1:k], axis=1)
    n_cell = np.median(neighbor_medians)
    return max(float(n_cell), float(default_floor))


def _tube_scores(region_coords, target_coords):
    region_coords = np.asarray(region_coords, dtype=np.float64)
    target_coords = np.asarray(target_coords, dtype=np.float64)

    center = np.median(region_coords, axis=0)
    mad = np.median(np.abs(region_coords - center), axis=0)
    scale = _MAD_SCALE * mad
    scale[scale <= _EPS] = 1.0

    region_w = (region_coords - center) / scale
    target_w = (target_coords - center) / scale

    v1, v2 = _principal_vectors(region_w)

    proj_region = region_w @ v1
    residual_region = region_w - np.outer(proj_region, v1)
    d_region = np.linalg.norm(residual_region, axis=1)

    spread = _MAD_SCALE * np.median(np.abs(d_region - np.median(d_region))) + _SPREAD_EPS
    if spread <= _SPREAD_EPS:
        spread = _SPREAD_EPS

    n_cell = _five_nn_floor(region_w, spread)

    proj_target = target_w @ v1
    residual_target = target_w - np.outer(proj_target, v1)
    d_target = np.linalg.norm(residual_target, axis=1)

    abs_score = np.maximum(d_target - n_cell, 0.0) / spread
    signed_score = np.sign(target_w @ v2) * abs_score

    abs_score = np.clip(abs_score, 0.0, _SCORE_CLIP)
    signed_score = np.clip(signed_score, -_SCORE_CLIP, _SCORE_CLIP)
    return abs_score.astype(np.float64), signed_score.astype(np.float64)


def _build_cell_to_key(z_idx, i_idx, nz, ni, counts):
    n_rows = z_idx.shape[0]
    num_cells = nz * ni
    cell_keys = np.empty(num_cells, dtype=object)

    nonempty_cells = np.flatnonzero(counts > 0)
    nonempty_z = nonempty_cells // ni
    nonempty_i = nonempty_cells % ni
    nonempty_count = nonempty_cells.size

    for zi in range(nz):
        row_start = zi * ni
        for ib in range(ni):
            cell_id = row_start + ib
            base_count = int(counts[zi, ib])

            if base_count >= _COUNT_FULL:
                cell_keys[cell_id] = ((zi,), ib, ib)
                continue

            i_lo = ib
            i_hi = ib
            merged_count = base_count
            max_radius = max(ib, ni - 1 - ib)
            for radius in range(1, max_radius + 1):
                lo = ib - radius
                if lo < 0:
                    lo = 0
                hi = ib + radius
                if hi >= ni:
                    hi = ni - 1
                merged_count = int(np.sum(counts[zi, lo:hi + 1]))
                i_lo = lo
                i_hi = hi
                if merged_count >= _COUNT_FULL:
                    break

            if merged_count < _COUNT_FULL:
                z_window = [zi]
                if zi > 0:
                    z_window.append(zi - 1)
                if zi + 1 < nz:
                    z_window.append(zi + 1)
                z_window = tuple(sorted(z_window))
                merged_count = int(np.sum(counts[np.ix_(z_window, np.arange(i_lo, i_hi + 1))]))

                if merged_count < _COUNT_MIN:
                    if n_rows == 0 or nonempty_count == 0:
                        cell_keys[cell_id] = _GLOBAL_KEY
                        continue

                    if base_count > 0 and nonempty_count > 1:
                        cand = nonempty_cells[nonempty_cells != cell_id]
                        if cand.size == 0:
                            cand = nonempty_cells
                    else:
                        cand = nonempty_cells

                    if cand.size == 0:
                        cell_keys[cell_id] = _GLOBAL_KEY
                        continue

                    cand_z = cand // ni
                    cand_i = cand % ni
                    dist = np.abs(cand_z - zi) + np.abs(cand_i - ib)
                    tie_z = np.abs(cand_z - zi)
                    tie_i = np.abs(cand_i - ib)
                    order = np.lexsort((tie_i, tie_z, dist))
                    nearest = cand[order[0]]
                    nearest_zi = int(nearest // ni)
                    nearest_i = int(nearest % ni)
                    cell_keys[cell_id] = ((nearest_zi,), nearest_i, nearest_i)
                    continue

                cell_keys[cell_id] = (z_window, i_lo, i_hi)
                continue

            cell_keys[cell_id] = ((zi,), i_lo, i_hi)

    return cell_keys


def add_error_deconvolved_locus_tube_residuals(raw, deps, aux):
    _ = deps
    _ = aux

    n_rows = len(raw)
    if n_rows == 0:
        return pd.DataFrame(
            {
                "A_abs": pd.Series(dtype=np.float64),
                "A_signed": pd.Series(dtype=np.float64),
                "B_abs": pd.Series(dtype=np.float64),
                "B_signed": pd.Series(dtype=np.float64),
                "g_2_4_3_0": pd.Series(dtype=np.float64),
                "g_3_5_plus": pd.Series(dtype=np.float64),
            },
            index=raw.index,
        )

    z = _to_float_array(raw["redshift"])
    i_band = _to_float_array(raw["i"])

    u = _to_float_array(raw["u"])
    g = _to_float_array(raw["g"])
    r = _to_float_array(raw["r"])
    z1 = _to_float_array(raw["z"])
    c1 = u - g
    c2 = g - r
    c3 = r - i_band
    c4 = i_band - z1

    z_edges = np.asarray(_Z_BIN_EDGES, dtype=np.float64)
    nz = z_edges.shape[0] - 1
    ni = int(round((_I_BIN_STOP - _I_BIN_START) / _I_BIN_WIDTH))
    if ni <= 0:
        ni = 1

    z_bin = np.searchsorted(z_edges, z, side="right") - 1
    z_bin = np.clip(z_bin, 0, nz - 1).astype(np.int64)

    i_bin = np.floor((i_band - _I_BIN_START) / _I_BIN_WIDTH).astype(np.int64)
    i_bin = np.clip(i_bin, 0, ni - 1)

    num_cells = nz * ni
    cell_ids = z_bin * ni + i_bin
    counts = np.bincount(cell_ids, minlength=num_cells).astype(np.int64)
    counts_mat = counts.reshape(nz, ni)

    row_order = np.argsort(cell_ids, kind="mergesort")
    offsets = np.empty(num_cells + 1, dtype=np.int64)
    offsets[0] = 0
    np.cumsum(counts, out=offsets[1:])

    nonempty_cells = np.flatnonzero(counts > 0)

    cell_to_key = _build_cell_to_key(z_bin, i_bin, nz, ni, counts_mat)

    key_to_cells = defaultdict(list)
    for cell in nonempty_cells:
        key_to_cells[cell_to_key[int(cell)]].append(int(cell))

    all_rows = np.arange(n_rows, dtype=np.int64)

    a_abs = np.empty(n_rows, dtype=np.float64)
    a_signed = np.empty(n_rows, dtype=np.float64)
    b_abs = np.empty(n_rows, dtype=np.float64)
    b_signed = np.empty(n_rows, dtype=np.float64)

    coords_a = np.column_stack((c1, c2, c3))
    coords_b = np.column_stack((c2, c3, c4))

    for key, cells in key_to_cells.items():
        target_rows = _collect_rows_for_cells(offsets, row_order, cells)
        if target_rows.size == 0:
            continue

        if len(key) == 1 and key[0] == _GLOBAL_KEY[0]:
            region_rows = all_rows
        else:
            zi_region, i_lo, i_hi = key
            region_rows = _collect_rows_for_region(offsets, row_order, zi_region, i_lo, i_hi, ni, ni)

        if region_rows.size == 0:
            region_rows = all_rows

        region_coords_a = coords_a[region_rows]
        target_coords_a = coords_a[target_rows]
        abs_a, signed_a = _tube_scores(region_coords_a, target_coords_a)

        region_coords_b = coords_b[region_rows]
        target_coords_b = coords_b[target_rows]
        abs_b, signed_b = _tube_scores(region_coords_b, target_coords_b)

        a_abs[target_rows] = abs_a
        a_signed[target_rows] = signed_a
        b_abs[target_rows] = abs_b
        b_signed[target_rows] = signed_b

    z_gate_low = (z >= 2.4) & (z < 3.0)
    z_gate_high = z >= 3.5

    abs_mean = 0.5 * (a_abs + b_abs)
    g_2_4_3_0 = abs_mean * z_gate_low.astype(np.float64)
    g_3_5_plus = abs_mean * z_gate_high.astype(np.float64)

    return pd.DataFrame(
        {
            "A_abs": a_abs,
            "A_signed": a_signed,
            "B_abs": b_abs,
            "B_signed": b_signed,
            "g_2_4_3_0": g_2_4_3_0,
            "g_3_5_plus": g_3_5_plus,
        },
        index=raw.index,
    )


FEATURE_GROUPS = [
    {
        "name": "error_deconvolved_locus_tube_residuals",
        "fn": add_error_deconvolved_locus_tube_residuals,
        "depends_on": [],
        "description": "Constructs adaptive redshift-brightness conditioned deconvolved tube residual and sign features from local color manifolds.",
    }
]