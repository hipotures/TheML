import numpy as np
import pandas as pd

GROUP_NAME = "class_conditional_color_density_posteriors"
REDSHIFT_BINS = (-0.01, 0.2, 0.7, 1.3, 2.0, 3.2, 4.8, 7.01)
IBIN_WIDTH = 0.50
COLOR_BIN_COUNT = 16
CLASS_ALPHA = 1.0
CLASS_BETA = 0.5
N_MIN = 200
EPSILON = 1e-12
_CHEB_CACHE = {}

def _to_float_array(values, fallback=0.0):
    arr = pd.to_numeric(values, errors="coerce").to_numpy(dtype=np.float64)
    if arr.size == 0:
        return arr
    finite = np.isfinite(arr)
    if np.any(~finite):
        if np.any(finite):
            fill = float(np.nanmedian(arr[finite]))
        else:
            fill = float(fallback)
        arr[~finite] = fill
    return arr

def _coerce_bool_mask(values):
    ser = pd.Series(values)
    if pd.api.types.is_bool_dtype(ser):
        return ser.fillna(False).astype(bool).to_numpy()
    if pd.api.types.is_numeric_dtype(ser):
        return ser.fillna(0).astype(int).astype(bool).to_numpy()
    lowered = ser.astype("string").str.lower().str.strip().fillna("")
    return ((lowered == "true") | (lowered == "train") | (lowered == "1") | (lowered == "yes")).to_numpy()

def _derive_fit_mask(raw, aux):
    n = len(raw)
    if n == 0:
        return np.array([], dtype=bool)
    if not isinstance(aux, pd.DataFrame) or aux.empty:
        return np.ones(n, dtype=bool)

    for col in ("is_train", "train", "train_mask", "is_training", "train_row"):
        if col in aux.columns:
            if aux[col].notna().any():
                return _coerce_bool_mask(aux[col])

    for col in ("is_valid", "is_validation", "is_test", "valid", "val"):
        if col in aux.columns:
            if aux[col].notna().any():
                return ~_coerce_bool_mask(aux[col])

    if "split" in aux.columns:
        split = (
            aux["split"]
            .astype("string")
            .str.lower()
            .str.strip()
            .fillna("")
        )
        if split.notna().any():
            return (split == "train").to_numpy()

    return np.ones(n, dtype=bool)

def _derive_class_indices(raw):
    n = len(raw)
    class_idx = np.full(n, 2, dtype=np.int8)  # STAR default
    if n == 0 or "spectral_type" not in raw.columns:
        return class_idx

    spectral = raw["spectral_type"].astype("string").str.upper().str.strip().fillna("")
    class_idx[spectral == "G/K"] = 0  # GALAXY
    class_idx[spectral == "O/B"] = 1  # QSO

    if "galaxy_population" in raw.columns:
        galaxy_pop = raw["galaxy_population"].astype("string").str.upper().str.strip().fillna("")
        class_idx[(spectral == "A/F") & (galaxy_pop == "RED_SEQUENCE")] = 0
        class_idx[(spectral == "") & (galaxy_pop == "RED_SEQUENCE")] = 0

    return class_idx

def _assign_redshift_bin(redshift):
    edges = np.asarray(REDSHIFT_BINS, dtype=np.float64)
    return np.clip(np.searchsorted(edges, redshift, side="right") - 1, 0, len(edges) - 2)

def _assign_i_band_bin(i_band, fit_mask):
    if fit_mask.size == 0 or not np.any(fit_mask):
        fit_vals = i_band
    else:
        fit_vals = i_band[fit_mask]

    finite = np.isfinite(fit_vals)
    if np.any(finite):
        i_min = float(np.min(fit_vals[finite]))
        i_max = float(np.max(fit_vals[finite]))
    else:
        finite_global = np.isfinite(i_band)
        if np.any(finite_global):
            i_min = float(np.min(i_band[finite_global]))
            i_max = float(np.max(i_band[finite_global]))
        else:
            i_min = 0.0
            i_max = 0.0

    n_bins = int(np.floor((i_max - i_min) / IBIN_WIDTH)) + 1
    if n_bins < 1:
        n_bins = 1

    i_bin = np.floor((i_band - i_min) / IBIN_WIDTH).astype(np.int64)
    i_bin = np.clip(i_bin, 0, n_bins - 1)
    return i_bin, n_bins

def _quantile_with_overflow_bins(values, fit_mask):
    values_fit = values[fit_mask] if fit_mask.size and np.any(fit_mask) else values
    finite_fit = np.isfinite(values_fit)
    if np.any(finite_fit):
        values_fit = values_fit[finite_fit]

    if values_fit.size == 0:
        lo = -1.0
        hi = 1.0
    else:
        lo = float(np.nanpercentile(values_fit, 0.5))
        hi = float(np.nanpercentile(values_fit, 99.5))
        if not np.isfinite(lo) or not np.isfinite(hi) or (hi <= lo):
            finite_all = np.isfinite(values)
            if np.any(finite_all):
                lo = float(np.min(values[finite_all]))
                hi = float(np.max(values[finite_all]))
            if not np.isfinite(lo) or not np.isfinite(hi) or (hi <= lo):
                lo = 0.0
                hi = 1.0

    clipped = np.clip(values, lo, hi)
    quant_probs = np.linspace(0.0, 1.0, COLOR_BIN_COUNT + 1)
    q = np.quantile(clipped, quant_probs)
    q = np.asarray(q, dtype=np.float64)

    for i in range(1, q.size):
        if not np.isfinite(q[i]) or q[i] <= q[i - 1]:
            q[i] = q[i - 1] + max(1e-12, 1e-9 * (i + 1))

    if q.size < COLOR_BIN_COUNT + 1:
        q = np.linspace(q[0], q[-1], COLOR_BIN_COUNT + 1)

    return np.concatenate((np.array([-np.inf], dtype=np.float64), q, np.array([np.inf], dtype=np.float64)))

def _digitize_color(value, edges):
    bins = len(edges) - 1
    idx = np.digitize(value, edges[1:-1], right=True)
    return np.clip(idx, 0, bins - 1).astype(np.int64)

def _get_chebyshev_neighbors(n_bins):
    cached = _CHEB_CACHE.get(n_bins)
    if cached is not None:
        return cached

    total = n_bins ** 3
    x = np.arange(total) // (n_bins * n_bins)
    y = (np.arange(total) // n_bins) % n_bins
    z = np.arange(total) % n_bins

    n1 = []
    w1 = []
    n2 = []
    w2 = []

    for cx, cy, cz in zip(x, y, z):
        neigh1 = []
        weight1 = []
        neigh2 = []
        weight2 = []
        for dx in (-2, -1, 0, 1, 2):
            nx = cx + dx
            if nx < 0 or nx >= n_bins:
                continue
            for dy in (-2, -1, 0, 1, 2):
                ny = cy + dy
                if ny < 0 or ny >= n_bins:
                    continue
                for dz in (-2, -1, 0, 1, 2):
                    nz = cz + dz
                    if nz < 0 or nz >= n_bins:
                        continue
                    r = max(abs(dx), abs(dy), abs(dz))
                    if r > 2:
                        continue
                    cell = (nx * n_bins + ny) * n_bins + nz
                    weight = 0.5 ** r
                    neigh2.append(cell)
                    weight2.append(weight)
                    if r <= 1:
                        neigh1.append(cell)
                        weight1.append(weight)

        n1.append(np.array(neigh1, dtype=np.int32))
        w1.append(np.array(weight1, dtype=np.float64))
        n2.append(np.array(neigh2, dtype=np.int32))
        w2.append(np.array(weight2, dtype=np.float64))

    max1 = max(len(v) for v in n1)
    max2 = max(len(v) for v in n2)

    neigh1_idx = np.full((total, max1), -1, dtype=np.int32)
    neigh1_w = np.zeros((total, max1), dtype=np.float64)
    neigh2_idx = np.full((total, max2), -1, dtype=np.int32)
    neigh2_w = np.zeros((total, max2), dtype=np.float64)

    for i in range(total):
        a1 = n1[i]
        neigh1_idx[i, :len(a1)] = a1
        neigh1_w[i, :len(a1)] = w1[i]

        a2 = n2[i]
        neigh2_idx[i, :len(a2)] = a2
        neigh2_w[i, :len(a2)] = w2[i]

    neigh1_sum = neigh1_w.sum(axis=1)
    neigh2_sum = neigh2_w.sum(axis=1)

    _CHEB_CACHE[n_bins] = (neigh1_idx, neigh1_w, neigh1_sum, neigh2_idx, neigh2_w, neigh2_sum)
    return _CHEB_CACHE[n_bins]

def _weighted_prob_and_mass(class_counts, neigh_idx, neigh_w, neigh_sum):
    n_strata, n_cells = class_counts.shape
    totals = class_counts.sum(axis=1)
    denom = totals + CLASS_ALPHA * n_cells
    prob = np.zeros((n_strata, n_cells), dtype=np.float64)
    mass = np.zeros((n_strata, n_cells), dtype=np.float64)

    if n_strata == 0:
        return prob, mass

    valid = denom > 0

    for cell in range(n_cells):
        ids = neigh_idx[cell]
        active = ids >= 0
        if not np.any(active):
            continue
        ids = ids[active].astype(np.int64)
        weights = neigh_w[cell, active]
        wsum = neigh_sum[cell]
        if wsum <= 0:
            continue

        loc = class_counts[:, ids]
        mass[:, cell] = np.dot(loc, weights)
        weighted_num = np.dot(loc + CLASS_ALPHA, weights)
        if np.any(valid):
            prob[valid, cell] = (weighted_num[valid] / wsum) / denom[valid]

    return prob, mass

def add_class_conditional_color_density_posteriors(raw, deps, aux):
    n = len(raw)
    index = raw.index

    if n == 0:
        return pd.DataFrame(index=index)

    required = ("redshift", "u", "g", "r", "i", "spectral_type", "galaxy_population")
    if any(col not in raw.columns for col in required):
        zeros = np.zeros(n, dtype=np.float64)
        return pd.DataFrame(
            {
                "s_GALAXY": zeros,
                "s_QSO": zeros,
                "s_STAR": zeros,
                "s_QSO_minus_STAR": zeros,
                "s_STAR_minus_GALAXY": zeros,
                "s_entropy": zeros,
            },
            index=index,
        )

    redshift = _to_float_array(raw["redshift"], fallback=0.0)
    u = _to_float_array(raw["u"], fallback=0.0)
    g = _to_float_array(raw["g"], fallback=0.0)
    r = _to_float_array(raw["r"], fallback=0.0)
    i_band = _to_float_array(raw["i"], fallback=0.0)

    fit_mask = _derive_fit_mask(raw, aux)
    class_idx = _derive_class_indices(raw)

    red_bin = _assign_redshift_bin(redshift)
    i_bin, n_i_bins = _assign_i_band_bin(i_band, fit_mask)

    c0 = u - g
    c1 = g - r
    c2 = r - i_band

    edges0 = _quantile_with_overflow_bins(c0, fit_mask)
    edges1 = _quantile_with_overflow_bins(c1, fit_mask)
    edges2 = _quantile_with_overflow_bins(c2, fit_mask)

    b0 = _digitize_color(c0, edges0)
    b1 = _digitize_color(c1, edges1)
    b2 = _digitize_color(c2, edges2)

    n_color_bins = len(edges0) - 1
    n_color_cells = n_color_bins ** 3

    color_cell = ((b0 * n_color_bins + b1) * n_color_bins + b2).astype(np.int64)

    n_red_bins = len(REDSHIFT_BINS) - 1
    n_strata = n_red_bins * n_i_bins
    stratum = red_bin.astype(np.int64) * n_i_bins + i_bin

    class_counts = np.zeros((3, n_strata, n_color_cells), dtype=np.float64)

    for c in range(3):
        cmask = class_idx == c
        if not np.any(cmask):
            continue
        key = stratum[cmask] * n_color_cells + color_cell[cmask]
        counts = np.bincount(key, minlength=n_strata * n_color_cells)
        class_counts[c] = counts.reshape(n_strata, n_color_cells)

    class_tot_by_strata = class_counts.sum(axis=2)
    total_by_strata = class_tot_by_strata.sum(axis=0)

    pi_strata = np.zeros((3, n_strata), dtype=np.float64)
    den_pi = total_by_strata + 3.0 * CLASS_BETA
    valid_pi = den_pi > 0
    if np.any(valid_pi):
        pi_strata[:, valid_pi] = (class_tot_by_strata[:, valid_pi] + CLASS_BETA) / den_pi[valid_pi]

    class_tot_global = class_tot_by_strata.sum(axis=1)
    global_total = class_tot_global.sum()
    if global_total > 0:
        pi_global = (class_tot_global + CLASS_BETA) / (global_total + 3.0 * CLASS_BETA)
    else:
        pi_global = np.array([1.0 / 3.0] * 3, dtype=np.float64)

    counts_red = class_counts.reshape(3, n_red_bins, n_i_bins, n_color_cells).sum(axis=2)

    p_red = np.zeros_like(counts_red, dtype=np.float64)
    for c in range(3):
        row_tot = counts_red[c].sum(axis=1)
        den_row = row_tot + CLASS_ALPHA * n_color_cells
        valid_row = den_row > 0
        if np.any(valid_row):
            p_red[c, valid_row] = (counts_red[c, valid_row] + CLASS_ALPHA) / den_row[valid_row, None]

    global_counts = class_counts.sum(axis=1)
    p_global = np.zeros_like(global_counts, dtype=np.float64)
    for c in range(3):
        gtot = global_counts[c].sum()
        if gtot > 0:
            p_global[c] = (global_counts[c] + CLASS_ALPHA) / (gtot + CLASS_ALPHA * n_color_cells)

    neigh1_idx, neigh1_w, neigh1_sum, neigh2_idx, neigh2_w, neigh2_sum = _get_chebyshev_neighbors(n_color_bins)

    local_prob = []
    neigh1_prob = []
    neigh1_mass = []
    neigh2_prob = []
    neigh2_mass = []

    for c in range(3):
        cc = class_counts[c]
        cc_tot = cc.sum(axis=1)
        den = cc_tot + CLASS_ALPHA * n_color_cells
        local = np.zeros((n_strata, n_color_cells), dtype=np.float64)
        valid = den > 0
        if np.any(valid):
            local[valid] = (cc[valid] + CLASS_ALPHA) / den[valid, None]

        p1, m1 = _weighted_prob_and_mass(cc, neigh1_idx, neigh1_w, neigh1_sum)
        p2, m2 = _weighted_prob_and_mass(cc, neigh2_idx, neigh2_w, neigh2_sum)

        local_prob.append(local)
        neigh1_prob.append(p1)
        neigh1_mass.append(m1)
        neigh2_prob.append(p2)
        neigh2_mass.append(m2)

    scores = np.zeros((n, 3), dtype=np.float64)

    for c in range(3):
        class_cell_counts = class_counts[c][stratum, color_cell]
        class_strata_counts = class_tot_by_strata[c][stratum]
        strata_counts = total_by_strata[stratum]

        p = local_prob[c][stratum, color_cell]
        pi = pi_strata[c][stratum]

        use_global_prior = np.zeros(n, dtype=bool)
        fallback = (strata_counts < N_MIN) | (class_cell_counts <= 0.0) | (class_strata_counts < 1.0)

        if np.any(fallback):
            m1_row = neigh1_mass[c][stratum, color_cell]
            p1_row = neigh1_prob[c][stratum, color_cell]
            can_backoff1 = fallback & (m1_row > 0.0)
            p[can_backoff1] = p1_row[can_backoff1]
            fallback = fallback & ~can_backoff1

            if np.any(fallback):
                m2_row = neigh2_mass[c][stratum, color_cell]
                p2_row = neigh2_prob[c][stratum, color_cell]
                can_backoff2 = fallback & (m2_row > 0.0)
                p[can_backoff2] = p2_row[can_backoff2]
                fallback = fallback & ~can_backoff2

                if np.any(fallback):
                    red_counts_row = counts_red[c][red_bin, color_cell]
                    pz_row = p_red[c][red_bin, color_cell]
                    can_backoff_red = fallback & (red_counts_row > 0.0)
                    p[can_backoff_red] = pz_row[can_backoff_red]
                    fallback = fallback & ~can_backoff_red

                    if np.any(fallback):
                        global_counts_row = global_counts[c][color_cell]
                        pg_row = p_global[c][color_cell]
                        can_backoff_global = fallback & (global_counts_row > 0.0)
                        p[can_backoff_global] = pg_row[can_backoff_global]
                        fallback = fallback & ~can_backoff_global

                        if np.any(fallback):
                            p[fallback] = 1.0
                            use_global_prior[fallback] = True

        if np.any(use_global_prior):
            pi[use_global_prior] = pi_global[c]

        scores[:, c] = np.log(np.maximum(pi, EPSILON)) + np.log(np.maximum(p, EPSILON))

    max_score = np.max(scores, axis=1, keepdims=True)
    exp_score = np.exp(scores - max_score)
    prob = exp_score / np.maximum(np.sum(exp_score, axis=1, keepdims=True), EPSILON)
    entropy = -np.sum(prob * np.log(np.maximum(prob, EPSILON)), axis=1) / np.log(3.0)

    return pd.DataFrame(
        {
            "s_GALAXY": scores[:, 0],
            "s_QSO": scores[:, 1],
            "s_STAR": scores[:, 2],
            "s_QSO_minus_STAR": scores[:, 1] - scores[:, 2],
            "s_STAR_minus_GALAXY": scores[:, 2] - scores[:, 0],
            "s_entropy": entropy,
        },
        index=index,
    )

FEATURE_GROUPS = [
    {
        "name": GROUP_NAME,
        "fn": add_class_conditional_color_density_posteriors,
        "depends_on": [],
        "description": "Build redshift- and i-band-conditioned class-conditional color-density posterior scores with deterministic fallback and entropy uncertainty features.",
    }
]