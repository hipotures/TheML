import numpy as np
import pandas as pd

STATE_SEPARATOR = "|"
COLOR_PAIRS = (("u", "g"), ("g", "r"), ("r", "i"), ("i", "z"))
COLOR_SUFFIXES = ("u_g", "g_r", "r_i", "i_z")
N_QUANTILES = 200
MIN_BIN_SIZE = 3000
MAX_FINAL_BINS = 24
MIN_STATE_BIN_COUNT = 150
MAX_NEIGHBOR_STEPS = 4
MAD_SCALE = 1.4826
SIGMA_FLOOR = 1e-3
REG_LAMBDA = 0.1
REG_EPS = 1e-6
DIST_LOW_Q = 0.01
DIST_HIGH_Q = 0.99
GROUP_NAME = "tag_redshift_compatibility_residuals"
FALLBACK_LABELS = ("direct", "neighbor", "state", "global")


def _build_redshift_bins(redshift_log, n_quantiles=N_QUANTILES, min_bin_size=MIN_BIN_SIZE, max_bins=MAX_FINAL_BINS):
    redshift_log = np.asarray(redshift_log, dtype=float)
    n_rows = redshift_log.shape[0]
    if n_rows == 0:
        return np.array([0.0, 1.0], dtype=float), np.zeros(0, dtype=np.int32)

    n_points = min(n_quantiles + 1, max(2, n_rows))
    raw_edges = np.quantile(redshift_log, np.linspace(0.0, 1.0, n_points))
    raw_edges = np.sort(np.unique(raw_edges[np.isfinite(raw_edges)]))
    if raw_edges.size < 2:
        center = float(np.nanmedian(redshift_log)) if np.isfinite(np.nanmedian(redshift_log)) else 0.0
        eps = 1.0 if center == 0.0 else abs(center) * 1e-6 + 1e-6
        raw_edges = np.array([center - eps, center + eps], dtype=float)
    elif raw_edges.size == 2 and np.isclose(raw_edges[0], raw_edges[1]):
        eps = abs(raw_edges[0]) * 1e-6 + 1e-6
        raw_edges = np.array([raw_edges[0] - eps, raw_edges[1] + eps], dtype=float)

    raw_bin = np.searchsorted(raw_edges, redshift_log, side="right") - 1
    raw_bin = np.clip(raw_bin, 0, raw_edges.size - 2)
    raw_counts = np.bincount(raw_bin, minlength=raw_edges.size - 1).astype(np.int64)

    segments = []
    start = 0
    acc = 0
    for i, c in enumerate(raw_counts):
        acc += int(c)
        if i == raw_counts.size - 1:
            segments.append((start, raw_counts.size))
        elif acc >= min_bin_size:
            segments.append((start, i + 1))
            start = i + 1
            acc = 0

    if len(segments) > 1:
        seg_start, seg_end = segments[-1]
        last_count = int(raw_counts[seg_start:seg_end].sum())
        if last_count < min_bin_size:
            prev_s, prev_e = segments[-2]
            segments[-2:] = [(prev_s, seg_end)]

    if len(segments) > 1:
        seg_counts = np.array([int(raw_counts[s:e].sum()) for s, e in segments], dtype=np.int64)
        while seg_counts.size > max_bins and seg_counts.size > 1:
            pair_counts = seg_counts[:-1] + seg_counts[1:]
            k = int(np.argmin(pair_counts))
            segments[k] = (segments[k][0], segments[k + 1][1])
            seg_counts[k] = pair_counts[k]
            segments.pop(k + 1)
            seg_counts = np.delete(seg_counts, k + 1)

    edges = np.empty(len(segments) + 1, dtype=float)
    edges[0] = raw_edges[segments[0][0]]
    for i, (_, end) in enumerate(segments):
        edges[i + 1] = raw_edges[end]
    if not np.all(np.isfinite(edges)):
        return np.array([0.0, 1.0], dtype=float), np.zeros(n_rows, dtype=np.int32)

    final_bin = np.clip(np.searchsorted(edges, redshift_log, side="right") - 1, 0, edges.size - 2).astype(np.int32)
    return edges, final_bin


def _robust_profile(rows, global_sigma2):
    rows = np.asarray(rows, dtype=float)
    if rows.shape[0] == 0:
        mu = np.zeros(4, dtype=float)
        sigma = np.full(4, SIGMA_FLOOR, dtype=float)
        cov = np.diag(np.square(sigma))
        cov = cov + np.diag(REG_LAMBDA * global_sigma2 + REG_EPS)
        inv_cov = np.linalg.pinv(cov)
        return mu, sigma, inv_cov

    mu = np.nanmedian(rows, axis=0)
    residual = rows - mu
    med_res = np.nanmedian(residual, axis=0)
    mad = np.nanmedian(np.abs(residual - med_res), axis=0)
    sigma = np.maximum(mad * MAD_SCALE, SIGMA_FLOOR)

    q_lo_hi = np.nanquantile(residual, [0.01, 0.99], axis=0)
    q_lo = np.where(np.isfinite(q_lo_hi[0]), q_lo_hi[0], -sigma)
    q_hi = np.where(np.isfinite(q_lo_hi[1]), q_lo_hi[1], sigma)
    clipped = np.clip(residual, q_lo, q_hi)

    if rows.shape[0] > 1:
        cov = np.cov(clipped, rowvar=False, bias=False)
    else:
        cov = np.zeros((4, 4), dtype=float)

    cov = cov + np.diag(REG_LAMBDA * global_sigma2 + REG_EPS)
    inv_cov = np.linalg.pinv(cov)
    return mu, sigma, inv_cov


def _resolve_profile_bins(state_bin_counts, state_totals):
    n_states, n_bins = state_bin_counts.shape
    resolved_bin = np.full((n_states, n_bins), -1, dtype=np.int32)
    resolved_level = np.full((n_states, n_bins), 3, dtype=np.int8)

    for s in range(n_states):
        total_state = int(state_totals[s])
        for b in range(n_bins):
            direct = int(state_bin_counts[s, b])
            if direct >= MIN_STATE_BIN_COUNT:
                resolved_bin[s, b] = b
                resolved_level[s, b] = 0
                continue

            found = False
            for step in range(1, MAX_NEIGHBOR_STEPS + 1):
                left = b - step
                if left >= 0 and state_bin_counts[s, left] >= MIN_STATE_BIN_COUNT:
                    resolved_bin[s, b] = left
                    resolved_level[s, b] = 1
                    found = True
                    break
                right = b + step
                if right < n_bins and state_bin_counts[s, right] >= MIN_STATE_BIN_COUNT:
                    resolved_bin[s, b] = right
                    resolved_level[s, b] = 1
                    found = True
                    break

            if found:
                continue

            if total_state > 0:
                resolved_level[s, b] = 2

    return resolved_bin, resolved_level


def add_tag_redshift_compatibility_residuals(raw, deps, aux):
    _ = deps
    _ = aux

    n_rows = len(raw)
    if n_rows == 0:
        return pd.DataFrame(index=raw.index)

    redshift = np.maximum(raw["redshift"].to_numpy(dtype=float), 0.0)
    redshift_log = np.log1p(redshift)
    _, red_bin = _build_redshift_bins(redshift_log)

    colors = np.column_stack(
        (
            raw["u"].to_numpy(dtype=float) - raw["g"].to_numpy(dtype=float),
            raw["g"].to_numpy(dtype=float) - raw["r"].to_numpy(dtype=float),
            raw["r"].to_numpy(dtype=float) - raw["i"].to_numpy(dtype=float),
            raw["i"].to_numpy(dtype=float) - raw["z"].to_numpy(dtype=float),
        )
    )

    spectral = raw["spectral_type"].to_numpy(dtype=object)
    galaxy_pop = raw["galaxy_population"].to_numpy(dtype=object)

    state_labels = np.char.add(spectral.astype(str), STATE_SEPARATOR)
    state_labels = np.char.add(state_labels, galaxy_pop.astype(str))

    spec_unique = tuple(sorted({str(s) for s in spectral}))
    pop_unique = tuple(sorted({str(p) for p in galaxy_pop}))
    all_states = tuple(spec + STATE_SEPARATOR + pop for spec in spec_unique for pop in pop_unique)
    n_states = len(all_states)
    if n_states == 0:
        return pd.DataFrame(index=raw.index)

    state_to_idx = {s: i for i, s in enumerate(all_states)}
    state_idx = np.fromiter((state_to_idx[str(s)] for s in state_labels), dtype=np.int32, count=n_rows)

    n_bins = int(red_bin.max()) + 1
    bin_totals = np.bincount(red_bin, minlength=n_bins).astype(np.int64)

    state_bin_counts = np.zeros((n_states, n_bins), dtype=np.int64)
    np.add.at(state_bin_counts, (state_idx, red_bin), 1)
    state_totals = state_bin_counts.sum(axis=1).astype(np.int64)

    global_color = np.nanmedian(colors, axis=0)
    global_resid = colors - global_color
    global_mad = np.nanmedian(np.abs(global_resid - np.nanmedian(global_resid, axis=0)), axis=0)
    global_sigma2 = np.square(np.maximum(global_mad * MAD_SCALE, SIGMA_FLOOR))

    global_mu, global_sigma, global_inv = _robust_profile(colors, global_sigma2)
    global_profile = (global_mu, global_sigma, global_inv)

    cell_mu = np.tile(global_mu, (n_states, n_bins, 1)).astype(float)
    cell_sigma = np.tile(global_sigma, (n_states, n_bins, 1)).astype(float)
    cell_inv = np.tile(global_inv, (n_states, n_bins, 1, 1)).astype(float)

    pair_key = (state_idx.astype(np.int64) * np.int64(n_bins + 1)) + red_bin.astype(np.int64)
    order = np.argsort(pair_key, kind="mergesort")
    sorted_keys = pair_key[order]
    uniq_keys, key_starts = np.unique(sorted_keys, return_index=True)
    key_starts = np.asarray(key_starts, dtype=np.int64)
    key_ends = np.r_[key_starts[1:], order.size]

    for u in range(uniq_keys.size):
        s = int(uniq_keys[u] // (n_bins + 1))
        b = int(uniq_keys[u] - (n_bins + 1) * s)
        if s >= n_states or b >= n_bins:
            continue
        rows = order[key_starts[u]:key_ends[u]]
        mu_c, sigma_c, inv_c = _robust_profile(colors[rows], global_sigma2)
        cell_mu[s, b] = mu_c
        cell_sigma[s, b] = sigma_c
        cell_inv[s, b] = inv_c

    state_mu = np.tile(global_mu, (n_states, 1)).astype(float)
    state_sigma = np.tile(global_sigma, (n_states, 1)).astype(float)
    state_inv = np.tile(global_inv, (n_states, 1, 1)).astype(float)

    for s in range(n_states):
        if state_totals[s] > 0:
            idx = np.flatnonzero(state_idx == s)
            mu_s, sigma_s, inv_s = _robust_profile(colors[idx], global_sigma2)
            state_mu[s] = mu_s
            state_sigma[s] = sigma_s
            state_inv[s] = inv_s

    profile_bin, profile_level = _resolve_profile_bins(state_bin_counts, state_totals)

    selected_mu = np.tile(global_profile[0], (n_states, n_bins, 1)).astype(float)
    selected_sigma = np.tile(global_profile[1], (n_states, n_bins, 1)).astype(float)
    selected_inv = np.tile(global_profile[2], (n_states, n_bins, 1, 1)).astype(float)
    prior_matrix = np.zeros((n_states, n_bins), dtype=float)

    for s in range(n_states):
        for b in range(n_bins):
            lvl = int(profile_level[s, b])
            if lvl in (0, 1):
                pb = int(profile_bin[s, b])
                selected_mu[s, b] = cell_mu[s, pb]
                selected_sigma[s, b] = cell_sigma[s, pb]
                selected_inv[s, b] = cell_inv[s, pb]
                denom = float(bin_totals[pb]) if bin_totals.size > pb else 0.0
                if denom > 0.0:
                    prior_matrix[s, b] = float(state_bin_counts[s, pb]) / denom
                else:
                    prior_matrix[s, b] = float(state_totals[s]) / float(n_rows) if n_rows else 0.0
            elif lvl == 2:
                selected_mu[s, b] = state_mu[s]
                selected_sigma[s, b] = state_sigma[s]
                selected_inv[s, b] = state_inv[s]
                prior_matrix[s, b] = float(state_totals[s]) / float(n_rows) if n_rows else 0.0
            else:
                prior_matrix[s, b] = float(state_totals[s]) / float(n_rows) if n_rows else 0.0

    distances = np.full((n_rows, n_states), np.nan, dtype=float)
    for s in range(n_states):
        for b in range(n_bins):
            row_idx = np.flatnonzero(red_bin == b)
            if row_idx.size == 0:
                continue
            diff = colors[row_idx] - selected_mu[s, b]
            inv_mat = selected_inv[s, b]
            distances[row_idx, s] = np.einsum("ij,jk,ik->i", diff, inv_mat, diff)

    dist_low = np.zeros(n_states, dtype=float)
    dist_high = np.ones(n_states, dtype=float)
    for s in range(n_states):
        col = distances[:, s]
        valid = np.isfinite(col)
        vals = col[valid]
        if vals.size == 0:
            dist_low[s] = 0.0
            dist_high[s] = 0.0
        elif vals.size == 1:
            dist_low[s] = float(vals[0])
            dist_high[s] = float(vals[0])
        else:
            dist_low[s] = float(np.quantile(vals, DIST_LOW_Q))
            dist_high[s] = float(np.quantile(vals, DIST_HIGH_Q))
        if dist_high[s] < dist_low[s]:
            dist_high[s] = dist_low[s]

    distances = np.where(np.isfinite(distances), np.clip(distances, dist_low[None, :], dist_high[None, :]), distances)

    d_self = distances[np.arange(n_rows), state_idx]
    p_self = prior_matrix[state_idx, red_bin]
    fallback_idx = profile_level[state_idx, red_bin]
    fallback = np.asarray(FALLBACK_LABELS, dtype=object)[fallback_idx]
    self_mu_row = selected_mu[state_idx, red_bin]
    self_sigma_row = np.maximum(selected_sigma[state_idx, red_bin], SIGMA_FLOOR)
    self_r = (colors - self_mu_row) / self_sigma_row

    alt1_d = np.full(n_rows, np.nan, dtype=float)
    alt2_d = np.full(n_rows, np.nan, dtype=float)
    alt1_r = np.full((n_rows, 4), np.nan, dtype=float)

    all_state_ids = np.arange(n_states, dtype=np.int32)
    if n_states > 1:
        alt_state_lookup = np.empty((n_states, n_states - 1), dtype=np.int32)
        for s in range(n_states):
            if s == 0:
                alt_state_lookup[s] = all_state_ids[1:]
            elif s == n_states - 1:
                alt_state_lookup[s] = all_state_ids[:-1]
            else:
                alt_state_lookup[s] = np.concatenate((all_state_ids[:s], all_state_ids[s + 1:]))

        for i in range(n_rows):
            s = int(state_idx[i])
            b = int(red_bin[i])
            alt_idx = alt_state_lookup[s]
            cand = distances[i, alt_idx]
            finite = np.isfinite(cand)
            if not np.any(finite):
                continue
            local_idx = np.flatnonzero(finite)
            local_order = np.argsort(cand[local_idx])
            top_local = local_idx[local_order]
            first_local = top_local[0]
            alt1_state = int(alt_idx[first_local])
            alt1_d[i] = float(cand[first_local])

            if top_local.size > 1:
                second_local = top_local[1]
                alt2_d[i] = float(cand[second_local])

            alt_mu = selected_mu[alt1_state, b]
            alt_sig = np.maximum(selected_sigma[alt1_state, b], SIGMA_FLOOR)
            alt1_r[i] = (colors[i] - alt_mu) / alt_sig

    m1 = np.where(np.isfinite(alt1_d) & np.isfinite(d_self), alt1_d - d_self, np.nan)
    m2 = np.where(np.isfinite(alt2_d) & np.isfinite(d_self), alt2_d - d_self, np.nan)

    out = pd.DataFrame(
        {
            f"{GROUP_NAME}_d_self": d_self,
            f"{GROUP_NAME}_d_alt1": alt1_d,
            f"{GROUP_NAME}_d_alt2": alt2_d,
            f"{GROUP_NAME}_margin_alt1": m1,
            f"{GROUP_NAME}_margin_alt2": m2,
            f"{GROUP_NAME}_fallback_level": fallback,
            f"{GROUP_NAME}_p_self_class": p_self,
        },
        index=raw.index,
    )

    for j, suffix in enumerate(COLOR_SUFFIXES):
        out[f"{GROUP_NAME}_r_self_{suffix}"] = self_r[:, j]
        out[f"{GROUP_NAME}_r_alt1_{suffix}"] = alt1_r[:, j]

    return out


FEATURE_GROUPS = [
    {
        "name": GROUP_NAME,
        "fn": add_tag_redshift_compatibility_residuals,
        "depends_on": [],
        "description": "Build redshift-binned catalog-state manifold compatibility distances, margins, residuals, fallback context, and class-prior signals.",
    }
]