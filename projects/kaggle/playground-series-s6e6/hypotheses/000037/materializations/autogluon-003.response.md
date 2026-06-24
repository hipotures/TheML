import numpy as np
import pandas as pd

GROUP_NAME = "population_color_manifold_drifts"
N_BINS = 24
N_MIN = 300
MAD_SCALE = 1.4826
SIGMA_EPS = 1e-8
Z_CLIP = 8.0
SMALL = 1e-6
RED_POP = "Red_Sequence"
BLUE_POP = "Blue_Cloud"
COLOR_KEYS = ("u_g", "g_r", "r_i", "i_z", "u_r")


def _resolve_train_mask(aux, expected_len):
    if not isinstance(aux, pd.DataFrame) or len(aux) == 0:
        return np.ones(expected_len, dtype=bool)

    for col in ("is_train", "train_mask", "is_train_mask", "is_train_row", "train"):
        if col not in aux.columns:
            continue
        vals = aux[col].to_numpy()
        if len(vals) != expected_len:
            continue

        if np.issubdtype(vals.dtype, np.bool_):
            return vals.astype(bool, copy=False)
        if np.issubdtype(vals.dtype, np.number):
            return np.asarray(vals, dtype=float) > 0.5

        lowered = pd.Series(vals).astype(str).str.strip().str.lower()
        return lowered.isin(("1", "true", "t", "yes", "y", "train")).to_numpy()

    return np.ones(expected_len, dtype=bool)


def _quantile_bin_edges(logz_train, n_bins):
    arr = np.asarray(logz_train, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        arr = np.array([0.0, 1.0], dtype=float)
    probs = np.linspace(0.0, 1.0, n_bins + 1, dtype=float)
    edges = np.quantile(arr, probs).astype(float)
    edges[0] = float(np.min(arr))
    edges[-1] = float(np.max(arr))
    if edges[0] == edges[-1]:
        half = 0.5 if edges[0] == 0.0 else abs(edges[0]) * 0.5 + 0.5
        edges = np.linspace(edges[0] - half, edges[0] + half, n_bins + 1)
    else:
        step = 1e-12
        for i in range(1, len(edges)):
            if (not np.isfinite(edges[i])) or edges[i] <= edges[i - 1]:
                edges[i] = edges[i - 1] + step
                step *= 10.0
    return edges


def _robust_location_scale(values):
    mu = np.nanmedian(values, axis=0)
    mad = np.nanmedian(np.abs(values - mu), axis=0)
    sigma = MAD_SCALE * mad
    sigma = np.where(np.isfinite(sigma), sigma, 0.0)
    return mu, sigma


def add_population_color_manifold_drifts(raw, deps, aux):
    n_rows = len(raw)
    row_index = raw.index
    raw = raw.copy(deep=False)

    colors_raw = np.column_stack(
        [
            raw["u"].to_numpy(dtype=float, copy=False) - raw["g"].to_numpy(dtype=float, copy=False),
            raw["g"].to_numpy(dtype=float, copy=False) - raw["r"].to_numpy(dtype=float, copy=False),
            raw["r"].to_numpy(dtype=float, copy=False) - raw["i"].to_numpy(dtype=float, copy=False),
            raw["i"].to_numpy(dtype=float, copy=False) - raw["z"].to_numpy(dtype=float, copy=False),
            raw["u"].to_numpy(dtype=float, copy=False) - raw["r"].to_numpy(dtype=float, copy=False),
        ],
        dtype=float,
    )
    n_colors = len(COLOR_KEYS)

    pop_series = raw["galaxy_population"].to_numpy(dtype=object)
    pop_idx = np.full(n_rows, -1, dtype=np.int8)
    pop_idx[pop_series == RED_POP] = 0
    pop_idx[pop_series == BLUE_POP] = 1
    unknown_pop = pop_idx < 0

    redshift = raw["redshift"].to_numpy(dtype=float, copy=False)
    safe_t = np.log1p(np.clip(redshift, -0.999999999, None))
    train_mask = _resolve_train_mask(aux, n_rows)
    if train_mask.sum() == 0:
        train_mask = np.ones(n_rows, dtype=bool)

    edges = _quantile_bin_edges(safe_t[train_mask], N_BINS)
    safe_t = np.clip(safe_t, edges[0], edges[-1])
    bin_idx = np.searchsorted(edges[1:-1], safe_t, side="right")
    bin_idx = np.clip(bin_idx, 0, N_BINS - 1).astype(np.int16)

    train_bins = bin_idx[train_mask]
    train_pop = pop_idx[train_mask]
    train_colors = colors_raw[train_mask]
    known_train = (train_pop == 0) | (train_pop == 1)

    if np.any(known_train):
        color_ref = train_colors[known_train]
    else:
        color_ref = train_colors

    if color_ref.size == 0:
        color_ref = colors_raw

    global_mu, global_sigma = _robust_location_scale(color_ref)
    global_sigma = np.where(global_sigma > SIGMA_EPS, global_sigma, EPS)

    mu_cell = np.full((N_BINS, 2, n_colors), np.nan, dtype=float)
    sigma_cell = np.full((N_BINS, 2, n_colors), np.nan, dtype=float)
    cell_count = np.zeros((N_BINS, 2), dtype=np.int32)

    for p in range(2):
        pp = train_pop[known_train] == p
        for b in range(N_BINS):
            mask = (train_bins == b) & pp
            if not np.any(mask):
                continue
            vals = train_colors[mask]
            n = vals.shape[0]
            cell_count[b, p] = n
            mu_cell[b, p, :] = np.nanmedian(vals, axis=0)
            sigma_cell[b, p, :] = MAD_SCALE * np.nanmedian(np.abs(vals - mu_cell[b, p, :]), axis=0)

    mu_tilde = np.full((N_BINS, 2, n_colors), np.nan, dtype=float)
    sigma_tilde = np.full((N_BINS, 2, n_colors), np.nan, dtype=float)

    for b in range(N_BINS):
        for p in range(2):
            n = cell_count[b, p]
            for fi in range(n_colors):
                mu_loc = mu_cell[b, p, fi]
                sigma_loc = sigma_cell[b, p, fi]

                base_mu = global_mu[fi]
                base_sigma = global_sigma[fi]
                if np.isfinite(mu_loc) and np.isfinite(sigma_loc):
                    w = 1.0 if n >= N_MIN else n / N_MIN
                    if sigma_loc <= SIGMA_EPS:
                        w = 0.0
                    base_mu = w * mu_loc + (1.0 - w) * global_mu[fi]
                    base_sigma = w * sigma_loc + (1.0 - w) * global_sigma[fi]

                needs_fallback = (n < N_MIN) or (not np.isfinite(mu_loc)) or (not np.isfinite(sigma_loc)) or (sigma_loc <= SIGMA_EPS)
                mu_est = base_mu
                sigma_est = base_sigma

                if needs_fallback:
                    left_n = 0.0
                    right_n = 0.0
                    left_mu = 0.0
                    right_mu = 0.0
                    left_sigma = 0.0
                    right_sigma = 0.0

                    if b > 0:
                        n_left = float(cell_count[b - 1, p])
                        mu_left = mu_cell[b - 1, p, fi]
                        sigma_left = sigma_cell[b - 1, p, fi]
                        if (n_left > 0) and np.isfinite(mu_left) and np.isfinite(sigma_left) and (sigma_left > SIGMA_EPS):
                            left_n = n_left
                            left_mu = mu_left
                            left_sigma = sigma_left

                    if b < N_BINS - 1:
                        n_right_local = float(cell_count[b + 1, p])
                        mu_right = mu_cell[b + 1, p, fi]
                        sigma_right = sigma_cell[b + 1, p, fi]
                        if (n_right_local > 0) and np.isfinite(mu_right) and np.isfinite(sigma_right) and (sigma_right > SIGMA_EPS):
                            right_n = n_right_local
                            right_mu = mu_right
                            right_sigma = sigma_right

                    adj_n = left_n + right_n
                    if adj_n > 0.0:
                        adj_mu = (left_mu * left_n + right_mu * right_n) / adj_n
                        adj_sigma = (left_sigma * left_n + right_sigma * right_n) / adj_n
                        if np.isfinite(adj_mu) and np.isfinite(adj_sigma) and (adj_sigma > SIGMA_EPS):
                            w_adj = 1.0 if adj_n >= N_MIN else adj_n / N_MIN
                            mu_est = w_adj * adj_mu + (1.0 - w_adj) * global_mu[fi]
                            sigma_est = w_adj * adj_sigma + (1.0 - w_adj) * global_sigma[fi]
                        else:
                            mu_est = global_mu[fi]
                            sigma_est = global_sigma[fi]

                mu_tilde[b, p, fi] = mu_est
                sigma_tilde[b, p, fi] = sigma_est

    safe_pop = np.where(pop_idx < 0, 0, pop_idx).astype(np.int8)
    safe_opp = np.where(pop_idx == 0, 1, np.where(pop_idx == 1, 0, 0)).astype(np.int8)

    assigned_mu = mu_tilde[bin_idx, safe_pop]
    assigned_sigma = sigma_tilde[bin_idx, safe_pop]
    opposite_mu = mu_tilde[bin_idx, safe_opp]
    opposite_sigma = sigma_tilde[bin_idx, safe_opp]
    red_mu = mu_tilde[bin_idx, 0]
    red_sigma = sigma_tilde[bin_idx, 0]
    blue_mu = mu_tilde[bin_idx, 1]
    blue_sigma = sigma_tilde[bin_idx, 1]

    d_assigned = np.zeros(n_rows, dtype=float)
    d_opposite = np.zeros(n_rows, dtype=float)

    feature_data = {}

    for fi, cname in enumerate(COLOR_KEYS):
        cvals = colors_raw[:, fi]

        mu_a = assigned_mu[:, fi]
        sig_a = assigned_sigma[:, fi]
        valid_a = (~unknown_pop) & np.isfinite(mu_a) & np.isfinite(sig_a) & (sig_a > SIGMA_EPS)
        z_a = np.zeros(n_rows, dtype=float)
        if np.any(valid_a):
            z_a[valid_a] = np.clip((cvals[valid_a] - mu_a[valid_a]) / sig_a[valid_a], -Z_CLIP, Z_CLIP)
        d_assigned += z_a * z_a

        mu_o = opposite_mu[:, fi]
        sig_o = opposite_sigma[:, fi]
        valid_o = (~unknown_pop) & np.isfinite(mu_o) & np.isfinite(sig_o) & (sig_o > SIGMA_EPS)
        z_o = np.zeros(n_rows, dtype=float)
        if np.any(valid_o):
            z_o[valid_o] = np.clip((cvals[valid_o] - mu_o[valid_o]) / sig_o[valid_o], -Z_CLIP, Z_CLIP)
        d_opposite += z_o * z_o

        feature_data[f"{GROUP_NAME}_assigned_z_{cname}"] = z_a
        feature_data[f"{GROUP_NAME}_opposite_z_{cname}"] = z_o
        feature_data[f"{GROUP_NAME}_assigned_missing_{cname}"] = ~valid_a
        feature_data[f"{GROUP_NAME}_opposite_missing_{cname}"] = ~valid_o

        mu_r = red_mu[:, fi]
        sig_r = red_sigma[:, fi]
        mu_b = blue_mu[:, fi]
        sig_b = blue_sigma[:, fi]

        valid_r = np.isfinite(mu_r) & np.isfinite(sig_r) & (sig_r > SIGMA_EPS)
        valid_b = np.isfinite(mu_b) & np.isfinite(sig_b) & (sig_b > SIGMA_EPS)
        delta_valid = valid_r & valid_b
        red_std = np.zeros(n_rows, dtype=float)
        blue_std = np.zeros(n_rows, dtype=float)
        if np.any(delta_valid):
            red_std[delta_valid] = np.clip((cvals[delta_valid] - mu_r[delta_valid]) / sig_r[delta_valid], -Z_CLIP, Z_CLIP)
            blue_std[delta_valid] = np.clip((cvals[delta_valid] - mu_b[delta_valid]) / sig_b[delta_valid], -Z_CLIP, Z_CLIP)
        delta_f = red_std - blue_std
        feature_data[f"{GROUP_NAME}_delta_{cname}"] = delta_f
        feature_data[f"{GROUP_NAME}_delta_missing_{cname}"] = ~delta_valid

        mid_f = 0.5 * (mu_r + mu_b)
        gap_f = mu_r - mu_b
        divider_valid = np.isfinite(mid_f) & np.isfinite(gap_f)
        m_f = np.zeros(n_rows, dtype=float)
        if np.any(divider_valid):
            m_f[divider_valid] = (cvals[divider_valid] - mid_f[divider_valid]) * np.sign(gap_f[divider_valid])

        feature_data[f"{GROUP_NAME}_mid_{cname}"] = mid_f
        feature_data[f"{GROUP_NAME}_gap_{cname}"] = gap_f
        feature_data[f"{GROUP_NAME}_margin_{cname}"] = m_f
        feature_data[f"{GROUP_NAME}_margin_missing_{cname}"] = ~divider_valid

    d_assigned = d_assigned / float(n_colors)
    d_opposite = d_opposite / float(n_colors)
    feature_data[f"{GROUP_NAME}_D_assigned"] = d_assigned
    feature_data[f"{GROUP_NAME}_D_opposite"] = d_opposite
    feature_data[f"{GROUP_NAME}_D_diff"] = np.log1p(d_opposite) - np.log1p(d_assigned)
    feature_data[f"{GROUP_NAME}_D_ratio"] = d_opposite / (d_assigned + SMALL)

    return pd.DataFrame(feature_data, index=row_index)


FEATURE_GROUPS = [
    {
        "name": GROUP_NAME,
        "fn": add_population_color_manifold_drifts,
        "depends_on": [],
        "description": "Builds redshift-bin manifold residual, cross-manifold, and separator geometry features from robust color statistics with shrinkage and adjacent-bin fallback.",
    }
]