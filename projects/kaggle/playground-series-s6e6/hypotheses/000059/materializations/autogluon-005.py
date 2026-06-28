import numpy as np
import pandas as pd

MIN_BIN_ROWS = 3000
MIN_PROFILE_ROWS = 150
N_CANDIDATE_BINS = 160
COV_SHRINKAGE = 0.35
JITTER_FRACTION = 0.0001
SIGMA_FLOOR_FRACTION = 0.03
DISTANCE_CLIP_LO = 0.5
DISTANCE_CLIP_HI = 99.5
TRAIN_ID_MAX = 577346
COLOR_COLUMNS = ("u_g", "g_r", "r_i", "i_z")
STATE_SEP = "__"

def _as_numeric_array(frame, columns):
    values = []
    for col in columns:
        values.append(pd.to_numeric(frame[col], errors="coerce").to_numpy(dtype=float))
    return values

def _safe_median(values, fallback=0.0):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float(fallback)
    return float(np.median(values))

def _safe_quantile(values, q, fallback):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float(fallback)
    return float(np.percentile(values, q))

def _make_redshift_bins(t_train):
    t_train = np.asarray(t_train, dtype=float)
    t_train = t_train[np.isfinite(t_train)]
    if t_train.size == 0:
        return np.array([-np.inf, np.inf], dtype=float)

    quantiles = np.linspace(0.0, 1.0, N_CANDIDATE_BINS + 1)
    edges = np.quantile(t_train, quantiles)
    edges = np.unique(edges)

    if edges.size < 2:
        center = float(edges[0]) if edges.size else 0.0
        return np.array([-np.inf, center, np.inf], dtype=float)

    candidate = np.searchsorted(edges[1:-1], t_train, side="right")
    counts = np.bincount(candidate, minlength=edges.size - 1)

    merged_edges = [float(edges[0])]
    running = 0
    for idx, count in enumerate(counts):
        running += int(count)
        is_last = idx == len(counts) - 1
        if running >= MIN_BIN_ROWS or is_last:
            merged_edges.append(float(edges[idx + 1]))
            running = 0

    if len(merged_edges) < 2:
        merged_edges = [float(edges[0]), float(edges[-1])]

    merged_edges[0] = -np.inf
    merged_edges[-1] = np.inf
    return np.asarray(merged_edges, dtype=float)

def _bin_values(t_values, edges):
    bins = np.searchsorted(edges[1:-1], t_values, side="right")
    return np.clip(bins, 0, max(0, len(edges) - 2)).astype(int)

def _profile_from_colors(colors, global_diag, sigma_floor):
    colors = np.asarray(colors, dtype=float)
    finite = np.all(np.isfinite(colors), axis=1)
    colors = colors[finite]
    n_rows = int(colors.shape[0])

    if n_rows == 0:
        mu = np.zeros(4, dtype=float)
        sigma = np.maximum(np.sqrt(np.maximum(global_diag, 0.0)), sigma_floor)
        inv_cov = np.diag(1.0 / np.maximum(global_diag, sigma_floor ** 2))
        return {"n": 0, "mu": mu, "sigma": sigma, "inv_cov": inv_cov}

    mu = np.median(colors, axis=0)
    mad = np.median(np.abs(colors - mu), axis=0)
    sigma = np.maximum(1.4826 * mad, sigma_floor)

    lo = np.percentile(colors, 1.0, axis=0)
    hi = np.percentile(colors, 99.0, axis=0)
    clipped = np.clip(colors, lo, hi)

    if n_rows >= 2:
        cov = np.cov(clipped, rowvar=False)
        if cov.shape == ():
            cov = np.diag(np.repeat(float(cov), 4))
    else:
        cov = np.diag(np.square(sigma))

    cov = np.asarray(cov, dtype=float)
    if cov.shape != (4, 4) or not np.all(np.isfinite(cov)):
        cov = np.diag(np.square(sigma))

    target = np.diag(np.maximum(global_diag, sigma_floor ** 2))
    shrunk = (1.0 - COV_SHRINKAGE) * cov + COV_SHRINKAGE * target
    jitter = max(
        _safe_median(global_diag, 1.0) * JITTER_FRACTION,
        _safe_median(np.square(sigma_floor), 1e-8) * JITTER_FRACTION,
        1e-8,
    )
    shrunk = shrunk + np.eye(4) * jitter

    try:
        inv_cov = np.linalg.pinv(shrunk)
    except np.linalg.LinAlgError:
        inv_cov = np.diag(1.0 / np.maximum(np.diag(shrunk), sigma_floor ** 2))

    return {"n": n_rows, "mu": mu, "sigma": sigma, "inv_cov": inv_cov}

def _state_key(spectral, population):
    return spectral.astype(str) + STATE_SEP + population.astype(str)

def _state_parts(state):
    parts = str(state).split(STATE_SEP, 1)
    if len(parts) == 2:
        return parts[0], parts[1]
    return str(state), ""

def add_tag_redshift_compatibility_residuals(raw, deps, aux):
    out_index = raw.index
    required = ["id", "u", "g", "r", "i", "z", "redshift", "spectral_type", "galaxy_population"]
    missing = [col for col in required if col not in raw.columns]
    if missing:
        return pd.DataFrame(index=out_index)

    frame = raw.loc[:, required].copy()

    ids = pd.to_numeric(frame["id"], errors="coerce")
    train_mask = ids.le(TRAIN_ID_MAX).to_numpy()
    if int(np.sum(train_mask)) < MIN_PROFILE_ROWS:
        train_mask = np.zeros(len(frame), dtype=bool)
        train_mask[: max(1, len(frame) // 2)] = True

    u, g, r, i_mag, z_mag = _as_numeric_array(frame, ["u", "g", "r", "i", "z"])
    colors = np.column_stack([u - g, g - r, r - i_mag, i_mag - z_mag])
    colors = np.where(np.isfinite(colors), colors, np.nan)

    redshift = pd.to_numeric(frame["redshift"], errors="coerce").to_numpy(dtype=float)
    redshift_t = np.log1p(np.maximum(np.nan_to_num(redshift, nan=0.0), 0.0))

    train_colors = colors[train_mask]
    finite_train_colors = train_colors[np.all(np.isfinite(train_colors), axis=1)]
    if finite_train_colors.size == 0:
        finite_train_colors = np.zeros((1, 4), dtype=float)

    global_diag = np.nanvar(finite_train_colors, axis=0)
    global_diag = np.where(np.isfinite(global_diag) & (global_diag > 1e-8), global_diag, 1.0)
    global_scale = np.sqrt(np.maximum(global_diag, 1e-8))
    sigma_floor = np.maximum(global_scale * SIGMA_FLOOR_FRACTION, 1e-4)

    edges = _make_redshift_bins(redshift_t[train_mask])
    bins = _bin_values(redshift_t, edges)
    n_bins = max(1, len(edges) - 1)

    states = _state_key(frame["spectral_type"], frame["galaxy_population"]).to_numpy()
    train_states = states[train_mask]
    unique_states = sorted(pd.unique(train_states).astype(str).tolist())
    if len(unique_states) == 0:
        unique_states = sorted(pd.unique(states).astype(str).tolist())

    profiles = {}
    state_profiles = {}
    global_profile = _profile_from_colors(finite_train_colors, global_diag, sigma_floor)

    for state in unique_states:
        state_train_mask = train_mask & (states == state)
        state_profiles[state] = _profile_from_colors(colors[state_train_mask], global_diag, sigma_floor)
        for bin_id in range(n_bins):
            cell_mask = state_train_mask & (bins == bin_id)
            if int(np.sum(cell_mask)) > 0:
                profiles[(state, bin_id)] = _profile_from_colors(colors[cell_mask], global_diag, sigma_floor)

    def select_profile(state, bin_id):
        exact = profiles.get((state, int(bin_id)))
        if exact is not None and exact["n"] >= MIN_PROFILE_ROWS:
            return exact, 0

        for radius in range(1, n_bins + 1):
            best = None
            best_count = -1
            left = int(bin_id) - radius
            right = int(bin_id) + radius
            for candidate_bin in (left, right):
                if 0 <= candidate_bin < n_bins:
                    candidate = profiles.get((state, candidate_bin))
                    if candidate is not None and candidate["n"] >= MIN_PROFILE_ROWS and candidate["n"] > best_count:
                        best = candidate
                        best_count = candidate["n"]
            if best is not None:
                return best, min(radius, 99)

        state_profile = state_profiles.get(state)
        if state_profile is not None and state_profile["n"] >= MIN_PROFILE_ROWS:
            return state_profile, 100

        return global_profile, 101

    n_rows = len(frame)
    n_states = len(unique_states)
    all_d2 = np.empty((n_rows, n_states), dtype=float)
    all_depth = np.empty((n_rows, n_states), dtype=float)
    all_z = np.empty((n_rows, n_states, 4), dtype=float)

    for state_idx, state in enumerate(unique_states):
        for bin_id in range(n_bins):
            row_mask = bins == bin_id
            if not np.any(row_mask):
                continue
            profile, depth = select_profile(state, bin_id)
            residual = colors[row_mask] - profile["mu"]
            residual = np.where(np.isfinite(residual), residual, 0.0)
            d2 = np.einsum("ij,jk,ik->i", residual, profile["inv_cov"], residual)
            all_d2[row_mask, state_idx] = d2
            all_depth[row_mask, state_idx] = float(depth)
            all_z[row_mask, state_idx, :] = residual / profile["sigma"]

    if n_states == 0:
        return pd.DataFrame(index=out_index)

    clip_lo = _safe_quantile(all_d2[train_mask].ravel(), DISTANCE_CLIP_LO, 0.0)
    clip_hi = _safe_quantile(all_d2[train_mask].ravel(), DISTANCE_CLIP_HI, max(clip_lo + 1.0, 1.0))
    if clip_hi <= clip_lo:
        clip_hi = clip_lo + 1.0
    all_d2 = np.clip(all_d2, clip_lo, clip_hi)

    state_to_idx = {state: idx for idx, state in enumerate(unique_states)}
    self_idx = np.array([state_to_idx.get(str(state), -1) for state in states], dtype=int)
    unknown_self = self_idx < 0
    if np.any(unknown_self):
        self_idx[unknown_self] = np.argmin(all_d2[unknown_self], axis=1)

    row_numbers = np.arange(n_rows)
    d_self = all_d2[row_numbers, self_idx]
    depth_self = all_depth[row_numbers, self_idx]
    z_self = all_z[row_numbers, self_idx, :]

    alt_d2 = all_d2.copy()
    alt_d2[row_numbers, self_idx] = np.inf
    best_alt_idx = np.argmin(alt_d2, axis=1)
    d_best_alt = alt_d2[row_numbers, best_alt_idx]

    alt_d2_second = alt_d2.copy()
    alt_d2_second[row_numbers, best_alt_idx] = np.inf
    d_second_alt = np.min(alt_d2_second, axis=1)
    if n_states <= 2:
        d_second_alt = d_best_alt.copy()

    depth_best_alt = all_depth[row_numbers, best_alt_idx]
    z_best_alt = all_z[row_numbers, best_alt_idx, :]

    rank_self = 1 + np.sum(all_d2 < d_self[:, None], axis=1)
    centered = all_d2 - np.min(all_d2, axis=1, keepdims=True)
    weights = np.exp(-0.5 * np.clip(centered, 0.0, 100.0))
    weights_sum = np.sum(weights, axis=1, keepdims=True)
    probs = weights / np.maximum(weights_sum, 1e-12)
    entropy = -np.sum(probs * np.log(np.maximum(probs, 1e-12)), axis=1)
    concentration = np.max(probs, axis=1)

    best_alt_states = np.array(unique_states, dtype=object)[best_alt_idx]
    best_alt_spectral = []
    best_alt_population = []
    for state in best_alt_states:
        spectral, population = _state_parts(state)
        best_alt_spectral.append(spectral)
        best_alt_population.append(population)

    result = pd.DataFrame(index=out_index)
    result["d_self"] = d_self
    result["d_best_alt"] = d_best_alt
    result["d_second_alt"] = d_second_alt
    result["margin_best_alt_minus_self"] = d_best_alt - d_self
    result["margin_second_alt_minus_self"] = d_second_alt - d_self
    result["self_distance_rank"] = rank_self.astype(np.int16)
    result["state_distance_entropy"] = entropy
    result["state_softmin_concentration"] = concentration

    for idx, col in enumerate(COLOR_COLUMNS):
        result["self_z_" + col] = np.clip(z_self[:, idx], -25.0, 25.0)
    for idx, col in enumerate(COLOR_COLUMNS):
        result["best_alt_z_" + col] = np.clip(z_best_alt[:, idx], -25.0, 25.0)

    result["fallback_depth_self"] = depth_self
    result["fallback_depth_best_alt"] = depth_best_alt
    result["best_alt_spectral_type"] = pd.Series(best_alt_spectral, index=out_index, dtype="object")
    result["best_alt_galaxy_population"] = pd.Series(best_alt_population, index=out_index, dtype="object")
    result["best_alt_matches_spectral_type"] = result["best_alt_spectral_type"].to_numpy() == frame["spectral_type"].astype(str).to_numpy()
    result["best_alt_matches_galaxy_population"] = result["best_alt_galaxy_population"].to_numpy() == frame["galaxy_population"].astype(str).to_numpy()

    return result

FEATURE_GROUPS = [
    {
        "name": "tag_redshift_compatibility_residuals",
        "fn": add_tag_redshift_compatibility_residuals,
        "depends_on": [],
        "description": "Robust redshift-binned tag compatibility distances and residual margins from broadband color profiles.",
    }
]