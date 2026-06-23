import numpy as np
import pandas as pd

_COLOR_COLUMNS = ("u", "g", "r", "i", "z")
_AUX_MISSING_SENTINEL_BASE = ("u", "g", "r", "i", "z", "redshift")
_TAG_STATE_DELIM = "__"
_MIN_BIN_SUPPORT = 150
_MAX_INITIAL_QUANTILE_BINS = 64
_CELL_SUPPORT_FLOOR = 40
_STATE_SUPPORT_FLOOR = 180
_MAD_FLOOR = 1e-6
_COV_JITTER_FACTOR = 1e-5
_PCT_LOW = 1.0
_PCT_HIGH = 99.0
_DISTANCE_CLIP_LOW = 1.0
_DISTANCE_CLIP_HIGH = 99.0
_MAG_MIN = -300.0
_MAG_MAX = 70.0
_REDSHIFT_MIN = -1.0
_REDSHIFT_MAX = 20.0
_COLOR_NAMES = ("u_g", "g_r", "r_i", "i_z")


def _to_float(df, col):
    return pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=np.float64)


def _build_color_matrix(u, g, r, i, z):
    return np.column_stack((u - g, g - r, r - i, i - z))


def _fit_robust_model(colors):
    x = np.asarray(colors, dtype=np.float64)
    if x.size == 0:
        return None
    x = x[np.isfinite(x).all(axis=1)]
    if x.shape[0] == 0:
        return None

    mu = np.nanmedian(x, axis=0)
    centered = x - mu
    mad = np.nanmedian(np.abs(centered), axis=0)
    mad = np.where(np.isfinite(mad), mad, 1.0)
    mad = np.maximum(mad, _MAD_FLOOR)

    scaled = centered / mad
    if scaled.shape[0] <= 3:
        cov = np.eye(4, dtype=np.float64)
    else:
        cov = np.cov(scaled, rowvar=False, bias=True)

    if not np.all(np.isfinite(cov)):
        cov = np.eye(4, dtype=np.float64)

    cov = 0.5 * (cov + cov.T)
    trace = float(np.trace(cov))
    jitter = max(trace * _COV_JITTER_FACTOR, _MAD_FLOOR)
    cov = cov + np.eye(4, dtype=np.float64) * jitter

    try:
        inv_cov = np.linalg.inv(cov)
    except Exception:
        inv_cov = np.linalg.pinv(cov)

    return {
        "mu": mu,
        "scale": mad,
        "inv_cov": inv_cov,
        "support": int(x.shape[0]),
    }


def _mahalanobis_distance(color, model):
    if model is None:
        return np.nan
    diff = np.asarray(color, dtype=np.float64) - model["mu"]
    scaled = diff / model["scale"]
    if not np.all(np.isfinite(scaled)):
        return np.nan
    d2 = float(np.dot(scaled, np.dot(model["inv_cov"], scaled)))
    if not np.isfinite(d2) or d2 < 0.0:
        return np.nan
    return np.sqrt(d2)


def _standardized_residual(color, model):
    if model is None:
        return None
    resid = (np.asarray(color, dtype=np.float64) - model["mu"]) / model["scale"]
    if not np.all(np.isfinite(resid)):
        return None
    return resid


def _build_merged_bins(t_values, min_support, aux_t=None):
    t = np.asarray(t_values, dtype=np.float64)
    t = t[np.isfinite(t)]
    if aux_t is not None:
        extra = np.asarray(aux_t, dtype=np.float64)
        extra = extra[np.isfinite(extra)]
        if extra.size:
            t = extra if t.size == 0 else np.concatenate((t, extra), axis=0)

    if t.size == 0:
        return np.array([0.0, 1.0], dtype=np.float64)
    if np.nanmin(t) == np.nanmax(t):
        center = float(np.nanmin(t))
        return np.array([center - 1e-6, center + 1e-6], dtype=np.float64)
    if t.size < 4:
        lo = float(np.nanmin(t))
        hi = float(np.nanmax(t))
        if lo == hi:
            return np.array([lo - 1e-6, hi + 1e-6], dtype=np.float64)
        return np.array([lo, hi], dtype=np.float64)

    desired = int(max(4, min(_MAX_INITIAL_QUANTILE_BINS, t.size // max(min_support, 1))))
    desired = min(desired, t.size)
    _, edges = pd.qcut(t, q=desired, duplicates="drop", retbins=True)
    edges = np.asarray(edges, dtype=np.float64)
    if edges.size < 3:
        return np.array([float(np.nanmin(t) - 1e-6), float(np.nanmax(t) + 1e-6)], dtype=np.float64)

    labels = pd.cut(t, edges, include_lowest=True, duplicates="drop")
    counts = np.bincount(labels.codes[labels.codes >= 0], minlength=len(edges) - 1)

    merged = [edges[0]]
    run = 0
    for i, c in enumerate(counts):
        run += int(c)
        if run >= min_support or i == len(counts) - 1:
            merged.append(edges[i + 1])
            run = 0

    merged_edges = np.unique(np.array(merged, dtype=np.float64))
    if merged_edges.size < 2 or merged_edges[0] == merged_edges[-1]:
        lo = float(np.nanmin(t)) - 1e-6
        hi = float(np.nanmax(t)) + 1e-6
        return np.array([lo, hi], dtype=np.float64)
    return merged_edges


def _clip_values(values, low=_DISTANCE_CLIP_LOW, high=_DISTANCE_CLIP_HIGH):
    arr = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(arr)
    if not finite.any():
        return arr
    lo = np.nanpercentile(arr[finite], low)
    hi = np.nanpercentile(arr[finite], high)
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        return arr
    return np.clip(arr, lo, hi)


def _init_outputs(n):
    self_dist = np.full(n, np.nan, dtype=np.float64)
    alt1_dist = np.full(n, np.nan, dtype=np.float64)
    alt2_dist = np.full(n, np.nan, dtype=np.float64)
    margin1 = np.full(n, np.nan, dtype=np.float64)
    margin2 = np.full(n, np.nan, dtype=np.float64)
    self_resid = np.full((n, 4), np.nan, dtype=np.float64)
    alt1_resid = np.full((n, 4), np.nan, dtype=np.float64)
    self_fb_state = np.zeros(n, dtype=np.int8)
    self_fb_global = np.zeros(n, dtype=np.int8)
    return self_dist, alt1_dist, alt2_dist, margin1, margin2, self_resid, alt1_resid, self_fb_state, self_fb_global


def _assemble_output(index, self_dist, alt1_dist, alt2_dist, margin1, margin2, self_resid, alt1_resid, fb_state, fb_global):
    out = pd.DataFrame(index=index)
    out["tag_rc_self_mahal"] = self_dist
    out["tag_rc_alt1_mahal"] = alt1_dist
    out["tag_rc_alt2_mahal"] = alt2_dist
    out["tag_rc_margin1"] = margin1
    out["tag_rc_margin2"] = margin2
    out["tag_rc_self_fallback_state"] = fb_state.astype(np.float64)
    out["tag_rc_self_fallback_global"] = fb_global.astype(np.float64)
    for k, name in enumerate(_COLOR_NAMES):
        out[f"tag_rc_self_resid_{name}"] = self_resid[:, k]
        out[f"tag_rc_alt1_resid_{name}"] = alt1_resid[:, k]
    return out


def _to_codes_from_cut(categorical_obj):
    if hasattr(categorical_obj, "codes"):
        return np.asarray(categorical_obj.codes, dtype=np.int64)
    if hasattr(categorical_obj, "cat"):
        return categorical_obj.cat.codes.to_numpy(dtype=np.int64)
    return pd.Series(categorical_obj).cat.codes.to_numpy(dtype=np.int64)


def add_tag_redshift_compatibility_residuals(raw, deps, aux):
    required = ("u", "g", "r", "i", "z", "redshift", "spectral_type", "galaxy_population")
    n = len(raw)
    outputs = _init_outputs(n)

    if not set(required).issubset(raw.columns):
        return _assemble_output(raw.index, *outputs)

    u = _to_float(raw, "u")
    g = _to_float(raw, "g")
    r = _to_float(raw, "r")
    i = _to_float(raw, "i")
    z = _to_float(raw, "z")
    red = _to_float(raw, "redshift")

    spec = raw["spectral_type"].astype("string").str.strip()
    pop = raw["galaxy_population"].astype("string").str.strip()
    state = (spec + _TAG_STATE_DELIM + pop).to_numpy(dtype=object)
    tag_mask = spec.notna().to_numpy() & pop.notna().to_numpy()

    raw_mask = (
        np.isfinite(u)
        & np.isfinite(g)
        & np.isfinite(r)
        & np.isfinite(i)
        & np.isfinite(z)
        & np.isfinite(red)
        & (u > _MAG_MIN)
        & (u < _MAG_MAX)
        & (g > _MAG_MIN)
        & (g < _MAG_MAX)
        & (r > _MAG_MIN)
        & (r < _MAG_MAX)
        & (i > _MAG_MIN)
        & (i < _MAG_MAX)
        & (z > _MAG_MIN)
        & (z < _MAG_MAX)
        & (red > _REDSHIFT_MIN)
        & (red < _REDSHIFT_MAX)
        & tag_mask
    )

    colors = _build_color_matrix(u, g, r, i, z)
    red_t = np.log1p(np.clip(np.where(np.isfinite(red), red, np.nan), 1e-6, None))
    raw_t_for_bins = red_t[raw_mask]

    aux_global_colors = None
    aux_t_for_bins = None
    aux_state_colors = None
    aux_state_bins = None
    aux_state_names = None

    if isinstance(aux, pd.DataFrame) and not aux.empty:
        if all(col in aux.columns for col in _AUX_MISSING_SENTINEL_BASE):
            au_u = _to_float(aux, "u")
            au_g = _to_float(aux, "g")
            au_r = _to_float(aux, "r")
            au_i = _to_float(aux, "i")
            au_z = _to_float(aux, "z")
            au_red = _to_float(aux, "redshift")
            au_mask = (
                np.isfinite(au_u)
                & np.isfinite(au_g)
                & np.isfinite(au_r)
                & np.isfinite(au_i)
                & np.isfinite(au_z)
                & np.isfinite(au_red)
                & (au_u > _MAG_MIN)
                & (au_u < _MAG_MAX)
                & (au_g > _MAG_MIN)
                & (au_g < _MAG_MAX)
                & (au_r > _MAG_MIN)
                & (au_r < _MAG_MAX)
                & (au_i > _MAG_MIN)
                & (au_i < _MAG_MAX)
                & (au_z > _MAG_MIN)
                & (au_z < _MAG_MAX)
                & (au_red > _REDSHIFT_MIN)
                & (au_red < _REDSHIFT_MAX)
            )

            au_colors = _build_color_matrix(au_u, au_g, au_r, au_i, au_z)
            if au_mask.any():
                au_t_all = np.log1p(np.clip(np.where(np.isfinite(au_red), au_red, np.nan), 1e-6, None))
                aux_global_colors = au_colors[au_mask]
                aux_t_for_bins = au_t_all[au_mask]

                if "spectral_type" in aux.columns and "galaxy_population" in aux.columns:
                    au_spec = aux["spectral_type"].astype("string").str.strip()
                    au_pop = aux["galaxy_population"].astype("string").str.strip()
                    au_state_mask = au_mask & au_spec.notna().to_numpy() & au_pop.notna().to_numpy()
                    if au_state_mask.any():
                        aux_state_colors = au_colors[au_state_mask]
                        aux_state_names = (au_spec[au_state_mask] + _TAG_STATE_DELIM + au_pop[au_state_mask]).to_numpy(dtype=object)
                        aux_state_bins = np.log1p(np.clip(np.where(np.isfinite(au_red[au_state_mask]), au_red[au_state_mask], np.nan), 1e-6, None))

    merged_edges = _build_merged_bins(raw_t_for_bins, _MIN_BIN_SUPPORT, aux_t_for_bins)

    zbin_cat = pd.cut(red_t, merged_edges, include_lowest=True, duplicates="drop")
    zbin = _to_codes_from_cut(zbin_cat)

    aux_state_zbin = None
    if aux_state_bins is not None and len(aux_state_bins):
        aux_state_zbin = _to_codes_from_cut(
            pd.cut(aux_state_bins, merged_edges, include_lowest=True, duplicates="drop")
        )

    base_idx = np.flatnonzero(raw_mask)
    if base_idx.size == 0:
        return _assemble_output(raw.index, *outputs)

    model_colors = [colors[raw_mask]]
    model_states = [state[raw_mask]]
    model_bins = zbin[raw_mask]

    if aux_state_colors is not None and aux_state_names is not None and aux_state_zbin is not None and aux_state_colors.shape[0]:
        aux_state_valid = (aux_state_zbin >= 0)
        if aux_state_valid.any():
            model_colors.append(aux_state_colors[aux_state_valid])
            model_states.append(aux_state_names[aux_state_valid])
            model_bins = np.concatenate((model_bins, aux_state_zbin[aux_state_valid]), axis=0)

    model_colors_arr = np.vstack(model_colors)
    model_states_arr = np.concatenate(model_states)
    model_bins_arr = np.asarray(model_bins, dtype=np.int64)

    valid_cell_mask = np.flatnonzero(model_bins_arr >= 0)
    model_colors_arr = model_colors_arr[valid_cell_mask]
    model_states_arr = model_states_arr[valid_cell_mask]
    model_bins_arr = model_bins_arr[valid_cell_mask]

    cell_models = {}
    state_models = {}

    if model_colors_arr.shape[0]:
        df_model = pd.DataFrame(
            {
                "state": model_states_arr,
                "zbin": model_bins_arr,
                "row": np.arange(model_colors_arr.shape[0], dtype=np.int64),
            }
        )
        for (st, zb), sub in df_model.groupby(["state", "zbin"], sort=False):
            if int(zb) < 0:
                continue
            block = model_colors_arr[sub["row"].to_numpy()]
            m = _fit_robust_model(block)
            if m is not None:
                cell_models[(st, int(zb))] = m

        for st, sub in df_model.groupby("state", sort=False):
            block = model_colors_arr[sub["row"].to_numpy()]
            m = _fit_robust_model(block)
            if m is not None:
                state_models[st] = m

    if aux_global_colors is not None and aux_global_colors.shape[0]:
        global_model_data = np.vstack((colors[raw_mask], aux_global_colors))
    else:
        global_model_data = colors[raw_mask]
    global_model = _fit_robust_model(global_model_data)

    zbin_to_state_models = {}
    for (st, zb), model in cell_models.items():
        if model["support"] >= _CELL_SUPPORT_FLOOR:
            zbin_to_state_models.setdefault(int(zb), {})[st] = model

    n_bins = max(int(len(merged_edges) - 1), 1)
    self_dist, alt1_dist, alt2_dist, margin1, margin2, self_resid, alt1_resid, fb_state, fb_global = outputs

    for pos in base_idx:
        st = state[pos]
        zb = int(zbin[pos]) if zbin[pos] >= 0 else -1
        if zb < 0:
            continue

        c = colors[pos]
        self_model = None
        fallback_state = 0
        fallback_global = 0

        candidate_self = cell_models.get((st, zb))
        if candidate_self is not None and candidate_self["support"] >= _CELL_SUPPORT_FLOOR:
            self_model = candidate_self
        else:
            candidate_state = state_models.get(st)
            if candidate_state is not None and candidate_state["support"] >= _STATE_SUPPORT_FLOOR:
                self_model = candidate_state
                fallback_state = 1
            elif global_model is not None:
                self_model = global_model
                fallback_global = 1

        if self_model is None:
            continue

        d_self = _mahalanobis_distance(c, self_model)
        self_dist[pos] = d_self
        if np.isfinite(d_self):
            fb_state[pos] = np.int8(fallback_state)
            fb_global[pos] = np.int8(fallback_global)

        sr = _standardized_residual(c, self_model)
        if sr is not None:
            self_resid[pos, :] = sr

        alt_candidates = []
        same_bin = zbin_to_state_models.get(zb, {})
        for s_alt, alt_model in same_bin.items():
            if s_alt != st:
                alt_candidates.append(alt_model)

        if not alt_candidates:
            for radius in range(1, n_bins + 1):
                nearest = []
                left = zb - radius
                right = zb + radius

                if left >= 0:
                    lm = zbin_to_state_models.get(left, {})
                    for s_alt, alt_model in lm.items():
                        if s_alt != st:
                            nearest.append(alt_model)
                if right < n_bins:
                    rm = zbin_to_state_models.get(right, {})
                    for s_alt, alt_model in rm.items():
                        if s_alt != st:
                            nearest.append(alt_model)

                if nearest:
                    alt_candidates = nearest
                    break

        if not alt_candidates:
            for s_alt, alt_model in state_models.items():
                if s_alt != st and alt_model["support"] >= _STATE_SUPPORT_FLOOR:
                    alt_candidates.append(alt_model)

        if alt_candidates:
            d_alt = []
            for m_alt in alt_candidates:
                d = _mahalanobis_distance(c, m_alt)
                if np.isfinite(d):
                    d_alt.append((d, m_alt))
            if d_alt:
                d_alt.sort(key=lambda p: p[0])
                d1, m1 = d_alt[0]
                alt1_dist[pos] = d1
                r1 = _standardized_residual(c, m1)
                if r1 is not None:
                    alt1_resid[pos, :] = r1
                if np.isfinite(d_self):
                    margin1[pos] = d1 - d_self

                if len(d_alt) > 1:
                    d2 = d_alt[1][0]
                    alt2_dist[pos] = d2
                    if np.isfinite(d_self):
                        margin2[pos] = d2 - d_self

    self_dist = _clip_values(self_dist, _PCT_LOW, _PCT_HIGH)
    alt1_dist = _clip_values(alt1_dist, _PCT_LOW, _PCT_HIGH)
    alt2_dist = _clip_values(alt2_dist, _PCT_LOW, _PCT_HIGH)
    margin1 = _clip_values(margin1, _PCT_LOW, _PCT_HIGH)
    margin2 = _clip_values(margin2, _PCT_LOW, _PCT_HIGH)

    return _assemble_output(
        raw.index,
        self_dist,
        alt1_dist,
        alt2_dist,
        margin1,
        margin2,
        self_resid,
        alt1_resid,
        fb_state,
        fb_global,
    )


FEATURE_GROUPS = [
    {
        "name": "tag_redshift_compatibility_residuals",
        "fn": add_tag_redshift_compatibility_residuals,
        "depends_on": [],
        "description": "Builds redshift-bin-aware color-manifold compatibility distances for each tag state and contrastive mismatch residual margins.",
    }
]