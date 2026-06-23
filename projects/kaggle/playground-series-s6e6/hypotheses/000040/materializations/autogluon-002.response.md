import numpy as np
import pandas as pd

_REDSHIFT_BINS = (0.0, 0.3, 1.2, 2.6, 4.0, 7.0)
_I_BIN_WIDTH = 0.5
_COLOR_BINS = 14
_COLOR_CELLS = _COLOR_BINS ** 3
_CLASS_ORDER = ("GALAXY", "QSO", "STAR")
_ALPHA = 1.0
_BETA = 1.0
_EPS = 1e-12


def _find_class_column(frame):
    if not isinstance(frame, pd.DataFrame):
        return None
    lower_map = {str(col).lower(): col for col in frame.columns}
    candidates = ("class", "target", "label", "y", "y_true", "target_class")
    for candidate in candidates:
        if candidate in lower_map:
            return lower_map[candidate]
    return None


def _normalize_class_series(series):
    if series is None:
        return pd.Series(dtype="string")

    numeric = pd.to_numeric(series, errors="coerce")
    normalized = pd.Series(pd.NA, index=series.index, dtype="string")
    unique_numeric = pd.Index(pd.Series(numeric).dropna().unique())

    if len(unique_numeric) == 3 and set(unique_numeric) == {0.0, 1.0, 2.0}:
        mapping = {0.0: "GALAXY", 1.0: "QSO", 2.0: "STAR"}
        normalized = numeric.map(mapping).astype("string")
    elif len(unique_numeric) == 3 and set(unique_numeric) == {1.0, 2.0, 3.0}:
        mapping = {1.0: "GALAXY", 2.0: "QSO", 3.0: "STAR"}
        normalized = numeric.map(mapping).astype("string")
    else:
        text = series.astype("string").str.strip().str.upper()
        mapped = pd.Series(pd.NA, index=series.index, dtype="string")
        mapped.loc[text.isin({"GALAXY", "GAL"})] = "GALAXY"
        mapped.loc[text.isin({"QSO", "QUASAR"})] = "QSO"
        mapped.loc[text.isin({"STAR"})] = "STAR"
        mapped.loc[text.str.contains("GALAXY", na=False)] = "GALAXY"
        mapped.loc[text.str.contains("QSO", na=False)] = "QSO"
        mapped.loc[text.str.contains("QUASAR", na=False)] = "QSO"
        mapped.loc[text.str.contains("STAR", na=False)] = "STAR"
        mapped = mapped.mask(mapped.eq("STAR"), "STAR")
        normalized = mapped

    if normalized.notna().any():
        normalized = normalized.where(~normalized.str.fullmatch(r"^\s*$", case=False, na=False), pd.NA)

    return normalized


def _build_reference_from(source):
    if not isinstance(source, pd.DataFrame) or source.empty:
        return pd.DataFrame(columns=["u", "g", "r", "i", "redshift", "class"])

    class_col = _find_class_column(source)
    required = ("u", "g", "r", "i", "redshift")
    if class_col is None:
        return pd.DataFrame(columns=["u", "g", "r", "i", "redshift", "class"])

    if not set(required).issubset(set(source.columns)):
        return pd.DataFrame(columns=["u", "g", "r", "i", "redshift", "class"])

    cls = _normalize_class_series(source[class_col])
    if cls.notna().sum() == 0:
        return pd.DataFrame(columns=["u", "g", "r", "i", "redshift", "class"])

    ref = source.loc[cls.notna(), list(required)].copy()
    ref["class"] = cls.loc[cls.notna()]
    for col in required:
        ref[col] = pd.to_numeric(ref[col], errors="coerce")
    ref = ref.replace([np.inf, -np.inf], np.nan)
    ref["class"] = ref["class"].astype("string")
    ref = ref.dropna(subset=list(required) + ["class"]).copy()
    ref["class"] = ref["class"].where(ref["class"].isin(_CLASS_ORDER), pd.NA)
    ref = ref.dropna(subset=["class"])
    return ref.loc[:, ["u", "g", "r", "i", "redshift", "class"]]


def _make_color_edges(values):
    arr = np.asarray(values, dtype=float)
    finite = np.isfinite(arr)
    vals = arr[finite]
    if vals.size == 0:
        return np.array([-1.0, 1.0], dtype=float)

    quantiles = np.linspace(0.0, 1.0, _COLOR_BINS + 1)
    edges = np.quantile(vals, quantiles).astype(float)
    edges[0] = edges[0] - _EPS
    edges[-1] = edges[-1] + _EPS
    for i in range(1, edges.size):
        if not np.isfinite(edges[i - 1]):
            edges[i - 1] = float(i - 1)
        if edges[i] <= edges[i - 1]:
            edges[i] = edges[i - 1] + _EPS
    return edges


def _i_edges(vals):
    arr = np.asarray(vals, dtype=float)
    finite = np.isfinite(arr)
    vals = arr[finite]
    if vals.size == 0:
        return np.array((0.0, _I_BIN_WIDTH), dtype=float)

    lo = np.floor(np.min(vals) / _I_BIN_WIDTH) * _I_BIN_WIDTH
    hi = np.ceil(np.max(vals) / _I_BIN_WIDTH) * _I_BIN_WIDTH
    if hi <= lo:
        hi = lo + _I_BIN_WIDTH
    return np.arange(lo, hi + _I_BIN_WIDTH, _I_BIN_WIDTH, dtype=float)


def _digitize_fixed(values, bin_edges):
    arr = np.asarray(values, dtype=float)
    out = np.full(arr.shape, -1, dtype=np.int16)
    finite = np.isfinite(arr)
    if not np.any(finite):
        return out

    clipped = arr[finite]
    clipped = np.clip(clipped, bin_edges[0], bin_edges[-1] - _EPS)
    out[finite] = np.digitize(clipped, np.asarray(bin_edges[1:-1], dtype=float), right=False).astype(np.int16)
    return out


def _digitize_with_fixed_bins(values, bins):
    arr = np.asarray(values, dtype=float)
    out = np.full(arr.shape, -1, dtype=np.int16)
    finite = np.isfinite(arr)
    if not np.any(finite):
        return out

    clipped = arr[finite]
    clipped = np.clip(clipped, bins[0], bins[-1] - _EPS)
    out[finite] = np.digitize(clipped, np.asarray(bins[1:-1], dtype=float), right=False).astype(np.int16)
    return out


def _neighbor_sum_3d(hist):
    padded = np.pad(hist, 1, mode="constant", constant_values=0.0)
    summed = np.zeros_like(hist, dtype=float)
    for dz in (-1, 0, 1):
        z0 = 1 + dz
        z1 = z0 + _COLOR_BINS
        for dy in (-1, 0, 1):
            y0 = 1 + dy
            y1 = y0 + _COLOR_BINS
            for dx in (-1, 0, 1):
                x0 = 1 + dx
                x1 = x0 + _COLOR_BINS
                summed += padded[x0:x1, y0:y1, z0:z1]
    return summed


def _build_hist_stats(reference):
    if reference.empty:
        return None

    u = reference["u"].to_numpy(dtype=float)
    g = reference["g"].to_numpy(dtype=float)
    r = reference["r"].to_numpy(dtype=float)
    i = reference["i"].to_numpy(dtype=float)
    z = reference["redshift"].to_numpy(dtype=float)
    cls = reference["class"].map({"GALAXY": 0, "QSO": 1, "STAR": 2}).to_numpy(dtype=np.int8)

    color_ug = u - g
    color_gr = g - r
    color_ri = r - i

    z_bin = _digitize_fixed(z, _REDSHIFT_BINS)
    i_edges = _i_edges(i)
    i_bin = _digitize_with_fixed_bins(i, i_edges)

    edges_ug = _make_color_edges(color_ug)
    edges_gr = _make_color_edges(color_gr)
    edges_ri = _make_color_edges(color_ri)

    cu = _digitize_with_fixed_bins(color_ug, edges_ug)
    cg = _digitize_with_fixed_bins(color_gr, edges_gr)
    cri = _digitize_with_fixed_bins(color_ri, edges_ri)

    valid = (cls >= 0) & (z_bin >= 0) & (i_bin >= 0) & (cu >= 0) & (cg >= 0) & (cri >= 0)
    if not np.any(valid):
        return None

    cls = cls[valid]
    z_bin = z_bin[valid]
    i_bin = i_bin[valid]
    cu = cu[valid]
    cg = cg[valid]
    cri = cri[valid]

    histograms = {}
    class_stratum_totals = {}
    stratum_counts = {}
    class_totals = np.zeros(3, dtype=float)

    for row in range(len(cls)):
        c = int(cls[row])
        zk = int(z_bin[row])
        ik = int(i_bin[row])
        key = (c, zk, ik)
        if key not in histograms:
            histograms[key] = np.zeros((_COLOR_BINS, _COLOR_BINS, _COLOR_BINS), dtype=float)
        histograms[key][int(cu[row]), int(cg[row]), int(cri[row])] += 1.0

        str_key = (zk, ik)
        if str_key not in stratum_counts:
            stratum_counts[str_key] = np.zeros(3, dtype=float)
        stratum_counts[str_key][c] += 1.0

        class_totals[c] += 1.0

    if class_totals.sum() == 0:
        return None

    class_stratum_totals = {k: float(v.sum()) for k, v in histograms.items()}
    stratum_log_priors = {}
    for key, vals in stratum_counts.items():
        total = float(vals.sum())
        if total <= 0:
            continue
        probs = (vals + _BETA) / (total + 3.0 * _BETA)
        stratum_log_priors[key] = np.log(np.maximum(probs, _EPS))

    global_total = float(class_totals.sum())
    global_prior = (class_totals + _BETA) / (global_total + 3.0 * _BETA)
    log_global_prior = np.log(np.maximum(global_prior, _EPS))

    neighbor_histograms = {k: _neighbor_sum_3d(v) for k, v in histograms.items()}

    return {
        "histograms": histograms,
        "neighbor_histograms": neighbor_histograms,
        "class_stratum_totals": class_stratum_totals,
        "stratum_log_priors": stratum_log_priors,
        "log_global_prior": log_global_prior,
        "i_edges": i_edges,
        "edges_ug": edges_ug,
        "edges_gr": edges_gr,
        "edges_ri": edges_ri,
    }


def _baseline_features(index):
    n = len(index)
    zero = np.zeros(n, dtype=float)
    entropy = np.full(n, np.log(3.0), dtype=float)
    return pd.DataFrame(
        {
            "s_galaxy": zero.copy(),
            "s_qso": zero.copy(),
            "s_star": zero.copy(),
            "margin_qso_minus_star": zero.copy(),
            "margin_star_minus_galaxy": zero.copy(),
            "posterior_entropy": entropy,
        },
        index=index,
    )


def add_class_conditional_color_density_posteriors(raw, deps, aux):
    required = ("u", "g", "r", "i", "redshift")
    if not set(required).issubset(set(raw.columns)):
        return _baseline_features(raw.index)

    ref = _build_reference_from(aux)
    if ref.empty:
        ref = _build_reference_from(raw)

    stats = _build_hist_stats(ref)
    if stats is None:
        return _baseline_features(raw.index)

    u = pd.to_numeric(raw["u"], errors="coerce").to_numpy(dtype=float)
    g = pd.to_numeric(raw["g"], errors="coerce").to_numpy(dtype=float)
    r = pd.to_numeric(raw["r"], errors="coerce").to_numpy(dtype=float)
    i = pd.to_numeric(raw["i"], errors="coerce").to_numpy(dtype=float)
    z = pd.to_numeric(raw["redshift"], errors="coerce").to_numpy(dtype=float)

    color_ug = u - g
    color_gr = g - r
    color_ri = r - i

    z_bin = _digitize_fixed(z, _REDSHIFT_BINS)
    i_bin = _digitize_with_fixed_bins(i, stats["i_edges"])
    cu = _digitize_with_fixed_bins(color_ug, stats["edges_ug"])
    cg = _digitize_with_fixed_bins(color_gr, stats["edges_gr"])
    cri = _digitize_with_fixed_bins(color_ri, stats["edges_ri"])

    n = len(raw)
    log_global = stats["log_global_prior"]
    scores = np.tile(log_global, (n, 1))

    valid = (z_bin >= 0) & (i_bin >= 0) & (cu >= 0) & (cg >= 0) & (cri >= 0)
    idxs = np.nonzero(valid)[0]

    histograms = stats["histograms"]
    neighbors = stats["neighbor_histograms"]
    class_stratum_totals = stats["class_stratum_totals"]
    stratum_log_priors = stats["stratum_log_priors"]

    for row in idxs:
        zb = int(z_bin[row])
        ib = int(i_bin[row])
        cu_i = int(cu[row])
        cg_i = int(cg[row])
        ri_i = int(cri[row])

        baseline = stratum_log_priors.get((zb, ib), log_global)
        scores[row, :] = baseline

        for c in (0, 1, 2):
            key = (c, zb, ib)
            hist = histograms.get(key)
            if hist is None:
                continue

            total = class_stratum_totals.get(key, 0.0)
            if total <= 0:
                continue

            cell = float(hist[cu_i, cg_i, ri_i])
            if cell > 0.0:
                p = (cell + _ALPHA) / (total + _ALPHA * _COLOR_CELLS)
                scores[row, c] = baseline[c] + np.log(p)
                continue

            neigh = neighbors.get(key)
            neigh_count = float(neigh[cu_i, cg_i, ri_i]) if neigh is not None else 0.0
            if neigh_count > 0.0:
                p = (neigh_count + _ALPHA) / (total + _ALPHA * _COLOR_CELLS)
                scores[row, c] = baseline[c] + np.log(p)

    probs = np.exp(scores - scores.max(axis=1, keepdims=True))
    probs = probs / (probs.sum(axis=1, keepdims=True) + _EPS)
    entropy = -np.sum(probs * np.log(np.maximum(probs, _EPS)), axis=1)

    return pd.DataFrame(
        {
            "s_galaxy": scores[:, 0],
            "s_qso": scores[:, 1],
            "s_star": scores[:, 2],
            "margin_qso_minus_star": scores[:, 1] - scores[:, 2],
            "margin_star_minus_galaxy": scores[:, 2] - scores[:, 0],
            "posterior_entropy": entropy,
        },
        index=raw.index,
    )


FEATURE_GROUPS = [
    {
        "name": "class_conditional_color_density_posteriors",
        "fn": add_class_conditional_color_density_posteriors,
        "depends_on": [],
        "description": "Builds stratum-conditioned color-histogram Bayesian scores for GALAXY/QSO/STAR with neighbor and prior backoff.",
    }
]