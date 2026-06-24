import numpy as np
import pandas as pd


_TRAIN_MASK_KEYS = ("is_train", "train_mask", "train", "is_training", "is_train_row", "split", "dataset")


def _to_float(raw, column):
    return pd.to_numeric(raw[column], errors="coerce").to_numpy(dtype=float)


def _candidate_bool_mask(frame, key, n):
    if not isinstance(frame, pd.DataFrame) or key not in frame.columns or len(frame) != n:
        return None

    series = frame[key]

    if pd.api.types.is_bool_dtype(series):
        mask = series.astype(bool).to_numpy(dtype=bool)
    elif pd.api.types.is_numeric_dtype(series):
        numeric = pd.to_numeric(series, errors="coerce").fillna(0.0)
        mask = (numeric != 0).to_numpy(dtype=bool)
    else:
        text = series.astype(str).str.strip().str.lower()
        if text.dropna().empty:
            return None
        if text.isin({"train", "1", "true", "t", "yes", "y"}).any():
            mask = text.eq("train").to_numpy(dtype=bool)
        else:
            return None

    if mask.ndim != 1 or mask.size != n:
        return None
    if not mask.any() or mask.all():
        return None
    return mask


def _infer_train_mask(raw, aux):
    n = len(raw)
    if n == 0:
        return np.array([], dtype=bool)

    if isinstance(aux, pd.DataFrame) and not aux.empty:
        for key in _TRAIN_MASK_KEYS:
            mask = _candidate_bool_mask(aux, key, n)
            if mask is not None:
                return mask

    for key in _TRAIN_MASK_KEYS:
        mask = _candidate_bool_mask(raw, key, n)
        if mask is not None:
            return mask

    # Fallback: no explicit training indicator is available.
    return np.ones(n, dtype=bool)


def _term_statistics(values, train_mask):
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        zero = np.array([0.0], dtype=float)
        return {"clipped": zero, "scale": 0.01, "span": 1.0}

    if len(train_mask) == len(arr):
        train = arr[train_mask]
    else:
        train = arr
    train = train[np.isfinite(train)]
    if train.size == 0:
        train = arr[np.isfinite(arr)]
    if train.size == 0:
        train = np.array([0.0], dtype=float)

    lo = float(np.nanpercentile(train, 0.5))
    hi = float(np.nanpercentile(train, 99.5))
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        lo = float(np.nanmin(train))
        hi = float(np.nanmax(train))
        if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
            lo = -1.0
            hi = 1.0

    clipped = np.clip(arr, lo, hi)
    clipped_train = clipped[np.isfinite(clipped)]
    if clipped_train.size == 0:
        clipped_train = np.array([0.0], dtype=float)

    center = float(np.nanmedian(clipped_train))
    mad = float(np.nanmedian(np.abs(clipped_train - center)))
    if not np.isfinite(mad):
        mad = 0.0

    span = float(np.nanmax(clipped_train) - np.nanmin(clipped_train))
    if not np.isfinite(span) or span <= 0.0:
        span = 1.0

    return {
        "clipped": clipped,
        "scale": max(mad, 0.01),
        "span": span,
    }


def _interval_penalty(x, low, high, scale):
    vals = np.asarray(x, dtype=float)
    out = np.zeros_like(vals, dtype=float)

    finite = np.isfinite(vals)
    below = finite & (vals < low)
    above = finite & (vals > high)

    if np.any(below):
        out[below] = (low - vals[below]) / scale
    if np.any(above):
        out[above] = (vals[above] - high) / scale
    return out


def _hi_penalty(x, k, scale):
    vals = np.asarray(x, dtype=float)
    thr = np.asarray(k, dtype=float)
    out = np.zeros_like(vals, dtype=float)

    diff = vals - thr
    finite = np.isfinite(vals) & np.isfinite(thr)
    active = finite & (diff > 0.0)
    out[active] = diff[active] / scale
    return out


def _lo_penalty(x, k, scale):
    vals = np.asarray(x, dtype=float)
    thr = np.asarray(k, dtype=float)
    out = np.zeros_like(vals, dtype=float)

    diff = thr - vals
    finite = np.isfinite(vals) & np.isfinite(thr)
    active = finite & (diff > 0.0)
    out[active] = diff[active] / scale
    return out


def add_segue_stellar_subtype_margins(raw, deps, aux):
    train_mask = _infer_train_mask(raw, aux)

    u = _to_float(raw, "u")
    g = _to_float(raw, "g")
    r = _to_float(raw, "r")
    i = _to_float(raw, "i")
    z = _to_float(raw, "z")
    redshift = _to_float(raw, "redshift")

    ug = u - g
    gr = g - r
    ri = r - i
    iz = i - z
    gi = g - i

    l_color = -0.436 * u + 1.129 * g - 0.119 * r - 0.574 * i + 0.1984
    s_color = -0.249 * u + 0.794 * g - 0.555 * r + 0.234
    p1 = 0.91 * ug + 0.415 * gr - 1.280
    v = 0.283 * ug - 0.354 * gr + 0.455 * ri + 0.766 * iz
    ug_plus_2gr = ug + 2.0 * gr

    terms = {
        "ug": ug,
        "gr": gr,
        "ri": ri,
        "iz": iz,
        "gi": gi,
        "r": r,
        "l_color": l_color,
        "s_color": s_color,
        "p1": p1,
        "v": v,
        "redshift": redshift,
        "ug_plus_2gr": ug_plus_2gr,
    }
    stats = {name: _term_statistics(values, train_mask) for name, values in terms.items()}

    ug_c = stats["ug"]["clipped"]
    gr_c = stats["gr"]["clipped"]
    ri_c = stats["ri"]["clipped"]
    iz_c = stats["iz"]["clipped"]
    gi_c = stats["gi"]["clipped"]
    r_c = stats["r"]["clipped"]
    l_c = stats["l_color"]["clipped"]
    v_c = stats["v"]["clipped"]
    redshift_c = stats["redshift"]["clipped"]
    ug2gr_c = stats["ug_plus_2gr"]["clipped"]

    s_ug = stats["ug"]["scale"]
    s_gr = stats["gr"]["scale"]
    s_ri = stats["ri"]["scale"]
    s_iz = stats["iz"]["scale"]
    s_gi = stats["gi"]["scale"]
    s_r = stats["r"]["scale"]
    s_l = stats["l_color"]["scale"]
    s_v = stats["v"]["scale"]

    span_ug = stats["ug"]["span"]
    span_gr = stats["gr"]["span"]
    span_ri = stats["ri"]["span"]
    span_iz = stats["iz"]["span"]
    span_gi = stats["gi"]["span"]
    span_r = stats["r"]["span"]
    span_l = stats["l_color"]["span"]

    w_ug = 1.0 / max(span_ug, 1e-6)
    w_gr = 1.0 / max(span_gr, 1e-6)
    w_ri = 1.0 / max(span_ri, 1e-6)
    w_iz = 1.0 / max(span_iz, 1e-6)
    w_gi = 1.0 / max(span_gi, 1e-6)
    w_r = 1.0 / max(span_r, 1e-6)
    w_l = 1.0 / max(span_l, 1e-6)
    w_v = 1.0 / max(max(stats["v"]["span"], 1e-6), 1e-6)
    w_ug2gr = 1.0 / max(stats["ug_plus_2gr"]["span"], 1e-6)
    w_r_hi = 1.0 / max(span_r, 1e-6)
    w_ri_hi = 1.0 / max(span_ri, 1e-6)
    w_gi_hi = 1.0 / max(span_gi, 1e-6)

    g_z = 1.0 + np.maximum(0.0, redshift_c - 0.30) / 0.30

    white_dwarf = (
        w_gr * _interval_penalty(gr_c, -1.0, -0.2, s_gr)
        + w_ug * _interval_penalty(ug_c, -1.0, 0.7, s_ug)
        + w_ug2gr * _hi_penalty(ug2gr_c, -0.1, s_ug)
    )

    cool_ug = np.where(
        gi_c < 0.6,
        _interval_penalty(ug_c, -0.4, 1.0, s_ug),
        _interval_penalty(ug_c, -0.2, 1.4, s_ug),
    )
    cool_white_dwarf = (
        w_r * _interval_penalty(r_c, 14.5, 20.5, s_r)
        + w_gi * _interval_penalty(gi_c, -2.0, 1.7, s_gi)
        + w_ug * cool_ug
    )

    a_bhb = (
        w_ug * _interval_penalty(ug_c, 0.8, 1.5, s_ug)
        + w_gr * _interval_penalty(gr_c, -0.5, 0.2, s_gr)
        + w_v * _interval_penalty(v_c, -0.15, 0.15, s_v)
    )

    k_giants = (
        w_gr * _interval_penalty(gr_c, 0.35, 0.8, s_gr)
        + w_ri * _interval_penalty(ri_c, 0.15, 0.6, s_ri)
        + w_l * _lo_penalty(l_c, 0.07, s_l)
    )

    low_metallicity = (
        w_gr * _interval_penalty(gr_c, -0.5, 0.75, s_gr)
        + w_ug * _interval_penalty(ug_c, 0.6, 3.0, s_ug)
        + w_l * _lo_penalty(l_c, 0.135, s_l)
    )

    m_subdwarf = (
        w_ri_hi * _hi_penalty(ri_c, 0.9, s_ri)
        + w_ri_hi * _hi_penalty(ri_c, 0.787 * gr_c - 0.356, s_ri)
        + w_gi_hi * _interval_penalty(gi_c, 1.8, 2.4, s_gi)
    )

    iz_piece = np.where(
        ri_c < 1.0,
        _interval_penalty(iz_c, -0.5, 1.0, s_iz),
        _interval_penalty(iz_c, -0.2, 0.7, s_iz),
    )
    main_sequence_white_dwarf_pairs = (
        w_ug * _hi_penalty(ug_c, 2.25, s_ug)
        + w_gr * _interval_penalty(gr_c, -0.2, 1.2, s_gr)
        + w_ri * _interval_penalty(ri_c, 0.5, 2.0, s_ri)
        + w_gr * _lo_penalty(gr_c, -1.0 * ri_c + 1.13, s_gr)
        + w_gr * _hi_penalty(gr_c, 0.95 * ri_c + 0.50, s_gr)
        + w_iz * iz_piece
    )

    white_dwarf *= g_z
    cool_white_dwarf *= g_z
    a_bhb *= g_z
    k_giants *= g_z
    low_metallicity *= g_z
    m_subdwarf *= g_z
    main_sequence_white_dwarf_pairs *= g_z

    prototype_matrix = np.column_stack(
        [
            white_dwarf,
            cool_white_dwarf,
            a_bhb,
            k_giants,
            low_metallicity,
            m_subdwarf,
            main_sequence_white_dwarf_pairs,
        ]
    )

    best_and_next = np.partition(prototype_matrix, 1, axis=1)
    star_margin_best = best_and_next[:, 0]
    star_margin_second = best_and_next[:, 1]
    star_margin_gap = star_margin_second - star_margin_best
    star_inlier_count = np.sum(prototype_matrix <= 0.02, axis=1).astype(np.int16)
    star_confidence = np.exp(-star_margin_best)

    return pd.DataFrame(
        {
            "star_margin_best": star_margin_best,
            "star_margin_second": star_margin_second,
            "star_margin_gap": star_margin_gap,
            "star_inlier_count": star_inlier_count,
            "star_confidence": star_confidence,
        },
        index=raw.index,
    )


FEATURE_GROUPS = [
    {
        "name": "segue_stellar_subtype_margins",
        "fn": add_segue_stellar_subtype_margins,
        "depends_on": [],
        "description": "Builds clipped SEGUE-inspired stellar subtype margin penalties and emits compact margin-gap summary features for robust star-locus proximity.",
    }
]