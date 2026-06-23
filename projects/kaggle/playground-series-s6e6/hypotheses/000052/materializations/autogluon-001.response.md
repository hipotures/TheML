import numpy as np
import pandas as pd

_EPS = 1e-6
_CLAUSE_CLIP = 6.0
_INSIDE_EPS = 1e-4
_REFERENCE_BANDS = ("u", "g", "r", "i", "z")


def _to_float(series):
    values = pd.to_numeric(series, errors="coerce").astype("float64")
    values = values.replace([np.inf, -np.inf], np.nan)
    if values.isna().all():
        return values.fillna(0.0)
    fill = values.median(skipna=True)
    if not np.isfinite(fill):
        fill = 0.0
    return values.fillna(fill)


def _mad_scale(values, min_scale=_EPS):
    arr = pd.to_numeric(values, errors="coerce").astype("float64").to_numpy()
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float(1.0)
    med = np.median(arr)
    mad = np.median(np.abs(arr - med))
    scale = 1.4826 * mad
    if (not np.isfinite(scale)) or scale < min_scale:
        scale = float(min_scale)
    return float(scale)


def _interval_margin(values, lower, upper):
    v = np.asarray(values, dtype=np.float64)
    return np.where(v < lower, lower - v, np.where(v > upper, v - upper, 0.0))


def _upper_margin(values, upper):
    v = np.asarray(values, dtype=np.float64)
    return np.maximum(0.0, v - upper)


def _lower_margin(values, lower):
    v = np.asarray(values, dtype=np.float64)
    return np.maximum(0.0, lower - v)


def _normalize_clause(margin, scale, width=1.0, weight=1.0, clip=_CLAUSE_CLIP):
    m = np.asarray(margin, dtype=np.float64)
    w = np.asarray(width, dtype=np.float64)
    s = np.asarray(scale, dtype=np.float64)
    denom = np.maximum(np.abs(s), _EPS) * np.maximum(np.abs(w), _EPS)
    norm = (m * float(weight)) / denom
    return np.clip(norm, 0.0, clip)


def _reference_series(raw, aux, name):
    raw_series = _to_float(raw[name])
    if isinstance(aux, pd.DataFrame) and not aux.empty and name in aux.columns:
        aux_series = _to_float(aux[name])
        return pd.concat([raw_series, aux_series], axis=0)
    return raw_series


def add_segue_stellar_subtype_margins(raw, deps, aux):
    idx = raw.index

    u = _to_float(raw["u"])
    g = _to_float(raw["g"])
    r = _to_float(raw["r"])
    i = _to_float(raw["i"])
    z = _to_float(raw["z"])

    u_ref = _reference_series(raw, aux, "u")
    g_ref = _reference_series(raw, aux, "g")
    r_ref = _reference_series(raw, aux, "r")
    i_ref = _reference_series(raw, aux, "i")
    z_ref = _reference_series(raw, aux, "z")

    ug = u - g
    gr = g - r
    ri = r - i
    iz = i - z
    gi = g - i

    ug_ref = u_ref - g_ref
    gr_ref = g_ref - r_ref
    ri_ref = r_ref - i_ref
    iz_ref = i_ref - z_ref

    v = 0.283 * ug - 0.354 * gr + 0.455 * (r - i) + 0.766 * iz
    l_color = -0.436 * u + 1.129 * g - 0.119 * r - 0.574 * i + 0.1984
    v_ref = 0.283 * ug_ref - 0.354 * gr_ref + 0.455 * (r_ref - i_ref) + 0.766 * iz_ref
    l_color_ref = -0.436 * u_ref + 1.129 * g_ref - 0.119 * r_ref - 0.574 * i_ref + 0.1984

    scale_u = _mad_scale(u_ref)
    scale_g = _mad_scale(g_ref)
    scale_r = _mad_scale(r_ref)
    scale_i = _mad_scale(i_ref)
    scale_z = _mad_scale(z_ref)
    scale_ug = _mad_scale(ug_ref)
    scale_gr = _mad_scale(gr_ref)
    scale_ri = _mad_scale(ri_ref)
    scale_iz = _mad_scale(iz_ref)
    scale_gi = _mad_scale(gi)
    scale_v = _mad_scale(v_ref)
    scale_l = _mad_scale(l_color_ref)
    scale_ug2gr = _mad_scale(ug_ref + 2.0 * gr_ref)
    scale_ri_line = _mad_scale(ri_ref - (0.787 * gr_ref - 0.356))

    # White dwarf
    wd_gr = _interval_margin(gr, -1.0, -0.2)
    wd_ug = _interval_margin(ug, -1.0, 0.7)
    wd_line = _upper_margin(ug + 2.0 * gr, -0.1)
    white_dwarf_margin = (
        _normalize_clause(wd_gr, scale_gr, width=0.8)
        + _normalize_clause(wd_ug, scale_ug, width=1.7)
        + _normalize_clause(wd_line, scale_ug2gr, width=1.0)
    )

    # Cool white dwarf with branch-adjusted gi envelope in r
    r_band = _interval_margin(r, 14.5, 20.5)
    r_branch = ((20.0 - r).clip(-4.0, 4.0) / 4.0)
    gi_low = -2.0 + 0.04 * r_branch
    gi_high = 1.7 - 0.04 * r_branch
    gi_width = np.maximum(gi_high - gi_low, 3.2)
    cw_ugi = _upper_margin(gi, gi_high)
    cw_lgi = _lower_margin(gi, gi_low)
    cool_white_dwarf_margin = (
        _normalize_clause(r_band, scale_r, width=6.0)
        + _normalize_clause(cw_lgi, scale_gi, width=gi_width)
        + _normalize_clause(cw_ugi, scale_gi, width=gi_width)
    )

    # A/BHB
    ab_ug = _interval_margin(ug, 0.8, 1.5)
    ab_gr = _interval_margin(gr, -0.5, 0.2)
    ab_v = np.abs(v)
    abhb_margin = (
        _normalize_clause(ab_ug, scale_ug, width=0.7)
        + _normalize_clause(ab_gr, scale_gr, width=0.7)
        + _normalize_clause(ab_v, scale_v, width=1.0)
    )

    # K giants
    kg_gr = _interval_margin(gr, 0.35, 0.8)
    kg_ri = _interval_margin(ri, 0.15, 0.6)
    kg_l = _lower_margin(l_color, 0.07)
    k_giants_margin = (
        _normalize_clause(kg_gr, scale_gr, width=0.45)
        + _normalize_clause(kg_ri, scale_ri, width=0.45)
        + _normalize_clause(kg_l, scale_l, width=0.07)
    )

    # Low metallicity
    lm_gr = _interval_margin(gr, -0.5, 0.75)
    lm_ug = _interval_margin(ug, 0.6, 3.0)
    lm_l = _lower_margin(l_color, 0.135)
    low_metallicity_margin = (
        _normalize_clause(lm_gr, scale_gr, width=1.25)
        + _normalize_clause(lm_ug, scale_ug, width=2.4)
        + _normalize_clause(lm_l, scale_l, width=0.135)
    )

    # M subdwarfs
    ms_ri = _upper_margin(ri, 0.9)
    ms_line = _upper_margin(ri - (0.787 * gr - 0.356), 0.0)
    ms_gi = _interval_margin(gi, 1.8, 2.4)
    m_subdwarf_margin = (
        _normalize_clause(ms_ri, scale_ri, width=0.9)
        + _normalize_clause(ms_line, scale_ri_line, width=0.6)
        + _normalize_clause(ms_gi, scale_gi, width=0.6)
    )

    # Main-sequence + white-dwarf pair track (with ri branch on iz side-conditions)
    mswd_ug = _upper_margin(ug, 2.25)
    mswd_gr = _interval_margin(gr, -0.2, 1.2)
    mswd_ri = _interval_margin(ri, 0.5, 2.0)
    mswd_lower = _lower_margin(gr, -19.78 * ri + 11.13)
    mswd_upper = _upper_margin(gr, 0.95 * ri + 0.5)
    iz_low = np.where(ri < 1.0, -0.50, np.where(ri < 1.5, -0.28, -0.18))
    iz_high = np.where(ri < 1.0, 0.55, np.where(ri < 1.5, 0.85, 1.05))
    iz_width = np.maximum(iz_high - iz_low, 0.2)
    mswd_iz_low = _lower_margin(iz, iz_low)
    mswd_iz_high = _upper_margin(iz, iz_high)
    main_sequence_white_dwarf_pairs_margin = (
        _normalize_clause(mswd_ug, scale_ug, width=2.25)
        + _normalize_clause(mswd_gr, scale_gr, width=1.4)
        + _normalize_clause(mswd_ri, scale_ri, width=1.5)
        + _normalize_clause(mswd_lower, scale_gr, width=1.0)
        + _normalize_clause(mswd_upper, scale_gr, width=1.0)
        + _normalize_clause(mswd_iz_low, scale_iz, width=iz_width)
        + _normalize_clause(mswd_iz_high, scale_iz, width=iz_width)
    )

    prototype_scores = pd.DataFrame(
        {
            "segue_white_dwarf_margin": pd.Series(white_dwarf_margin, index=idx),
            "segue_cool_white_dwarf_margin": pd.Series(cool_white_dwarf_margin, index=idx),
            "segue_ahb_margin": pd.Series(abhb_margin, index=idx),
            "segue_k_giants_margin": pd.Series(k_giants_margin, index=idx),
            "segue_low_metallicity_margin": pd.Series(low_metallicity_margin, index=idx),
            "segue_m_subdwarf_margin": pd.Series(m_subdwarf_margin, index=idx),
            "segue_main_sequence_white_dwarf_pairs_margin": pd.Series(
                main_sequence_white_dwarf_pairs_margin, index=idx
            ),
        },
        index=idx,
    )

    sorted_scores = np.sort(prototype_scores.to_numpy(dtype=np.float64), axis=1)
    best = sorted_scores[:, 0]
    second = sorted_scores[:, 1]
    margin_gap = second - best
    count_inside = (prototype_scores <= _INSIDE_EPS).sum(axis=1).astype(np.int16)

    return pd.DataFrame(
        {
            "segue_best_star_margin": best,
            "segue_second_best_margin": second,
            "segue_margin_gap": margin_gap,
            "segue_count_inside_prototypes": count_inside.astype(np.int16),
        },
        index=idx,
    )


FEATURE_GROUPS = [
    {
        "name": "segue_stellar_subtype_margins",
        "fn": add_segue_stellar_subtype_margins,
        "depends_on": [],
        "description": "Creates normalized SEGUE-inspired signed-distance stellar-subtype geometry margins and prototype-aggregate proximity features in calibrated color space.",
    }
]