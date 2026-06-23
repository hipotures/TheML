import numpy as np
import pandas as pd

REDDENING_COEFFS = (5.155, 3.793, 2.751, 2.086, 1.479)
REDDENING_DIRECTION = (1.362, 1.042, 0.665, 0.607)
EBV_CLIP_RANGE = (0.0, 1.2)
FINE_EBV_BIN_DEG = 1.0
COARSE_EBV_BIN_DEG = 10.0
CLIP_PERCENTILES = (0.5, 99.5)


def _normalize_name(name):
    return "".join(ch for ch in str(name).lower() if ch.isalnum())


def _find_column(frame, candidates):
    normalized = {_normalize_name(col): col for col in frame.columns}
    for cand in candidates:
        key = _normalize_name(cand)
        if key in normalized:
            return normalized[key]
    return None


def _to_float_array(values):
    return pd.to_numeric(values, errors="coerce").to_numpy(dtype="float64")


def _equatorial_to_galactic_l_b(ra_deg, dec_deg):
    ra = np.deg2rad(_to_float_array(ra_deg))
    dec = np.deg2rad(_to_float_array(dec_deg))

    x_eq = np.cos(dec) * np.cos(ra)
    y_eq = np.cos(dec) * np.sin(ra)
    z_eq = np.sin(dec)

    x_gal = -0.0548755604 * x_eq - 0.8734370902 * y_eq - 0.4838350155 * z_eq
    y_gal = 0.4941094279 * x_eq - 0.4448296300 * y_eq + 0.7469822445 * z_eq
    z_gal = -0.8676661490 * x_eq - 0.1980763734 * y_eq + 0.4559837762 * z_eq

    l = np.degrees(np.arctan2(y_gal, x_gal))
    l = np.mod(l, 360.0)
    b = np.degrees(np.arcsin(np.clip(z_gal, -1.0, 1.0)))
    return l, b


def _build_aux_ebv_map(aux):
    if aux is None or getattr(aux, "empty", True):
        return None

    ebv_col = _find_column(
        aux,
        (
            "ebv",
            "e_bv",
            "ebvsf",
            "ebvsfd",
            "ebvschlegel",
            "ebv_sfd",
            "ebvreddening",
            "av",
            "a_v",
            "extinction",
            "sfd_ebv",
        ),
    )
    if ebv_col is None:
        return None

    ebv = _to_float_array(aux[ebv_col])

    l_col = _find_column(
        aux,
        ("l", "galactic_l", "l_galactic", "lgal", "l_deg", "l_deg_deg"),
    )
    b_col = _find_column(
        aux,
        ("b", "galactic_b", "b_galactic", "bgal", "b_deg", "b_deg_deg"),
    )

    if l_col is not None and b_col is not None:
        l = _to_float_array(aux[l_col])
        b = _to_float_array(aux[b_col])
    else:
        alpha_col = _find_column(aux, ("alpha", "ra", "rightascension", "right_ascension"))
        delta_col = _find_column(aux, ("delta", "dec", "declination"))
        if alpha_col is None or delta_col is None:
            return None
        l, b = _equatorial_to_galactic_l_b(aux[alpha_col], aux[delta_col])

    keep = np.isfinite(ebv) & np.isfinite(l) & np.isfinite(b)
    if not np.any(keep):
        return None

    return pd.DataFrame(
        {
            "l": np.mod(l[keep], 360.0),
            "b": np.clip(b[keep], -90.0, 90.0),
            "ebv": ebv[keep],
        }
    )


def _estimate_ebv_from_aux(alpha_deg, delta_deg, aux):
    l_raw, b_raw = _equatorial_to_galactic_l_b(alpha_deg, delta_deg)
    n = len(l_raw)
    aux_map = _build_aux_ebv_map(aux)

    if aux_map is None or aux_map.empty:
        return np.zeros(n, dtype="float64"), np.ones(n, dtype=bool)

    b_raw = np.clip(b_raw, -90.0, 90.0)
    l_map = np.mod(aux_map["l"].to_numpy(dtype="float64"), 360.0)
    b_map = np.clip(aux_map["b"].to_numpy(dtype="float64"), -90.0, 90.0)
    ebv_map_vals = aux_map["ebv"].to_numpy(dtype="float64")

    fine_l = np.floor_divide(l_map, FINE_EBV_BIN_DEG).astype("int16")
    fine_b = np.floor_divide(b_map + 90.0, FINE_EBV_BIN_DEG).astype("int16")
    fine_df = pd.DataFrame(
        {
            "l_bin": fine_l,
            "b_bin": fine_b,
            "ebv": ebv_map_vals,
        }
    )
    fine_lookup = fine_df.groupby(["l_bin", "b_bin"])["ebv"].median()

    raw_l_fine = np.floor_divide(np.mod(l_raw, 360.0), FINE_EBV_BIN_DEG).astype("int16")
    raw_b_fine = np.floor_divide(b_raw + 90.0, FINE_EBV_BIN_DEG).astype("int16")
    raw_fine_idx = pd.MultiIndex.from_arrays(
        (raw_l_fine, raw_b_fine), names=["l_bin", "b_bin"]
    )
    ebv = fine_lookup.reindex(raw_fine_idx).to_numpy(dtype="float64")
    missing_primary = np.isnan(ebv)

    if np.any(missing_primary):
        coarse_l = np.floor_divide(l_map, COARSE_EBV_BIN_DEG).astype("int16")
        coarse_b = np.floor_divide(b_map + 90.0, COARSE_EBV_BIN_DEG).astype("int16")
        coarse_df = pd.DataFrame(
            {
                "l_bin10": coarse_l,
                "b_bin10": coarse_b,
                "ebv": ebv_map_vals,
            }
        )
        coarse_lookup = coarse_df.groupby(["l_bin10", "b_bin10"])["ebv"].median()
        raw_l_coarse = np.floor_divide(np.mod(l_raw, 360.0), COARSE_EBV_BIN_DEG).astype("int16")
        raw_b_coarse = np.floor_divide(b_raw + 90.0, COARSE_EBV_BIN_DEG).astype("int16")
        raw_coarse_idx = pd.MultiIndex.from_arrays(
            (raw_l_coarse, raw_b_coarse), names=["l_bin10", "b_bin10"]
        )
        ebv_coarse = coarse_lookup.reindex(raw_coarse_idx).to_numpy(dtype="float64")
        ebv = np.where(missing_primary, ebv_coarse, ebv)

    if np.any(np.isnan(ebv)):
        global_ebv = np.nanmedian(ebv_map_vals)
        if not np.isfinite(global_ebv):
            global_ebv = 0.0
        ebv = np.where(np.isnan(ebv), global_ebv, ebv)

    ebv = np.clip(ebv, EBV_CLIP_RANGE[0], EBV_CLIP_RANGE[1])
    return ebv, missing_primary


def _quantize_6_bins(values, index):
    arr = np.asarray(values, dtype="float64")
    n = len(arr)
    default = pd.Series(np.zeros(n, dtype="int16"), index=index)

    finite = np.isfinite(arr)
    if finite.sum() < 2:
        return default

    edges = np.nanpercentile(arr, [0.0, 100.0 / 6.0, 2.0 * 100.0 / 6.0, 3.0 * 100.0 / 6.0, 4.0 * 100.0 / 6.0, 5.0 * 100.0 / 6.0, 100.0])
    edges = np.asarray(edges, dtype="float64")
    if not np.all(np.isfinite(edges)):
        return default

    # enforce strict monotonic edges
    for i in range(1, len(edges)):
        if edges[i] <= edges[i - 1]:
            edges[i] = edges[i - 1] + 1e-9 * (abs(edges[i - 1]) + 1.0)

    if edges[0] == edges[-1]:
        return default

    bins = edges[1:-1]
    codes = np.digitize(arr, bins, right=False)
    codes = np.clip(codes, 0, 5)
    return pd.Series(codes.astype("int16"), index=index)


def _clip_0_5_99_5(values):
    arr = np.asarray(values, dtype="float64")
    finite = np.isfinite(arr)
    if finite.sum() < 2:
        return arr
    lo = np.nanpercentile(arr, CLIP_PERCENTILES[0])
    hi = np.nanpercentile(arr, CLIP_PERCENTILES[1])
    if not np.isfinite(lo) or not np.isfinite(hi) or lo >= hi:
        return arr
    return np.clip(arr, lo, hi)


def add_foreground_reddening_geometry(raw, deps, aux):
    idx = raw.index
    n = len(raw)

    alpha = raw["alpha"]
    delta = raw["delta"]

    l_deg, b_deg = _equatorial_to_galactic_l_b(alpha, delta)
    abs_b = np.abs(b_deg)

    ebv, ebv_missing = _estimate_ebv_from_aux(alpha, delta, aux)

    u = _to_float_array(raw["u"])
    g = _to_float_array(raw["g"])
    r = _to_float_array(raw["r"])
    i = _to_float_array(raw["i"])
    z = _to_float_array(raw["z"])
    redshift = _to_float_array(raw["redshift"])

    u0 = u - REDDENING_COEFFS[0] * ebv
    g0 = g - REDDENING_COEFFS[1] * ebv
    r0 = r - REDDENING_COEFFS[2] * ebv
    i0 = i - REDDENING_COEFFS[3] * ebv
    z0 = z - REDDENING_COEFFS[4] * ebv

    c_ug = u - g
    c_gr = g - r
    c_ri = r - i
    c_iz = i - z

    c0_ug = u0 - g0
    c0_gr = g0 - r0
    c0_ri = r0 - i0
    c0_iz = i0 - z0

    d0, d1, d2, d3 = REDDENING_DIRECTION
    norm2 = d0 * d0 + d1 * d1 + d2 * d2 + d3 * d3

    t = (c_ug * d0 + c_gr * d1 + c_ri * d2 + c_iz * d3) / norm2
    t0 = (c0_ug * d0 + c0_gr * d1 + c0_ri * d2 + c0_iz * d3) / norm2
    delta_t = t - t0

    r0_1 = c0_ug - t0 * d0
    r0_2 = c0_gr - t0 * d1
    r0_3 = c0_ri - t0 * d2

    ebv_bin = _quantize_6_bins(ebv, idx)
    delta_t_x_abs_b = delta_t * abs_b
    ebv_bin_x_redshift = ebv_bin.to_numpy(dtype="float64") * redshift

    out = pd.DataFrame(index=idx)
    out["ebv"] = ebv
    out["ebv_missing"] = ebv_missing.astype("uint8")
    out["u0"] = u0
    out["g0"] = g0
    out["r0"] = r0
    out["i0"] = i0
    out["z0"] = z0
    out["color_ug"] = c_ug
    out["color_gr"] = c_gr
    out["color_ri"] = c_ri
    out["color_iz"] = c_iz
    out["color0_ug"] = c0_ug
    out["color0_gr"] = c0_gr
    out["color0_ri"] = c0_ri
    out["color0_iz"] = c0_iz
    out["dust_proj_t"] = t
    out["dust_proj_t0"] = t0
    out["dust_delta_t"] = delta_t
    out["dust_res_r0_1"] = r0_1
    out["dust_res_r0_2"] = r0_2
    out["dust_res_r0_3"] = r0_3
    out["abs_galactic_b"] = abs_b
    out["ebv_bin6"] = ebv_bin
    out["delta_t_x_abs_b"] = delta_t_x_abs_b
    out["ebv_bin6_x_redshift"] = ebv_bin_x_redshift
    out["l_deg"] = l_deg
    out["b_deg"] = b_deg

    numeric_cols = out.select_dtypes(include=["number"]).columns.tolist()
    for c in numeric_cols:
        out[f"{c}_clipped"] = _clip_0_5_99_5(out[c].to_numpy())

    return out


FEATURE_GROUPS = [
    {
        "name": "foreground_reddening_geometry",
        "fn": add_foreground_reddening_geometry,
        "depends_on": [],
        "description": "Builds a dust-aware reddening geometry block by estimating E(B-V), dereddening ugriz magnitudes, projecting dust and intrinsic color directions, and adding stabilized clipped interaction features.",
    }
]