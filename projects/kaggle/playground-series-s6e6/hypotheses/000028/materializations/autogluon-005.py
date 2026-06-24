import numpy as np
import pandas as pd

SFD98_SCALE = 0.86
EXTINCTION_COEFFS = (5.155, 3.793, 2.751, 2.086, 1.479)
EQ_TO_GAL_MATRIX = (
    (-0.0548755604, -0.8734370902, -0.4838350155),
    (0.4941094279, -0.4448296300, 0.7469822445),
    (-0.8676661490, -0.1980763734, 0.4559837762),
)
BIN_SIZE_DEG = 5.0
EBV_MIN = 0.0
EBV_MAX = 1.2
CLIP_Q_LOW = 0.005
CLIP_Q_HIGH = 0.995
LAT_BIN_COUNT = 10
NEAREST_BIN_MAX_DEG = 10.0

def _to_series(values, index):
    return pd.Series(pd.to_numeric(values, errors="coerce"), index=index, dtype="float64")

def _normalize_name(name):
    return "".join(ch for ch in str(name).lower() if ch.isalnum())

def _find_column(columns, candidates):
    norm = {_normalize_name(c): c for c in columns}
    for cand in candidates:
        if cand in norm:
            return norm[cand]
    for cand in candidates:
        if len(cand) <= 1:
            continue
        for key, col in norm.items():
            if cand in key:
                return col
    return None

def _to_boolean_mask(values, index):
    s = pd.Series(values, index=index)
    if pd.api.types.is_bool_dtype(s):
        return s.fillna(False).astype(bool)
    num = pd.to_numeric(s, errors="coerce")
    if num.notna().any():
        return num.fillna(0).gt(0).astype(bool)
    text = s.astype(str).str.strip().str.lower()
    return text.isin({"1", "true", "t", "yes", "y", "train", "training", "tr"})

def _infer_train_mask(raw_index, aux):
    if aux is None or len(aux) == 0:
        return pd.Series(True, index=raw_index)
    if len(aux) != len(raw_index):
        return pd.Series(True, index=raw_index)

    candidates = (
        "is_train",
        "is_train_row",
        "is_train_mask",
        "is_train_flag",
        "trainmask",
        "train_flag",
        "split",
        "fold",
        "set",
    )
    col = _find_column(aux.columns, candidates)
    if col is not None:
        return pd.Series(
            _to_boolean_mask(aux[col].to_numpy(), aux.index).to_numpy(),
            index=raw_index,
            dtype=bool,
        )
    return pd.Series(True, index=raw_index, dtype=bool)

def _equatorial_to_galactic(ra_deg, dec_deg):
    ra = pd.to_numeric(ra_deg, errors="coerce").to_numpy(dtype=float)
    dec = pd.to_numeric(dec_deg, errors="coerce").to_numpy(dtype=float)
    idx = np.isfinite(ra) & np.isfinite(dec)

    l = np.full(len(ra), np.nan, dtype=float)
    b = np.full(len(dec), np.nan, dtype=float)

    if idx.any():
        ra_r = np.radians(ra[idx])
        dec_r = np.radians(dec[idx])

        cosd = np.cos(dec_r)
        x = cosd * np.cos(ra_r)
        y = cosd * np.sin(ra_r)
        z = np.sin(dec_r)

        m00, m01, m02 = EQ_TO_GAL_MATRIX[0]
        m10, m11, m12 = EQ_TO_GAL_MATRIX[1]
        m20, m21, m22 = EQ_TO_GAL_MATRIX[2]

        xg = m00 * x + m01 * y + m02 * z
        yg = m10 * x + m11 * y + m12 * z
        zg = m20 * x + m21 * y + m22 * z

        l[idx] = (np.degrees(np.arctan2(yg, xg)) % 360.0)
        b[idx] = np.degrees(np.arcsin(np.clip(zg, -1.0, 1.0)))

    return pd.Series(l, index=ra_deg.index), pd.Series(b, index=ra_deg.index)

def _five_deg_bins(l_deg, b_deg):
    l_arr = np.mod(np.asarray(l_deg, dtype=float), 360.0)
    b_arr = np.asarray(b_deg, dtype=float)

    l_valid = np.isfinite(l_arr)
    b_valid = np.isfinite(b_arr)

    l_bin = np.full(len(l_arr), -1, dtype=int)
    b_bin = np.full(len(b_arr), -1, dtype=int)

    if np.any(l_valid):
        l_bin[l_valid] = np.floor(l_arr[l_valid] / BIN_SIZE_DEG).astype(int)
    if np.any(b_valid):
        b_bin[b_valid] = np.floor((b_arr[b_valid] + 90.0) / BIN_SIZE_DEG).astype(int)
        b_bin = np.clip(b_bin, 0, 35)

    return pd.Series(l_bin, index=l_deg.index), pd.Series(b_bin, index=b_deg.index)

def _to_xyz(l_deg, b_deg):
    l_rad = np.radians(np.asarray(l_deg, dtype=float))
    b_rad = np.radians(np.asarray(b_deg, dtype=float))
    cb = np.cos(b_rad)
    return np.column_stack((cb * np.cos(l_rad), cb * np.sin(l_rad), np.sin(b_rad)))

def _lookup_map_by_5deg_bins(raw_l, raw_b, map_l, map_b, map_ebv):
    raw_l_bin, raw_b_bin = _five_deg_bins(raw_l, raw_b)

    map_l = pd.to_numeric(map_l, errors="coerce")
    map_b = pd.to_numeric(map_b, errors="coerce")
    map_e = pd.to_numeric(map_ebv, errors="coerce") * SFD98_SCALE
    valid_map = map_e.notna()

    if valid_map.sum() == 0:
        return pd.Series(np.nan, index=raw_l.index)

    map_l_bin, map_b_bin = _five_deg_bins(map_l, map_b)
    map_df = pd.DataFrame(
        {"l_bin": map_l_bin, "b_bin": map_b_bin, "ebv": map_e},
        index=map_l.index,
    )
    map_df = map_df.loc[valid_map]
    if len(map_df) == 0:
        return pd.Series(np.nan, index=raw_l.index)

    map_median = map_df.groupby(["l_bin", "b_bin"], as_index=True)["ebv"].median()
    lookup_keys = pd.MultiIndex.from_arrays(
        [raw_l_bin.to_numpy(), raw_b_bin.to_numpy()],
        names=["l_bin", "b_bin"],
    )
    return pd.Series(lookup_keys.map(map_median), index=raw_l.index)

def _extract_ebv_from_aux(raw_index, aux, raw_l, raw_b):
    if aux is None or len(aux) == 0:
        return pd.Series(np.nan, index=raw_index)

    ebv_col = _find_column(
        aux.columns,
        (
            "ebv",
            "ebv_sfd98",
            "sfd98_ebv",
            "sfd_ebv",
            "ebv_sfd",
            "sfd",
            "extinction",
            "dust",
        ),
    )
    if len(aux) == len(raw_index) and ebv_col is not None:
        return _to_series(aux[ebv_col], raw_index) * SFD98_SCALE

    l_col = _find_column(
        aux.columns,
        ("galacticlongitude", "longitude", "glon", "l", "ldeg", "lon", "ra"),
    )
    b_col = _find_column(
        aux.columns,
        ("galacticlatitude", "latitude", "glat", "b", "bdeg", "lat"),
    )

    if ebv_col is None or l_col is None or b_col is None:
        return pd.Series(np.nan, index=raw_index)

    return _lookup_map_by_5deg_bins(
        raw_l=raw_l,
        raw_b=raw_b,
        map_l=aux[l_col],
        map_b=aux[b_col],
        map_ebv=aux[ebv_col],
    )

def _nearest_bin_fill(target_bins, median_by_bin):
    if len(target_bins) == 0:
        return np.array([], dtype=float)
    if median_by_bin is None or len(median_by_bin) == 0:
        return np.full(len(target_bins), np.nan, dtype=float)

    pair_arr = np.asarray(target_bins, dtype=float)
    out = np.full(len(pair_arr), np.nan, dtype=float)

    full_bins = np.asarray(list(median_by_bin.index), dtype=float)
    full_vals = np.asarray(median_by_bin.to_numpy(), dtype=float)
    if full_bins.ndim == 1:
        full_bins = np.array([full_bins], dtype=float)

    full_mask = (
        (full_bins[:, 0] >= 0)
        & (full_bins[:, 0] <= 71.5)
        & (full_bins[:, 1] >= 0)
        & (full_bins[:, 1] <= 35.5)
    )
    if not full_mask.all():
        full_bins = full_bins[full_mask]
        full_vals = full_vals[full_mask]

    if len(full_bins) == 0:
        return out

    valid = (
        (pair_arr[:, 0] >= 0)
        & (pair_arr[:, 0] <= 71.5)
        & (pair_arr[:, 1] >= 0)
        & (pair_arr[:, 1] <= 35.5)
    )
    if not np.any(valid):
        return out

    valid_pairs = pair_arr[valid]
    t_cent = np.column_stack((valid_pairs[:, 0] + 0.5, valid_pairs[:, 1] + 0.5))
    t_l = t_cent[:, 0] * BIN_SIZE_DEG
    t_b = t_cent[:, 1] * BIN_SIZE_DEG - 90.0
    f_cent = np.column_stack((full_bins[:, 0] + 0.5, full_bins[:, 1] + 0.5))
    f_l = f_cent[:, 0] * BIN_SIZE_DEG
    f_b = f_cent[:, 1] * BIN_SIZE_DEG - 90.0

    t_vec = _to_xyz(t_l, t_b)
    f_vec = _to_xyz(f_l, f_b)

    cosang = t_vec @ f_vec.T
    cosang = np.clip(cosang, -1.0, 1.0)
    best = np.argmax(cosang, axis=1)
    best_ang = np.degrees(np.arccos(cosang[np.arange(len(best)), best]))
    chosen = np.full(len(best), np.nan, dtype=float)
    near_mask = best_ang <= NEAREST_BIN_MAX_DEG
    chosen[near_mask] = full_vals[best[near_mask]]

    out[np.where(valid)[0]] = chosen
    return out

def _impute_ebv(raw_ebv, raw_l, raw_b, train_mask, index):
    ebv = _to_series(raw_ebv, index)
    is_imputed = pd.Series(0, index=index, dtype="int8")

    missing_initial = ~np.isfinite(ebv.to_numpy(dtype=float))
    if not missing_initial.any():
        return ebv.clip(EBV_MIN, EBV_MAX), is_imputed

    l_bin, b_bin = _five_deg_bins(raw_l, raw_b)
    l_arr = l_bin.to_numpy(dtype=int)
    b_arr = b_bin.to_numpy(dtype=int)

    train_mask = pd.Series(train_mask, index=index, dtype=bool)
    df = pd.DataFrame(
        {
            "l_bin": l_arr,
            "b_bin": b_arr,
            "ebv": ebv,
        },
        index=index,
    )

    train_rows = df.loc[train_mask & df["ebv"].notna(), ["l_bin", "b_bin", "ebv"]]
    if len(train_rows):
        bin_median = train_rows.groupby(["l_bin", "b_bin"])["ebv"].median()
    else:
        bin_median = pd.Series(dtype=float)

    keys = pd.MultiIndex.from_arrays([l_arr, b_arr], names=["l_bin", "b_bin"])
    key_map = pd.Series(keys.map(bin_median), index=index, dtype=float)

    same_bin_fill = missing_initial & key_map.notna()
    ebv.loc[same_bin_fill] = key_map.loc[same_bin_fill]
    is_imputed.loc[same_bin_fill] = 1

    remaining = missing_initial & ebv.isna()
    if remaining.any():
        rem_idx = np.where(remaining.to_numpy())[0]
        rem_pairs = np.unique(
            np.column_stack((l_arr[rem_idx], b_arr[rem_idx])),
            axis=0,
        )
        rem_vals = _nearest_bin_fill(rem_pairs, bin_median)
        rem_lookup = {
            (float(pair[0]), float(pair[1])): val for pair, val in zip(rem_pairs, rem_vals)
        }
        rem_filled = np.array(
            [rem_lookup.get((float(l_arr[i]), float(b_arr[i])), np.nan) for i in rem_idx],
            dtype=float,
        )
        ebv.iloc[rem_idx] = rem_filled
        is_imputed.iloc[rem_idx] = pd.Series(
            pd.notna(rem_filled),
            index=ebv.iloc[rem_idx].index,
            dtype="int8",
        ).to_numpy()

    remaining = ebv.isna()
    if remaining.any():
        train_valid = df.loc[train_mask & df["ebv"].notna(), "ebv"]
        if len(train_valid) > 0:
            global_median = float(pd.to_numeric(train_valid, errors="coerce").median())
            if not np.isfinite(global_median):
                global_median = 0.0
        else:
            global_median = 0.0
        ebv.loc[remaining] = global_median
        is_imputed.loc[remaining] = 1

    return ebv.clip(EBV_MIN, EBV_MAX), is_imputed

def _latitude_decile_bin(abs_b, train_mask, index):
    abs_b = pd.to_numeric(abs_b, errors="coerce")
    train_mask = pd.Series(train_mask, index=index, dtype=bool)
    train_vals = abs_b.loc[train_mask].to_numpy(dtype=float)

    if not np.isfinite(train_vals).any():
        train_vals = abs_b.to_numpy(dtype=float)

    finite_train = train_vals[np.isfinite(train_vals)]
    if finite_train.size == 0:
        return pd.Series(np.zeros(len(abs_b), dtype="int64"), index=index)

    if np.unique(finite_train).size == 1:
        return pd.Series(np.zeros(len(abs_b), dtype="int64"), index=index)

    probs = np.linspace(0.0, 1.0, LAT_BIN_COUNT + 1)
    edges = np.quantile(finite_train, probs)
    edges_clean = []
    prev = None
    for edge in edges:
        if not np.isfinite(edge):
            continue
        edge = float(edge)
        if prev is None:
            prev = edge
            edges_clean.append(edge)
            continue
        if edge <= prev:
            edge = np.nextafter(prev, np.inf)
        edges_clean.append(edge)
        prev = edge

    if len(edges_clean) < 2:
        return pd.Series(np.zeros(len(abs_b), dtype="int64"), index=index)

    edges_clean = np.array(edges_clean, dtype=float)
    min_v = float(np.nanmin(finite_train))
    max_v = float(np.nanmax(finite_train))
    edges_clean[0] = min(min_v, edges_clean[0])
    edges_clean[-1] = max(max_v, edges_clean[-1])

    if np.any(np.diff(edges_clean) <= 0.0):
        edges_clean = np.linspace(min_v, max_v, LAT_BIN_COUNT + 1)

    bins = pd.cut(abs_b, bins=edges_clean, include_lowest=True, labels=False)
    return bins.fillna(-1).astype(int)

def _clip_continuous_features(features, train_mask):
    train_mask = pd.Series(train_mask, index=features.index, dtype=bool)
    numeric_cols = [
        c
        for c in features.columns
        if pd.api.types.is_float_dtype(features[c].dtype)
    ]

    train_ok = train_mask.copy()
    if not train_ok.any():
        train_ok[:] = True

    for col in numeric_cols:
        vals = pd.to_numeric(features.loc[train_ok, col], errors="coerce")
        if vals.notna().sum() == 0:
            continue
        q = vals.quantile([CLIP_Q_LOW, CLIP_Q_HIGH])
        lo = float(q.iloc[0])
        hi = float(q.iloc[1])
        if np.isfinite(lo) and np.isfinite(hi) and lo < hi:
            features[col] = features[col].clip(lower=lo, upper=hi)

    return features

def add_foreground_reddening_geometry(raw, deps, aux):
    _ = deps
    index = raw.index

    alpha = _to_series(raw["alpha"], index)
    delta = _to_series(raw["delta"], index)
    l_deg, b_deg = _equatorial_to_galactic(alpha, delta)

    train_mask = _infer_train_mask(index, aux)

    ebv_raw = _extract_ebv_from_aux(index, aux, l_deg, b_deg)
    ebv, is_ebv_imputed = _impute_ebv(ebv_raw, l_deg, b_deg, train_mask, index)

    u = _to_series(raw["u"], index)
    g = _to_series(raw["g"], index)
    r = _to_series(raw["r"], index)
    i = _to_series(raw["i"], index)
    z = _to_series(raw["z"], index)
    redshift = _to_series(raw["redshift"], index)

    au, ag, ar, ai, az = EXTINCTION_COEFFS

    u0 = u - au * ebv
    g0 = g - ag * ebv
    r0 = r - ar * ebv
    i0 = i - ai * ebv
    z0 = z - az * ebv

    color_ug = u - g
    color_gr = g - r
    color_ri = r - i
    color_iz = i - z

    color0_ug = u0 - g0
    color0_gr = g0 - r0
    color0_ri = r0 - i0
    color0_iz = i0 - z0

    c = np.column_stack(
        [
            color_ug.to_numpy(dtype=float),
            color_gr.to_numpy(dtype=float),
            color_ri.to_numpy(dtype=float),
            color_iz.to_numpy(dtype=float),
        ]
    )
    c0 = np.column_stack(
        [
            color0_ug.to_numpy(dtype=float),
            color0_gr.to_numpy(dtype=float),
            color0_ri.to_numpy(dtype=float),
            color0_iz.to_numpy(dtype=float),
        ]
    )

    d = np.array([au - ag, ag - ar, ar - ai, ai - az], dtype=float)
    dnorm = float(np.linalg.norm(d))
    if dnorm > 0.0:
        d_hat = d / dnorm
    else:
        d_hat = np.zeros(4, dtype=float)

    t = pd.Series(c.dot(d_hat), index=index)
    t0 = pd.Series(c0.dot(d_hat), index=index)
    dt = t - t0

    proj0 = np.outer(t0.to_numpy(dtype=float), d_hat)
    resid0 = c0 - proj0
    resid0_0 = pd.Series(resid0[:, 0], index=index)
    resid0_1 = pd.Series(resid0[:, 1], index=index)
    resid0_2 = pd.Series(resid0[:, 2], index=index)
    resid0_norm = pd.Series(np.sqrt(np.sum(resid0 * resid0, axis=1)), index=index)

    abs_b = b_deg.abs()
    lat_bin = _latitude_decile_bin(abs_b, train_mask, index)

    features = pd.DataFrame(
        {
            "ebv_clipped": ebv,
            "is_ebv_imputed": is_ebv_imputed.astype("int8"),
            "u0": u0,
            "g0": g0,
            "r0": r0,
            "i0": i0,
            "z0": z0,
            "color_ug": color_ug,
            "color_gr": color_gr,
            "color_ri": color_ri,
            "color_iz": color_iz,
            "color0_ug": color0_ug,
            "color0_gr": color0_gr,
            "color0_ri": color0_ri,
            "color0_iz": color0_iz,
            "reddening_proj_obs": t,
            "reddening_proj_dered": t0,
            "reddening_proj_delta": dt,
            "reddening_residual0": resid0_0,
            "reddening_residual1": resid0_1,
            "reddening_residual2": resid0_2,
            "reddening_residual_norm": resid0_norm,
            "gal_b_abs": abs_b,
            "gal_b_decile_bin": lat_bin,
            "reddening_delta_times_absb": dt * abs_b,
            "redshift_times_ebv": redshift * ebv,
            "ebv_times_absb": ebv * abs_b,
        },
        index=index,
    )

    return _clip_continuous_features(features, train_mask)

FEATURE_GROUPS = [
    {
        "name": "foreground_reddening_geometry",
        "fn": add_foreground_reddening_geometry,
        "depends_on": [],
        "description": "Build extinction-aware color geometry features using Galactic E(B-V) with fallback imputation and orthogonal residual decomposition.",
    },
]