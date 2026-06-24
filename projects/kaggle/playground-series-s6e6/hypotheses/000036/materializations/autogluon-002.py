import numpy as np
import pandas as pd

DECILE_QUANTILES = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
MIN_HIERARCHY_COUNT = 800
MAD_SCALE = 1.4826
MAD_EPSILON = 1e-6
Z_SCORE_LIMIT = 8.0
MISSING_CATEGORY = "__missing__"
TRAIN_MASK_CANDIDATES = ("is_train", "train_mask", "is_train_mask", "is_train_row", "train")
ZCOLOR_COEFFS = {
    "p1": (0.91, 0.415, -1.280),  # p1 = 0.91*ug + 0.415*gr - 1.280
    "v": (0.283, -0.354, 0.455, 0.766),  # v = 0.283*ug -0.354*gr +0.455*ri +0.766*iz
    "l": (-0.436, 1.129, -0.119, -0.574, 0.1984),  # l = -0.436*u +1.129*g -0.119*r -0.574*i +0.1984
}


def _to_numeric_safe(series):
    arr = pd.to_numeric(series, errors="coerce").astype(float)
    return arr.replace([np.inf, -np.inf], np.nan)


def _mad_scaled(values):
    values = pd.Series(values, dtype=float).dropna()
    if values.empty:
        return np.nan
    med = values.median()
    if not np.isfinite(med):
        return np.nan
    mad = (values - med).abs().median()
    if not np.isfinite(mad):
        return np.nan
    return MAD_SCALE * mad


def _coerce_bool_mask(obj):
    s = pd.Series(obj)
    numeric = pd.to_numeric(s, errors="coerce")
    if numeric.notna().any():
        return numeric.notna() & numeric.astype(float).astype(bool)

    text = s.astype(str).str.strip().str.lower()
    text_mask = text.isin(("1", "true", "t", "yes", "y", "train"))
    return text_mask & text.ne("nan")


def _resolve_train_mask(raw, aux):
    for column in TRAIN_MASK_CANDIDATES:
        if column in raw.columns:
            mask = _coerce_bool_mask(raw[column]).to_numpy(dtype=bool)
            if mask.any():
                return mask

    if isinstance(aux, pd.DataFrame) and len(aux) == len(raw):
        for column in TRAIN_MASK_CANDIDATES:
            if column in aux.columns:
                mask = _coerce_bool_mask(aux[column]).to_numpy(dtype=bool)
                if mask.any():
                    return mask

    return np.ones(len(raw), dtype=bool)


def _compute_redshift_bins(redshift, train_mask):
    z = _to_numeric_safe(redshift)
    train_values = z.iloc[train_mask].dropna() if np.any(train_mask) else z.dropna()

    if train_values.empty:
        train_values = z.dropna()
    if train_values.empty:
        return pd.Series(np.zeros(len(z), dtype=np.int16), index=z.index)

    quantiles = train_values.quantile(np.array(DECILE_QUANTILES), interpolation="linear")
    edges = np.asarray(quantiles.to_numpy(dtype=float), dtype=float)

    z_min = z.min(skipna=True)
    z_max = z.max(skipna=True)

    if not np.isfinite(z_min):
        z_min = 0.0
    if not np.isfinite(z_max):
        z_max = z_min + 1.0
    if z_max <= z_min:
        z_max = z_min + 1.0

    edges[0] = float(z_min)
    edges[-1] = float(z_max)

    for i in range(1, len(edges)):
        if (not np.isfinite(edges[i])) or (edges[i] <= edges[i - 1]):
            edges[i] = np.nextafter(edges[i - 1], np.inf)

    z_vals = z.to_numpy(dtype=float)
    fill_z = np.nanmedian(z_vals)
    if not np.isfinite(fill_z):
        fill_z = z_min
    z_vals = np.where(np.isfinite(z_vals), z_vals, fill_z)

    zbin = np.digitize(z_vals, bins=edges[1:-1], right=True)
    zbin = np.clip(zbin, 0, len(DECILE_QUANTILES) - 2)
    return pd.Series(zbin.astype(np.int16), index=z.index)


def _hierarchical_robust_features(values, zbin, spectral_type, galaxy_population, train_mask):
    x = _to_numeric_safe(values)
    idx = x.index

    if np.asarray(train_mask, dtype=bool).sum() == 0 or len(train_mask) != len(x):
        train_mask = np.ones(len(x), dtype=bool)
    train_mask = np.asarray(train_mask, dtype=bool)

    frame = pd.DataFrame(
        {
            "x": x,
            "zbin": pd.Series(zbin, index=idx).astype("int16"),
            "spectral_type": pd.Series(spectral_type, index=idx).astype(str),
            "galaxy_population": pd.Series(galaxy_population, index=idx).astype(str),
        },
        index=idx,
    )

    train_frame = frame.iloc[train_mask]
    if train_frame.empty:
        train_frame = frame

    global_median = train_frame["x"].median()
    global_mad = _mad_scaled(train_frame["x"])
    if not np.isfinite(global_median):
        global_median = 0.0
    if not np.isfinite(global_mad) or global_mad < MAD_EPSILON:
        global_mad = MAD_EPSILON

    g3 = train_frame.groupby(["zbin", "spectral_type", "galaxy_population"])["x"]
    med3 = g3.median()
    mad3 = g3.apply(_mad_scaled)
    cnt3 = g3.count()
    keep3 = cnt3 >= MIN_HIERARCHY_COUNT
    med3 = med3[keep3]
    mad3 = mad3[keep3]

    g2 = train_frame.groupby(["zbin", "spectral_type"])["x"]
    med2 = g2.median()
    mad2 = g2.apply(_mad_scaled)
    cnt2 = g2.count()
    keep2 = cnt2 >= MIN_HIERARCHY_COUNT
    med2 = med2[keep2]
    mad2 = mad2[keep2]

    g1 = train_frame.groupby(["spectral_type"])["x"]
    med1 = g1.median()
    mad1 = g1.apply(_mad_scaled)
    cnt1 = g1.count()
    keep1 = cnt1 >= MIN_HIERARCHY_COUNT
    med1 = med1[keep1]
    mad1 = mad1[keep1]

    k3 = pd.Series(list(zip(frame["zbin"], frame["spectral_type"], frame["galaxy_population"])), index=idx)
    k2 = pd.Series(list(zip(frame["zbin"], frame["spectral_type"])), index=idx)
    k1 = pd.Series(frame["spectral_type"], index=idx)

    med = k3.map(med3.to_dict())
    med = med.fillna(k2.map(med2.to_dict()))
    med = med.fillna(k1.map(med1.to_dict()))
    med = med.fillna(global_median)

    mad = k3.map(mad3.to_dict())
    mad = mad.fillna(k2.map(mad2.to_dict()))
    mad = mad.fillna(k1.map(mad1.to_dict()))
    mad = mad.fillna(global_mad)
    mad = mad.where(mad >= MAD_EPSILON, MAD_EPSILON)

    z_scores = ((x - med) / mad).replace([np.inf, -np.inf], np.nan).clip(-Z_SCORE_LIMIT, Z_SCORE_LIMIT).fillna(0.0)
    raw_vals = x.fillna(global_median).replace([np.inf, -np.inf], global_median).fillna(global_median)

    return raw_vals, z_scores


def add_segue_stellar_atmospheric_indices(raw, deps, aux):
    _ = deps
    raw_index = raw.index

    train_mask = _resolve_train_mask(raw, aux)

    u = _to_numeric_safe(raw["u"])
    g = _to_numeric_safe(raw["g"])
    r = _to_numeric_safe(raw["r"])
    i = _to_numeric_safe(raw["i"])
    zmag = _to_numeric_safe(raw["z"])
    redshift = _to_numeric_safe(raw["redshift"])

    spectral_type = raw["spectral_type"].fillna(MISSING_CATEGORY).astype(str)
    galaxy_population = raw["galaxy_population"].fillna(MISSING_CATEGORY).astype(str)

    zbin = _compute_redshift_bins(redshift, train_mask)

    ug = u - g
    gr = g - r
    ri = r - i
    iz = i - zmag

    a1, a2, a3 = ZCOLOR_COEFFS["p1"]
    b1, b2, b3, b4 = ZCOLOR_COEFFS["v"]
    c1, c2, c3, c4, c5 = ZCOLOR_COEFFS["l"]

    p1_raw = a1 * ug + a2 * gr + a3
    v_raw = b1 * ug + b2 * gr + b3 * ri + b4 * iz
    l_raw = c1 * u + c2 * g + c3 * r + c4 * i + c5

    p1, z_p1 = _hierarchical_robust_features(p1_raw, zbin, spectral_type, galaxy_population, train_mask)
    v, z_v = _hierarchical_robust_features(v_raw, zbin, spectral_type, galaxy_population, train_mask)
    l, z_l = _hierarchical_robust_features(l_raw, zbin, spectral_type, galaxy_population, train_mask)

    p1_violation_left = (-0.70 - p1).clip(lower=0.0)
    p1_violation_right = (p1 + 0.25).clip(lower=0.0)
    p1_violation = p1_violation_left + p1_violation_right

    l_tail_low = (0.07 - l).clip(lower=0.0)
    l_tail_high = (l - 0.135).clip(lower=0.0)
    l_tail = l_tail_low + l_tail_high

    v_abs = v.abs()
    abs_z_mean = (z_p1.abs() + z_v.abs() + z_l.abs()) / 3.0
    locus_dist = np.sqrt(z_p1 * z_p1 + z_v * z_v + z_l * z_l)
    cross_vp = v_abs * z_p1
    cross_vl = v_abs * z_l
    residual_sign_balance = np.sign(z_p1) * np.sign(z_v) * np.sign(z_l)

    new_features = pd.DataFrame(
        {
            "p1": p1,
            "v": v,
            "l": l,
            "z_p1": z_p1,
            "z_v": z_v,
            "z_l": z_l,
            "p1_violation_left": p1_violation_left,
            "p1_violation_right": p1_violation_right,
            "p1_violation": p1_violation,
            "l_tail_low": l_tail_low,
            "l_tail_high": l_tail_high,
            "l_tail": l_tail,
            "v_abs": v_abs,
            "abs_z_mean": abs_z_mean,
            "locus_dist": locus_dist,
            "cross_vp": cross_vp,
            "cross_vl": cross_vl,
            "residual_sign_balance": residual_sign_balance,
        },
        index=raw_index,
    )

    zscore_columns = {"z_p1", "z_v", "z_l", "abs_z_mean", "locus_dist", "cross_vp", "cross_vl", "residual_sign_balance"}
    for column in new_features.columns:
        series = new_features[column].replace([np.inf, -np.inf], np.nan)
        if column in zscore_columns:
            new_features[column] = series.fillna(0.0)
        else:
            median_value = series.median(skipna=True)
            if not np.isfinite(median_value):
                median_value = 0.0
            new_features[column] = series.fillna(median_value)

    return new_features


FEATURE_GROUPS = [
    {
        "name": "segue_stellar_atmospheric_indices",
        "fn": add_segue_stellar_atmospheric_indices,
        "depends_on": [],
        "description": "Build robust redshift- and subtype-hierarchical SDSS color residual features from atmospheric manifold indices.",
    }
]