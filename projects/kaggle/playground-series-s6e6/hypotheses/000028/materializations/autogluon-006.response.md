import numpy as np
import pandas as pd


TRAIN_ID_MAX = 577346
SFD98_TO_SCHLAFLY_FINKBEINER = 0.86
EBV_CLIP_MIN = 0.0
EBV_CLIP_MAX = 1.2
GALACTIC_BIN_DEGREES = 5.0
NEAREST_BIN_MAX_DEGREES = 10.0
CLIP_LOW_PERCENTILE = 0.5
CLIP_HIGH_PERCENTILE = 99.5

ICRS_TO_GALACTIC_MATRIX = (
    (-0.0548755604162154, -0.8734370902348850, -0.4838350155487132),
    (0.4941094278755837, -0.4448296299600112, 0.7469822444972189),
    (-0.8676661490190047, -0.1980763734312015, 0.4559837761750669),
)

SDSS_EXTINCTION_COEFFICIENTS = (5.155, 3.793, 2.751, 2.086, 1.479)
COLOR_EXCESS_VECTOR = (1.362, 1.042, 0.665, 0.607)

AUX_EBV_COLUMNS = (
    "ebv",
    "EBV",
    "e_bv",
    "E_BV",
    "E(B-V)",
    "sfd_ebv",
    "SFD_EBV",
    "sfd98_ebv",
    "SFD98_EBV",
    "dust_ebv",
    "reddening",
)


def _as_float_array(frame, column):
    return pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=np.float64)


def _training_mask(raw):
    n_rows = len(raw)
    if "id" not in raw.columns:
        return np.ones(n_rows, dtype=bool)

    ids = pd.to_numeric(raw["id"], errors="coerce").to_numpy(dtype=np.float64)
    mask = np.isfinite(ids) & (ids <= TRAIN_ID_MAX)
    if not np.any(mask):
        return np.ones(n_rows, dtype=bool)
    return mask


def _galactic_coordinates(alpha_deg, delta_deg):
    alpha = np.deg2rad(alpha_deg)
    delta = np.deg2rad(delta_deg)

    cos_delta = np.cos(delta)
    x_eq = cos_delta * np.cos(alpha)
    y_eq = cos_delta * np.sin(alpha)
    z_eq = np.sin(delta)

    matrix = np.asarray(ICRS_TO_GALACTIC_MATRIX, dtype=np.float64)
    x_gal = matrix[0, 0] * x_eq + matrix[0, 1] * y_eq + matrix[0, 2] * z_eq
    y_gal = matrix[1, 0] * x_eq + matrix[1, 1] * y_eq + matrix[1, 2] * z_eq
    z_gal = matrix[2, 0] * x_eq + matrix[2, 1] * y_eq + matrix[2, 2] * z_eq

    lon = np.mod(np.rad2deg(np.arctan2(y_gal, x_gal)), 360.0)
    lat = np.rad2deg(np.arcsin(np.clip(z_gal, -1.0, 1.0)))
    return lon, lat


def _find_aux_ebv_column(aux):
    if not isinstance(aux, pd.DataFrame) or aux.empty:
        return None

    exact_columns = set(aux.columns)
    lower_to_column = {str(col).lower(): col for col in aux.columns}

    for column in AUX_EBV_COLUMNS:
        if column in exact_columns:
            return column
        lowered = column.lower()
        if lowered in lower_to_column:
            return lower_to_column[lowered]

    for column in aux.columns:
        lowered = str(column).lower()
        if "ebv" in lowered or "e_bv" in lowered or "reddening" in lowered:
            return column

    return None


def _lookup_ebv_from_aux(raw, aux):
    values = np.full(len(raw), np.nan, dtype=np.float64)
    if not isinstance(aux, pd.DataFrame) or aux.empty:
        return values

    ebv_column = _find_aux_ebv_column(aux)
    if ebv_column is None:
        return values

    aux_ebv = pd.to_numeric(aux[ebv_column], errors="coerce")

    if "id" in raw.columns and "id" in aux.columns:
        aux_ids = pd.to_numeric(aux["id"], errors="coerce")
        id_frame = pd.DataFrame({"id": aux_ids, "ebv": aux_ebv})
        id_frame = id_frame.dropna(subset=["id"])
        if not id_frame.empty:
            ebv_by_id = id_frame.groupby("id", sort=False)["ebv"].median()
            raw_ids = pd.to_numeric(raw["id"], errors="coerce")
            return raw_ids.map(ebv_by_id).to_numpy(dtype=np.float64)

    if len(aux) == len(raw):
        if aux.index.equals(raw.index):
            return aux_ebv.reindex(raw.index).to_numpy(dtype=np.float64)
        return aux_ebv.reset_index(drop=True).to_numpy(dtype=np.float64)

    if aux.index.is_unique:
        aligned = aux_ebv.reindex(raw.index)
        if aligned.notna().any():
            return aligned.to_numpy(dtype=np.float64)

    return values


def _galactic_bin_keys(lon_deg, lat_deg):
    lon_bin = np.floor(np.mod(lon_deg, 360.0) / GALACTIC_BIN_DEGREES).astype(np.int16)
    lat_bin = np.floor((np.clip(lat_deg, -90.0, 90.0) + 90.0) / GALACTIC_BIN_DEGREES).astype(np.int16)
    lat_bin = np.clip(lat_bin, 0, 35).astype(np.int16)
    return (lon_bin.astype(np.int32) * 100 + lat_bin.astype(np.int32)).astype(np.int32)


def _impute_ebv(ebv_values, lon_deg, lat_deg, train_mask):
    ebv = np.asarray(ebv_values, dtype=np.float64).copy()
    valid = np.isfinite(ebv)
    train_valid = train_mask & valid

    if np.any(train_valid):
        global_median = float(np.nanmedian(ebv[train_valid]))
    else:
        global_median = 0.0

    imputed = ~valid
    if not np.any(imputed):
        return ebv, imputed

    keys = _galactic_bin_keys(lon_deg, lat_deg)

    if np.any(train_valid):
        train_bins = pd.DataFrame({"key": keys[train_valid], "ebv": ebv[train_valid]})
        bin_medians = train_bins.groupby("key", sort=False)["ebv"].median()
    else:
        bin_medians = pd.Series(dtype=np.float64)

    if not bin_medians.empty:
        direct_fill = pd.Series(keys).map(bin_medians).to_numpy(dtype=np.float64)
        direct_mask = imputed & np.isfinite(direct_fill)
        ebv[direct_mask] = direct_fill[direct_mask]

        remaining = imputed & ~np.isfinite(ebv)
        if np.any(remaining):
            median_keys = bin_medians.index.to_numpy(dtype=np.int32)
            median_values = bin_medians.to_numpy(dtype=np.float64)
            median_lon_bins = median_keys // 100
            median_lat_bins = median_keys % 100
            median_lon_centers = median_lon_bins.astype(np.float64) * GALACTIC_BIN_DEGREES + 0.5 * GALACTIC_BIN_DEGREES
            median_lat_centers = median_lat_bins.astype(np.float64) * GALACTIC_BIN_DEGREES - 90.0 + 0.5 * GALACTIC_BIN_DEGREES

            nearest_by_key = {}
            for key in np.unique(keys[remaining]):
                lon_bin = key // 100
                lat_bin = key % 100
                lon_center = float(lon_bin) * GALACTIC_BIN_DEGREES + 0.5 * GALACTIC_BIN_DEGREES
                lat_center = float(lat_bin) * GALACTIC_BIN_DEGREES - 90.0 + 0.5 * GALACTIC_BIN_DEGREES

                lon_delta = np.abs(median_lon_centers - lon_center)
                lon_delta = np.minimum(lon_delta, 360.0 - lon_delta)
                lat_delta = median_lat_centers - lat_center
                distances = np.sqrt(lon_delta * lon_delta + lat_delta * lat_delta)
                nearest_idx = int(np.argmin(distances))
                if distances[nearest_idx] <= NEAREST_BIN_MAX_DEGREES:
                    nearest_by_key[int(key)] = median_values[nearest_idx]

            if nearest_by_key:
                nearest_fill = pd.Series(keys).map(nearest_by_key).to_numpy(dtype=np.float64)
                nearest_mask = remaining & np.isfinite(nearest_fill)
                ebv[nearest_mask] = nearest_fill[nearest_mask]

    remaining = imputed & ~np.isfinite(ebv)
    if np.any(remaining):
        ebv[remaining] = global_median

    return ebv, imputed


def _decile_bins(values, train_mask):
    arr = np.asarray(values, dtype=np.float64)
    train_values = arr[train_mask & np.isfinite(arr)]
    if train_values.size == 0:
        return np.full(arr.shape[0], -1, dtype=np.int8)

    cuts = np.nanpercentile(train_values, np.arange(10.0, 100.0, 10.0))
    cuts = np.unique(cuts[np.isfinite(cuts)])

    bins = np.searchsorted(cuts, arr, side="right").astype(np.int16)
    bins[~np.isfinite(arr)] = -1
    return np.clip(bins, -1, 9).astype(np.int8)


def _clip_to_train_percentiles(values, train_mask):
    arr = np.asarray(values, dtype=np.float64)
    train_values = arr[train_mask & np.isfinite(arr)]
    if train_values.size == 0:
        return arr.astype(np.float32)

    low, high = np.nanpercentile(train_values, (CLIP_LOW_PERCENTILE, CLIP_HIGH_PERCENTILE))
    if not np.isfinite(low) or not np.isfinite(high):
        return arr.astype(np.float32)
    if low > high:
        low, high = high, low

    return np.clip(arr, low, high).astype(np.float32)


def add_foreground_reddening_geometry(raw, deps, aux):
    train_mask = _training_mask(raw)

    alpha = _as_float_array(raw, "alpha")
    delta = _as_float_array(raw, "delta")
    u = _as_float_array(raw, "u")
    g = _as_float_array(raw, "g")
    r = _as_float_array(raw, "r")
    i_mag = _as_float_array(raw, "i")
    z = _as_float_array(raw, "z")
    redshift = _as_float_array(raw, "redshift")

    gal_l, gal_b = _galactic_coordinates(alpha, delta)

    ebv_lookup = _lookup_ebv_from_aux(raw, aux) * SFD98_TO_SCHLAFLY_FINKBEINER
    ebv_raw, is_ebv_imputed = _impute_ebv(ebv_lookup, gal_l, gal_b, train_mask)
    ebv_clipped = np.clip(ebv_raw, EBV_CLIP_MIN, EBV_CLIP_MAX)

    extinction_coeffs = np.asarray(SDSS_EXTINCTION_COEFFICIENTS, dtype=np.float64)
    a_u = extinction_coeffs[0] * ebv_clipped
    a_g = extinction_coeffs[1] * ebv_clipped
    a_r = extinction_coeffs[2] * ebv_clipped
    a_i = extinction_coeffs[3] * ebv_clipped
    a_z = extinction_coeffs[4] * ebv_clipped

    u0 = u - a_u
    g0 = g - a_g
    r0 = r - a_r
    i0 = i_mag - a_i
    z0 = z - a_z

    obs_ug = u - g
    obs_gr = g - r
    obs_ri = r - i_mag
    obs_iz = i_mag - z

    dered_ug = u0 - g0
    dered_gr = g0 - r0
    dered_ri = r0 - i0
    dered_iz = i0 - z0

    color_excess = np.asarray(COLOR_EXCESS_VECTOR, dtype=np.float64)
    reddening_direction = color_excess / np.sqrt(np.sum(color_excess * color_excess))

    parallel_obs = (
        obs_ug * reddening_direction[0]
        + obs_gr * reddening_direction[1]
        + obs_ri * reddening_direction[2]
        + obs_iz * reddening_direction[3]
    )
    parallel_dered = (
        dered_ug * reddening_direction[0]
        + dered_gr * reddening_direction[1]
        + dered_ri * reddening_direction[2]
        + dered_iz * reddening_direction[3]
    )
    dust_parallel = parallel_obs - parallel_dered

    residual_ug = dered_ug - parallel_dered * reddening_direction[0]
    residual_gr = dered_gr - parallel_dered * reddening_direction[1]
    residual_ri = dered_ri - parallel_dered * reddening_direction[2]
    residual_iz = dered_iz - parallel_dered * reddening_direction[3]
    residual_norm = np.sqrt(
        residual_ug * residual_ug
        + residual_gr * residual_gr
        + residual_ri * residual_ri
        + residual_iz * residual_iz
    )

    abs_b = np.abs(gal_b)
    signed_b = gal_b

    ebv_decile = _decile_bins(ebv_clipped, train_mask)
    abs_b_decile = _decile_bins(abs_b, train_mask)

    population = raw["galaxy_population"].astype("string").str.strip().str.lower()
    red_sequence = population.eq("red_sequence").to_numpy(dtype=np.float64)
    blue_cloud = population.eq("blue_cloud").to_numpy(dtype=np.float64)

    continuous_features = {
        "ebv_raw": ebv_raw,
        "ebv_clipped": ebv_clipped,
        "u0": u0,
        "g0": g0,
        "r0": r0,
        "i0": i0,
        "z0": z0,
        "obs_ug": obs_ug,
        "obs_gr": obs_gr,
        "obs_ri": obs_ri,
        "obs_iz": obs_iz,
        "dered_ug": dered_ug,
        "dered_gr": dered_gr,
        "dered_ri": dered_ri,
        "dered_iz": dered_iz,
        "parallel_obs": parallel_obs,
        "parallel_dered": parallel_dered,
        "dust_parallel": dust_parallel,
        "residual_ug": residual_ug,
        "residual_gr": residual_gr,
        "residual_ri": residual_ri,
        "residual_iz": residual_iz,
        "residual_norm": residual_norm,
        "abs_b": abs_b,
        "signed_b": signed_b,
        "dust_parallel_x_abs_b": dust_parallel * abs_b,
        "ebv_x_redshift": ebv_clipped * redshift,
        "ebv_x_red_sequence": ebv_clipped * red_sequence,
        "ebv_x_blue_cloud": ebv_clipped * blue_cloud,
    }

    out = pd.DataFrame(index=raw.index)
    for name, values in continuous_features.items():
        out[name] = _clip_to_train_percentiles(values, train_mask)

    out["is_ebv_imputed"] = is_ebv_imputed.astype(bool)
    out["ebv_decile"] = ebv_decile
    out["abs_b_decile"] = abs_b_decile

    return out


FEATURE_GROUPS = [
    {
        "name": "foreground_reddening_geometry",
        "fn": add_foreground_reddening_geometry,
        "depends_on": [],
        "description": "Derives dust-corrected color geometry, Galactic latitude context, and reddening interactions from sky position and broadband photometry.",
    }
]