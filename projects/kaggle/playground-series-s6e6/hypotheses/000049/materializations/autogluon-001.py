import numpy as np
import pandas as pd

_BANDS = ("u", "g", "r", "i", "z")
_SOFTENING = (1.4e-10, 0.9e-10, 1.2e-10, 1.8e-10, 7.4e-10)
_M10_MAG = (22.12, 22.60, 22.29, 21.85, 20.32)
_M0_MAG = (24.63, 25.11, 24.80, 24.36, 22.83)
_QUANTILE_BOUNDS = (0.002, 0.998)
_LN10_DIV_2PT5 = 0.9210340371976184
_LOG_EPS = 1e-12
_BAND_POSITIONS = (0.0, 1.0, 2.0, 3.0, 4.0)


def _to_band_float_frame(df):
    bands = df.loc[:, _BANDS].copy()
    return bands.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)


def _build_reference_frame(raw_bands, aux):
    if not isinstance(aux, pd.DataFrame) or aux.empty:
        return raw_bands
    aux_bands = aux.reindex(columns=_BANDS)
    if aux_bands.empty:
        return raw_bands
    aux_bands = aux_bands.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    aux_bands = aux_bands.dropna(how="all")
    if aux_bands.empty:
        return raw_bands
    return pd.concat([raw_bands, aux_bands], axis=0, ignore_index=True)


def _asinh_flux_ratio(mag_series, softening):
    mag = pd.to_numeric(mag_series, errors="coerce")
    return 2.0 * softening * np.sinh(-(mag * _LN10_DIV_2PT5) - np.log(softening))


def _clip_to_reference_quantiles(series, reference_series, lower_q, upper_q):
    lower = reference_series.quantile(lower_q)
    upper = reference_series.quantile(upper_q)
    if pd.isna(lower) or pd.isna(upper):
        return series
    if not np.isfinite(lower) or not np.isfinite(upper):
        return series
    if lower == upper:
        return series
    if lower > upper:
        lower, upper = upper, lower
    return series.clip(lower=lower, upper=upper)


def _longest_true_run_1d(matrix):
    arr = np.asarray(matrix, dtype=np.bool_)
    if arr.ndim != 2 or arr.shape[1] != 5:
        return np.zeros(arr.shape[0], dtype=np.uint8)

    run5 = arr.all(axis=1)
    run4 = np.logical_or(
        np.all(arr[:, 0:4], axis=1),
        np.all(arr[:, 1:5], axis=1),
    )
    run3 = np.logical_or.reduce((
        np.all(arr[:, 0:3], axis=1),
        np.all(arr[:, 1:4], axis=1),
        np.all(arr[:, 2:5], axis=1),
    ))
    run2 = np.logical_or.reduce((
        np.all(arr[:, 0:2], axis=1),
        np.all(arr[:, 1:3], axis=1),
        np.all(arr[:, 2:4], axis=1),
        np.all(arr[:, 3:5], axis=1),
    ))
    run1 = arr.any(axis=1)

    return np.select(
        (run5, run4, run3, run2, run1),
        (5, 4, 3, 2, 1),
        default=0,
    ).astype(np.uint8)


def add_asinh_censoring_regime_geometry(raw, deps, aux):
    raw_bands = _to_band_float_frame(raw)
    ref_bands = _build_reference_frame(raw_bands, aux)
    q_low, q_high = _QUANTILE_BOUNDS

    flux = {}
    q_ratio = {}
    detect = {}
    neg = {}
    delta = {}

    for band, softening, m10, m0 in zip(_BANDS, _SOFTENING, _M10_MAG, _M0_MAG):
        ref_flux = _asinh_flux_ratio(ref_bands[band], softening).replace([np.inf, -np.inf], np.nan)
        raw_flux = _asinh_flux_ratio(raw_bands[band], softening).replace([np.inf, -np.inf], np.nan)
        clipped_flux = _clip_to_reference_quantiles(raw_flux, ref_flux, q_low, q_high)

        flux[band] = clipped_flux
        q_ratio[band] = clipped_flux / (2.0 * softening)
        detect[band] = (clipped_flux > (10.0 * softening)).astype(np.int8)
        neg[band] = (clipped_flux < 0.0).astype(np.int8)
        delta[band] = ((raw_bands[band] - m10) / (m0 - m10)).clip(0.0, 1.0)

    flux_df = pd.DataFrame(flux, index=raw.index)
    q_df = pd.DataFrame(q_ratio, index=raw.index)
    detect_df = pd.DataFrame(detect, index=raw.index)
    neg_df = pd.DataFrame(neg, index=raw.index)
    delta_df = pd.DataFrame(delta, index=raw.index)

    detect_arr = detect_df.to_numpy(dtype=np.int8)
    missing_arr = (detect_arr == 0)
    detected_count = detect_arr.sum(axis=1).astype(np.int8)
    detect_frac = detected_count.astype(np.float32) / float(len(_BANDS))

    non_detected_redder_than_blue = np.zeros(raw.shape[0], dtype=np.int8)
    for i in range(1, len(_BANDS)):
        has_bluer_detected = detect_arr[:, :i].any(axis=1)
        non_detected_redder_than_blue += (missing_arr[:, i] & has_bluer_detected).astype(np.int8)

    longest_missing_run = _longest_true_run_1d(missing_arr)

    u_detected = detect_arr[:, 0].astype(bool)
    g_detected = detect_arr[:, 1].astype(bool)
    r_detected = detect_arr[:, 2].astype(bool)
    i_detected = detect_arr[:, 3].astype(bool)
    z_detected = detect_arr[:, 4].astype(bool)

    u_drop = (missing_arr[:, 0] & g_detected & r_detected & i_detected & z_detected).astype(np.int8)
    g_drop = (u_detected & missing_arr[:, 1] & r_detected & i_detected & z_detected).astype(np.int8)
    ug_drop = (missing_arr[:, 0] & missing_arr[:, 1] & r_detected & i_detected & z_detected).astype(np.int8)

    detected_flux = flux_df.to_numpy(dtype=np.float64) * detect_arr.astype(np.float64)
    abs_detected_flux = np.abs(detected_flux)
    log_abs_flux = np.log10(np.maximum(abs_detected_flux, _LOG_EPS))

    slope_ug = pd.Series(log_abs_flux[:, 0] - log_abs_flux[:, 1], index=raw.index)
    slope_gr = pd.Series(log_abs_flux[:, 1] - log_abs_flux[:, 2], index=raw.index)
    slope_ri = pd.Series(log_abs_flux[:, 2] - log_abs_flux[:, 3], index=raw.index)
    slope_iz = pd.Series(log_abs_flux[:, 3] - log_abs_flux[:, 4], index=raw.index)

    slope_ug = _clip_to_reference_quantiles(slope_ug, slope_ug, q_low, q_high)
    slope_gr = _clip_to_reference_quantiles(slope_gr, slope_gr, q_low, q_high)
    slope_ri = _clip_to_reference_quantiles(slope_ri, slope_ri, q_low, q_high)
    slope_iz = _clip_to_reference_quantiles(slope_iz, slope_iz, q_low, q_high)

    curv_ug_gr = _clip_to_reference_quantiles(slope_ug - slope_gr, slope_ug - slope_gr, q_low, q_high)
    curv_gr_ri = _clip_to_reference_quantiles(slope_gr - slope_ri, slope_gr - slope_ri, q_low, q_high)
    curv_ri_iz = _clip_to_reference_quantiles(slope_ri - slope_iz, slope_ri - slope_iz, q_low, q_high)

    abs_sum = abs_detected_flux.sum(axis=1)
    valid = abs_sum > 0
    mass_center = np.zeros(raw.shape[0], dtype=np.float64)
    mass_entropy = np.zeros(raw.shape[0], dtype=np.float64)

    if np.any(valid):
        weights = np.zeros_like(abs_detected_flux, dtype=np.float64)
        weights[valid] = abs_detected_flux[valid] / abs_sum[valid][:, None]
        positions = np.array(_BAND_POSITIONS, dtype=np.float64)
        mass_center[valid] = weights[valid] @ positions
        ent = -(weights[valid] * np.log(np.maximum(weights[valid], np.finfo(np.float64).tiny))).sum(axis=1)
        mass_entropy[valid] = ent / np.log(5.0)

    mass_center_norm = (mass_center / 4.0).astype(np.float32)
    mass_entropy = np.clip(mass_entropy.astype(np.float32), 0.0, 1.0)

    new_features = pd.DataFrame(index=raw.index)
    new_features["arcg_detected_count"] = detected_count.astype(np.int16)
    new_features["arcg_detected_fraction"] = detect_frac.astype(np.float32)
    new_features["arcg_non_detected_redder_than_blue_count"] = non_detected_redder_than_blue.astype(np.int16)
    new_features["arcg_longest_missing_run"] = longest_missing_run.astype(np.int16)
    new_features["arcg_u_dropout"] = u_drop
    new_features["arcg_g_dropout"] = g_drop
    new_features["arcg_ug_dropout"] = ug_drop
    new_features["arcg_detected_flux_mass_center_norm"] = mass_center_norm
    new_features["arcg_detected_flux_mass_entropy"] = mass_entropy

    for band in _BANDS:
        new_features[f"arcg_fratio_{band}"] = flux_df[band]
        new_features[f"arcg_q_{band}"] = q_df[band]
        new_features[f"arcg_detect10_{band}"] = detect_df[band]
        new_features[f"arcg_neg_{band}"] = neg_df[band]
        new_features[f"arcg_delta_{band}"] = delta_df[band]

    new_features["arcg_absflux_slope_ug"] = slope_ug.astype(np.float64)
    new_features["arcg_absflux_slope_gr"] = slope_gr.astype(np.float64)
    new_features["arcg_absflux_slope_ri"] = slope_ri.astype(np.float64)
    new_features["arcg_absflux_slope_iz"] = slope_iz.astype(np.float64)
    new_features["arcg_absflux_curv_ug_gr"] = curv_ug_gr.astype(np.float64)
    new_features["arcg_absflux_curv_gr_ri"] = curv_gr_ri.astype(np.float64)
    new_features["arcg_absflux_curv_ri_iz"] = curv_ri_iz.astype(np.float64)

    return new_features


FEATURE_GROUPS = [
    {
        "name": "asinh_censoring_regime_geometry",
        "fn": add_asinh_censoring_regime_geometry,
        "depends_on": [],
        "description": "Compute asinh-based detection-regime flags, dropout geometry, and detection-only flux-shape descriptors across ugriz.",
    },
]