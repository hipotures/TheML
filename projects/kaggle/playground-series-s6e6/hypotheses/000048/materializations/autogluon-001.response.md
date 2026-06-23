import numpy as np
import pandas as pd

_LUPT_BANDS = ("u", "g", "r", "i", "z")
_LUPT_SOFTENING = (1.4e-10, 0.9e-10, 1.2e-10, 1.8e-10, 7.4e-10)
_LUPT_WAVELENGTHS = (3543.0, 4770.0, 6231.0, 7625.0, 9134.0)
_LUPT_U_AB_OFFSET = 0.04
_LN10 = 2.302585092994046
_EPS = 1e-12
_ADJ_PAIRS = ((0, 1, "ug"), (1, 2, "gr"), (2, 3, "ri"), (3, 4, "iz"))


def _extract_band_matrix(raw):
    n = len(raw)
    cols = []
    for band in _LUPT_BANDS:
        if band in raw.columns:
            cols.append(pd.to_numeric(raw[band], errors="coerce").to_numpy(dtype=float))
        else:
            cols.append(np.full(n, np.nan, dtype=float))
    return np.column_stack(cols) if cols else np.empty((n, 0), dtype=float)


def _normalize_label(value):
    text = str(value).strip().lower().replace(" ", "_").replace("/", "_")
    safe = []
    for char in text:
        if char.isalnum() or char in ("_", "-"):
            safe.append(char)
        else:
            safe.append("_")
    safe_text = "".join(safe)
    return safe_text[:40] if safe_text else "class"


def add_luptitude_regime_flux_features(raw, deps, aux):
    _ = deps
    idx = raw.index
    n = len(raw)
    features = pd.DataFrame(index=idx)

    if n == 0:
        return features

    mags = _extract_band_matrix(raw)
    if mags.shape[1] != len(_LUPT_BANDS):
        return features

    adj_mags = mags.copy()
    adj_mags[:, 0] = adj_mags[:, 0] - _LUPT_U_AB_OFFSET

    softening = np.array(_LUPT_SOFTENING, dtype=float)
    flux_arg = (-0.4 * _LN10) * adj_mags - np.log(softening)
    flux = 2.0 * softening * np.sinh(flux_arg)

    finite_flux = np.isfinite(flux)
    abs_flux = np.abs(flux)
    soft_flag = np.where(finite_flux, (abs_flux <= (10.0 * softening)).astype(float), np.nan)

    sign = np.where(np.isnan(flux), np.nan, np.where(flux >= 0.0, 1.0, -1.0))
    soft_flux = np.where(finite_flux, sign * np.maximum(abs_flux, softening), np.nan)
    log_soft_flux = np.log(np.abs(soft_flux) + _EPS)

    features["lupt_u_ab"] = adj_mags[:, 0]

    for i, band in enumerate(_LUPT_BANDS):
        features[f"lupt_flux_{band}"] = flux[:, i]
        features[f"lupt_soft_regime_{band}"] = soft_flag[:, i]
        features[f"lupt_soft_flux_{band}"] = soft_flux[:, i]
        features[f"lupt_log_soft_flux_{band}"] = log_soft_flux[:, i]

    soft_count = np.nansum(soft_flag, axis=1)
    soft_fraction = soft_count / float(len(_LUPT_BANDS))
    soft_regime_weight = 1.0 - soft_fraction
    negative_flux_count = np.nansum((flux < 0.0).astype(float), axis=1)

    features["lupt_soft_count"] = soft_count
    features["lupt_soft_fraction"] = soft_fraction
    features["lupt_soft_regime_weight"] = soft_regime_weight
    features["lupt_negative_flux_count"] = negative_flux_count

    for left, right, pair in _ADJ_PAIRS:
        features[f"lupt_flux_log_color_{pair}"] = log_soft_flux[:, left] - log_soft_flux[:, right]
        features[f"lupt_delta_c_{pair}"] = (adj_mags[:, left] - mags[:, right]) - (
            log_soft_flux[:, left] - log_soft_flux[:, right]
        )
        features[f"lupt_flux_sign_mismatch_{pair}"] = (flux[:, left] * flux[:, right] < 0.0).astype(float)

    wave_log = np.log10(np.array(_LUPT_WAVELENGTHS, dtype=float))
    dx = np.diff(wave_log)

    first = (log_soft_flux[:, 1:] - log_soft_flux[:, :-1]) / dx
    features["lupt_flux_slope_ug"] = first[:, 0]
    features["lupt_flux_slope_gr"] = first[:, 1]
    features["lupt_flux_slope_ri"] = first[:, 2]
    features["lupt_flux_slope_iz"] = first[:, 3]

    second_span = np.array([wave_log[2] - wave_log[0], wave_log[3] - wave_log[1], wave_log[4] - wave_log[2]])
    second = (first[:, 1:] - first[:, :-1]) / second_span
    features["lupt_flux_second_ugr"] = second[:, 0]
    features["lupt_flux_second_gri"] = second[:, 1]
    features["lupt_flux_second_riz"] = second[:, 2]

    x = wave_log
    x2 = x * x
    x3 = x2 * x
    x4 = x2 * x2
    npts = float(len(_LUPT_BANDS))
    xtx = np.array(
        [
            [npts, np.sum(x), np.sum(x2)],
            [np.sum(x), np.sum(x2), np.sum(x3)],
            [np.sum(x2), np.sum(x3), np.sum(x4)],
        ],
        dtype=float,
    )
    xtx_inv = np.linalg.inv(xtx)

    rhs = np.column_stack(
        (
            np.nansum(log_soft_flux, axis=1),
            np.nansum(log_soft_flux * x[None, :], axis=1),
            np.nansum(log_soft_flux * x2[None, :], axis=1),
        )
    )
    quad_coef = rhs @ xtx_inv
    ls_slope = quad_coef[:, 1] + 2.0 * quad_coef[:, 2] * (np.sum(x) / npts)
    ls_curvature = 2.0 * quad_coef[:, 2]

    features["lupt_lsq_slope"] = ls_slope
    features["lupt_lsq_curvature"] = ls_curvature
    features["lupt_lsq_slope_high_snr"] = ls_slope * soft_regime_weight
    features["lupt_lsq_slope_low_snr"] = ls_slope * soft_fraction
    features["lupt_lsq_curvature_high_snr"] = ls_curvature * soft_regime_weight
    features["lupt_lsq_curvature_low_snr"] = ls_curvature * soft_fraction

    abs_soft = np.abs(soft_flux)
    sum_abs_soft = np.nansum(abs_soft, axis=1)
    row_has_data = np.isfinite(sum_abs_soft)
    share = abs_soft / (sum_abs_soft[:, None] + _EPS)

    features["lupt_soft_abs_sum"] = sum_abs_soft
    features["lupt_soft_share_max"] = np.where(row_has_data, np.max(share, axis=1), np.nan)
    features["lupt_soft_entropy"] = -np.nansum(share * np.log(share + _EPS), axis=1)
    features["lupt_soft_l2"] = np.where(row_has_data, np.nansum(share * share, axis=1), np.nan)

    for i, band in enumerate(_LUPT_BANDS):
        features[f"lupt_soft_share_{band}"] = share[:, i]

    if isinstance(aux, pd.DataFrame) and not aux.empty and {"u", "g", "r", "i", "z", "class"}.issubset(aux.columns):
        aux_mags = _extract_band_matrix(aux)
        if aux_mags.shape[1] == len(_LUPT_BANDS):
            aux_mask = np.isfinite(aux_mags).all(axis=1)
            if np.any(aux_mask):
                aux_adj = aux_mags.copy()
                aux_adj[:, 0] = aux_adj[:, 0] - _LUPT_U_AB_OFFSET
                aux_arg = (-0.4 * _LN10) * aux_adj - np.log(softening)
                aux_flux = 2.0 * softening * np.sinh(aux_arg)
                aux_finite = np.isfinite(aux_flux)
                aux_sign = np.where(np.isnan(aux_flux), np.nan, np.where(aux_flux >= 0.0, 1.0, -1.0))
                aux_abs = np.abs(aux_flux)
                aux_soft_flux = np.where(aux_finite, aux_sign * np.maximum(aux_abs, softening), np.nan)
                aux_log_soft = np.log(np.abs(aux_soft_flux) + _EPS)

                aux_class = aux.loc[aux_mask, "class"].astype(str).to_numpy()
                seen = []

                for cls in pd.unique(aux_class):
                    if cls in ("nan", "None", "null"):
                        continue
                    m = aux_class == cls
                    if not np.any(m):
                        continue
                    aux_rows = aux_log_soft[aux_mask][m]
                    if aux_rows.size == 0:
                        continue
                    centroid = np.nanmean(aux_rows, axis=0)
                    if np.any(np.isfinite(centroid)):
                        dist = np.sqrt(np.nansum((log_soft_flux - centroid[None, :]) ** 2, axis=1))
                        safe_cls = _normalize_label(cls)
                        features[f"lupt_aux_class_dist_{safe_cls}"] = dist
                        features[f"lupt_aux_class_similarity_{safe_cls}"] = 1.0 / (1.0 + dist)
                        seen.append(safe_cls)

                if seen:
                    sim_cols = [f"lupt_aux_class_similarity_{c}" for c in seen]
                    sim = features.loc[:, sim_cols].to_numpy(dtype=float)
                    features["lupt_aux_class_best_similarity"] = np.nanmax(sim, axis=1)

    return features


FEATURE_GROUPS = [
    {
        "name": "luptitude_regime_flux_features",
        "fn": add_luptitude_regime_flux_features,
        "depends_on": [],
        "description": "Generate luptitude regime-aware flux-domain descriptors with soft clipping, slope/curvature structure, and mismatch features for adjacent color pairs.",
    }
]