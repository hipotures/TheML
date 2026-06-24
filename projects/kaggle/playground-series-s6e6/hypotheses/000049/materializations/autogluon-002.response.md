import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype

BANDS = ("u", "g", "r", "i", "z")
ASINH_SOFTENING = (1.4e-10, 0.9e-10, 1.2e-10, 1.8e-10, 7.4e-10)
M10 = (22.12, 22.60, 22.29, 21.85, 20.32)
M0 = (24.63, 25.11, 24.80, 24.36, 22.83)
TRAIN_MAX_ID = 577346
EPS = 1e-12


def _infer_train_mask(raw, aux):
    n = len(raw)
    if n == 0:
        return np.array([], dtype=bool)

    if isinstance(aux, pd.DataFrame) and not aux.empty and len(aux) == n:
        for col in (
            "is_train",
            "is_train_mask",
            "train_mask",
            "train",
            "split_train",
            "train_flag",
        ):
            if col not in aux.columns:
                continue

            values = aux[col]
            if is_bool_dtype(values):
                return values.fillna(False).astype(bool).to_numpy()

            if is_numeric_dtype(values):
                numeric = pd.to_numeric(values, errors="coerce")
                finite = numeric.dropna()
                if not finite.empty and set(finite.unique()).issubset({0, 1}):
                    return (numeric.fillna(0) > 0).to_numpy(bool)

            text = values.astype("string").str.strip().str.lower()
            unique_text = set(text.dropna().unique())
            if {"train", "test"}.issubset(unique_text) or {"train", "valid"}.issubset(unique_text):
                return (text == "train").to_numpy(bool)

    if "id" in raw.columns and is_numeric_dtype(raw["id"]):
        ids = pd.to_numeric(raw["id"], errors="coerce")
        if ids.notna().all():
            id_min = int(ids.min())
            id_max = int(ids.max())
            if id_min == 0 and id_max == TRAIN_MAX_ID:
                return np.ones(n, dtype=bool)
            if id_min == 0 and id_max > TRAIN_MAX_ID:
                return (ids.to_numpy() <= TRAIN_MAX_ID).astype(bool)
            if id_min > TRAIN_MAX_ID and id_min == id_max - n + 1:
                return np.zeros(n, dtype=bool)

    return np.ones(n, dtype=bool)


def _max_true_run(values):
    current = 0
    best = 0
    for value in values:
        if value:
            current += 1
            if current > best:
                best = current
        else:
            current = 0
    return best


def add_asinh_censoring_regime_geometry(raw, deps, aux):
    idx = raw.index
    n = len(raw)
    if n == 0:
        return pd.DataFrame(index=idx)

    train_mask = _infer_train_mask(raw, aux)
    if train_mask.size != n:
        train_mask = np.ones(n, dtype=bool)
    if not train_mask.any():
        train_mask = np.ones(n, dtype=bool)

    mag = np.column_stack([pd.to_numeric(raw[band], errors="coerce").to_numpy(dtype=float) for band in BANDS])

    m10 = np.asarray(M10, dtype=float)
    m0 = np.asarray(M0, dtype=float)
    soft = np.asarray(ASINH_SOFTENING, dtype=float)

    ln10_over_2p5 = np.log(10.0) / 2.5
    flux = 2.0 * soft * np.sinh(-(ln10_over_2p5 * mag) - np.log(soft))
    flux_df = pd.DataFrame(flux, columns=BANDS, index=idx)

    for i, band in enumerate(BANDS):
        train_band = flux_df.loc[train_mask, band]
        if train_band.notna().sum() >= 3:
            q_low = float(train_band.quantile(0.001))
            q_high = float(train_band.quantile(0.999))
            if np.isfinite(q_low) and np.isfinite(q_high):
                if q_low > q_high:
                    q_low, q_high = q_high, q_low
                flux_df.loc[:, band] = flux_df[band].clip(lower=q_low, upper=q_high)

    state = np.full((n, len(BANDS)), -1, dtype=np.int8)
    for i in range(len(BANDS)):
        m = mag[:, i]
        finite = np.isfinite(m)
        state[finite & (m <= m10[i]), i] = 2
        state[finite & (m > m10[i]) & (m <= m0[i]), i] = 1
        state[finite & (m > m0[i]), i] = 0

    regime = pd.DataFrame(state, columns=BANDS, index=idx)
    conf_df = pd.DataFrame(1.0 / (1.0 + np.exp((mag - m10) / 0.10)), columns=BANDS, index=idx)
    ratio_df = pd.DataFrame(np.clip((mag - m10) / (m0 - m10), 0.0, 1.0), columns=BANDS, index=idx)

    state_arr = state
    detected = state_arr > 0
    eq0 = state_arr == 0
    le1 = (state_arr >= 0) & (state_arr <= 1)

    count_high = (state_arr == 2).sum(axis=1).astype(np.int16)
    count_low = (state_arr == 1).sum(axis=1).astype(np.int16)
    count_nondetect = (state_arr == 0).sum(axis=1).astype(np.int16)
    detected_band_count = detected.sum(axis=1).astype(np.int16)

    u_drop = (state_arr[:, 0] == 0).astype(np.int8)
    g_drop_after_u = ((state_arr[:, 1] == 0) & (state_arr[:, 0] > 0)).astype(np.int8)
    ug_drop_with_redder = ((state_arr[:, 0] == 0) & (state_arr[:, 1] == 0) & (detected[:, 2:].any(axis=1))).astype(np.int8)
    late_dropout_count = (u_drop + g_drop_after_u + ug_drop_with_redder).astype(np.int16)

    non_detected_redder_count = np.zeros(n, dtype=np.int8)
    for j in range(1, len(BANDS)):
        prior_detected = detected[:, :j].any(axis=1)
        non_detected_redder_count += ((state_arr[:, j] == 0) & prior_detected).astype(np.int8)

    det_any = detected.any(axis=1)
    first_detected_band = np.where(det_any, detected.argmax(axis=1), -1).astype(np.int8)
    last_detected_band = np.where(det_any, len(BANDS) - 1 - np.fliplr(detected).argmax(axis=1), -1).astype(np.int8)

    max_run_le1 = np.fromiter((_max_true_run(row) for row in le1), dtype=np.int8, count=n)
    max_run_eq0 = np.fromiter((_max_true_run(row) for row in eq0), dtype=np.int8, count=n)

    flux_arr = flux_df.to_numpy(dtype=float)
    valid = detected
    pair_mask = valid[:, 1:] & valid[:, :-1]
    adjacent_pair_count = pair_mask.sum(axis=1).astype(np.int8)
    shape_ok = adjacent_pair_count >= 2

    log_slopes = np.log10((flux_arr[:, 1:] + EPS) / (flux_arr[:, :-1] + EPS))
    slope_u_g = np.where(shape_ok & pair_mask[:, 0], log_slopes[:, 0], 0.0)
    slope_g_r = np.where(shape_ok & pair_mask[:, 1], log_slopes[:, 1], 0.0)
    slope_r_i = np.where(shape_ok & pair_mask[:, 2], log_slopes[:, 2], 0.0)
    slope_i_z = np.where(shape_ok & pair_mask[:, 3], log_slopes[:, 3], 0.0)

    curvature_u_g_r = np.where(shape_ok & pair_mask[:, 0] & pair_mask[:, 1], log_slopes[:, 1] - log_slopes[:, 0], 0.0)
    curvature_g_r_i = np.where(shape_ok & pair_mask[:, 1] & pair_mask[:, 2], log_slopes[:, 2] - log_slopes[:, 1], 0.0)
    curvature_r_i_z = np.where(shape_ok & pair_mask[:, 2] & pair_mask[:, 3], log_slopes[:, 3] - log_slopes[:, 2], 0.0)

    abs_flux = np.where(valid, np.abs(flux_arr), 0.0)
    mass = abs_flux.sum(axis=1)
    safe_mass = np.where(mass > 0.0, mass, 1.0)
    p = np.where(valid, abs_flux / safe_mass[:, None], 0.0)
    entropy = -np.sum(np.where(p > 0, p * np.log(p), 0.0), axis=1)
    centroid = np.sum(p * np.arange(len(BANDS), dtype=float), axis=1)
    entropy = np.where(shape_ok, entropy, 0.0)
    centroid = np.where(shape_ok, centroid, 0.0)

    features = {
        "asinh_flux_u_winsorized": flux_df["u"],
        "asinh_flux_g_winsorized": flux_df["g"],
        "asinh_flux_r_winsorized": flux_df["r"],
        "asinh_flux_i_winsorized": flux_df["i"],
        "asinh_flux_z_winsorized": flux_df["z"],
        "regime_state_u": regime["u"],
        "regime_state_g": regime["g"],
        "regime_state_r": regime["r"],
        "regime_state_i": regime["i"],
        "regime_state_z": regime["z"],
        "regime_confidence_u": conf_df["u"],
        "regime_confidence_g": conf_df["g"],
        "regime_confidence_r": conf_df["r"],
        "regime_confidence_i": conf_df["i"],
        "regime_confidence_z": conf_df["z"],
        "censor_depth_ratio_u": ratio_df["u"],
        "censor_depth_ratio_g": ratio_df["g"],
        "censor_depth_ratio_r": ratio_df["r"],
        "censor_depth_ratio_i": ratio_df["i"],
        "censor_depth_ratio_z": ratio_df["z"],
        "regime_count_highsnr": pd.Series(count_high, index=idx, dtype=np.int16),
        "regime_count_lowsnr": pd.Series(count_low, index=idx, dtype=np.int16),
        "regime_count_nondetect": pd.Series(count_nondetect, index=idx, dtype=np.int16),
        "detected_band_count": pd.Series(detected_band_count, index=idx, dtype=np.int16),
        "first_detected_band_idx": pd.Series(first_detected_band, index=idx, dtype=np.int8),
        "last_detected_band_idx": pd.Series(last_detected_band, index=idx, dtype=np.int8),
        "u_dropout": pd.Series(u_drop, index=idx, dtype=np.int8),
        "g_dropout_after_u": pd.Series(g_drop_after_u, index=idx, dtype=np.int8),
        "u_g_dropout_with_redder_detected": pd.Series(ug_drop_with_redder, index=idx, dtype=np.int8),
        "late_dropout_count": pd.Series(late_dropout_count, index=idx, dtype=np.int16),
        "non_detected_redder_than_blue_count": pd.Series(non_detected_redder_count, index=idx, dtype=np.int8),
        "max_run_r_le1": pd.Series(max_run_le1, index=idx, dtype=np.int8),
        "max_run_r_eq0": pd.Series(max_run_eq0, index=idx, dtype=np.int8),
        "valid_adjacent_pairs": pd.Series(adjacent_pair_count, index=idx, dtype=np.int8),
        "shape_slope_u_g": pd.Series(slope_u_g, index=idx, dtype=float),
        "shape_slope_g_r": pd.Series(slope_g_r, index=idx, dtype=float),
        "shape_slope_r_i": pd.Series(slope_r_i, index=idx, dtype=float),
        "shape_slope_i_z": pd.Series(slope_i_z, index=idx, dtype=float),
        "shape_curvature_u_g_r": pd.Series(curvature_u_g_r, index=idx, dtype=float),
        "shape_curvature_g_r_i": pd.Series(curvature_g_r_i, index=idx, dtype=float),
        "shape_curvature_r_i_z": pd.Series(curvature_r_i_z, index=idx, dtype=float),
        "flux_mass_entropy": pd.Series(entropy, index=idx, dtype=float),
        "flux_mass_centroid": pd.Series(centroid, index=idx, dtype=float),
    }

    return pd.DataFrame(features, index=idx)


FEATURE_GROUPS = [
    {
        "name": "asinh_censoring_regime_geometry",
        "fn": add_asinh_censoring_regime_geometry,
        "depends_on": [],
        "description": "Builds SDSS asinh inversion detection-regime topology and censored-band geometry descriptors across u,g,r,i,z.",
    }
]