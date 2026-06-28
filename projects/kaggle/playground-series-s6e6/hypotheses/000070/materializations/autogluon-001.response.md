import numpy as np
import pandas as pd


OPTICAL_BANDS = ("u", "g", "r", "i", "z")
RED_TAIL_BANDS = ("r", "i", "z")
QUAD_BANDS = ("g", "r", "i", "z")
VIRTUAL_IR_ANCHORS = ("J", "K", "W1", "W2")

EFFECTIVE_WAVELENGTH_MICRON = {
    "u": 0.355,
    "g": 0.477,
    "r": 0.623,
    "i": 0.762,
    "z": 0.913,
    "J": 1.25,
    "K": 2.20,
    "W1": 3.40,
    "W2": 4.60,
}

SPLIT_FLAG_COLUMNS = (
    "is_train",
    "_is_train",
    "train",
    "_train",
    "split_train",
    "__is_train",
)

TRAIN_MAX_ID = 577346
LOW_Q = 0.001
HIGH_Q = 0.999
EPS = 1.0e-12


def _as_numeric_series(df, name, index):
    if name in df.columns:
        return pd.to_numeric(df[name], errors="coerce").astype("float64")
    return pd.Series(np.nan, index=index, dtype="float64")


def _training_mask(raw, aux):
    index = raw.index

    if isinstance(aux, pd.DataFrame) and len(aux) == len(raw):
        for col in SPLIT_FLAG_COLUMNS:
            if col in aux.columns:
                mask = aux[col].astype(bool)
                return pd.Series(mask.to_numpy(), index=index)

    if "id" in raw.columns:
        ids = pd.to_numeric(raw["id"], errors="coerce")
        mask = ids <= TRAIN_MAX_ID
        if bool(mask.any()):
            return pd.Series(mask.to_numpy(), index=index)

    return pd.Series(True, index=index)


def _finite_quantiles(values, train_mask, low_q=LOW_Q, high_q=HIGH_Q):
    train_values = values.loc[train_mask]
    train_values = train_values.replace([np.inf, -np.inf], np.nan).dropna()

    if train_values.empty:
        all_values = values.replace([np.inf, -np.inf], np.nan).dropna()
        if all_values.empty:
            return -1.0, 1.0
        train_values = all_values

    lo = float(train_values.quantile(low_q))
    hi = float(train_values.quantile(high_q))

    if not np.isfinite(lo):
        lo = float(train_values.min())
    if not np.isfinite(hi):
        hi = float(train_values.max())
    if not np.isfinite(lo):
        lo = -1.0
    if not np.isfinite(hi):
        hi = 1.0
    if hi <= lo:
        spread = max(abs(lo), 1.0) * 1.0e-6
        lo -= spread
        hi += spread

    return lo, hi


def _weighted_polyfit_per_row(x, y, w, degree):
    n_rows = y.shape[0]
    n_coef = degree + 1
    coefs = np.full((n_rows, n_coef), np.nan, dtype="float64")
    good = np.zeros(n_rows, dtype=bool)

    for row_idx in range(n_rows):
        yi = y[row_idx]
        wi = w[row_idx]
        valid = np.isfinite(yi) & np.isfinite(wi) & (wi > 0.0)

        if int(valid.sum()) < n_coef:
            continue

        xv = x[valid]
        yv = yi[valid]
        sw = np.sqrt(wi[valid])

        design_cols = []
        for power in range(degree, -1, -1):
            design_cols.append(xv ** power)
        design = np.vstack(design_cols).T

        weighted_design = design * sw[:, None]
        weighted_y = yv * sw

        try:
            coef, _, rank, _ = np.linalg.lstsq(weighted_design, weighted_y, rcond=None)
        except np.linalg.LinAlgError:
            continue

        if rank == n_coef and np.all(np.isfinite(coef)):
            coefs[row_idx] = coef
            good[row_idx] = True

    return coefs, good


def add_virtual_infrared_tail_extrapolation(raw, deps, aux):
    index = raw.index
    train_mask = _training_mask(raw, aux)

    mags = pd.DataFrame(index=index)
    for band in OPTICAL_BANDS:
        mags[band] = _as_numeric_series(raw, band, index)

    clipped = pd.DataFrame(index=index)
    mag_clip_width = pd.DataFrame(index=index)

    for band in OPTICAL_BANDS:
        lo, hi = _finite_quantiles(mags[band], train_mask)
        clipped[band] = mags[band].clip(lo, hi)
        mag_clip_width[band] = (mags[band] - clipped[band]).abs()

    median_mag = clipped.loc[:, OPTICAL_BANDS].median(axis=1)
    rel_log_flux = pd.DataFrame(index=index)

    for band in OPTICAL_BANDS:
        rel_log_flux[band] = -0.4 * (clipped[band] - median_mag)

    colors = pd.DataFrame(index=index)
    color_pairs = (
        ("u", "g"),
        ("g", "r"),
        ("r", "i"),
        ("i", "z"),
        ("g", "z"),
        ("r", "z"),
    )

    for left, right in color_pairs:
        name = left + "_minus_" + right
        raw_color = clipped[left] - clipped[right]
        lo, hi = _finite_quantiles(raw_color, train_mask)
        colors[name] = raw_color.clip(lo, hi)

    faint_center = {}
    faint_scale = {}
    for band in OPTICAL_BANDS:
        center, high = _finite_quantiles(clipped[band], train_mask, 0.50, 0.995)
        scale = max(high - center, 0.10)
        faint_center[band] = center
        faint_scale[band] = scale

    weights = pd.DataFrame(index=index)
    for band in OPTICAL_BANDS:
        faintness = (clipped[band] - faint_center[band]) / faint_scale[band]
        weights[band] = 1.0 / (1.0 + np.exp(3.0 * faintness.clip(-8.0, 8.0)))
        weights[band] = weights[band].clip(0.05, 1.0)

    log_wave = {}
    for band, wave in EFFECTIVE_WAVELENGTH_MICRON.items():
        log_wave[band] = float(np.log(wave))

    red_x = np.asarray([log_wave[band] for band in RED_TAIL_BANDS], dtype="float64")
    quad_x = np.asarray([log_wave[band] for band in QUAD_BANDS], dtype="float64")

    red_y = rel_log_flux.loc[:, RED_TAIL_BANDS].to_numpy(dtype="float64")
    red_w = weights.loc[:, RED_TAIL_BANDS].to_numpy(dtype="float64")
    quad_y = rel_log_flux.loc[:, QUAD_BANDS].to_numpy(dtype="float64")
    quad_w = weights.loc[:, QUAD_BANDS].to_numpy(dtype="float64")

    red_coef, red_good = _weighted_polyfit_per_row(red_x, red_y, red_w, 1)
    quad_coef, quad_good = _weighted_polyfit_per_row(quad_x, quad_y, quad_w, 2)

    red_slope = red_coef[:, 0]
    red_intercept = red_coef[:, 1]
    quad_a = quad_coef[:, 0]
    quad_b = quad_coef[:, 1]
    quad_c = quad_coef[:, 2]

    red_reliability = np.nanmean(red_w, axis=1)
    quad_reliability = np.nanmean(quad_w, axis=1)
    fit_reliability = np.where(quad_good, quad_reliability, red_reliability * 0.50)

    out = pd.DataFrame(index=index)

    redshift = _as_numeric_series(raw, "redshift", index)
    gate_redshift = redshift.clip(lower=0.0)

    low_z = gate_redshift < 0.05
    mid_z = (gate_redshift >= 0.05) & (gate_redshift < 2.15)
    high_z = (gate_redshift >= 2.15) & (gate_redshift < 3.50)
    extreme_z = gate_redshift >= 3.50

    out["red_tail_slope_logflux_logwave"] = red_slope
    out["red_tail_spectral_index"] = red_slope / np.log(10.0)
    out["red_tail_intercept"] = red_intercept
    out["red_tail_reliability"] = red_reliability
    out["quad_tail_curvature"] = quad_a
    out["quad_tail_linear_term"] = quad_b
    out["quad_fit_good"] = quad_good
    out["virtual_ir_fit_reliability"] = fit_reliability

    z_rel_flux = rel_log_flux["z"].to_numpy(dtype="float64")
    i_rel_flux = rel_log_flux["i"].to_numpy(dtype="float64")

    virtual_values = {}
    red_virtual_values = {}

    for anchor in VIRTUAL_IR_ANCHORS:
        x_anchor = log_wave[anchor]

        red_pred = red_slope * x_anchor + red_intercept
        quad_pred = quad_a * x_anchor * x_anchor + quad_b * x_anchor + quad_c
        use_quad = quad_good & np.isfinite(quad_pred)
        pred = np.where(use_quad, quad_pred, red_pred)

        pred_series = pd.Series(pred, index=index)
        lo, hi = _finite_quantiles(pred_series, train_mask)
        pred = np.clip(pred, lo, hi)

        red_pred_series = pd.Series(red_pred, index=index)
        red_lo, red_hi = _finite_quantiles(red_pred_series, train_mask)
        red_pred = np.clip(red_pred, red_lo, red_hi)

        virtual_values[anchor] = pred
        red_virtual_values[anchor] = red_pred

        out["virtual_" + anchor + "_rel_log_flux"] = pred
        out["virtual_" + anchor + "_redline_rel_log_flux"] = red_pred
        out["z_to_virtual_" + anchor + "_color_surrogate"] = -2.5 * (z_rel_flux - pred)
        out["i_to_virtual_" + anchor + "_color_surrogate"] = -2.5 * (i_rel_flux - pred)
        out["virtual_" + anchor + "_quad_redline_disagreement"] = pred - red_pred

    j_flux = virtual_values["J"]
    k_flux = virtual_values["K"]
    w1_flux = virtual_values["W1"]
    w2_flux = virtual_values["W2"]

    out["virtual_J_minus_K_color_surrogate"] = -2.5 * (j_flux - k_flux)
    out["virtual_K_minus_W1_color_surrogate"] = -2.5 * (k_flux - w1_flux)
    out["virtual_W1_minus_W2_color_surrogate"] = -2.5 * (w1_flux - w2_flux)
    out["virtual_z_minus_W1_color_surrogate"] = -2.5 * (z_rel_flux - w1_flux)
    out["virtual_z_minus_W2_color_surrogate"] = -2.5 * (z_rel_flux - w2_flux)

    out["red_tail_rayleigh_jeans_margin"] = red_slope + 0.8
    out["red_tail_powerlaw_margin"] = red_slope + 0.4
    out["virtual_ir_long_tail_drop"] = j_flux - w2_flux
    out["virtual_ir_short_long_ratio"] = (j_flux - z_rel_flux) / (w2_flux - z_rel_flux + EPS)
    out["virtual_ir_mean_disagreement"] = (
        np.abs(virtual_values["J"] - red_virtual_values["J"])
        + np.abs(virtual_values["K"] - red_virtual_values["K"])
        + np.abs(virtual_values["W1"] - red_virtual_values["W1"])
        + np.abs(virtual_values["W2"] - red_virtual_values["W2"])
    ) / 4.0

    out["optical_red_tail_color_ri"] = colors["r_minus_i"]
    out["optical_red_tail_color_iz"] = colors["i_minus_z"]
    out["optical_red_tail_color_rz"] = colors["r_minus_z"]
    out["blue_to_red_baseline_gz"] = colors["g_minus_z"]
    out["blue_to_red_baseline_ug"] = colors["u_minus_g"]
    out["red_tail_mag_clip_amount"] = mag_clip_width.loc[:, RED_TAIL_BANDS].mean(axis=1)

    redshift_gate_specs = (
        ("z_lt_005", low_z),
        ("z_005_215", mid_z),
        ("z_215_350", high_z),
        ("z_ge_350", extreme_z),
    )

    for gate_name, mask in redshift_gate_specs:
        gate = mask.astype("float64").to_numpy()
        out[gate_name + "_red_tail_spectral_index"] = out["red_tail_spectral_index"].to_numpy(dtype="float64") * gate
        out[gate_name + "_virtual_z_minus_W1_color"] = out["virtual_z_minus_W1_color_surrogate"].to_numpy(dtype="float64") * gate
        out[gate_name + "_virtual_W1_minus_W2_color"] = out["virtual_W1_minus_W2_color_surrogate"].to_numpy(dtype="float64") * gate
        out[gate_name + "_quad_redline_disagreement"] = out["virtual_ir_mean_disagreement"].to_numpy(dtype="float64") * gate
        out[gate_name + "_fit_reliability"] = out["virtual_ir_fit_reliability"].to_numpy(dtype="float64") * gate

    out = out.replace([np.inf, -np.inf], np.nan)
    return out


FEATURE_GROUPS = [
    {
        "name": "virtual_infrared_tail_extrapolation",
        "fn": add_virtual_infrared_tail_extrapolation,
        "depends_on": [],
        "description": "Extrapolates optical ugriz continuum shape into virtual infrared anchors and summarizes red-tail behavior.",
    }
]