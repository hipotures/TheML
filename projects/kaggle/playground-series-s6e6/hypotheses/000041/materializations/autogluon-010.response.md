import math
import numpy as np
import pandas as pd

_EPS = 1e-12
_LOG_TWO_PI = 1.8378770664093453
_LOG_THREE = 1.0986122886681098
_MAD_TO_STD = 1.482602218505602

_CLASS_ORDER = ("galaxy", "star", "qso")
_OUTPUT_COLUMNS = (
    "uniform_galaxy_probability",
    "uniform_star_probability",
    "uniform_qso_probability",
    "local_prior_galaxy_probability",
    "local_prior_star_probability",
    "local_prior_qso_probability",
    "qso_vs_star_log_odds",
    "qso_vs_galaxy_log_odds",
    "star_vs_galaxy_log_odds",
    "top_probability",
    "top_two_probability_gap",
    "posterior_entropy",
    "effective_local_support",
)

_REDSHIFT_EDGES = (-0.02, 0.8, 2.5, 4.0, 7.2)

_RAW_TEMPLATE_CENTERS = {
    "galaxy": (-0.95, -0.42, -0.16, 0.08),
    "star": (-0.62, -0.25, -0.08, 0.04),
    "qso": (-0.18, -0.06, -0.01, 0.02),
}

_RAW_TEMPLATE_SCALES = {
    "galaxy": (0.45, 0.25, 0.16, 0.14),
    "star": (0.38, 0.22, 0.14, 0.12),
    "qso": (0.50, 0.28, 0.18, 0.16),
}

_STAR_REDSHIFT_COMPONENTS = (
    (0.92, 0.0, 0.055),
    (0.08, 0.0, 0.18),
)

_GALAXY_REDSHIFT_COMPONENTS = (
    (0.32, 0.18, 0.18),
    (0.46, 0.62, 0.34),
    (0.22, 1.15, 0.60),
)

_QSO_REDSHIFT_COMPONENTS = (
    (0.23, 1.20, 0.55),
    (0.34, 2.15, 0.75),
    (0.27, 3.65, 0.95),
    (0.16, 5.25, 1.10),
)


def _empty_output(raw):
    return pd.DataFrame(
        {name: np.asarray([], dtype=float) for name in _OUTPUT_COLUMNS},
        index=raw.index,
    )


def _numeric_column(raw, name, default):
    if name not in raw.columns:
        return np.full(len(raw), default, dtype=float)
    values = pd.to_numeric(raw[name], errors="coerce").to_numpy(dtype=float, copy=True)
    values[~np.isfinite(values)] = np.nan
    return values


def _fill_nonfinite(values, default):
    out = np.asarray(values, dtype=float).copy()
    finite = np.isfinite(out)
    if finite.any():
        fill_value = float(np.median(out[finite]))
    else:
        fill_value = float(default)
    out[~finite] = fill_value
    return out


def _uppercase_array(raw, name):
    if name not in raw.columns:
        return np.full(len(raw), "", dtype=object)
    return raw[name].astype(str).str.upper().to_numpy(copy=True)


def _sigmoid(values):
    clipped = np.clip(np.asarray(values, dtype=float), -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def _winsorize_columns(values, low_q, high_q):
    out = np.asarray(values, dtype=float).copy()
    lows = np.nanquantile(out, low_q, axis=0)
    highs = np.nanquantile(out, high_q, axis=0)
    lows = np.where(np.isfinite(lows), lows, 0.0)
    highs = np.where(np.isfinite(highs), highs, lows + 1.0)
    highs = np.where(highs > lows, highs, lows + 1.0)
    return np.clip(out, lows, highs)


def _robust_standardize_columns(values):
    x = np.asarray(values, dtype=float)
    med = np.nanmedian(x, axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    filled = np.where(np.isfinite(x), x, med)
    mad = np.nanmedian(np.abs(filled - med), axis=0)
    std = np.nanstd(filled, axis=0)
    scale = _MAD_TO_STD * mad
    scale = np.where(np.isfinite(scale) & (scale > 1e-8), scale, std)
    scale = np.where(np.isfinite(scale) & (scale > 1e-8), scale, 1.0)
    return (filled - med) / scale, med, scale


def _robust_standardize_1d(values, default):
    clean = _fill_nonfinite(values, default)
    med = float(np.median(clean))
    mad = float(np.median(np.abs(clean - med)))
    std = float(np.std(clean))
    scale = _MAD_TO_STD * mad
    if not np.isfinite(scale) or scale <= 1e-8:
        scale = std
    if not np.isfinite(scale) or scale <= 1e-8:
        scale = 1.0
    return (clean - med) / scale, clean, med, scale


def _weighted_center_scale(values, weights, fallback_center, fallback_scale):
    x = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    w = np.where(np.isfinite(w) & (w > 0.0), w, 0.0)
    total = float(np.sum(w))

    fallback_center_arr = np.asarray(fallback_center, dtype=float)
    fallback_scale_arr = np.asarray(fallback_scale, dtype=float)

    if not np.isfinite(total) or total <= _EPS:
        return fallback_center_arr.copy(), fallback_scale_arr.copy()

    mean = np.sum(x * w[:, None], axis=0) / total
    diff = x - mean
    var = np.sum(diff * diff * w[:, None], axis=0) / total
    scale = np.sqrt(np.maximum(var, 0.04))

    center = 0.78 * mean + 0.22 * fallback_center_arr
    scale = 0.78 * scale + 0.22 * fallback_scale_arr
    scale = np.where(np.isfinite(scale) & (scale > 0.20), scale, fallback_scale_arr)
    scale = np.clip(scale, 0.25, 4.0)

    return center, scale


def _diag_logpdf(values, center, scale):
    x = np.asarray(values, dtype=float)
    c = np.asarray(center, dtype=float)
    s = np.maximum(np.asarray(scale, dtype=float), 1e-6)
    z = (x - c) / s
    return -0.5 * np.sum(z * z + 2.0 * np.log(s) + _LOG_TWO_PI, axis=1)


def _normal_logpdf(values, mean, scale):
    x = np.asarray(values, dtype=float)
    s = max(float(scale), 1e-6)
    z = (x - float(mean)) / s
    return -0.5 * (z * z + 2.0 * math.log(s) + _LOG_TWO_PI)


def _logsumexp_vectors(vectors):
    stacked = np.vstack(vectors).T
    max_value = np.max(stacked, axis=1)
    shifted = np.clip(stacked - max_value[:, None], -745.0, 80.0)
    return max_value + np.log(np.sum(np.exp(shifted), axis=1))


def _mixture_logpdf_1d(values, components):
    terms = []
    for weight, mean, scale in components:
        terms.append(math.log(max(float(weight), _EPS)) + _normal_logpdf(values, mean, scale))
    return _logsumexp_vectors(terms)


def _softmax_rows(log_scores):
    scores = np.asarray(log_scores, dtype=float).copy()
    scores = np.where(np.isfinite(scores), scores, -745.0)
    max_scores = np.max(scores, axis=1, keepdims=True)
    shifted = np.clip(scores - max_scores, -745.0, 80.0)
    exp_scores = np.exp(shifted)
    denom = np.sum(exp_scores, axis=1, keepdims=True)
    probs = exp_scores / np.maximum(denom, _EPS)

    bad_rows = ~np.all(np.isfinite(probs), axis=1)
    if bad_rows.any():
        probs[bad_rows, :] = 1.0 / probs.shape[1]

    return probs


def _effective_local_support(redshift, i_mag):
    n_rows = len(i_mag)
    if n_rows == 0:
        return np.asarray([], dtype=float), np.asarray([], dtype=float)

    z_clean = _fill_nonfinite(redshift, 0.0)
    i_clean = _fill_nonfinite(i_mag, 20.0)

    i_min = float(np.min(i_clean))
    i_max = float(np.max(i_clean))
    start = math.floor(i_min * 5.0) / 5.0 - 0.2
    stop = math.ceil(i_max * 5.0) / 5.0 + 0.4
    if not np.isfinite(start) or not np.isfinite(stop) or stop <= start:
        start = 0.0
        stop = 1.0

    i_edges = np.arange(start, stop + 0.200001, 0.2)
    if len(i_edges) < 2:
        i_edges = np.asarray((start, start + 0.2), dtype=float)

    i_idx = np.searchsorted(i_edges, i_clean, side="right") - 1
    i_idx = np.clip(i_idx, 0, len(i_edges) - 2)

    z_edges = np.asarray(_REDSHIFT_EDGES, dtype=float)
    z_clipped = np.clip(z_clean, z_edges[0], z_edges[-1])
    z_idx = np.searchsorted(z_edges, z_clipped, side="right") - 1
    z_idx = np.clip(z_idx, 0, len(z_edges) - 2)

    n_i_bins = len(i_edges) - 1
    n_z_bins = len(z_edges) - 1
    cell_id = z_idx * n_i_bins + i_idx
    counts = np.bincount(cell_id, minlength=n_i_bins * n_z_bins)
    support = counts[cell_id].astype(float)
    strength = support / (support + 300.0)

    return support, np.clip(strength, 0.0, 1.0)


def _clean_vector(values, default):
    arr = np.asarray(values, dtype=float)
    return np.where(np.isfinite(arr), arr, float(default))


def add_xdqso_inspired_flux_density_scores(raw, deps, aux):
    if len(raw) == 0:
        return _empty_output(raw)

    u = _fill_nonfinite(_numeric_column(raw, "u", 20.0), 20.0)
    g = _fill_nonfinite(_numeric_column(raw, "g", 20.0), 20.0)
    r = _fill_nonfinite(_numeric_column(raw, "r", 20.0), 20.0)
    i = _fill_nonfinite(_numeric_column(raw, "i", 20.0), 20.0)
    z_mag = _fill_nonfinite(_numeric_column(raw, "z", 20.0), 20.0)
    redshift = _fill_nonfinite(_numeric_column(raw, "redshift", 0.0), 0.0)

    flux_shape = np.column_stack(
        (
            -0.4 * (u - i),
            -0.4 * (g - i),
            -0.4 * (r - i),
            -0.4 * (z_mag - i),
        )
    )
    flux_shape = _winsorize_columns(flux_shape, 0.005, 0.995)
    x_norm, x_median, x_scale = _robust_standardize_columns(flux_shape)

    i_norm, i_clean, i_median, i_scale = _robust_standardize_1d(i, 20.0)
    z_clean = np.clip(redshift, _REDSHIFT_EDGES[0], _REDSHIFT_EDGES[-1])

    support, support_strength = _effective_local_support(z_clean, i_clean)

    spectral_type = _uppercase_array(raw, "spectral_type")
    galaxy_population = _uppercase_array(raw, "galaxy_population")

    spec_m = (spectral_type == "M").astype(float)
    spec_gk = ((spectral_type == "G/K") | (spectral_type == "GK") | (spectral_type == "G-K")).astype(float)
    spec_ob = ((spectral_type == "O/B") | (spectral_type == "OB") | (spectral_type == "O-B")).astype(float)
    spec_af = ((spectral_type == "A/F") | (spectral_type == "AF") | (spectral_type == "A-F")).astype(float)

    red_sequence = (galaxy_population == "RED_SEQUENCE").astype(float)
    blue_cloud = (galaxy_population == "BLUE_CLOUD").astype(float)

    blue_flux = 0.55 * x_norm[:, 0] + 0.35 * x_norm[:, 1] + 0.10 * x_norm[:, 2]
    red_flux = -blue_flux
    uv_excess = x_norm[:, 0] - x_norm[:, 1]

    near_zero_z = np.exp(-0.5 * np.square(z_clean / 0.08))
    mid_z = _sigmoid((z_clean - 0.02) / 0.08) * _sigmoid((1.75 - z_clean) / 0.55)
    high_z = _sigmoid((z_clean - 0.85) / 0.30)
    very_high_z = _sigmoid((z_clean - 2.20) / 0.45)

    blue_proxy = _sigmoid(blue_flux / 0.75)
    red_proxy = _sigmoid(red_flux / 0.75)
    uv_proxy = _sigmoid(uv_excess / 0.70)
    faint_proxy = _sigmoid((i_norm + 0.15) / 0.75)
    bright_proxy = _sigmoid((-i_norm + 0.10) / 0.80)

    star_proxy = (
        (near_zero_z + 0.02)
        * (1.0 + 0.18 * bright_proxy + 0.12 * spec_m + 0.10 * spec_gk + 0.08 * spec_af + 0.06 * spec_ob)
        * (1.0 + 0.08 * red_proxy)
    )
    galaxy_proxy = (
        (mid_z + 0.04)
        * (1.0 + 0.30 * red_sequence + 0.18 * blue_cloud)
        * (1.0 + 0.15 * red_proxy + 0.05 * faint_proxy)
    )
    qso_proxy = (
        (high_z + 0.04 + 0.25 * very_high_z)
        * (1.0 + 0.28 * blue_proxy + 0.20 * uv_proxy + 0.15 * faint_proxy)
        * (1.0 + 0.12 * blue_cloud + 0.08 * spec_ob + 0.05 * spec_af)
    )

    galaxy_template_center = (np.asarray(_RAW_TEMPLATE_CENTERS["galaxy"], dtype=float) - x_median) / x_scale
    star_template_center = (np.asarray(_RAW_TEMPLATE_CENTERS["star"], dtype=float) - x_median) / x_scale
    qso_template_center = (np.asarray(_RAW_TEMPLATE_CENTERS["qso"], dtype=float) - x_median) / x_scale

    galaxy_template_scale = np.maximum(np.asarray(_RAW_TEMPLATE_SCALES["galaxy"], dtype=float) / x_scale, 0.35)
    star_template_scale = np.maximum(np.asarray(_RAW_TEMPLATE_SCALES["star"], dtype=float) / x_scale, 0.35)
    qso_template_scale = np.maximum(np.asarray(_RAW_TEMPLATE_SCALES["qso"], dtype=float) / x_scale, 0.35)

    galaxy_center, galaxy_scale = _weighted_center_scale(
        x_norm, galaxy_proxy, galaxy_template_center, galaxy_template_scale
    )
    star_center, star_scale = _weighted_center_scale(
        x_norm, star_proxy, star_template_center, star_template_scale
    )
    qso_center, qso_scale = _weighted_center_scale(
        x_norm, qso_proxy, qso_template_center, qso_template_scale
    )

    i_matrix = i_norm[:, None]
    galaxy_i_center, galaxy_i_scale = _weighted_center_scale(i_matrix, galaxy_proxy, (0.0,), (1.0,))
    star_i_center, star_i_scale = _weighted_center_scale(i_matrix, star_proxy, (-0.15,), (1.0,))
    qso_i_center, qso_i_scale = _weighted_center_scale(i_matrix, qso_proxy, (0.20,), (1.0,))

    global_center = np.zeros(x_norm.shape[1], dtype=float)
    global_scale = np.maximum(np.nanstd(x_norm, axis=0), 0.80)
    global_shape = _diag_logpdf(x_norm, global_center, global_scale)

    galaxy_shape = support_strength * _diag_logpdf(x_norm, galaxy_center, galaxy_scale) + (1.0 - support_strength) * global_shape
    star_shape = support_strength * _diag_logpdf(x_norm, star_center, star_scale) + (1.0 - support_strength) * global_shape
    qso_shape = support_strength * _diag_logpdf(x_norm, qso_center, qso_scale) + (1.0 - support_strength) * global_shape

    galaxy_z_log = _mixture_logpdf_1d(z_clean, _GALAXY_REDSHIFT_COMPONENTS)
    star_z_log = _mixture_logpdf_1d(z_clean, _STAR_REDSHIFT_COMPONENTS)
    qso_z_log = _mixture_logpdf_1d(z_clean, _QSO_REDSHIFT_COMPONENTS)

    galaxy_i_log = _normal_logpdf(i_norm, galaxy_i_center[0], galaxy_i_scale[0])
    star_i_log = _normal_logpdf(i_norm, star_i_center[0], star_i_scale[0])
    qso_i_log = _normal_logpdf(i_norm, qso_i_center[0], qso_i_scale[0])

    galaxy_score = galaxy_shape + 0.70 * galaxy_z_log + 0.20 * galaxy_i_log
    star_score = star_shape + 0.70 * star_z_log + 0.20 * star_i_log
    qso_score = qso_shape + 0.70 * qso_z_log + 0.20 * qso_i_log

    log_scores = np.column_stack((galaxy_score, star_score, qso_score))
    fallback_scores = np.column_stack(
        (
            global_shape + 0.35 * galaxy_z_log,
            global_shape + 0.35 * star_z_log,
            global_shape + 0.35 * qso_z_log,
        )
    )
    log_scores = np.where(np.isfinite(log_scores), log_scores, fallback_scores)
    log_scores = np.where(np.isfinite(log_scores), log_scores, 0.0)

    local_prior_raw = np.column_stack((galaxy_proxy, star_proxy, qso_proxy))
    prior_sum = np.sum(local_prior_raw, axis=1, keepdims=True)
    local_prior = local_prior_raw / np.maximum(prior_sum, _EPS)
    local_prior = np.where(np.isfinite(local_prior), local_prior, 1.0 / 3.0)
    local_prior = 0.8 * local_prior + (0.2 / 3.0)
    local_prior = support_strength[:, None] * local_prior + (1.0 - support_strength[:, None]) / 3.0
    local_prior = local_prior / np.maximum(np.sum(local_prior, axis=1, keepdims=True), _EPS)

    uniform_prob = _softmax_rows(log_scores)
    local_prob = _softmax_rows(log_scores + np.log(np.clip(local_prior, _EPS, 1.0)))

    uniform_log_prob = np.log(np.clip(uniform_prob, _EPS, 1.0))
    sorted_local_prob = np.sort(local_prob, axis=1)
    top_probability = sorted_local_prob[:, -1]
    top_two_gap = sorted_local_prob[:, -1] - sorted_local_prob[:, -2]
    entropy = -np.sum(local_prob * np.log(np.clip(local_prob, _EPS, 1.0)), axis=1) / _LOG_THREE

    data = {
        "uniform_galaxy_probability": _clean_vector(uniform_prob[:, 0], 1.0 / 3.0),
        "uniform_star_probability": _clean_vector(uniform_prob[:, 1], 1.0 / 3.0),
        "uniform_qso_probability": _clean_vector(uniform_prob[:, 2], 1.0 / 3.0),
        "local_prior_galaxy_probability": _clean_vector(local_prob[:, 0], 1.0 / 3.0),
        "local_prior_star_probability": _clean_vector(local_prob[:, 1], 1.0 / 3.0),
        "local_prior_qso_probability": _clean_vector(local_prob[:, 2], 1.0 / 3.0),
        "qso_vs_star_log_odds": _clean_vector(uniform_log_prob[:, 2] - uniform_log_prob[:, 1], 0.0),
        "qso_vs_galaxy_log_odds": _clean_vector(uniform_log_prob[:, 2] - uniform_log_prob[:, 0], 0.0),
        "star_vs_galaxy_log_odds": _clean_vector(uniform_log_prob[:, 1] - uniform_log_prob[:, 0], 0.0),
        "top_probability": _clean_vector(top_probability, 1.0 / 3.0),
        "top_two_probability_gap": _clean_vector(top_two_gap, 0.0),
        "posterior_entropy": _clean_vector(entropy, 1.0),
        "effective_local_support": _clean_vector(support, 0.0),
    }

    return pd.DataFrame(data, index=raw.index)


FEATURE_GROUPS = [
    {
        "name": "xdqso_inspired_flux_density_scores",
        "fn": add_xdqso_inspired_flux_density_scores,
        "depends_on": [],
        "description": "Covariate-only XDQSO-inspired flux-shape likelihood proxy scores conditioned on redshift, brightness, and local support.",
    }
]