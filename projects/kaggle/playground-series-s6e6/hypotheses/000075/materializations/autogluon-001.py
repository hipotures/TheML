import numpy as np
import pandas as pd


FG_SUPPORT_GR_MIN = 0.0
FG_CORE_GR_MIN = 0.2
FG_CORE_GR_MAX = 0.6
FG_SUPPORT_GR_MAX = 0.8

TEFF_DOC_GR_MIN = -0.3
TEFF_DOC_GR_MAX = 1.3
TEFF_LOG_REFERENCE = 3.75

METAL_GR_MIN = 0.2
METAL_GR_MAX = 0.4
UG_METAL_MIN = 0.8
UG_METAL_MAX = 1.4
UG_LOW_MIN = 0.8
UG_LOW_MAX = 1.0
UV_EXCESS_UG_THRESHOLD = 0.6

FEH_PLAUSIBLE_MIN = -2.5
FEH_PLAUSIBLE_MAX = 0.3
FEH_HALO_MIN = -2.2
FEH_HALO_MAX = -1.0
FEH_DISK_MIN = -1.0
FEH_DISK_MAX = 0.3

COLOR_DISTANCE_CLIP = 5.0
METALLICITY_MARGIN_CLIP = 5.0


def _finite_numeric(raw, column, default_value):
    series = pd.to_numeric(raw[column], errors="coerce").astype("float64")
    series = series.replace([np.inf, -np.inf], np.nan)
    if series.isna().any():
        fill_value = series.median(skipna=True)
        if not np.isfinite(fill_value):
            fill_value = default_value
        series = series.fillna(float(fill_value))
    return series


def _support_gate(values, support_lower, core_lower, core_upper, support_upper):
    rising = ((values - support_lower) / (core_lower - support_lower)).clip(lower=0.0, upper=1.0)
    falling = ((support_upper - values) / (support_upper - core_upper)).clip(lower=0.0, upper=1.0)
    support = rising.where(rising <= falling, falling)
    return support.clip(lower=0.0, upper=1.0).fillna(0.0)


def _closed_interval_indicator(values, lower, upper):
    return ((values >= lower) & (values <= upper)).astype("float64")


def _strict_interval_indicator(values, lower, upper):
    return ((values > lower) & (values < upper)).astype("float64")


def _signed_interval_margin(values, lower, upper, clip_value):
    lower_margin = values - lower
    upper_margin = upper - values
    margin = lower_margin.where(lower_margin <= upper_margin, upper_margin)
    return margin.clip(lower=-clip_value, upper=clip_value).fillna(-clip_value)


def _outside_interval_distance(values, lower, upper, clip_value):
    below = (lower - values).clip(lower=0.0)
    above = (values - upper).clip(lower=0.0)
    distance = below + above
    return distance.clip(lower=0.0, upper=clip_value).fillna(clip_value)


def add_uv_blanketing_metallicity_plausibility(raw, deps, aux):
    u_mag = _finite_numeric(raw, "u", 22.0)
    g_mag = _finite_numeric(raw, "g", 21.0)
    r_mag = _finite_numeric(raw, "r", 20.5)

    ug_color = u_mag - g_mag
    gr_color = g_mag - r_mag

    fg_gr_support = _support_gate(
        gr_color,
        FG_SUPPORT_GR_MIN,
        FG_CORE_GR_MIN,
        FG_CORE_GR_MAX,
        FG_SUPPORT_GR_MAX,
    )
    fg_gr_core_indicator = _closed_interval_indicator(gr_color, FG_CORE_GR_MIN, FG_CORE_GR_MAX)
    fg_gr_support_indicator = _closed_interval_indicator(gr_color, FG_SUPPORT_GR_MIN, FG_SUPPORT_GR_MAX)
    fg_gr_taper_indicator = ((fg_gr_support_indicator > 0.0) & (fg_gr_core_indicator < 1.0)).astype("float64")

    gr_core_clipped = gr_color.clip(lower=FG_CORE_GR_MIN, upper=FG_CORE_GR_MAX)
    gr_broad_clipped = gr_color.clip(lower=TEFF_DOC_GR_MIN, upper=TEFF_DOC_GR_MAX)
    log_teff_core_boundary = 3.872 - 0.264 * gr_core_clipped
    log_teff_broad_boundary = (
        3.882
        - 0.316 * gr_broad_clipped
        + 0.0488 * gr_broad_clipped * gr_broad_clipped
        + 0.0283 * gr_broad_clipped * gr_broad_clipped * gr_broad_clipped
    )
    teff_core_k_boundary = 10.0 ** log_teff_core_boundary
    teff_broad_k_boundary = 10.0 ** log_teff_broad_boundary
    log_teff_core_minus_broad = log_teff_core_boundary - log_teff_broad_boundary

    ug_metal_clipped = ug_color.clip(lower=UG_METAL_MIN, upper=UG_METAL_MAX)
    ug_low_clipped = ug_color.clip(lower=UG_LOW_MIN, upper=UG_LOW_MAX)
    feh_uv_blanketing_poly_boundary = (
        -21.88
        + 47.39 * ug_metal_clipped
        - 35.50 * ug_metal_clipped * ug_metal_clipped
        + 9.018 * ug_metal_clipped * ug_metal_clipped * ug_metal_clipped
    )
    feh_low_metallicity_proxy_boundary = 5.14 * ug_low_clipped - 6.10

    metal_hot_gr_indicator = _strict_interval_indicator(gr_color, METAL_GR_MIN, METAL_GR_MAX)
    metal_ug_window_indicator = _strict_interval_indicator(ug_color, UG_METAL_MIN, UG_METAL_MAX)
    metal_calibration_support = metal_hot_gr_indicator * metal_ug_window_indicator
    low_metal_proxy_support = _strict_interval_indicator(ug_color, UG_LOW_MIN, UG_LOW_MAX) * fg_gr_support_indicator

    gr_hot_metal_distance = _outside_interval_distance(
        gr_color,
        METAL_GR_MIN,
        METAL_GR_MAX,
        COLOR_DISTANCE_CLIP,
    )
    ug_metal_window_distance = _outside_interval_distance(
        ug_color,
        UG_METAL_MIN,
        UG_METAL_MAX,
        COLOR_DISTANCE_CLIP,
    )

    uv_excess_conflict_margin = (UV_EXCESS_UG_THRESHOLD - ug_color).clip(lower=0.0, upper=COLOR_DISTANCE_CLIP)
    uv_excess_indicator = (ug_color < UV_EXCESS_UG_THRESHOLD).astype("float64")
    uv_excess_fg_indicator = ((ug_color < UV_EXCESS_UG_THRESHOLD) & (fg_gr_support > 0.0)).astype("float64")

    return pd.DataFrame(
        {
            "ug_color": ug_color,
            "gr_color": gr_color,
            "fg_gr_support": fg_gr_support,
            "fg_gr_core_indicator": fg_gr_core_indicator,
            "fg_gr_taper_indicator": fg_gr_taper_indicator,
            "fg_gr_core_margin": _signed_interval_margin(gr_color, FG_CORE_GR_MIN, FG_CORE_GR_MAX, COLOR_DISTANCE_CLIP),
            "fg_gr_support_margin": _signed_interval_margin(gr_color, FG_SUPPORT_GR_MIN, FG_SUPPORT_GR_MAX, COLOR_DISTANCE_CLIP),
            "fg_gr_documented_teff_margin": _signed_interval_margin(gr_color, TEFF_DOC_GR_MIN, TEFF_DOC_GR_MAX, COLOR_DISTANCE_CLIP),
            "fg_gr_core_distance": _outside_interval_distance(gr_color, FG_CORE_GR_MIN, FG_CORE_GR_MAX, COLOR_DISTANCE_CLIP),
            "fg_gr_support_distance": _outside_interval_distance(gr_color, FG_SUPPORT_GR_MIN, FG_SUPPORT_GR_MAX, COLOR_DISTANCE_CLIP),
            "fg_gr_documented_teff_distance": _outside_interval_distance(gr_color, TEFF_DOC_GR_MIN, TEFF_DOC_GR_MAX, COLOR_DISTANCE_CLIP),
            "log_teff_core_boundary": log_teff_core_boundary,
            "teff_core_k_boundary": teff_core_k_boundary,
            "log_teff_broad_boundary": log_teff_broad_boundary,
            "teff_broad_k_boundary": teff_broad_k_boundary,
            "log_teff_core_minus_broad": log_teff_core_minus_broad,
            "fg_support_weighted_log_teff_centered": fg_gr_support * (log_teff_core_boundary - TEFF_LOG_REFERENCE),
            "fg_support_weighted_teff_mismatch": fg_gr_support * log_teff_core_minus_broad,
            "feh_uv_blanketing_poly_boundary": feh_uv_blanketing_poly_boundary,
            "feh_low_metallicity_proxy_boundary": feh_low_metallicity_proxy_boundary,
            "metal_hot_gr_indicator": metal_hot_gr_indicator,
            "metal_ug_window_indicator": metal_ug_window_indicator,
            "metal_calibration_support": metal_calibration_support,
            "low_metal_proxy_support": low_metal_proxy_support,
            "ug_metal_window_margin": _signed_interval_margin(ug_color, UG_METAL_MIN, UG_METAL_MAX, COLOR_DISTANCE_CLIP),
            "ug_low_proxy_window_margin": _signed_interval_margin(ug_color, UG_LOW_MIN, UG_LOW_MAX, COLOR_DISTANCE_CLIP),
            "gr_hot_metal_window_margin": _signed_interval_margin(gr_color, METAL_GR_MIN, METAL_GR_MAX, COLOR_DISTANCE_CLIP),
            "combined_metal_window_distance": (gr_hot_metal_distance + ug_metal_window_distance).clip(lower=0.0, upper=COLOR_DISTANCE_CLIP),
            "feh_poly_plausible_margin": _signed_interval_margin(feh_uv_blanketing_poly_boundary, FEH_PLAUSIBLE_MIN, FEH_PLAUSIBLE_MAX, METALLICITY_MARGIN_CLIP),
            "feh_poly_halo_margin": _signed_interval_margin(feh_uv_blanketing_poly_boundary, FEH_HALO_MIN, FEH_HALO_MAX, METALLICITY_MARGIN_CLIP),
            "feh_poly_disk_margin": _signed_interval_margin(feh_uv_blanketing_poly_boundary, FEH_DISK_MIN, FEH_DISK_MAX, METALLICITY_MARGIN_CLIP),
            "feh_low_proxy_plausible_margin": _signed_interval_margin(feh_low_metallicity_proxy_boundary, FEH_PLAUSIBLE_MIN, FEH_PLAUSIBLE_MAX, METALLICITY_MARGIN_CLIP),
            "metal_support_weighted_feh_poly": metal_calibration_support * feh_uv_blanketing_poly_boundary,
            "metal_support_weighted_feh_plausible_margin": metal_calibration_support * _signed_interval_margin(
                feh_uv_blanketing_poly_boundary,
                FEH_PLAUSIBLE_MIN,
                FEH_PLAUSIBLE_MAX,
                METALLICITY_MARGIN_CLIP,
            ),
            "low_proxy_support_weighted_feh": low_metal_proxy_support * feh_low_metallicity_proxy_boundary,
            "uv_excess_conflict_margin": uv_excess_conflict_margin,
            "uv_excess_fg_weighted_conflict": fg_gr_support * uv_excess_conflict_margin,
            "uv_excess_indicator": uv_excess_indicator,
            "uv_excess_fg_indicator": uv_excess_fg_indicator,
        },
        index=raw.index,
    )


FEATURE_GROUPS = [
    {
        "name": "uv_blanketing_metallicity_plausibility",
        "fn": add_uv_blanketing_metallicity_plausibility,
        "depends_on": [],
        "description": "Encodes SDSS ugr F/G-star UV blanketing, temperature, metallicity plausibility, calibration support, and UV-excess conflicts.",
    }
]