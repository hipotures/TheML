import numpy as np
import pandas as pd


NUMERIC_COLUMNS = (
    "sleep_duration",
    "heart_rate",
    "bmi",
    "calorie_expenditure",
    "step_count",
    "exercise_duration",
    "water_intake",
)

TRAIN_ID_MAX = 690087
EPSILON = 1e-6


def _as_numeric_series(raw, column):
    if column in raw.columns:
        return pd.to_numeric(raw[column], errors="coerce")
    return pd.Series(np.nan, index=raw.index, dtype="float64")


def _is_multiple(series, step):
    values = series.to_numpy(dtype="float64", copy=False)
    out = np.zeros(len(series), dtype=bool)
    mask = np.isfinite(values)
    if mask.any():
        ratio = values[mask] / step
        out[mask] = np.abs(ratio - np.round(ratio)) <= EPSILON
    return pd.Series(out, index=series.index)


def _decimal_tiers(series):
    values = series.to_numpy(dtype="float64", copy=False)
    finite = np.isfinite(values)

    integer_like = np.zeros(len(series), dtype=bool)
    one_decimal_like = np.zeros(len(series), dtype=bool)
    two_decimal_like = np.zeros(len(series), dtype=bool)

    if finite.any():
        finite_values = values[finite]
        integer_like[finite] = np.abs(finite_values - np.round(finite_values)) <= EPSILON
        one_decimal_like[finite] = np.abs(finite_values * 10.0 - np.round(finite_values * 10.0)) <= EPSILON
        two_decimal_like[finite] = np.abs(finite_values * 100.0 - np.round(finite_values * 100.0)) <= EPSILON

    integer_tier = integer_like
    one_decimal_tier = one_decimal_like & ~integer_like
    two_decimal_tier = two_decimal_like & ~one_decimal_like
    finer_tier = finite & ~two_decimal_like

    return integer_tier, one_decimal_tier, two_decimal_tier, finer_tier, finite


def _training_mask(raw):
    if "id" not in raw.columns:
        return pd.Series(True, index=raw.index)

    ids = pd.to_numeric(raw["id"], errors="coerce")
    mask = ids <= TRAIN_ID_MAX
    if mask.any():
        return mask.fillna(False)
    return pd.Series(True, index=raw.index)


def _safe_fraction(numerator, denominator):
    return np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator, dtype="float64"),
        where=denominator > 0,
    )


def add_report_granularity_clipping_signature(raw, deps, aux):
    features = pd.DataFrame(index=raw.index)
    train_mask = _training_mask(raw)

    observed_count = np.zeros(len(raw), dtype="float64")
    coarse_count = np.zeros(len(raw), dtype="float64")
    integer_count = np.zeros(len(raw), dtype="float64")
    tenth_count = np.zeros(len(raw), dtype="float64")
    hundredth_count = np.zeros(len(raw), dtype="float64")
    fine_count = np.zeros(len(raw), dtype="float64")

    boundary_hit_count = np.zeros(len(raw), dtype="float64")
    outside_count = np.zeros(len(raw), dtype="float64")
    pressure_sum = np.zeros(len(raw), dtype="float64")
    pressure_max = np.zeros(len(raw), dtype="float64")

    for column in NUMERIC_COLUMNS:
        series = _as_numeric_series(raw, column)
        finite = series.notna().to_numpy()
        observed_count += finite.astype("float64")

        if column == "sleep_duration":
            coarse = _is_multiple(series, 0.5).to_numpy()
            features[column + "_whole_hour_heap"] = _is_multiple(series, 1.0).astype("int8")
            features[column + "_half_hour_heap"] = coarse.astype("int8")
        elif column == "heart_rate":
            coarse = _is_multiple(series, 5.0).to_numpy()
            features[column + "_multiple_5_heap"] = coarse.astype("int8")
            features[column + "_multiple_10_heap"] = _is_multiple(series, 10.0).astype("int8")
        elif column == "bmi":
            coarse = _is_multiple(series, 0.5).to_numpy()
            features[column + "_whole_heap"] = _is_multiple(series, 1.0).astype("int8")
            features[column + "_half_heap"] = coarse.astype("int8")
            features[column + "_one_decimal_heap"] = _is_multiple(series, 0.1).astype("int8")
        elif column == "calorie_expenditure":
            coarse = _is_multiple(series, 50.0).to_numpy()
            features[column + "_multiple_50_heap"] = coarse.astype("int8")
            features[column + "_multiple_100_heap"] = _is_multiple(series, 100.0).astype("int8")
        elif column == "step_count":
            coarse = _is_multiple(series, 500.0).to_numpy()
            features[column + "_multiple_500_heap"] = coarse.astype("int8")
            features[column + "_multiple_1000_heap"] = _is_multiple(series, 1000.0).astype("int8")
        elif column == "exercise_duration":
            coarse = _is_multiple(series, 5.0).to_numpy()
            features[column + "_multiple_5_heap"] = coarse.astype("int8")
            features[column + "_multiple_10_heap"] = _is_multiple(series, 10.0).astype("int8")
            features[column + "_multiple_30_heap"] = _is_multiple(series, 30.0).astype("int8")
        else:
            coarse = _is_multiple(series, 0.25).to_numpy()
            features[column + "_quarter_liter_heap"] = coarse.astype("int8")
            features[column + "_half_liter_heap"] = _is_multiple(series, 0.5).astype("int8")

        coarse_count += (coarse & finite).astype("float64")

        integer_tier, one_decimal_tier, two_decimal_tier, finer_tier, finite_tier = _decimal_tiers(series)
        features[column + "_precision_integer"] = integer_tier.astype("int8")
        features[column + "_precision_one_decimal"] = one_decimal_tier.astype("int8")
        features[column + "_precision_two_decimal"] = two_decimal_tier.astype("int8")
        features[column + "_precision_finer"] = finer_tier.astype("int8")

        integer_count += integer_tier.astype("float64")
        tenth_count += one_decimal_tier.astype("float64")
        hundredth_count += two_decimal_tier.astype("float64")
        fine_count += finer_tier.astype("float64")

        train_values = series.loc[train_mask & series.notna()]
        if train_values.empty:
            lower = np.nan
            upper = np.nan
            iqr = np.nan
        else:
            lower = float(train_values.min())
            upper = float(train_values.max())
            q1 = float(train_values.quantile(0.25))
            q3 = float(train_values.quantile(0.75))
            iqr = q3 - q1

        values = series.to_numpy(dtype="float64", copy=False)
        if np.isfinite(lower) and np.isfinite(upper):
            train_range = upper - lower
            scale = max(0.02 * train_range, 0.05 * iqr if np.isfinite(iqr) else 0.0, EPSILON)
            lower_hit = finite & (np.abs(values - lower) <= EPSILON)
            upper_hit = finite & (np.abs(values - upper) <= EPSILON)
            outside = finite & ((values < lower) | (values > upper))
            distance_to_bound = np.minimum(np.abs(values - lower), np.abs(values - upper))
            pressure = np.clip(1.0 - distance_to_bound / scale, 0.0, 1.0)
            pressure = np.where(finite, pressure, 0.0)
            pressure = np.where(outside, 1.0, pressure)
        else:
            lower_hit = np.zeros(len(raw), dtype=bool)
            upper_hit = np.zeros(len(raw), dtype=bool)
            outside = np.zeros(len(raw), dtype=bool)
            pressure = np.zeros(len(raw), dtype="float64")

        features[column + "_train_lower_bound_hit"] = lower_hit.astype("int8")
        features[column + "_train_upper_bound_hit"] = upper_hit.astype("int8")
        features[column + "_outside_train_envelope"] = outside.astype("int8")
        features[column + "_boundary_pressure"] = pressure.astype("float32")

        boundary_hit_count += (lower_hit | upper_hit).astype("float64")
        outside_count += outside.astype("float64")
        pressure_sum += pressure
        pressure_max = np.maximum(pressure_max, pressure)

    features["observed_numeric_count"] = observed_count.astype("float32")
    features["coarse_heaped_fraction"] = _safe_fraction(coarse_count, observed_count).astype("float32")
    features["integer_like_fraction"] = _safe_fraction(integer_count, observed_count).astype("float32")
    features["tenth_like_fraction"] = _safe_fraction(tenth_count, observed_count).astype("float32")
    features["hundredth_like_fraction"] = _safe_fraction(hundredth_count, observed_count).astype("float32")
    features["fine_grained_fraction"] = _safe_fraction(fine_count, observed_count).astype("float32")
    features["boundary_hit_count"] = boundary_hit_count.astype("float32")
    features["outside_train_envelope_count"] = outside_count.astype("float32")
    features["max_boundary_pressure"] = pressure_max.astype("float32")
    features["mean_boundary_pressure"] = _safe_fraction(pressure_sum, observed_count).astype("float32")

    return features


FEATURE_GROUPS = [
    {
        "name": "report_granularity_clipping_signature",
        "fn": add_report_granularity_clipping_signature,
        "depends_on": [],
        "description": "Measurement granularity, heaping, and train-boundary clipping signatures for raw student health measurements.",
    }
]