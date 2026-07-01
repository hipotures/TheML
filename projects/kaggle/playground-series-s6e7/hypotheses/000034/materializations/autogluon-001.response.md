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

DEMAND_COLUMNS = (
    "calorie_expenditure",
    "step_count",
    "exercise_duration",
)

TRAIN_ID_MAX = 690087

SLEEP_QUALITY_SCORES = {
    "poor": 0.0,
    "average": 0.5,
    "good": 1.0,
}

STRESS_LEVEL_SCORES = {
    "high": 0.0,
    "medium": 0.5,
    "low": 1.0,
}

DIET_BALANCE_SCORES = {
    "non-veg": 0.35,
    "veg": 0.65,
    "balanced": 1.0,
}


def _numeric(raw, column):
    if column not in raw.columns:
        return pd.Series(np.nan, index=raw.index, dtype="float64")
    return pd.to_numeric(raw[column], errors="coerce").astype("float64")


def _category_score(raw, column, mapping, neutral):
    if column not in raw.columns:
        return pd.Series(neutral, index=raw.index, dtype="float64")
    values = raw[column].astype("string").str.lower().str.strip()
    return values.map(mapping).astype("float64").fillna(neutral)


def _stats_frame(raw):
    if "id" not in raw.columns:
        return raw

    ids = pd.to_numeric(raw["id"], errors="coerce")
    train_like = ids.le(TRAIN_ID_MAX)
    if train_like.any():
        return raw.loc[train_like]

    return raw


def _median_from(frame, column, fallback):
    values = pd.to_numeric(frame[column], errors="coerce") if column in frame.columns else pd.Series(dtype="float64")
    median = values.median(skipna=True)
    if pd.isna(median):
        return fallback
    return float(median)


def _percentile_rank(values, reference_values):
    clean_ref = pd.to_numeric(reference_values, errors="coerce").dropna().sort_values()
    filled = values.fillna(clean_ref.median() if len(clean_ref) else 0.0)

    if len(clean_ref) < 2:
        return pd.Series(0.5, index=values.index, dtype="float64")

    ranks = np.searchsorted(clean_ref.to_numpy(), filled.to_numpy(), side="right") / float(len(clean_ref))
    return pd.Series(ranks, index=values.index, dtype="float64").clip(0.0, 1.0)


def add_energy_availability_phase(raw, deps, aux):
    stats = _stats_frame(raw)

    sleep = _numeric(raw, "sleep_duration")
    heart_rate = _numeric(raw, "heart_rate")
    bmi = _numeric(raw, "bmi")
    calories = _numeric(raw, "calorie_expenditure")
    steps = _numeric(raw, "step_count")
    exercise = _numeric(raw, "exercise_duration")
    water = _numeric(raw, "water_intake")

    sleep_filled = sleep.fillna(_median_from(stats, "sleep_duration", 7.5))
    heart_rate_filled = heart_rate.fillna(_median_from(stats, "heart_rate", 72.0))
    bmi_filled = bmi.fillna(_median_from(stats, "bmi", 22.5))
    water_filled = water.fillna(_median_from(stats, "water_intake", 2.2))

    calorie_pct = _percentile_rank(calories, stats["calorie_expenditure"] if "calorie_expenditure" in stats.columns else calories)
    step_pct = _percentile_rank(steps, stats["step_count"] if "step_count" in stats.columns else steps)
    exercise_pct = _percentile_rank(exercise, stats["exercise_duration"] if "exercise_duration" in stats.columns else exercise)
    heart_rate_pct = _percentile_rank(heart_rate, stats["heart_rate"] if "heart_rate" in stats.columns else heart_rate)

    all_demand_missing = calories.isna() & steps.isna() & exercise.isna()
    demand_score = pd.concat((calorie_pct, step_pct, exercise_pct), axis=1).mean(axis=1).clip(0.0, 1.0)
    demand_score = demand_score.mask(all_demand_missing, 0.5)

    sleep_band_support = (1.0 - ((sleep_filled - 8.0).abs() / 2.5)).clip(0.0, 1.0)
    water_support = ((water_filled - 1.0) / 2.0).clip(0.0, 1.0)
    sleep_quality_support = _category_score(raw, "sleep_quality", SLEEP_QUALITY_SCORES, 0.5)
    stress_support = _category_score(raw, "stress_level", STRESS_LEVEL_SCORES, 0.5)
    diet_support = _category_score(raw, "diet_type", DIET_BALANCE_SCORES, 0.5)

    recovery_support = pd.concat(
        (
            sleep_band_support,
            water_support,
            sleep_quality_support,
            stress_support,
            diet_support,
        ),
        axis=1,
    ).mean(axis=1).clip(0.0, 1.0)

    weak_recovery = (1.0 - recovery_support).clip(0.0, 1.0)
    low_bmi_margin = ((18.5 - bmi_filled) / 2.5).clip(0.0, 1.0)
    healthy_bmi_band = np.minimum((bmi_filled - 18.5) / 2.0, (25.0 - bmi_filled) / 3.0)
    healthy_bmi_band = pd.Series(healthy_bmi_band, index=raw.index, dtype="float64").clip(0.0, 1.0)
    high_bmi_margin = ((bmi_filled - 25.0) / 7.5).clip(0.0, 1.0)

    elevated_hr = heart_rate_pct.clip(0.0, 1.0)
    non_elevated_hr = (1.0 - elevated_hr).clip(0.0, 1.0)

    demand_minus_support_gap = (demand_score - recovery_support).clip(-1.0, 1.0)
    under_supported_exertion_pressure = (
        demand_score * 0.35
        + weak_recovery * 0.25
        + low_bmi_margin * 0.20
        + elevated_hr * 0.20
    ).clip(0.0, 1.0)
    low_throughput_surplus_pressure = (
        high_bmi_margin * 0.40
        + (1.0 - demand_score).clip(0.0, 1.0) * 0.30
        + weak_recovery * 0.30
    ).clip(0.0, 1.0)
    supported_high_throughput_score = (
        demand_score * 0.35
        + healthy_bmi_band * 0.25
        + recovery_support * 0.25
        + non_elevated_hr * 0.15
    ).clip(0.0, 1.0)

    pressure_stack = pd.concat(
        (
            under_supported_exertion_pressure.rename("under_supported_exertion"),
            low_throughput_surplus_pressure.rename("low_throughput_surplus"),
            supported_high_throughput_score.rename("supported_high_throughput"),
        ),
        axis=1,
    )
    phase_label = pressure_stack.idxmax(axis=1).astype("object")
    phase_label = phase_label.mask(pressure_stack.max(axis=1).lt(0.45), "neutral")

    return pd.DataFrame(
        {
            "activity_demand_score": demand_score.astype("float32"),
            "recovery_support_score": recovery_support.astype("float32"),
            "demand_minus_support_gap": demand_minus_support_gap.astype("float32"),
            "low_bmi_reserve_deficit": low_bmi_margin.astype("float32"),
            "healthy_bmi_band": healthy_bmi_band.astype("float32"),
            "high_bmi_surplus_margin": high_bmi_margin.astype("float32"),
            "heart_rate_percentile": heart_rate_pct.astype("float32"),
            "under_supported_exertion_pressure": under_supported_exertion_pressure.astype("float32"),
            "low_throughput_surplus_pressure": low_throughput_surplus_pressure.astype("float32"),
            "supported_high_throughput_score": supported_high_throughput_score.astype("float32"),
            "energy_availability_phase": phase_label.astype("category"),
            "all_demand_inputs_missing": all_demand_missing.astype("int8"),
        },
        index=raw.index,
    )


FEATURE_GROUPS = [
    {
        "name": "energy_availability_phase",
        "fn": add_energy_availability_phase,
        "depends_on": [],
        "description": "Energy availability phase features combining activity demand, recovery support, body context, and heart-rate strain.",
    }
]