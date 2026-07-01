import numpy as np
import pandas as pd


TRAIN_ID_MAX = 690087
MIN_CELL_ROWS = 500
BMI_BINS = (-np.inf, 18.5, 25.0, 30.0, np.inf)
BMI_LABELS = ("underweight", "normal", "overweight", "obese")
STEP_BINS = (-np.inf, 5000.0, 8000.0, 12000.0, np.inf)
STEP_LABELS = ("low_steps", "some_steps", "active_steps", "high_steps")
RESIDUAL_BINS = (-np.inf, -1.0, -0.25, 0.25, 1.0, np.inf)
RESIDUAL_LABELS = (
    "very_low_residual",
    "low_residual",
    "near_expected",
    "high_residual",
    "very_high_residual",
)


def _numeric_series(raw, name, default_value=np.nan):
    if name in raw.columns:
        return pd.to_numeric(raw[name], errors="coerce")
    return pd.Series(default_value, index=raw.index, dtype="float64")


def _train_mask(raw):
    if "id" not in raw.columns:
        return pd.Series(True, index=raw.index)

    ids = pd.to_numeric(raw["id"], errors="coerce")
    mask = ids.le(TRAIN_ID_MAX)
    if bool(mask.any()):
        return mask.fillna(False)
    return pd.Series(True, index=raw.index)


def _safe_median(values, fallback):
    med = values.dropna().median()
    if pd.isna(med):
        return fallback
    return float(med)


def _exercise_band(values):
    out = pd.Series("missing_exercise", index=values.index, dtype="object")
    observed = values.notna()
    out.loc[observed & values.eq(0)] = "none"
    out.loc[observed & values.gt(0) & values.lt(30)] = "short"
    out.loc[observed & values.ge(30) & values.lt(60)] = "medium"
    out.loc[observed & values.ge(60)] = "long"
    return out


def _lookup_expected(stats, keys, fallback):
    expected = pd.Series(np.nan, index=keys.index, dtype="float64")

    levels = [
        ["body_band", "step_band", "exercise_band", "gender_band"],
        ["body_band", "step_band", "exercise_band"],
        ["step_band", "exercise_band"],
    ]

    for level in levels:
        table = stats.get(tuple(level))
        if table is None or table.empty:
            continue

        lookup_frame = keys[level].copy()
        lookup_frame["_row_order"] = np.arange(len(lookup_frame))
        merged = lookup_frame.merge(table, on=level, how="left", sort=False)
        merged = merged.sort_values("_row_order")
        values = pd.Series(merged["expected"].to_numpy(), index=keys.index)
        expected = expected.fillna(values)

    return expected.fillna(fallback)


def add_nonexercise_energy_residual(raw, deps, aux):
    train_mask = _train_mask(raw)

    sleep = _numeric_series(raw, "sleep_duration")
    bmi = _numeric_series(raw, "bmi")
    steps = _numeric_series(raw, "step_count")
    exercise = _numeric_series(raw, "exercise_duration")
    calories = _numeric_series(raw, "calorie_expenditure")

    train_fallback = train_mask & (
        sleep.notna()
        | bmi.notna()
        | steps.notna()
        | exercise.notna()
        | calories.notna()
    )

    sleep_med = _safe_median(sleep.loc[train_fallback], 7.0)
    bmi_med = _safe_median(bmi.loc[train_fallback], 23.0)
    steps_med = _safe_median(steps.loc[train_fallback], 8000.0)
    exercise_med = _safe_median(exercise.loc[train_fallback], 30.0)
    calories_med = _safe_median(calories.loc[train_fallback], 2200.0)

    sleep_calc = sleep.fillna(sleep_med)
    bmi_calc = bmi.fillna(bmi_med)
    steps_calc = steps.fillna(steps_med)
    exercise_calc = exercise.fillna(exercise_med)

    waking_hours = (24.0 - sleep_calc).clip(lower=14.0, upper=21.0)

    body_band = pd.cut(
        bmi_calc,
        bins=list(BMI_BINS),
        labels=list(BMI_LABELS),
        include_lowest=True,
    ).astype("object")

    step_band = pd.cut(
        steps_calc,
        bins=list(STEP_BINS),
        labels=list(STEP_LABELS),
        include_lowest=True,
    ).astype("object")

    exercise_band = _exercise_band(exercise_calc)

    if "gender" in raw.columns:
        gender_band = raw["gender"].astype("object").where(raw["gender"].notna(), "unknown")
    else:
        gender_band = pd.Series("unknown", index=raw.index, dtype="object")

    keys = pd.DataFrame(
        {
            "body_band": body_band,
            "step_band": step_band,
            "exercise_band": exercise_band,
            "gender_band": gender_band,
        },
        index=raw.index,
    )

    fit_mask = train_mask & calories.notna()
    fit_frame = keys.loc[fit_mask].copy()
    fit_frame["calorie_expenditure"] = calories.loc[fit_mask].astype("float64")

    stats = {}
    for level in (
        ["body_band", "step_band", "exercise_band", "gender_band"],
        ["body_band", "step_band", "exercise_band"],
        ["step_band", "exercise_band"],
    ):
        grouped = (
            fit_frame.groupby(level, dropna=False, observed=False)["calorie_expenditure"]
            .agg(["median", "size"])
            .reset_index()
        )
        grouped = grouped.loc[grouped["size"].ge(MIN_CELL_ROWS), level + ["median"]]
        grouped = grouped.rename(columns={"median": "expected"})
        stats[tuple(level)] = grouped

    expected = _lookup_expected(stats, keys, calories_med)
    residual_observed = calories.notna()
    residual = (calories - expected).where(residual_observed, 0.0)

    train_residual = residual.loc[fit_mask]
    q25 = train_residual.quantile(0.25)
    q75 = train_residual.quantile(0.75)
    residual_iqr = float(q75 - q25) if pd.notna(q75) and pd.notna(q25) else 1.0
    if residual_iqr <= 0:
        residual_iqr = 1.0

    residual_z = residual / residual_iqr
    residual_per_waking_hour = residual / waking_hours.replace(0, np.nan)
    residual_to_exercise = residual / (exercise_calc + 10.0)

    residual_bin = pd.cut(
        residual_z,
        bins=list(RESIDUAL_BINS),
        labels=list(RESIDUAL_LABELS),
        include_lowest=True,
    ).astype("object")
    residual_bin = residual_bin.where(residual_observed, "unknown_energy")

    return pd.DataFrame(
        {
            "expected_calorie_expenditure": expected.astype("float64"),
            "energy_residual": residual.astype("float64"),
            "energy_residual_iqr_z": residual_z.astype("float64"),
            "energy_residual_per_waking_hour": residual_per_waking_hour.astype("float64"),
            "energy_residual_to_exercise": residual_to_exercise.astype("float64"),
            "energy_residual_bin": residual_bin.astype("object"),
        },
        index=raw.index,
    )


FEATURE_GROUPS = [
    {
        "name": "nonexercise_energy_residual",
        "fn": add_nonexercise_energy_residual,
        "depends_on": [],
        "description": "Target-free residual energy expenditure after backing out coarse body, step, exercise, and gender context.",
    }
]