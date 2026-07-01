import numpy as np
import pandas as pd

MOVEMENT_DENSITY_LABELS = ("missing", "lt_250", "250_500", "500_750", "gte_750")
EXERCISE_DENSITY_LABELS = ("missing", "zero", "gt0_1", "1_3", "gte_3")


def add_waking_day_behavior_density(raw, deps, aux):
    sleep = pd.to_numeric(raw.get("sleep_duration"), errors="coerce")
    steps = pd.to_numeric(raw.get("step_count"), errors="coerce")
    exercise = pd.to_numeric(raw.get("exercise_duration"), errors="coerce")
    calories = pd.to_numeric(raw.get("calorie_expenditure"), errors="coerce")
    water = pd.to_numeric(raw.get("water_intake"), errors="coerce")

    valid_sleep = sleep.notna() & np.isfinite(sleep) & (sleep > 0) & (sleep < 24)
    clipped_sleep = sleep.where(valid_sleep).clip(lower=1.0, upper=23.0)
    waking_hours = 24.0 - clipped_sleep

    valid_waking = valid_sleep & waking_hours.notna() & np.isfinite(waking_hours) & (waking_hours > 0)
    waking_hours = waking_hours.where(valid_waking)

    exercise_minutes_per_wake_hour = exercise / waking_hours
    steps_per_wake_hour = steps / waking_hours
    calories_per_wake_hour = calories / waking_hours
    water_per_wake_hour = water / waking_hours
    exercise_wake_share = exercise / (60.0 * waking_hours)

    kcal_thousands = (calories / 1000.0).clip(lower=0.001)
    hydration_per_1000_kcal = water / kcal_thousands

    sleep_to_exercise_hours = clipped_sleep / (1.0 + exercise / 60.0)
    sleep_to_exercise_hours = sleep_to_exercise_hours.where(valid_waking)

    movement_density_band = pd.Series(MOVEMENT_DENSITY_LABELS[0], index=raw.index, dtype="object")
    movement_density_band = movement_density_band.mask(
        steps_per_wake_hour < 250.0, MOVEMENT_DENSITY_LABELS[1]
    )
    movement_density_band = movement_density_band.mask(
        (steps_per_wake_hour >= 250.0) & (steps_per_wake_hour < 500.0),
        MOVEMENT_DENSITY_LABELS[2],
    )
    movement_density_band = movement_density_band.mask(
        (steps_per_wake_hour >= 500.0) & (steps_per_wake_hour < 750.0),
        MOVEMENT_DENSITY_LABELS[3],
    )
    movement_density_band = movement_density_band.mask(
        steps_per_wake_hour >= 750.0, MOVEMENT_DENSITY_LABELS[4]
    )
    movement_density_band = movement_density_band.where(
        steps_per_wake_hour.notna(), pd.NA
    )

    exercise_density_band = pd.Series(EXERCISE_DENSITY_LABELS[0], index=raw.index, dtype="object")
    exercise_density_band = exercise_density_band.mask(
        exercise_minutes_per_wake_hour == 0.0, EXERCISE_DENSITY_LABELS[1]
    )
    exercise_density_band = exercise_density_band.mask(
        (exercise_minutes_per_wake_hour > 0.0) & (exercise_minutes_per_wake_hour < 1.0),
        EXERCISE_DENSITY_LABELS[2],
    )
    exercise_density_band = exercise_density_band.mask(
        (exercise_minutes_per_wake_hour >= 1.0) & (exercise_minutes_per_wake_hour < 3.0),
        EXERCISE_DENSITY_LABELS[3],
    )
    exercise_density_band = exercise_density_band.mask(
        exercise_minutes_per_wake_hour >= 3.0, EXERCISE_DENSITY_LABELS[4]
    )
    exercise_density_band = exercise_density_band.where(
        exercise_minutes_per_wake_hour.notna(), pd.NA
    )

    return pd.DataFrame(
        {
            "estimated_waking_hours": waking_hours,
            "exercise_wake_share": exercise_wake_share,
            "steps_per_wake_hour": steps_per_wake_hour,
            "calories_per_wake_hour": calories_per_wake_hour,
            "water_per_wake_hour": water_per_wake_hour,
            "hydration_per_1000_kcal": hydration_per_1000_kcal,
            "sleep_to_exercise_hours": sleep_to_exercise_hours,
            "movement_density_band": movement_density_band,
            "exercise_density_band": exercise_density_band,
        },
        index=raw.index,
    )


FEATURE_GROUPS = [
    {
        "name": "waking_day_behavior_density",
        "fn": add_waking_day_behavior_density,
        "depends_on": [],
        "description": "Waking-time-normalized behavior density features for activity, exercise, hydration, calories, and sleep allocation.",
    }
]