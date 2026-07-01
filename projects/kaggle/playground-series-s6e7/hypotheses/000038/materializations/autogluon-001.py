import numpy as np
import pandas as pd


def _numeric_column(frame, column_name):
    if column_name in frame.columns:
        return pd.to_numeric(frame[column_name], errors="coerce")
    return pd.Series(np.nan, index=frame.index, dtype="float64")


def add_hydration_linked_pulse_load(raw, deps, aux):
    step_count = _numeric_column(raw, "step_count")
    exercise_duration = _numeric_column(raw, "exercise_duration")
    calorie_expenditure = _numeric_column(raw, "calorie_expenditure")
    water_intake = _numeric_column(raw, "water_intake")
    heart_rate = _numeric_column(raw, "heart_rate")
    bmi = _numeric_column(raw, "bmi")

    steps_scale = ((step_count - 5000.0) / 5000.0).clip(0.0, 1.0)
    exercise_scale = (exercise_duration / 30.0).clip(0.0, 1.0)
    calories_scale = ((calorie_expenditure - 1800.0) / 1000.0).clip(0.0, 1.0)

    activity_components = pd.concat(
        [steps_scale, exercise_scale, calories_scale],
        axis=1,
        copy=False,
    )
    activity_demand = activity_components.mean(axis=1, skipna=True)
    all_activity_missing = activity_components.isna().all(axis=1)

    bmi_increment = (0.15 * ((bmi - 25.0) / 5.0).clip(0.0, 2.0)).where(bmi.notna(), 0.0)
    fluid_demand_liters = 1.8 + (0.7 * activity_demand) + bmi_increment

    hydration_gap = (water_intake - fluid_demand_liters).clip(-2.0, 2.0)
    pulse_load = ((heart_rate - 80.0) / 20.0).clip(0.0, 2.0)

    hydration_deficit = (-hydration_gap).clip(0.0, 2.0)
    dehydration_pulse_strain = pulse_load * hydration_deficit
    exertional_dehydration_strain = dehydration_pulse_strain * activity_demand
    resting_dehydration_strain = dehydration_pulse_strain * (1.0 - activity_demand)

    supported_active_pulse = (
        (activity_demand >= 0.6)
        & (hydration_gap >= 0.0)
        & heart_rate.between(50.0, 80.0, inclusive="both")
    )

    stage = pd.Series(0, index=raw.index, dtype="int8")
    unknown = water_intake.isna() | heart_rate.isna() | all_activity_missing

    stage.loc[
        (activity_demand >= 0.6)
        & (hydration_gap < -0.3)
        & (heart_rate >= 80.0)
    ] = 2
    stage.loc[
        (activity_demand < 0.4)
        & (hydration_gap < -0.3)
        & (heart_rate >= 80.0)
    ] = 3
    stage.loc[
        (hydration_gap >= 0.0)
        & (heart_rate >= 90.0)
    ] = 4
    stage.loc[supported_active_pulse] = 1
    stage.loc[unknown] = -1

    return pd.DataFrame(
        {
            "activity_demand": activity_demand,
            "fluid_demand_liters": fluid_demand_liters,
            "hydration_gap": hydration_gap,
            "pulse_load": pulse_load,
            "dehydration_pulse_strain": dehydration_pulse_strain,
            "exertional_dehydration_strain": exertional_dehydration_strain,
            "resting_dehydration_strain": resting_dehydration_strain,
            "supported_active_pulse": supported_active_pulse.astype("int8"),
            "hydropulse_stage": stage,
        },
        index=raw.index,
    )


FEATURE_GROUPS = [
    {
        "name": "hydration_linked_pulse_load",
        "fn": add_hydration_linked_pulse_load,
        "depends_on": [],
        "description": "NaN-aware hydration, exertion, and pulse-load interactions that separate supported active profiles from dehydration-linked cardiovascular strain.",
    }
]