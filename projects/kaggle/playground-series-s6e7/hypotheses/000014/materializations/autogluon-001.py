import numpy as np
import pandas as pd


BMI_UNDERWEIGHT_MAX = 18.5
BMI_HEALTHY_MAX = 25.0
BMI_OVERWEIGHT_MAX = 30.0
HR_LOW_MAX = 60.0
HR_NORMAL_MAX = 80.0
HR_HIGH_NORMAL_MAX = 100.0
ATHLETIC_STEPS_MIN = 10000.0
ATHLETIC_EXERCISE_MIN = 45.0


def _numeric_column(raw, name):
    if name in raw.columns:
        return pd.to_numeric(raw[name], errors="coerce")
    return pd.Series(np.nan, index=raw.index, dtype="float64")


def _string_column(raw, name):
    if name in raw.columns:
        return raw[name].astype("string").str.strip().str.lower()
    return pd.Series(pd.NA, index=raw.index, dtype="string")


def add_fitness_contextualized_cardiometabolic_stage(raw, deps, aux):
    bmi = _numeric_column(raw, "bmi")
    heart_rate = _numeric_column(raw, "heart_rate")
    step_count = _numeric_column(raw, "step_count")
    exercise_duration = _numeric_column(raw, "exercise_duration")
    activity_level = _string_column(raw, "physical_activity_level")

    bmi_category = pd.Series("bmi_missing", index=raw.index, dtype="object")
    bmi_category = bmi_category.mask(bmi < BMI_UNDERWEIGHT_MAX, "underweight")
    bmi_category = bmi_category.mask((bmi >= BMI_UNDERWEIGHT_MAX) & (bmi < BMI_HEALTHY_MAX), "healthy")
    bmi_category = bmi_category.mask((bmi >= BMI_HEALTHY_MAX) & (bmi < BMI_OVERWEIGHT_MAX), "overweight")
    bmi_category = bmi_category.mask(bmi >= BMI_OVERWEIGHT_MAX, "obesity")

    heart_category = pd.Series("heart_rate_missing", index=raw.index, dtype="object")
    heart_category = heart_category.mask(heart_rate < HR_LOW_MAX, "low")
    heart_category = heart_category.mask((heart_rate >= HR_LOW_MAX) & (heart_rate < HR_NORMAL_MAX), "normal")
    heart_category = heart_category.mask((heart_rate >= HR_NORMAL_MAX) & (heart_rate < HR_HIGH_NORMAL_MAX), "high_normal")
    heart_category = heart_category.mask(heart_rate >= HR_HIGH_NORMAL_MAX, "tachycardia_like")

    athletic_evidence = (
        activity_level.eq("active").fillna(False)
        | step_count.ge(ATHLETIC_STEPS_MIN).fillna(False)
        | exercise_duration.ge(ATHLETIC_EXERCISE_MIN).fillna(False)
    )

    contextual_heart_category = heart_category.copy()
    low_heart_rate = heart_category.eq("low")
    contextual_heart_category = contextual_heart_category.mask(low_heart_rate & athletic_evidence, "athletic_low")
    contextual_heart_category = contextual_heart_category.mask(low_heart_rate & ~athletic_evidence, "unexplained_low")

    bmi_points = pd.Series(0, index=raw.index, dtype="int16")
    bmi_points = bmi_points.mask(bmi_category.eq("underweight"), 1)
    bmi_points = bmi_points.mask(bmi_category.eq("overweight"), 1)
    bmi_points = bmi_points.mask(bmi_category.eq("obesity"), 2)

    heart_points = pd.Series(0, index=raw.index, dtype="int16")
    heart_points = heart_points.mask(contextual_heart_category.eq("unexplained_low"), 1)
    heart_points = heart_points.mask(contextual_heart_category.eq("high_normal"), 1)
    heart_points = heart_points.mask(contextual_heart_category.eq("tachycardia_like"), 2)

    stage_score = (bmi_points + heart_points).astype("int16")

    high_pulse = contextual_heart_category.isin(["high_normal", "tachycardia_like"])

    new_features = pd.DataFrame(index=raw.index)
    new_features["bmi_category"] = bmi_category.astype("category")
    new_features["contextual_heart_rate_category"] = contextual_heart_category.astype("category")
    new_features["bmi_x_contextual_heart_rate"] = (
        bmi_category.astype("string") + "__" + contextual_heart_category.astype("string")
    ).astype("category")
    new_features["cardiometabolic_stage_score"] = stage_score
    new_features["healthy_athletic_low_pulse"] = (
        bmi_category.eq("healthy") & contextual_heart_category.eq("athletic_low")
    ).astype("int8")
    new_features["heavy_high_pulse"] = (
        bmi_category.isin(["overweight", "obesity"]) & high_pulse
    ).astype("int8")
    new_features["underweight_high_pulse"] = (
        bmi_category.eq("underweight") & high_pulse
    ).astype("int8")
    new_features["unexplained_low_pulse"] = contextual_heart_category.eq("unexplained_low").astype("int8")

    return new_features


FEATURE_GROUPS = [
    {
        "name": "fitness_contextualized_cardiometabolic_stage",
        "fn": add_fitness_contextualized_cardiometabolic_stage,
        "depends_on": [],
        "description": "Clinical-threshold BMI and resting-pulse staging with activity-contextualized low heart rate interactions.",
    }
]